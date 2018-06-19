import argparse
import sys
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist
import time

FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

# CUDA_VISIBLE_DEVICES='' python dist_mnist.py --job_name=ps --task_index=0 --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --sync_replicas=True
# CUDA_VISIBLE_DEVICES=0 python dist_mnist.py --job_name=worker --task_index=0 --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --batch_size=128 --sync_replicas=True
# CUDA_VISIBLE_DEVICES=1 python dist_mnist.py --job_name=worker --task_index=1 --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --batch_size=128 --sync_replicas=True

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([mnist.IMAGE_PIXELS])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.data_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
      [image, label], batch_size=batch_size, num_threads=2,
      capacity=1000 + 3 * batch_size,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=1000)

    return images, sparse_labels


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    is_chief = (FLAGS.task_index == 0)
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      # Input images and labels.
      images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                              num_epochs=FLAGS.num_epochs)

      # Build a Graph that computes predictions from the inference model.
      logits = mnist.inference(images,
                               FLAGS.hidden1,
                               FLAGS.hidden2)

      # Add to the Graph the loss calculation.
      loss = mnist.loss(logits, labels)
      tf.summary.scalar(loss.op.name, loss)
      global_step = tf.contrib.framework.get_or_create_global_step()

      # opt = tf.train.AdagradOptimizer(0.01)
      opt = tf.train.GradientDescentOptimizer(0.001)
      num_workers = len(worker_hosts)
      if FLAGS.sync_replicas:
        if FLAGS.replicas_to_aggregate is None:
          replicas_to_aggregate = num_workers
        else:
          replicas_to_aggregate = FLAGS.replicas_to_aggregate

        opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          name="mnist_sync_replicas")
      train_op = opt.minimize(loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.

    hooks = [tf.train.StopAtStepHook(last_step=100000)]
    if FLAGS.sync_replicas:
      hooks=[opt.make_session_run_hook(is_chief)]
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False)
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           checkpoint_dir=FLAGS.train_dir,
                                           hooks=hooks,
                                           save_summaries_steps=10,
                                           config=config) as mon_sess:
      start_time = time.time()
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        _, loss_value, step = mon_sess.run([train_op, loss, global_step])
        # Print an overview fairly often.
        if step % 100 == 0:
          duration = time.time() - start_time
          print('Step %d: loss = %.5f (%.3f sec)' % (step, loss_value,
                                                     duration))
          start_time = time.time()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
    "--ps_hosts",
    type=str,
    default="",
    help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
    "--worker_hosts",
    type=str,
    default="",
    help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
    "--job_name",
    type=str,
    default="",
    help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job"
  )
  parser.add_argument(
    "--replicas_to_aggregate",
    type=int,
    default=2,
    help="replicas_to_aggregate"
  )
  parser.add_argument(
    "--sync_replicas",
    type=bool,
    default=False,
    help="sync_replicas"
  )
  parser.add_argument(
    "--train_dir",
    type=str,
    default="/tmp/train_logs/",
    help="train_dir"
  )
  parser.add_argument(
    "--data_dir",
    type=str,
    default="/export/fanlu",
    help="data_dir"
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="batch_size"
  )
  parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="num_epochs"
  )
  parser.add_argument(
    "--hidden1",
    type=int,
    default=128,
    help="hidden1"
  )
  parser.add_argument(
    "--hidden2",
    type=int,
    default=128,
    help="hidden2"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
