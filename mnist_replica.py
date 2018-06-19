# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""

# CUDA_VISIBLE_DEVICES='' python mnist_replica.py --train_dir=/tmp/mnist_dist --data_dir=/export/fanlu/ --job_name=ps --task_index=0 --sync_replicas=True --ps_hosts=localhost:1121 --worker_hosts=localhost:1122,localhost:1123
# CUDA_VISIBLE_DEVICES=0 python mnist_replica.py --train_dir=/tmp/mnist_dist --data_dir=/Users/lonica/Downloads/data/ --job_name=worker --task_index=0 --sync_replicas=True --ps_hosts=localhost:1121 --worker_hosts=localhost:1122,localhost:1123 --num_epochs=10
# CUDA_VISIBLE_DEVICES=1 python mnist_replica.py --train_dir=/tmp/mnist_dist --data_dir=/Users/lonica/Downloads/data/ --job_name=worker --task_index=1 --sync_replicas=True --ps_hosts=localhost:1121 --worker_hosts=localhost:1122,localhost:1123 --num_epochs=10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tempfile
import time
import os
import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_string("train_dir", "/tmp/mnist-train",
                    "Directory for storing ckpt and summary")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 1,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("num_epochs", 2,
                     "Number of epochs")
flags.DEFINE_integer("hidden1", 128,
                     "Number of units in the 1st hidden layer of the NN")
flags.DEFINE_integer("hidden2", 128,
                     "Number of units in the 2nd hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
  "existing_servers", False, "Whether servers already exists. If True, "
                             "will use the worker hosts via their GRPC URLs (one client process "
                             "per worker host). Otherwise, will create an in-process TensorFlow "
                             "server.")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


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
      [image, label], batch_size=batch_size, num_threads=4,
      capacity=1000 + 3 * batch_size,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=1000)

    return images, sparse_labels


def main(unused_argv):
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # if FLAGS.download_only:
  #   sys.exit(0)
  print(FLAGS)
  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  # Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)

  cluster = tf.train.ClusterSpec({
    "ps": ps_spec,
    "worker": worker_spec})

  server = tf.train.Server(
    cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()
  else:
    is_chief = (FLAGS.task_index == 0)
    worker_device = "/job:worker/task:%d" % (FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU
    with tf.device(
        tf.train.replica_device_setter(
          worker_device=worker_device,
          cluster=cluster)):
      global_step = tf.contrib.framework.get_or_create_global_step()

      images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                              num_epochs=FLAGS.num_epochs)
      logits = mnist.inference(images, FLAGS.hidden1, FLAGS.hidden2)
      loss = mnist.loss(logits, labels)
      tf.summary.scalar(loss.op.name, loss)

      opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

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

      if is_chief:
        # Initial token and chief queue runners required by the sync_replicas mode
        chief_queue_runner = opt.get_chief_queue_runner()
        sync_init_op = opt.get_init_tokens_op()

      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      my_summary_op = tf.summary.merge_all()

      sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=FLAGS.train_dir,
        summary_op=None,
        init_op=init_op,
        recovery_wait_secs=1,
        global_step=global_step,
        save_model_secs=60,
        save_summaries_secs=60
      )

      sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
        # device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

      with sv.managed_session(master=server.target, config=sess_config) as sess:
        start_time = time.time()
        step = 1

        # if is_chief:
        #   if FLAGS.train_dir:
        #     sv.start_standard_services(sess)

        queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
        sv.start_queue_runners(sess, queue_runners)

        if is_chief:
          # Chief worker will start the chief queue runner and call the init op.
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(sync_init_op)
        try:
          while not sv.should_stop():
            if step>0 and step % 100 == 0:
              # Create the summary every 100 chief steps.
              _, loss_value, global_step_value, summ = sess.run([train_op, loss, global_step, my_summary_op])
              if is_chief:
                sv.summary_computed(sess, summ)
              duration = time.time() - start_time
              sec_per_batch = duration / (global_step_value * num_workers)
              format_str = ("After %d training steps (%d global steps), "
                            "loss on training batch is %g.  "
                            "(%.3f sec/batch)")
              print(format_str % (step, global_step_value,
                                  loss_value, sec_per_batch))
            else:
              # Train normally
              _, loss_value, global_step_value = sess.run([train_op, loss, global_step])
            step += 1
        except errors.OutOfRangeError:
          # OutOfRangeError is thrown when epoch limit per
          # tf.train.limit_epochs is reached.
          print('Caught OutOfRangeError. Stopping Training.')

      # sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
      # # q = sv.__getattribute__("_graph").get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      # print("Worker %d: Session initialization complete." % FLAGS.task_index)
      #
      # # Start the queue runners.
      # queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      # # print(len(q), len(queue_runners))
      # sv.start_queue_runners(sess, queue_runners)
      # print('Started %d queues for processing input data' % (len(queue_runners)))
      #
      # # if is_chief:
      # #   if FLAGS.train_dir:
      # #     sv.start_standard_services(sess)
      # #
      # # sv.start_queue_runners(sess)
      #
      # if is_chief:
      #   # Chief worker will start the chief queue runner and call the init op.
      #   sv.start_queue_runners(sess, [chief_queue_runner])
      #   sess.run(sync_init_op)
      #
      # # Perform training
      # time_begin = time.time()
      # print("Training begins @ %f" % time_begin)
      #
      # step = 0
      # start_time = time.time()
      # try:
      #   while not sv.should_stop():
      #     if step > 0 and step % 100 == 0:
      #       _, loss_value, global_step_value, summ = sess.run(
      #         [train_op, loss, global_step, my_summary_op])
      #       if is_chief:
      #         sv.summary_computed(sess, summ)
      #       duration = time.time() - start_time
      #       sec_per_batch = duration / (global_step_value * num_workers)
      #       format_str = ("After %d training steps (%d global steps), "
      #                     "loss on training batch is %g.  "
      #                     "(%.3f sec/batch)")
      #       print(format_str % (step, global_step_value,
      #                           loss_value, sec_per_batch))
      #     else:
      #       _, loss_value, global_step_value = sess.run([train_op, loss, global_step])
      #     if global_step_value >= 10000: break
      #     step += 1
      # except errors.OutOfRangeError:
      #     # OutOfRangeError is thrown when epoch limit per
      #     # tf.train.limit_epochs is reached.
      #     print('Caught OutOfRangeError. Stopping Training.')
      # print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, global_step_value))

          # time_end = time.time()
          # print("Training ends @ %f" % time_end)
          # training_time = time_end - time_begin
          # print("Training elapsed time: %f s" % training_time)
          #
          # # Validation feed
          # val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
          # val_xent = sess.run(cross_entropy, feed_dict=val_feed)
          # print("After %d training step(s), validation cross entropy = %g" %
          #       (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()
