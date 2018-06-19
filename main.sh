#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='' python dist_mnist.py --train_dir=/tmp/mnist_train/ --data_dir=/export/fanlu/mnist/ --job_name=ps --task_index=0 --ps_hosts=localhost:1121 --worker_hosts=localhost:1122,localhost:1123
CUDA_VISIBLE_DEVICES=0 python dist_mnist.py --train_dir=/tmp/mnist_train/ --data_dir=/export/fanlu/mnist/ --job_name=worker --task_index=0 --ps_hosts=localhost:1121 --worker_hosts=localhost:1122,localhost:1123
CUDA_VISIBLE_DEVICES=1 python dist_mnist.py --train_dir=/tmp/mnist_train/ --data_dir=/export/fanlu/mnist/ --job_name=worker --task_index=1 --ps_hosts=localhost:1121 --worker_hosts=localhost:1122,localhost:1123
