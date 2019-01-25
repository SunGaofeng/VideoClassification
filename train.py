#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import time
import shutil
import logging
import argparse
import numpy as np
import paddle.fluid as fluid

from tools.train_utils import train_with_pyreader, train_without_pyreader
import models

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument('--model-name', type=str, default='AttentionCluster',
                        help='name of model to train.')
    parser.add_argument('--config', type=str, default='configs/attention_cluster.txt',
                        help='path to config file of model')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='traing batch size per GPU. None to use config file setting.')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='learning rate use for training. None to use config file setting.')
    parser.add_argument('--use-cpu', action='store_true', default=False,
                        help='default use gpu, set this to use cpu')
    parser.add_argument('--no-parallel', action='store_true', default=False,
                        help='whether to use parallel executor')
    parser.add_argument('--no-use-pyreader', action='store_true', default=False,
                        help='whether to use pyreader')
    parser.add_argument('--no-memory-optimize', action='store_true', default=False,
                        help='whether to use memory optimize in train')
    parser.add_argument('--epoch-num', type=int, default=0,
                        help='epoch number, 0 for read from config file')
    parser.add_argument('--valid-interval', type=int, default=1,
                        help='validation epoch interval, 0 for no validation.')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='save checkpoints epoch interval.')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='directory name to save train snapshoot')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args

def train(train_model, valid_model, args):
    train_prog = fluid.Program()
    train_startup = fluid.Program()
    with fluid.program_guard(train_prog, train_startup):
        with fluid.unique_name.guard():
            train_model.build_input(not args.no_use_pyreader)
            train_model.build_model()
            # for the input, has the form [data1, data2,..., label], so train_feeds[-1] is label
            train_feeds = train_model.feeds()
            train_feeds[-1].persistable = True
            # for the output of classification model, has the form [pred]
            train_outputs = train_model.outputs()
            for output in train_outputs:
                output.persistable = True
            train_loss = train_model.loss()
            train_loss.persistable = True
            # outputs, loss, label should be fetched, so set persistable to be true
            optimizer = train_model.optimizer()
            optimizer.minimize(train_loss)
            train_reader = train_model.reader()
            train_metrics = train_model.metrics()
            train_pyreader = train_model.pyreader()

    if not args.no_memory_optimize:
        fluid.memory_optimize(train_prog)

    valid_prog = fluid.Program()
    valid_startup = fluid.Program()
    with fluid.program_guard(valid_prog, valid_startup):
        with fluid.unique_name.guard():
            valid_model.build_input(not args.no_use_pyreader)
            valid_model.build_model()
            valid_feeds = valid_model.feeds()
            valid_outputs = valid_model.outputs()
            valid_loss = valid_model.loss()
            valid_reader = valid_model.reader()
            valid_metrics = valid_model.metrics()
            valid_pyreader = valid_model.pyreader()

    place = fluid.CPUPlace() if args.use_cpu else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(valid_startup)

    pretrain_base = train_model.pretrain_base()
    if pretrain_base:
        train_model.load_pretrained_params(exe, pretrain_base, train_prog, place)

    if args.no_parallel:
        train_exe = exe
        valid_exe = exe
    else:
        train_exe = fluid.ParallelExecutor(use_cuda=(not args.use_cpu), loss_name=train_loss.name, main_program=train_prog)
        valid_exe = fluid.ParallelExecutor(use_cuda=(not args.use_cpu), share_vars_from=train_exe, main_program=valid_prog)

    train_fetch_list = [train_loss.name] + [x.name for x in train_outputs] + [train_feeds[-1].name]
    valid_fetch_list = [valid_loss.name] + [x.name for x in valid_outputs] + [valid_feeds[-1].name]

    epochs = args.epoch_num or train_model.epoch_num()

    if args.no_use_pyreader:
        train_feeder = fluid.DataFeeder(place=place, feed_list=train_feeds)
        valid_feeder = fluid.DataFeeder(place=place, feed_list=valid_feeds)
        train_without_pyreader(exe, train_prog, train_exe, train_reader, train_feeder, \
                               train_fetch_list, train_metrics, epochs = epochs, \
                               log_interval = args.log_interval, valid_interval = args.valid_interval, \
                               save_dir = args.save_dir, save_model_name = args.model_name, \
                               test_exe = valid_exe, test_reader = valid_reader, test_feeder = valid_feeder, \
                               test_fetch_list = valid_fetch_list, test_metrics = valid_metrics)
    else:
        train_pyreader.decorate_paddle_reader(train_reader)
        valid_pyreader.decorate_paddle_reader(valid_reader)
        train_with_pyreader(exe, train_prog, train_exe, train_pyreader, train_fetch_list, train_metrics, \
                            epochs = epochs, log_interval = args.log_interval, \
                            valid_interval = args.valid_interval, \
                            save_dir = args.save_dir, save_model_name = args.model_name, \
                            test_exe = valid_exe, test_pyreader = valid_pyreader, \
                            test_fetch_list = valid_fetch_list, test_metrics = valid_metrics)


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_model = models.get_model(args.model_name, args.config, mode='train')
    valid_model = models.get_model(args.model_name, args.config, mode='valid')
    train_model.merge_configs('train', vars(args))
    valid_model.merge_configs('valid', vars(args))
    train(train_model, valid_model, args)
