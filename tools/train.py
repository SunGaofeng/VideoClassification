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
import time
import shutil
import logging
import argparse
import numpy as np
import paddle.fluid as fluid

import models

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

def save_model(exe, program, args, postfix=None):
    model_path = os.path.join(args.save_dir, args.model_name + postfix)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    fluid.io.save_persistables(exe, model_path, main_program=program)

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

    if not args.no_memory_optimize:
        # fluid.memory_optimize(fluid.default_main_program())
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

    place = fluid.CPUPlace() if args.use_cpu else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(valid_startup)

    pretrain_base = train_model.pretrain_base()
    if pretrain_base:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrain_base, var.name))
        fluid.io.load_vars(exe, pretrain_base, prediccate=if_exist, main_program=train_startup)

    if args.no_parallel:
        train_exe = exe
        valid_exe = exe
    else:
        train_exe = fluid.ParallelExecutor(use_cuda=(not args.use_cpu), loss_name=train_loss.name, main_program=train_prog)
        valid_exe = fluid.ParallelExecutor(use_cuda=(not args.use_cpu), share_vars_from=train_exe, main_program=train_prog)

    train_fetch_list = [train_loss.name] + [x.name for x in train_outputs] + [train_feeds[-1].name]
    valid_fetch_list = [valid_loss.name] + [x.name for x in valid_outputs] + [valid_feeds[-1].name]

    epochs = args.epoch_num or train_model.epoch_num()

    if args.no_use_pyreader:
        train_feeder = fluid.DataFeeder(place=place, feed_list=train_feeds)
        valid_feeder = fluid.DataFeeder(place=place, feed_list=valid_feeds)

        def valid_without_pyreader():
            epoch_period = []
            for valid_iter, data in enumerate(valid_reader()):
                valid_outs = valid_exe.run(valid_fetch_list, feed=valid_feeder.feed(data))
                if valid_iter % args.log_interval == 0:
                    # get eval string here
                    valid_iter_result = "hahaha"
                    logger.info("[TEST] iter {:<6}: {}".format(valid_iter, valid_iter_result))
            valid_iter_result = "hahaha finish"
            logger.info("[TEST] Finish: {}".format(valid_iter_result))

        def train_without_pyreader():
            cur_time = time.time()
            for epoch in range(epochs):
                epoch_periods = []
                for train_iter, data in enumerate(train_reader()):
                    train_outs = train_exe.run(train_fetch_list, feed=train_feeder.feed(data))
                    prev_time = cur_time
                    cur_time = time.time()
                    period = cur_time - prev_time
                    epoch_periods.append(period)
                    loss = np.mean(train_outs[0])
                    if train_iter % args.log_interval == 0:
                        # eval here
                        train_eval_result = "hohoho"
                        logger.info("[TRAIN] Epoch {:<3} Batch {:<6}, loss: {:<12}, {}".format(epoch, train_iter, loss, train_eval_result))
                    train_iter += 1
                logger.info('[TRAIN] Epoch {} training finished, average time: {}'.format(epoch, np.mean(epoch_periods)))
                epoch_periods = []
                if (epoch + 1) % args.save_interval == 0:
                    save_model(exe, train_prog, args, "_epoch{}".format(epoch))
                if (epoch + 1) % args.valid_interval == 0:
                    valid_without_pyreader()

        # start to train
        train_without_pyreader()
    else:
        def valid_with_pyreader():
            valid_pyreader = valid_model.pyreader()
            if not valid_pyreader:
                logger.error("[TEST] get pyreader failed.")
            valid_pyreader.decorate_paddle_reader(valid_reader)
            valid_pyreader.start()
            valid_metrics.reset()
            valid_iter = 0
            try:
                while True:
                    valid_outs = valid_exe.run(fetch_list=valid_fetch_list)
                    loss = np.array(valid_outs[0])
                    pred = np.array(valid_outs[1])
                    label = np.array(valid_outs[-1])
                    valid_metrics.accumulate(loss, pred, label)
                    # do eval here
                    valid_iter += 1
                    #if valid_iter % arg.log_interval == 0:
                    #    # get eval string here
                    #    valid_iter_result = "hahaha"
                    #    logger.info("[TEST] iter {:<6}: {}".format(valid_iter, valid_iter_result))
            except fluid.core.EOFException:
                # get eval string here
                #valid_iter_result = "hahaha finish"
                #logger.info("[TEST] Finish: {}".format(valid_iter_result))
                valid_metrics.finalize_and_log_out("[TEST] Finish")
            finally:
                valid_pyreader.reset()

        def train_with_pyreader():
            train_pyreader = train_model.pyreader()
            if not train_pyreader:
                logger.error("[TRAIN] get pyreader failed.")
            train_pyreader.decorate_paddle_reader(train_reader)

            for epoch in range(epochs):
                train_pyreader.start()
                train_metrics.reset()
                try:
                    train_iter = 0
                    epoch_periods = []
                    cur_time = time.time()
                    while True:
                        train_outs = train_exe.run(fetch_list=train_fetch_list)
                        prev_time = cur_time
                        cur_time = time.time()
                        period = cur_time - prev_time
                        epoch_periods.append(period)
                        #loss = np.mean(train_outs[0])
                        loss = np.array(train_outs[0])
                        pred = np.array(train_outs[1])
                        label = np.array(train_outs[-1])
                        if train_iter % args.log_interval == 0:
                            # eval here
                            #train_eval_result = "hohoho"
                            train_metrics.calculate_and_log_out(loss, pred, label, \
                                        info = '[TRAIN] Epoch {}, iter {} '.format(epoch, train_iter))
                        train_iter += 1
                except fluid.core.EOFException:
                    # eval here
                    #train_eval_result = "hohoho finish"
                    logger.info('[TRAIN] Epoch {} training finished, average time: {}'.format(epoch, np.mean(epoch_periods)))
                    if (epoch + 1) % args.save_interval == 0:
                        save_model(exe, train_prog, args, "_epoch{}".format(epoch))
                    if (epoch + 1) % args.valid_interval == 0:
                        valid_with_pyreader()
                finally:
                    epoch_period = []
                    train_pyreader.reset()

        # start to train
        train_with_pyreader()

    save_model(exe, train_prog, args, "_final")
    logger.info('[TRAIN] train total {} epoch finished.'.format(epochs))

    
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
