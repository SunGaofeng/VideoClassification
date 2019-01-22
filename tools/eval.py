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
import logging
import argparse
import numpy as np
import paddle.fluid as fluid

import models

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='AttentionCluster',
                        help='name of model to train.')
    parser.add_argument('--config', type=str, default='configs/attention_cluster.txt',
                        help='path to config file of model')
    parser.add_argument('--use-cpu', action='store_true', default=False,
                        help='default use gpu, set this to use cpu')
    parser.add_argument('--weights', type=str, default=None,
                        help='weight path, None to use weights from Paddle.')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def eval(eval_model, args):
    eval_model.build_input(use_pyreader=False)
    eval_model.build_model()
    eval_feeds = eval_model.feeds()
    eval_outputs = eval_model.outputs()
    eval_reader = eval_model.reader()
    loss = eval_model.loss()

    place = fluid.CPUPlace() if args.use_cpu else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    weights = args.weights or eval_model.get_weights()
    def if_exist(var):
        return os.path.exists(os.path.join(weights, var.name))
    fluid.io.load_vars(exe, weights, predicate=if_exist)

    eval_feeder = fluid.DataFeeder(place=place, feed_list=eval_feeds)
    fetch_list = [loss.name] + [x.name for x in eval_outputs]

    def _eval_loop():
        epoch_loss = []
        epoch_period = []
        cur_time = time.time()
        for eval_iter, data in enumerate(eval_reader()):
            eval_outs = exe.run(fetch_list=fetch_list, feed=eval_feeder.feed(data))
            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            epoch_period.append(period)
            loss = np.mean(eval_outs[0])
            epoch_loss.append(loss)

            # metric here
            result = "example"
            if eval_iter % args.log_interval == 0:
                logger.info('[EVAL] Batch {:<3}, {}'.format(eval_iter, result))
        logger.info('[EVAL] eval finished.')

    # start eval loop
    _eval_loop()

    
if __name__ == "__main__":
    args = parse_args()
    
    eval_model = models.get_model(args.model_name, args.config, is_training=False)
    eval(eval_model, args)
