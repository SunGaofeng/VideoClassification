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
import logging
import argparse
import numpy as np
import paddle.fluid as fluid
import models

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='AttentionCluster',
                        help='name of model to train.')
    parser.add_argument('--config', type=str, default='configs/attention_cluster.txt',
                        help='path to config file of model')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='traing batch size per GPU. None to use config file setting.')
    parser.add_argument('--use-cpu', action='store_true', default=False,
                        help='default use gpu, set this to use cpu')
    parser.add_argument('--weights', type=str, default=None,
                        help='weight path, None to use weights from Paddle.')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def test(test_model, args):
    test_model.build_input(use_pyreader=False)
    test_model.build_model()
    test_feeds = test_model.feeds()
    test_outputs = test_model.outputs()
    test_reader = test_model.reader()
    test_metrics = test_model.metrics()
    loss = test_model.loss()

    place = fluid.CPUPlace() if args.use_cpu else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    weights = args.weights or test_model.get_weights()
    def if_exist(var):
        return os.path.exists(os.path.join(weights, var.name))
    fluid.io.load_vars(exe, weights, predicate=if_exist)

    test_feeder = fluid.DataFeeder(place=place, feed_list=test_feeds)
    fetch_list = [loss.name] + [x.name for x in test_outputs] + [test_feeds[-1].name]

    def _test_loop():
        epoch_period = []
        for test_iter, data in enumerate(test_reader()):
            cur_time = time.time()
            test_outs = exe.run(fetch_list=fetch_list, feed=test_feeder.feed(data))
            period = time.time() - cur_time
            epoch_period.append(period)
            loss = np.array(test_outs[0])
            pred = np.array(test_outs[1])
            label = np.array(test_outs[-1])
            test_metrics.accumulate(loss, pred, label)

            # metric here
            if test_iter % args.log_interval == 0:
                info_str = '[EVAL] Batch {}'.format(test_iter)
                test_metrics.calculate_and_log_out(loss, pred, label, info_str)
        test_metrics.finalize_and_log_out("[EVAL] eval finished. ")


    # start eval loop
    _test_loop()

    
if __name__ == "__main__":
    args = parse_args()
    
    test_model = models.get_model(args.model_name, args.config, mode='test')
    test_model.merge_configs('TEST', vars(args))
    test(test_model, args)
