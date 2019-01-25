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
    parser.add_argument('--batch-size', type=int, default=1,
                        help='sample number in a batch for inference.')
    parser.add_argument('--filelist', type=str, default=None,
                        help='path to inferenece data file lists file.')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def infer(infer_model, args):
    infer_model.build_input(use_pyreader=False)
    infer_model.build_model()
    infer_feeds = infer_model.feeds()
    infer_outputs = infer_model.outputs()

    place = fluid.CPUPlace() if args.use_cpu else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # get infer reader
    if not args.filelist:
        logger.error("[INFER] --filelist unset.")
        return
    assert os.path.exists(args.filelist), "{} not exist.".format(args.filelist)
    # infer_reader = infer_model.reader(args.filelist)
    infer_reader = infer_model.reader()

    # if no weight files specified, download weights from paddle
    weights = args.weights or infer_model.get_weights()
    def if_exist(var):
        return os.path.exists(os.path.join(weights, var.name))
    fluid.io.load_vars(exe, weights, predicate=if_exist)

    infer_feeder = fluid.DataFeeder(place=place, feed_list=infer_feeds)
    fetch_list = [infer_feeds[-1].name] + [x.name for x in infer_outputs]

    def _infer_loop():
        periods = []
        cur_time = time.time()
        for infer_iter, data in enumerate(infer_reader()):
            infer_outs = exe.run(fetch_list=fetch_list, feed=infer_feeder.feed(data))
            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)

            video_id = np.array(infer_outs[0])
            pred = np.array(infer_outs[1])
            label = np.array(infer_outs[2])
            # print result here
            # print_label(video_id, label)
            print("niconiconi")
        logger.info('[INFER] infer finished. average time: {}'.format(np.mean(periods)))

    # start infer loop
    _infer_loop()

    
if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    
    infer_model = models.get_model(args.model_name, args.config, 'infer')
    infer(infer_model, args)

