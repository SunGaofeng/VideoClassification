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

import paddle.fluid as fluid
from paddle.fluid import ParamAttr

from ..model import ModelBase
from tsn_res_model import TSN_ResNet

__all__ = ["TSN"]

class TSN(ModelBase):
    def __init__(self, name, cfg, mode='train'):
        super(TSN, self).__init__(
                name, cfg, mode = mode)
        self.get_config()

    def get_config(self):
        self.format = self.get_config_from_sec('model', 'format')
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.seg_num = self.get_config_from_sec('model', 'seg_num')
        self.seglen = self.get_config_from_sec('model', 'seglen')
        self.image_mean = self.get_config_from_sec('model', 'image_mean')
        self.image_std = self.get_config_from_sec('model', 'image_std')
        self.num_layers = self.get_config_from_sec('model', 'num_layers')

        self.num_epochs = self.get_config_from_sec('train', 'num_epochs')
        self.total_videos = self.get_config_from_sec('train', 'total_videos')
        self.base_learning_rate = self.get_config_from_sec('train', 'base_learning_rate')
        self.learning_rate_decay = self.get_config_from_sec('train', 'learning_rate_decay')
        self.l2_weight_decay = self.get_config_from_sec('train', 'l2_weight_decay')
        self.momentum = self.get_config_from_sec('train', 'momentum')

        self.short_size = self.get_config_from_sec(self.mode, 'short_size')
        self.target_size = self.get_config_from_sec(self.mode, 'target_size')
        self.num_reader_threads = self.get_config_from_sec(self.mode, 'num_reader_threads')
        self.buf_size = self.get_config_from_sec(self.mode, 'buf_size')
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size')
        self.filelist = self.get_config_from_sec(self.mode, 'list')


    def build_input(self, use_pyreader = True):
        image_shape = [3, self.target_size, self.target_size]
        image_shape[0] = image_shape[0] * self.seglen
        image_shape = [self.seg_num] + image_shape
        self.use_pyreader = use_pyreader
        if use_pyreader:
            py_reader = fluid.layers.py_reader(
                           capacity = 100,
                           shapes = [[-1] + image_shape, [-1] + [1]],
                           dtypes = ['float32', 'int64'],
                           name = 'train_py_reader' if self.is_training else 'test_py_reader',
                           use_double_buffer = True)
            image, label = fluid.layers.read_file(py_reader)
            self.py_reader = py_reader
        else:
            image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        self.feature_input = [image]
        self.label_input = label

    def create_model_args(self):
        cfg = {}
        cfg['layers'] = self.num_layers
        cfg['class_dim'] = self.num_classes
        cfg['seg_num'] = self.seg_num
        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        videomodel = TSN_ResNet(layers=cfg['layers'], seg_num=cfg['seg_num'])
        out = videomodel.net(input = self.feature_input[0], class_dim = cfg['class_dim'])
        self.network_outputs = [out]


    def optimizer(self):
        epoch_points = [self.num_epochs / 3, self.num_epochs * 2 / 3]
        total_videos = self.total_videos
        step = int(total_videos / self.batch_size + 1)
        bd = [e * step for e in epoch_points]
        base_lr = self.base_learning_rate
        lr_decay = self.learning_rate_decay
        lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
        l2_weight_decay = self.l2_weight_decay
        momentum = self.momentum
        optimizer = fluid.optimizer.Momentum(
                       learning_rate=fluid.layers.piecewise_decay(
                                boundaries=bd, values=lr),
                       momentum = momentum,
                       regularization=fluid.regularizer.L2Decay(l2_weight_decay))

        return optimizer


    def loss(self):
        cost = fluid.layers.cross_entropy(input=self.network_outputs[0], \
                           label=self.label_input, ignore_index=-1)
        self.loss_ = fluid.layers.mean(x=cost)
        return self.loss_


    def outputs(self):
        return self.network_outputs


    def feeds(self):
        return self.feature_input + [self.label_input]


    def weights_info(self):
        return None


    def create_dataset_args(self):
        cfg = {}
        cfg['format'] = self.format
        cfg['num_classes'] = self.num_classes
        cfg['seg_num'] = self.seg_num
        cfg['seglen'] = self.seglen
        cfg['short_size'] = self.short_size
        cfg['target_size'] = self.target_size
        cfg['num_reader_threads'] = self.num_reader_threads
        cfg['buf_size'] = self.buf_size
        cfg['image_mean'] = self.image_mean
        cfg['image_std'] = self.image_std
        cfg['list'] = self.filelist
        cfg['batch_size'] = self.batch_size

        return cfg


    def create_metrics_args(self):
        cfg = {}
        return cfg


