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
from .lstm_attention import LSTMAttentionModel

__all__ = ["AttentionLSTM"]

class AttentionLSTM(ModelBase):
    def __init__(self, name, cfg, mode='train', args=None):
        super(AttentionLSTM, self).__init__(name, cfg, mode, args=args)
        self.get_config()

    def get_config(self):
        # get model configs
        self.feature_num = self.cfg.MODEL.feature_num
        self.feature_names = self.cfg.MODEL.feature_names
        self.feature_dims = self.cfg.MODEL.feature_dims
        self.class_num = self.cfg.MODEL.class_num
        self.embedding_size = self.cfg.MODEL.embedding_size
        self.lstm_size = self.cfg.MODEL.lstm_size

        # get mode configs
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size', 1)
        self.use_gpu = self.get_config_from_sec(self.mode, 'use_gpu', False)
        self.gpu_num = self.get_config_from_sec(self.mode, 'gpu_num', 1)

        if self.mode == 'train':
            self.learning_rate = self.get_config_from_sec('train', 'learning_rate', 1e-3)
            self.weight_decay = self.get_config_from_sec('train', 'weight_decay', 8e-4)
            self.num_samples = self.get_config_from_sec('train', 'num_samples', 5000000)
            self.decay_epochs = self.get_config_from_sec('train', 'decay_epochs', [5])
            self.decay_gamma = self.get_config_from_sec('train', 'decay_gamma', 0.1)

    def build_input(self, use_pyreader):
        if use_pyreader:
            assert self.mode != 'infer', \
                'pyreader is not recommendated when infer, please set use_pyreader to be false.'
            shapes = []
            for dim in self.feature_dims:
                shapes.append([-1, dim])
            shapes.append([-1, self.class_num]) # label
            self.py_reader = fluid.layers.py_reader(
                capacity = 1024,
                shapes = shapes,
                lod_levels = [1] * self.feature_num + [0],
                dtypes = ['float32'] * (self.feature_num + 1),
                name = 'train_py_reader' if self.is_training else 'test_py_reader',
                use_double_buffer = True)
            inputs = fluid.layers.read_file(self.py_reader)
            self.feature_input = inputs[:self.feature_num]
            self.label_input = inputs[-1]
        else:
            self.feature_input = []
            for name, dim in zip(self.feature_names, self.feature_dims):
                self.feature_input.append(fluid.layers.data(shape=[dim], lod_level=1, dtype='float32', name=name))
            if self.mode == 'infer':
                self.label_input = None
            else:
                self.label_input = fluid.layers.data(shape=[self.class_num], dtype='float32', name='label')
        
    def build_model(self):
        att_outs = []
        for i, (input_dim, feature) in enumerate(zip(self.feature_dims, self.feature_input)):
            att = LSTMAttentionModel(input_dim, self.embedding_size, self.lstm_size)
            att_out = att.forward(feature, is_training = (self.mode == 'train'))
            att_outs.append(att_out)
        out = fluid.layers.concat(att_outs, axis=1)

        fc1 = fluid.layers.fc(input=out, size=8192, act='relu', 
                              bias_attr=ParamAttr(regularizer=fluid.regularizer.L2Decay(0.0),
                                                  initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        fc2 = fluid.layers.fc(input=fc1, size=4096, act='tanh', 
                              bias_attr=ParamAttr(regularizer=fluid.regularizer.L2Decay(0.0),
                                                  initializer=fluid.initializer.NormalInitializer(scale=0.0)))

        self.logit = fluid.layers.fc(input=fc2, size=self.class_num, act=None, \
                              bias_attr=ParamAttr(regularizer=fluid.regularizer.L2Decay(0.0),
                                                  initializer=fluid.initializer.NormalInitializer(scale=0.0)))

        self.output = fluid.layers.sigmoid(self.logit)


    def optimizer(self):
        assert self.mode == 'train', "optimizer only can be get in train mode"
        values = [self.learning_rate * (self.decay_gamma ** i) for i in range(len(self.decay_epochs) + 1)]
        iter_per_epoch = self.num_samples / self.batch_size
        boundaries = [e * iter_per_epoch for e in self.decay_epochs]
        return fluid.optimizer.RMSProp(
                learning_rate=fluid.layers.piecewise_decay(values=values, boundaries=boundaries),
                centered=True,
                regularization=fluid.regularizer.L2Decay(self.weight_decay))

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        cost = fluid.layers.sigmoid_cross_entropy_with_logits(x=self.logit, label=self.label_input)
        cost = fluid.layers.reduce_sum(cost, dim=-1)
        sum_cost = fluid.layers.reduce_sum(cost)
        self.loss_ = fluid.layers.scale(sum_cost, scale=self.gpu_num, bias_after_scale=False)
        return self.loss_

    def outputs(self):
        return [self.output, self.logit]

    def feeds(self):
        return self.feature_input if self.mode == 'infer' else self.feature_input + [self.label_input]

    def weights_info(self):
        return (None, None)
    
    def create_dataset_args(self):
        dataset_args = {}
        dataset_args['num_classes'] = self.class_num
        dataset_args['list'] = self.get_config_from_sec(self.mode, 'filelist')

        if self.use_gpu and self.py_reader:
            dataset_args['batch_size'] = int(self.batch_size / self.gpu_num)
        else:
            dataset_args['batch_size'] = self.batch_size

        return dataset_args
        
    def create_metrics_args(self):
        metrics_args = {}
        metrics_args['num_classes'] = self.class_num
        metrics_args['topk'] = 20
        return metrics_args

