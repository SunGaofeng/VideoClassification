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
from lstm_attention import LSTMAttentionModel

__all__ = ["AttentionLSTM"]

class AttentionLSTM(ModelBase):
    def __init__(self, name, cfg, mode='train'):
        super(AttentionLSTM, self).__init__(name, cfg, mode)

    def build_input(self, use_pyreader):
        if use_pyreader:
            shapes = []
            for dim in self.cfg.MODEL.feature_dims:
                shapes.append([-1, dim])
            if self.mode == 'infer':
                shapes.append([-1, 1]) # video id
                self.py_reader = fluid.layers.py_reader(
                    capacity = 1024,
                    shapes = shapes,
                    lod_levels = [1] * self.cfg.MODEL.feature_num + [0],
                    dtypes = ['float32'] * self.cfg.MODEL.feature_num + ['int32'],
                    name = 'train_py_reader' if self.is_training else 'test_py_reader',
                    use_double_buffer = True)
                inputs = fluid.layers.read_file(self.py_reader)
                self.feature_input = inputs[:self.cfg.MODEL.feature_num]
                self.video_id = inputs[-1]
            else:
                shapes.append([-1, self.cfg.MODEL.class_num]) # label
                self.py_reader = fluid.layers.py_reader(
                    capacity = 1024,
                    shapes = shapes,
                    lod_levels = [1] * self.cfg.MODEL.feature_num + [0],
                    dtypes = ['float32'] * (self.cfg.MODEL.feature_num + 1),
                    name = 'train_py_reader' if self.is_training else 'test_py_reader',
                    use_double_buffer = True)
                inputs = fluid.layers.read_file(self.py_reader)
                self.feature_input = inputs[:self.cfg.MODEL.feature_num]
                self.label_input = inputs[-1]
        else:
            self.feature_input = []
            for name, dim in zip(self.cfg.MODEL.feature_names, self.cfg.MODEL.feature_dims):
                self.feature_input.append(fluid.layers.data(shape=[dim], lod_level=1, dtype='float32', name=name))
            self.label_input = fluid.layers.data(shape=[self.cfg.MODEL.class_num], dtype='float32', name='label')
            self.video_id = fluid.layers.data(shape=[1], dtype='int32', name='video_id')
        
    def build_model(self):
        att_outs = []
        for i, (input_dim, feature) in enumerate(zip(self.cfg.MODEL.feature_dims, self.feature_input)):
            att = LSTMAttentionModel(input_dim, self.cfg.MODEL.embedding_size, self.cfg.MODEL.lstm_size)
            att_out = att.forward(feature)
            att_outs.append(att_out)
        out = fluid.layers.concat(att_outs, axis=1)

        fc1 = fluid.layers.fc(input=out, size=8192, act='relu', 
                              bias_attr=ParamAttr(regularizer=fluid.regularizer.L2Decay(0.0),
                                                  initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        fc2 = fluid.layers.fc(input=fc1, size=4096, act='tanh', 
                              bias_attr=ParamAttr(regularizer=fluid.regularizer.L2Decay(0.0),
                                                  initializer=fluid.initializer.NormalInitializer(scale=0.0)))

        self.logit = fluid.layers.fc(input=fc2, size=self.cfg.MODEL.class_num, act=None, \
                              bias_attr=ParamAttr(regularizer=fluid.regularizer.L2Decay(0.0),
                                                  initializer=fluid.initializer.NormalInitializer(scale=0.0)))

        self.output = fluid.layers.sigmoid(self.logit)


    def optimizer(self):
        assert self.mode == 'train', "optimizer only can be get in train mode"
        decay_epochs = self.cfg.TRAIN.decay_epochs
        decay_gamma = self.cfg.TRAIN.decay_gamma
        values = [self.cfg.TRAIN.learning_rate * (decay_gamma ** i) for i in range(len(decay_epochs) + 1)]
        iter_per_epoch = self.cfg.TRAIN.num_samples / (self.cfg.TRAIN.batch_size * self.cfg.TRAIN.gpu_num)
        boundaries = [e * iter_per_epoch for e in decay_epochs]
        return fluid.optimizer.RMSProp(
                learning_rate=fluid.layers.piecewise_decay(values=values, boundaries=boundaries),
                centered=True,
                regularization=fluid.regularizer.L2Decay(self.cfg.TRAIN.weight_decay))

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        cost = fluid.layers.sigmoid_cross_entropy_with_logits(x = self.logit, label = self.label_input)
        cost = fluid.layers.reduce_sum(cost, dim = -1)
        self.loss_ = fluid.layers.mean(x = cost)
        return self.loss_

    def outputs(self):
        return [self.output, self.logit]

    def feeds(self):
        if self.mode == 'infer':
            return self.feature_input + [self.video_id]
        else:
            return self.feature_input + [self.label_input]

    def weights_info(self):
        return (None, None)
    
    def create_dataset_args(self):
        dataset_args = {}
        dataset_args['num_classes'] = self.cfg.MODEL.class_num
        dataset_args['batch_size'] = self.get_config_from_sec(self.mode, 'batch_size')
        dataset_args['list'] = self.get_config_from_sec(self.mode, 'filelist')
        # dataset_args['eigen_file'] = self.eigen_file
        return dataset_args
        
    def create_metrics_args(self):
        metrics_args = {}
        metrics_args['num_classes'] = self.cfg.MODEL.class_num
        metrics_args['topk'] = 20
        return metrics_args

