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
from shifting_attention import ShiftingAttentionModel
from logistic_model import LogisticModel

__all__ = ["AttentionCluster"]

class AttentionCluster(ModelBase):
    def __init__(self, name, cfg, is_training=True):
        super(AttentionCluster, self).__init__(
                name, cfg, is_training)
        self.get_config()

    def get_config(self):
        self.feature_names = self.get_model_config('feature_names')
        self.feature_dims = self.get_model_config('feature_dims')
        self.feature_num = len(self.feature_dims)
        self.seg_num = self.get_model_config('seg_num')
        self.cluster_nums = self.get_model_config('cluster_nums')
        self.class_num = self.get_model_config('class_num')
        self.drop_rate = self.get_model_config('drop_rate')
        
        self.epochs = self.get_train_config('epoch')
        self.learning_rate = self.get_train_config('learning_rate')

    def build_input(self, use_pyreader):
        if use_pyreader:
            shapes = []
            for dim in self.feature_dims:
                shapes.append([-1, self.seg_num, dim])
            shapes.append([-1, self.class_num])
            self.py_reader = fluid.layers.py_reader(
                capacity = 640,
                shapes = shapes,
                lod_levels = [0, 0, 0],
                dtypes = ['float32', 'float32', 'float32'],
                # lod_levels = [0] * (self.feature_num + 1),
                # dtypes = ['float32'] * (self.feature_num + 1),
                name = 'train_py_reader' if self.is_training else 'test_py_reader',
                use_double_buffer = True)
            inputs = fluid.layers.read_file(self.py_reader)
            self.feature_input = inputs[:self.feature_num]
            self.label_input = inputs[-1]
        else:
            self.feature_input = []
            for name, dim in zip(self.feature_names, self.feature_dims):
                self.feature_input.append(fluid.layers.data(shape=[self.seg_num, dim], dtype='float32', name=name))
            self.label_input = fluid.layers.data(shape=[self.class_num], dtype='float32', name='label')
        
    def build_model(self):
        att_outs = []
        for i, (input_dim, cluster_num, feature) in enumerate(zip(self.feature_dims, self.cluster_nums, self.feature_input)):
            att = ShiftingAttentionModel(input_dim, self.seg_num, cluster_num, "satt{}".format(i))
            att_out = att.forward(feature)
            att_outs.append(att_out)
        out = fluid.layers.concat(att_outs, axis=1)

        if self.drop_rate > 0.:
          out = fluid.layers.dropout(out, self.drop_rate, is_test=(not self.is_training))

        fc1 = fluid.layers.fc(out, size=1024, act='tanh',
                              param_attr=ParamAttr(name="fc1.weights",
                                initializer=fluid.initializer.MSRA(uniform=False)),
                              bias_attr=ParamAttr(name="fc1.bias",
                                initializer=fluid.initializer.MSRA()))
        fc2 = fluid.layers.fc(fc1, size=4096, act='tanh',
                              param_attr=ParamAttr(name="fc2.weights",
                                initializer=fluid.initializer.MSRA(uniform=False)),
                              bias_attr=ParamAttr(name="fc2.bias",
                                initializer=fluid.initializer.MSRA()))

        aggregate_model = LogisticModel

        self.output, self.logit = aggregate_model().build_model(
                                    model_input = fc2,
                                    vocab_size = self.class_num,
                                    is_training = self.is_training)

        cost = fluid.layers.sigmoid_cross_entropy_with_logits(x = self.logit, label = self.label_input)
        cost = fluid.layers.reduce_sum(cost, dim = -1)
        self.loss_ = fluid.layers.mean(x = cost)

    def optimizer(self):
        return fluid.optimizer.AdamOptimizer(self.learning_rate)

    def loss(self):
        return self.loss_

    def outputs(self):
        return [self.output, self.logit]

    def feeds(self):
        return self.feature_input + [self.label_input]

    def weights_info(self):
        return ("imagenet_resnet50_fusebn", "http://paddlemodels.bj.bcebos.com/faster_rcnn/imagenet_resnet50_fusebn.tar.gz")
        
    def reader(self):
        import numpy as np
        def reader_():
            for i in range(10):
                yield [(np.random.random((1, self.seg_num, 1024)),
                        np.random.random((1, self.seg_num, 128)),
                        np.random.random((1, self.class_num)))]
        return reader_
    

