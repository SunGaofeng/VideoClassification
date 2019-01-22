
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
try:
  from configparser import ConfigParser
except:
  from ConfigParser import ConfigParser

import utils

WEIGHT_DIR = "~/.paddle/weights"

class NotImplementError(Exception):
    "Error: model function not implement"
    def __init__(self, model, function):
        super(NotImplementError, self).__init__()
        self.model = model.__class__.__name__
        self.function = function.__name__

    def __str__(self):
        return "Function {}() is not implemented in model {}".format(
                self.function, self.model)

class ModelNotFoundError(Exception):
    "Error: model not found"
    def __init__(self, model_name, avail_models):
        super(ModelNotFoundError, self).__init__()
        self.model_name = model_name
        self.avail_models = avail_models

    def __str__(self):
        msg = "Model {} Not Found.\nAvailiable models:\n".format(self.model_name)
        for model in self.avail_models:
            msg += "  {}\n".format(model)
        return msg


class ModelConfig(object):
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.cfg = ConfigParser()
    def parse(self):
        self.cfg.read(self.cfg_file)
        for sec in self.cfg.sections():
            sec_dict = {}
            for k, v in self.cfg.items(sec):
                try:
                    v = eval(v)
                except:
                    pass
                sec_dict[k] = v
            setattr(self, sec.lower(), sec_dict)


class ModelBase(object):
    def __init__(self, name, cfg, is_training=True, split = 'train'):
        self.name = name
        self.is_training = is_training
        self.split = split
        self.py_reader = None

        # parse config
        assert os.path.exists(cfg), "Config file {} not exists".format(cfg)
        self._config = ModelConfig(cfg)
        self._config.parse()

    def build_model(self):
        "build model struct"
        raise NotImplementError(self, self.build_model)

    def build_input(self, use_pyreader):
        "build input Variable"
        raise NotImplementError(self, self.build_input)

    def optimizer(self):
        "get model optimizer"
        raise NotImplementError(self, self.optimizer)

    def outputs():
        "get output variable"
        raise notimplementerror(self, self.outputs)
        
    def loss(self):
        "get loss variable"
        raise notimplementerror(self, self.loss)

    def feeds(self):
        "get feed inputs list"
        raise NotImplementError(self, self.feeds)

    def reader(self):
        "get model reader"
        raise NotImplementError(self, self.reader)

    def weights_info(self):
        "get model weight default path and download url"
        raise NotImplementError(self, self.weights_info)

    def get_weights(self, logger=None):
        "get model weight file path, download weight from Paddle if not exist"
        path, url = self.weights_info()
        path = os.path.join(WEIGHT_DIR, path)
        if os.path.exists(path):
            return path

        if logger:
            logger.info("Download weights of {} from {}".format(self.name, url))
        utils.download(url, path)
        return path

    def pyreader(self):
        return self.py_reader

    def epoch_num(self):
        "get train epoch num"
        return self.get_train_config('epoch')

    def pretrain_base(self):
        "get pretrain base model directory"
        return self.get_train_config('pretrain_base')

    def _get_config(self, sec, item, default=None):
        try:
            sec_dict = getattr(self._config, sec)
            for k, v in sec_dict.items():
                if k == item:
                    return v
            return default
        except:
            return default

    def get_model_config(self, item, default=None):
        "Get config item in seciton MODEL"
        return self._get_config('model', item, default)

    def get_reader_config(self, item, default=None):
        "Get config item in seciton READER"
        return self._get_config('reader', item, default)

    def get_train_config(self, item, default=None):
        "Get config item in seciton TRAIN"
        return self._get_config('train', item, default)

    def get_sec_config(self, sec, item, default=None):
        "Get config item in seciton sec"
        return self._get_config(sec, item, default)

class ModelZoo(object):
    def __init__(self):
        self.model_zoo = {}

    def regist(self, name, model):
        assert model.__base__ == ModelBase, "Unknow model type {}".format(type(model))
        self.model_zoo[name] = model

    def get(self, name, cfg, is_training=True, split = 'train'):
        for k, v in self.model_zoo.items():
            if k == name:
                return v(name, cfg, is_training, split)
        raise ModelNotFoundError(name, self.model_zoo.keys())

# singleton model_zoo
model_zoo = ModelZoo()

def regist_model(name, model):
    model_zoo.regist(name, model)

def get_model(name, cfg, is_training=True, split = 'train'):
    return model_zoo.get(name, cfg, is_training, split)

if __name__ == "__main__":
    class TestModel(ModelBase):
        pass
    model_zoo.regist('test', TestModel)
    m = model_zoo.get('test', './config.txt')
    print(m.get_train_config('batch_size'))
    m.build_model()
    m = model_zoo.get('test2', './config.txt')
        
