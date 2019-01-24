
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

from datareader import get_reader
from  metrics import get_metrics
import utils
from utils import download, AttrDict

WEIGHT_DIR = os.path.expanduser("~/.paddle/weights")

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
        self.parser = ConfigParser()
        self.cfg = AttrDict()

    def parse(self):
        self.parser.read(self.cfg_file)
        for sec in self.parser.sections():
            sec_dict = AttrDict()
            for k, v in self.parser.items(sec):
                try:
                    v = eval(v)
                except:
                    pass
                setattr(sec_dict, k, v)
            setattr(self.cfg, sec.upper(), sec_dict)

    def merge_configs(self, sec, cfg_dict):
        sec_dict = getattr(self.cfg, sec.upper())
        for k, v in cfg_dict.items():
            if v is None:
                continue
            if hasattr(sec_dict, k):
                setattr(sec_dict, k, v)

    def get_config_from_sec(self, sec, item):
        if not hasattr(self.cfg, sec):
            return None

        sec_dict = getattr(self.cfg, sec)
        if not hasattr(sec_dict, item):
            return None

        return getattr(sec_dict, item)

    def get_configs(self):
        return self.cfg


class ModelBase(object):
    def __init__(self, name, cfg, mode='train'):
        assert mode in ['train', 'valid', 'test', 'infer'], \
                "Unknown mode type {}".format(mode)
        self.name = name
        self.is_training = (mode=='train')
        self.mode = mode
        self.py_reader = None

        # parse config
        assert os.path.exists(cfg), \
                "Config file {} not exists".format(cfg)
        self._config = ModelConfig(cfg)
        self._config.parse()
        self.cfg = self._config.get_configs()

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

    def create_dataset_args(self):
        "get model reader"
        raise NotImplementError(self, self.create_dataset_args)

    def reader(self):
        dataset_args = self.create_dataset_args()
        return get_reader(self.name.upper(), self.mode, **dataset_args)

    def create_metrics_args(self):
        "get model reader"
        raise NotImplementError(self, self.create_metrics_args)

    def metrics(self):
        metrics_args = self.create_metrics_args()
        return get_metrics(self.name.upper(), self.mode, **metrics_args)

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
        download(url, path)
        return path

    def pyreader(self):
        return self.py_reader

    def epoch_num(self):
        "get train epoch num"
        return self.cfg.TRAIN.epoch

    def pretrain_base(self):
        "get pretrain base model directory"
        if 'pretrain_base' in self.cfg.TRAIN.keys():
            return self.cfg.TRAIN.pretrain_base
        else:
            return None

    def get_pretrain_weights(self, logger=None):
        "get model weight file path, download weight from Paddle if not exist"
        path, url = self.pretrain_info()
        if not path:
            return None

        path = os.path.join(WEIGHT_DIR, path)
        if os.path.exists(path):
            return path

        if logger:
            logger.info("Download pretrain weights of {} from {}".format(self.name, url))
        utils.download(url, path)
        return path

    def merge_configs(self, sec, cfg_dict):
        return self._config.merge_configs(sec, cfg_dict)

    def get_config_from_sec(self, sec, item, default=None):
        cfg_item = self._config.get_config_from_sec(sec.upper(), item) or default
        return cfg_item


class ModelZoo(object):
    def __init__(self):
        self.model_zoo = {}

    def regist(self, name, model):
        assert model.__base__ == ModelBase, "Unknow model type {}".format(type(model))
        self.model_zoo[name] = model

    def get(self, name, cfg, mode='train'):
        for k, v in self.model_zoo.items():
            if k == name:
                return v(name, cfg, mode)
        raise ModelNotFoundError(name, self.model_zoo.keys())

# singleton model_zoo
model_zoo = ModelZoo()

def regist_model(name, model):
    model_zoo.regist(name, model)

def get_model(name, cfg, mode='train'):
    return model_zoo.get(name, cfg, mode)

if __name__ == "__main__":
    class TestModel(ModelBase):
        pass
    model_zoo.regist('test', TestModel)
    m = model_zoo.get('test', './config.txt')
    print(m.get_train_config('batch_size'))
    m.build_model()
    m = model_zoo.get('test2', './config.txt')
        
