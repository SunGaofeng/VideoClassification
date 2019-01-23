from feature_reader import FeatureReader
from kinetics_reader import KineticsReader
from nonlocal_reader import NonlocalReader

reader_dict = {'NEXTVLAD': FeatureReader, \
               'LSTM': FeatureReader, \
               'ATTENTION_CLUSTER': FeatureReader, \
               'TSN': KineticsReader, \
               'TSM': KineticsReader, \
               'STNET': KineticsReader, \
               'NONLOCAL': NonlocalReader}

def get_datareader(model_name, phase = 'train', **cfg):
    assert model_name in reader_dict.keys(), 'model_name {} not supported'.format(model_name)
    model_reader = reader_dict[model_name](model_name, phase, cfg)
    return model_reader.create_reader()

