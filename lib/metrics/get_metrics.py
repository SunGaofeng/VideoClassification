from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging

import numpy as np
from metrics.youtube8m import eval_util as youtube8m_metrics
from metrics.kinetics import accuracy_metrics as kinetics_metrics
from metrics.nonlocal import nonlocal_test_metrics as nonlocal_test_metrics

logger = logging.getLogger(__name__)

class Metrics(object):
    def __init__(self, name, phase, **metrics_args):
        """Not implemented"""
        pass
    def calculate_and_log_out(self, loss, pred, label, info = ''):
        """Not implemented"""
        pass
    def accumulate(self, loss, pred, label, info = ''):
        """Not implemented"""
        pass
    def finalize_and_log_out(self, info = ''):
        """Not implemented"""
        pass
    def reset(self):
        """Not implemented"""
        pass

class Youtube8mMetrics(Metrics):
    def __init__(self, name, phase, **metrics_args):
        self.name = name
        self.phase = phase
        self.metrics_args = metrics_args
        self.num_classes = metrics_args['num_classes']
        self.topk = metrics_args['topk']
        self.calculator = youtube8m_metrics.EvaluationMetrics(self.num_classes, self.topk)
    def calculate_and_log_out(self, loss, pred, label, info = ''):
        loss = np.mean(np.array(loss))
        hit_at_one = youtube8m_metrics.calculate_hit_at_one(pred, label)
        perr = youtube8m_metrics.calculate_precision_at_equal_recall_rate(pred, label)
        gap = youtube8m_metrics.calculate_gap(pred, label)
        logger.info(info + ' , loss = {0}, Hit@1 = {1}, PERR = {2}, GAP = {3}'.format(\
                     '%.6f' % loss, '%.2f' % hit_at_one, '%.2f' % perr, '%.2f' % gap))

    def accumulate(self, loss, pred, label, info = ''):
        self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info = ''):
        epoch_info_dict = self.calculator.get()
        print(info + 'avg_hit_at_one: {0},\tavg_perr: {1},\tavg_loss :{2},\taps: {3},\tgap:{4}'\
                     .format(epoch_info_dict['avg_hit_at_one'], epoch_info_dict['avg_perr'], \
                             epoch_info_dict['avg_loss'], epoch_info_dict['aps'], epoch_info_dict['gap']))

    def reset(self):
        self.calculator.clear() #reset()


class Kinetics400Metrics(Metrics):
    def __init__(self, name, phase, **metrics_args):
        self.name = name
        self.phase = phase
        self.metrics_args = metrics_args
        self.calculator = kinetics_metrics.MetricsCalculator(name, phase.lower())

    def calculate_and_log_out(self, loss, pred, label, info = ''):
        if loss is not None:
            loss = np.mean(np.array(loss))
        else:
            loss = 0.
        acc1, acc5 = self.calculator.calculate_metrics(loss, pred, label)
        logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                       '%.2f' % acc1, '%.2f' % acc5))

    def accumulate(self, loss, pred, label, info = ''):
        self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info = ''):
        self.calculator.finalize_metrics()
        metrics_dict = self.calculator.get_computed_metrics()
        loss = metrics_dict['avg_loss']
        acc1 = metrics_dict['avg_acc1']
        acc5 = metrics_dict['avg_acc5']
        logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                       '%.2f' % acc1, '%.2f' % acc5))

    def reset(self):
        self.calculator.reset()

class NonlocalMetrics(Metrics):
    def __init__(self, name, phase, **metrics_args):
        self.name = name
        self.phase = phase
        self.metrics_args = metrics_args
        if phase == 'test':
            self.calculator = nonlocal_test_metrics.MetricsCalculator(name, phase.lower(), **metrics_args)
        else:
            self.calculator = kinetics_metrics.MetricsCalculator(name, phase.lower())

    def calculate_and_log_out(self, loss, pred, label, info = ''):
        if self.phase == 'test':
            pass
        else:
            if loss is not None:
                loss = np.mean(np.array(loss))
            else:
                loss = 0.
            acc1, acc5 = self.calculator.calculate_metrics(loss, pred, label)
            logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                                   '%.2f' % acc1, '%.2f' % acc5))

    def accumulate(self, loss, pred, label):
        self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info = ''):
        if self.phase == 'test':
            self.calculator.finalize_metrics()
        else:
            self.calculator.finalize_metrics()
            metrics_dict = self.calculator.get_computed_metrics()
            loss = metrics_dict['avg_loss']
            acc1 = metrics_dict['avg_acc1']
            acc5 = metrics_dict['avg_acc5']
            logger.info(info + '\tLoss: {},\ttop1_acc: {}, \ttop5_acc: {}'.format('%.6f' % loss, \
                           '%.2f' % acc1, '%.2f' % acc5))

    def reset(self):
        self.calculator.reset()


metrics_dict = {'NEXTVLAD': Youtube8mMetrics, \
                'LSTM': Youtube8mMetrics, \
                'ATTENTION_CLUSTER': Youtube8mMetrics, \
                'TSN': Kinetics400Metrics, \
                'TSM': Kinetics400Metrics, \
                'STNET': Kinetics400Metrics, \
                'NONLOCAL': NonlocalMetrics}


def get_metrics_model(name, phase, **metrics_args):
    assert name in metrics_dict.keys()
    metrics = metrics_dict[name](name, phase, **metrics_args)
    return metrics

