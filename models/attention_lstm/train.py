import os
import sys
import time
import numpy as np
import shutil
import paddle
import paddle.fluid as fluid
import reader
import eval_util
from lstm_attnet import LSTM_AttNet

import argparse
import functools
from paddle.fluid.framework import Parameter
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,    128,            "Minibatch size.")
add_arg('learning_rate',    float,  1e-3,           "Learning rate.")
add_arg('weight_decay',     float,  8e-4,           "Weight decay.")
add_arg('num_epochs',       int,    10,             "Number of epochs.")
add_arg('class_dim',        int,    3862,           "Number of class.")
add_arg('lstm_size',        int,    1024,           "Size of lstm hidden state.")
add_arg('embedding_size',   int,    512,            "Size of fc embedding.")
add_arg('model_save_dir',   str,    "output",       "Model save directory.")
add_arg('trainlist',        str,    "train.list",   "Train data list.")
add_arg('testlist',         str,    "test.list",    "Test data list.")
# yapf: enable

rgb_shape = [1024]
audio_shape = [128]
gpu_nums = 8

# set input data
def build_program(args, is_train):
    py_reader = fluid.layers.py_reader(
        capacity = 2048,
        shapes = [[-1] + rgb_shape, [-1] + audio_shape, [-1] + [args.class_dim]],
        lod_levels = [1, 1, 0],
        dtypes = ['float32', 'float32', 'float32'],
        name = 'train_py_reader' if is_train else 'test_py_reader',
        use_double_buffer = True)
        
    rgb, audio, label = fluid.layers.read_file(py_reader)

    # build model
    model = LSTM_AttNet(embedding_size=args.embedding_size, lstm_size=args.lstm_size)
    out = model.net_twostream(rgb=rgb, audio=audio, class_dim=args.class_dim)

    logits = out['logits']
    predictions = out['predictions']

    # calculate loss
    cost = fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=label)
    cost = fluid.layers.reduce_sum(cost, dim=-1)
    avg_cost = fluid.layers.reduce_sum(cost)
    avg_cost = fluid.layers.scale(avg_cost, scale=gpu_nums, bias_after_scale=False)

    return py_reader, avg_cost, rgb, audio, label, predictions 

def train(args):
    model_save_dir = args.model_save_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    trainlist = args.trainlist
    testlist = args.testlist
    class_dim = args.class_dim

    train_prog = fluid.Program()
    train_startup = fluid.Program()

    # py reader
    with fluid.program_guard(train_prog, train_startup):
        with fluid.unique_name.guard():
            train_py_reader, train_loss, train_rgb, train_audio, train_label, train_predictions = build_program(args,True)
    
        #optimizer = fluid.optimizer.RMSProp(
        #        learning_rate=args.learning_rate,
        #        centered=True,
        #        regularization=fluid.regularizer.L2Decay(args.weight_decay))
        optimizer = fluid.optimizer.RMSProp(
                learning_rate=fluid.layers.piecewise_decay(
                    values=[args.learning_rate, args.learning_rate / 10], boundaries=[5 * 5000000 / 1024]),
                centered=True,
                regularization=fluid.regularizer.L2Decay(args.weight_decay))
        opts = optimizer.minimize(train_loss)
   
    fluid.memory_optimize(fluid.default_main_program())

    #
    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        with fluid.unique_name.guard():
            test_py_reader, test_loss, test_rgb, test_audio, test_label, test_predictions = build_program(args, False)
    test_prog = test_prog.clone(for_test=True)

    # initialize
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(test_startup)

    ## load model
    #def is_parameter(var):
    #    if isinstance(var, Parameter):
    #        return isinstance(var, Parameter)

    #pretrained_model = "output_1e-3/5/"
    #if pretrained_model is not None:
    #    vars = filter(is_parameter, test_prog.list_vars())
    #    fluid.io.load_vars(exe, pretrained_model, vars=vars)
   
    # setup executor
    train_exe = fluid.ParallelExecutor(main_program=train_prog, 
            use_cuda=True, loss_name=train_loss.name)
    test_exe = fluid.ParallelExecutor(main_program=test_prog,
            use_cuda=True, share_vars_from=train_exe)

    # setup reader 
    train_reader = reader.train(trainlist, batch_size, class_dim)
    test_reader = reader.test(testlist, batch_size, class_dim)
    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)

    # set fetch list
    train_fetch_list = [train_loss.name, train_label.name, train_predictions.name]
    test_fetch_list = [test_loss.name, test_label.name, test_predictions.name]

    # save model
    def save_model(postfix):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=train_prog)

    # test
    def test_with_py_reader_2(test_exe, test_fetch_list, test_py_reader, eval_metrics):
        test_py_reader.start()
        test_epoch_loss = []
        test_iter_id = 0
        examples_processed = 0
        eval_metrics.clear()
        try:
            while True:
                loss, test_label_eval, test_predictions_eval = test_exe.run(fetch_list = test_fetch_list)
                loss = np.array(loss) / batch_size / gpu_nums
                examples_processed += test_label_eval.shape[0]
                print ('=== processed %d examples ...' % examples_processed)
                iteration_info_dict = eval_metrics.accumulate(test_predictions_eval, test_label_eval, loss)
                test_iter_id += 1
                if test_iter_id % 10 == 0:
                    print('Test iter {0}'.format(test_iter_id))
        except fluid.core.EOFException:
            epoch_info_dict = eval_metrics.get()
            print('Test finished, avg_hit_at_one: {0},\tavg_perr: {1},\tavg_loss :{2},\tgap:{3},\ttest samples:{4}'\
                     .format(epoch_info_dict['avg_hit_at_one'], epoch_info_dict['avg_perr'], \
                             epoch_info_dict['avg_loss'], epoch_info_dict['gap'], examples_processed))
            test_py_reader.reset()

        return epoch_info_dict

    # train
    for pass_id in range(num_epochs):
        train_py_reader.start()
        try:
            iter_id = 0
            while True:
                t1 = time.time()
                loss, train_label_eval, train_predictions_eval = train_exe.run(fetch_list=train_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(np.array(loss) / batch_size / gpu_nums)
                if iter_id % 10 == 0:
                    hit_at_one = eval_util.calculate_hit_at_one(train_predictions_eval, train_label_eval)
                    perr = eval_util.calculate_precision_at_equal_recall_rate(train_predictions_eval, train_label_eval)
                    gap = eval_util.calculate_gap(train_predictions_eval, train_label_eval)

                    print("[TRAIN] Pass: {0}\ttrainbatch: {1}\tloss: {2}\tHit@1: {3}\tPERR: {4}\tGAP: {5}\trun time: {6}"
                          .format(pass_id, iter_id, '%.6f' % loss, '%.2f' % hit_at_one, '%.2f' % perr, '%.2f' % gap, "%2.2f sec" % period))
                    sys.stdout.flush()
                iter_id += 1
        except fluid.core.EOFException:
            print('End of Train Epoch ', pass_id)
            train_py_reader.reset()

        # save model
        save_model(str(pass_id))

        # test
        eval_metrics = eval_util.EvaluationMetrics(class_dim, 20)

        test_with_py_reader_2(test_exe, test_fetch_list, test_py_reader, eval_metrics)

    save_model('model_final')

def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)

if __name__ == '__main__':
    main()

