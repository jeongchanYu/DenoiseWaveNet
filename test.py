import tensorflow as tf
import denoisewavenet as wavenet
import numpy as np
import wav
import json
import customfunction as cf
import time
import datetime
import math
import os

# prevent GPU overflow
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.InteractiveSession(config=gpu_config)

# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

default_float = config['default_float']
previous_size = int(config['previous_size'])
current_size = int(config['current_size'])
future_size = int(config['future_size'])
receptive_size = previous_size + current_size + future_size
load_check_point_name = config['load_check_point_name']
test_source_path = config['test_source_path']
test_target_path = config['test_target_path']

# training_source_path is path or file?
source_path_isdir = os.path.isdir(test_source_path)

# target path is exist?
if test_target_path == '':
    test_target_path_exist = False
    print("ISSUE: Test target file is not exist. Loss will be shown 0.")

else:
    test_target_path_exist = True
    if source_path_isdir:
        if not cf.compare_path_list(test_target_path, test_source_path, 'wav'):
            raise Exception("ERROR: Target and source file list is not same")

# make model
model = wavenet.DenoiseWaveNet(config['dilation'], config['relu_alpha'], config['default_float'])

# load model
if load_check_point_name != "":
    model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_path(), load_check_point_name))
else:
    raise Exception("ERROR: 'load_check_point_name' is empty. Test need check point.")

loss_object = tf.keras.losses.MeanAbsoluteError()
test_loss = tf.keras.metrics.Mean(name='test_loss')


# test function
@tf.function
def test_step(x, y):
    y_pred = model(x)
    if test_target_path_exist:
        loss = loss_object(y, y_pred)*2
    else:
        loss = loss_object(y_pred, y_pred)*2
    test_loss(loss)
    return y_pred


# make test data
if source_path_isdir:
    test_source_file_list = cf.read_path_list(test_source_path, "wav")
    if test_target_path_exist:
        test_target_file_list = cf.read_path_list(test_target_path, "wav")
    else:
        test_target_file_list = test_source_file_list
else:
    test_source_file_list = [test_source_path]
    if test_target_path_exist:
        test_target_file_list = [test_target_path]
    else:
        test_target_file_list = test_source_file_list


for i in range(len(test_source_file_list)):
    test_source_signal, test_source_sample_rate = wav.read_wav(test_source_file_list[i])
    test_target_signal, test_target_sample_rate = wav.read_wav(test_target_file_list[i])
    test_target_signal = np.array(test_target_signal)
    test_source_signal = np.array(test_source_signal)
    test_size_of_source = test_source_signal.size
    test_size_of_target = test_target_signal.size

    if test_size_of_source != test_size_of_target:
        raise Exception("ERROR: Test input, output size mismatch")

    test_mod = (current_size - (test_size_of_source % current_size)) % current_size
    test_target_signal_padded = np.concatenate([np.zeros(previous_size), test_target_signal, np.zeros(future_size + test_mod)]).astype(default_float)
    test_source_signal_padded = np.concatenate([np.zeros(previous_size), test_source_signal, np.zeros(future_size + test_mod)]).astype(default_float)

    # test run
    result = []
    result_noise = []
    frame=0
    sample = 0
    start = time.time()

    while sample < test_size_of_source:
        print("\rTest({}) : frame {}/{}".format(test_source_file_list[i], frame + 1, math.ceil(test_size_of_source / current_size)), end='')
        y_pred = test_step(test_source_signal_padded[sample:sample + previous_size + current_size + future_size],
                               test_target_signal_padded[sample:sample + previous_size + current_size + future_size])
        y_pred = np.array(y_pred, dtype=default_float)
        b_pred = np.array(test_source_signal_padded[sample:sample + previous_size + current_size + future_size], dtype=default_float) - y_pred
        y_pred = y_pred.tolist()
        b_pred = b_pred.tolist()
        result.extend(y_pred[previous_size:previous_size + current_size])
        result_noise.extend(b_pred[previous_size:previous_size + current_size])
        sample += current_size
        frame += 1
    print(" | loss : {}".format(test_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

    # save output
    result_path = "{}/test_result/{}/result/{}".format(cf.load_path(), load_check_point_name, os.path.dirname(test_source_file_list[i].replace(test_source_path, "")))
    result_noise_path = "{}/test_result/{}/result_noise/{}".format(cf.load_path(), load_check_point_name, os.path.dirname(test_source_file_list[i].replace(test_source_path, "")))
    file_name = os.path.basename(test_source_file_list[i])
    cf.createFolder(result_path)
    cf.createFolder(result_noise_path)
    wav.write_wav(result[:len(result) - test_mod], "{}/{}".format(result_path, file_name), test_source_sample_rate)
    wav.write_wav(result_noise[:len(result_noise) - test_mod], "{}/{}".format(result_noise_path, file_name), test_source_sample_rate)

    test_loss.reset_states()