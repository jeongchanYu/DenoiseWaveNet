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

# make test data
test_source_signal, test_source_sample_rate = wav.read_wav(config['test_source_file'])
if config['test_target_file'] == '':
    test_target_file_exist = False
    test_target_signal = test_source_signal
    print("ISSUE: Test target file is not exist")
else:
    test_target_file_exist = True
    test_target_signal, test_target_sample_rate = wav.read_wav(config['test_target_file'])
test_target_signal = np.array(test_target_signal)
test_source_signal = np.array(test_source_signal)
test_size_of_source = test_source_signal.size
test_size_of_target = test_target_signal.size

if test_size_of_source != test_size_of_target:
    raise Exception("ERROR: Test input, output size mismatch")

test_mod = (current_size - (test_size_of_source % current_size)) % current_size
test_target_signal_padded = np.concatenate([np.zeros(previous_size), test_target_signal, np.zeros(future_size + test_mod)]).astype(default_float)
test_source_signal_padded = np.concatenate([np.zeros(previous_size), test_source_signal, np.zeros(future_size + test_mod)]).astype(default_float)

# make model
model = wavenet.DenoiseWaveNet(config['dilation'], config['relu_alpha'], config['default_float'])

# load model
if config['load_check_point_name'] != "":
    model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_path(), config['load_check_point_name']))

loss_object = tf.keras.losses.MeanAbsoluteError()
test_loss = tf.keras.metrics.Mean(name='test_loss')


# test function
@tf.function
def test_step(x, y):
    y_pred = model(x)
    if test_target_file_exist:
        loss = loss_object(y, y_pred)*2
    else:
        loss = loss_object(y_pred, y_pred)*2
    test_loss(loss)
    return y_pred

# test run
result = []
result_noise = []
i=0
sample = 0
start = time.time()
while sample < test_size_of_source:
    print("\rTest : training {}/{}".format(i + 1, math.ceil(test_size_of_source / current_size)), end='')
    y_pred = test_step(test_source_signal_padded[sample:sample + previous_size + current_size + future_size],
                           test_target_signal_padded[sample:sample + previous_size + current_size + future_size])
    y_pred = np.array(y_pred, dtype=default_float)
    b_pred = np.array(test_source_signal_padded[sample:sample + previous_size + current_size + future_size], dtype=default_float) - y_pred
    y_pred = y_pred.tolist()
    b_pred = b_pred.tolist()
    result.extend(y_pred[previous_size:previous_size + current_size])
    result_noise.extend(b_pred[previous_size:previous_size + current_size])
    sample += current_size
    i += 1
print(" | loss : {}".format(test_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))
test_loss.reset_states()

# save output
cf.createFolder("{}/test_result".format(cf.load_path()))
wav.write_wav(result[:len(result) - test_mod], "{}/test_result/result.wav".format(cf.load_path()), test_source_sample_rate)
wav.write_wav(result_noise[:len(result_noise) - test_mod], "{}/test_result/result_noise.wav".format(cf.load_path()), test_source_sample_rate)