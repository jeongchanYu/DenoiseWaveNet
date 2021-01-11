import tensorflow as tf
import denoisewavenet as wavenet
import numpy as np
import wav
import json
import customfunction as cf
import time
import datetime
import math


# prevent GPU overflow
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.InteractiveSession(config=gpu_config)


# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

previous_size = int(config['previous_size'])
current_size = int(config['current_size'])
future_size = int(config['future_size'])
receptive_size = previous_size + current_size + future_size
shift_size = int(config['shift_size'])
batch_size = int(config['batch_size'])
epochs = int(config['epochs'])

# read train data file
target_signal, target_sample_rate = wav.read_wav(config['training_target_file'])
source_signal, source_sample_rate = wav.read_wav(config['training_source_file'])
target_signal = np.array(target_signal)
source_signal = np.array(source_signal)
size_of_target = target_signal.size
size_of_source = source_signal.size


# source & target file incorrect
if size_of_source != size_of_target:
    raise Exception("ERROR: Train input, output size mismatch")
if size_of_source < current_size:
    raise Exception("ERROR: Input file length is too small")
if shift_size <= 0:
    raise Exception("ERROR: Shift size is smaller or same with 0")

# padding
mod = (shift_size - (size_of_source % shift_size)) % shift_size
target_signal_padded = np.concatenate([np.zeros(previous_size), target_signal, np.zeros(future_size+mod)]).astype('float32')
source_signal_padded = np.concatenate([np.zeros(previous_size), source_signal, np.zeros(future_size+mod)]).astype('float32')
if shift_size < current_size:
    dif = current_size-shift_size
    target_signal_padded = np.concatenate([target_signal_padded, np.zeros(dif)]).astype('float32')
    source_signal_padded = np.concatenate([source_signal_padded, np.zeros(dif)]).astype('float32')

# make dataset
x_signal, y_signal = [], []
number_of_frames = math.ceil(size_of_source/shift_size)
for i in range(number_of_frames):
    x_signal.append(source_signal_padded[i*shift_size:(i*shift_size) + receptive_size])
    y_signal.append(target_signal_padded[i*shift_size:(i*shift_size) + receptive_size])
train_dataset = tf.data.Dataset.from_tensor_slices((x_signal, y_signal)).batch(batch_size)

# make test data
if config['training_source_file'] != config['test_source_file']:
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
    if test_source_sample_rate != source_sample_rate:
        raise Exception("ERROR: Train, test sample rate mismatch")

    test_mod = (current_size - (test_size_of_source % current_size)) % current_size
    test_target_signal_padded = np.concatenate([np.zeros(previous_size), test_target_signal, np.zeros(future_size + test_mod)]).astype('float32')
    test_source_signal_padded = np.concatenate([np.zeros(previous_size), test_source_signal, np.zeros(future_size + test_mod)]).astype('float32')
else:
    test_target_file_exist = True
    test_target_signal = target_signal
    test_source_signal = source_signal
    test_target_sample_rate = target_sample_rate
    test_source_sample_rate = source_sample_rate
    test_size_of_target = size_of_target
    test_size_of_source = size_of_source
    test_mod = (current_size - (size_of_source % current_size)) % current_size
    test_target_signal_padded = target_signal_padded
    test_source_signal_padded = source_signal_padded

# make model
model = wavenet.DenoiseWaveNet(config['dilation'], config['relu_alpha'], config['default_float'])

# load model
if config['load_check_point_name'] != "":
    model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_path(), config['load_check_point_name']))

loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
train_loss =tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

# train function
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_object(y, y_pred, 2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

# test function
@tf.function
def test_step(x, y):
    y_pred = model(x)
    if test_target_file_exist:
        loss = loss_object(y, y_pred, 2)
    else:
        loss = loss_object(y_pred, y_pred, 2)
    test_loss(loss)
    return y_pred

# train and test run
for epoch in range(epochs):
    i = 0
    start = time.time()
    for x, y in train_dataset:
        print("\rTrain : epoch {}/{}, frame {}/{}".format(epoch + 1, epochs, i + 1, math.ceil(number_of_frames / batch_size)), end='')
        train_step(x, y)
        i += 1
    print(" | loss : {}".format(train_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

    result = []
    result_noise = []
    i = 0
    sample = 0
    start = time.time()
    while sample < test_size_of_source:
        print("\rTest : epoch {}/{}, frame {}/{}".format(epoch + 1, epochs, i + 1, math.ceil(test_size_of_source/current_size)), end='')
        y_pred = test_step(test_source_signal_padded[sample:sample+previous_size+current_size+future_size],
                           test_target_signal_padded[sample:sample+previous_size+current_size+future_size])
        y_pred = np.array(y_pred, dtype='float64')
        b_pred = np.array(test_source_signal_padded[sample:sample+previous_size+current_size+future_size], dtype='float64')-y_pred
        y_pred = y_pred.tolist()
        b_pred = b_pred.tolist()
        result.extend(y_pred[previous_size:previous_size+current_size])
        result_noise.extend(b_pred[previous_size:previous_size+current_size])
        sample += current_size
        i += 1
    print(" | loss : {}".format(test_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

    # save checkpoint
    cf.createFolder("{}/checkpoint/{}_{}".format(cf.load_path(), config['save_check_point_name'], epoch+1))
    model.save_weights('{}/checkpoint/{}_{}/data.ckpt'.format(cf.load_path(), config['save_check_point_name'], epoch+1))

    # save output
    cf.createFolder("{}/train_result".format(cf.load_path()))
    wav.write_wav(result[:len(result)-test_mod], "{}/train_result/result{}.wav".format(cf.load_path(), epoch + 1), test_source_sample_rate)
    wav.write_wav(result_noise[:len(result_noise)-test_mod], "{}/train_result/result_noise{}.wav".format(cf.load_path(), epoch + 1), test_source_sample_rate)