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
shift_size = int(config['shift_size'])
batch_size = int(config['batch_size'])
epochs = int(config['epochs'])
training_target_path = config['training_target_path']
training_source_path = config['training_source_path']
save_check_point_name = config['save_check_point_name']
load_check_point_name = config['load_check_point_name']

# training_target_path is path or file?
target_path_isdir = os.path.isdir(training_target_path)
source_path_isdir = os.path.isdir(training_source_path)
if target_path_isdir != source_path_isdir:
    raise Exception("ERROR: Target and source path is incorrect")
if target_path_isdir:
    if not cf.compare_path_list(training_target_path, training_source_path, 'wav'):
        raise Exception("ERROR: Target and source file list is not same")
    training_target_file_list = cf.read_path_list(training_target_path, "wav")
    training_source_file_list = cf.read_path_list(training_source_path, "wav")
else:
    training_target_file_list = [training_target_path]
    training_source_file_list = [training_source_path]

x_signal, y_signal = [], []
num_of_total_frame = 0
for i in range(len(training_target_file_list)):
    # read train data file
    target_signal, target_sample_rate = wav.read_wav(training_target_file_list[i])
    source_signal, source_sample_rate = wav.read_wav(training_source_file_list[i])

    target_signal = np.array(target_signal)
    source_signal = np.array(source_signal)
    size_of_target = target_signal.size
    size_of_source = source_signal.size

    # source & target file incorrect
    if size_of_source != size_of_target:
        raise Exception("ERROR: Input, output size mismatch")
    if size_of_source < current_size:
        raise Exception("ERROR: Input file length is too small")
    if shift_size <= 0:
        raise Exception("ERROR: Shift size is smaller or same with 0")

    # padding
    mod = (shift_size - (size_of_source % shift_size)) % shift_size
    target_signal_padded = np.concatenate([np.zeros(previous_size), target_signal, np.zeros(future_size+mod)]).astype(default_float)
    source_signal_padded = np.concatenate([np.zeros(previous_size), source_signal, np.zeros(future_size+mod)]).astype(default_float)
    if shift_size < current_size:
        dif = current_size-shift_size
        target_signal_padded = np.concatenate([target_signal_padded, np.zeros(dif)]).astype(default_float)
        source_signal_padded = np.concatenate([source_signal_padded, np.zeros(dif)]).astype(default_float)

    # make dataset
    number_of_frames = math.ceil(size_of_source/shift_size)
    num_of_total_frame += number_of_frames
    for j in range(number_of_frames):
        x_signal.append(source_signal_padded[j*shift_size:(j*shift_size) + receptive_size])
        y_signal.append(target_signal_padded[j*shift_size:(j*shift_size) + receptive_size])


train_dataset = tf.data.Dataset.from_tensor_slices((x_signal, y_signal)).shuffle(number_of_frames).batch(batch_size)

# in train with output, test data must be a file not a directory
if os.path.isdir(config['test_source_file']) or os.path.isdir(config['test_target_file']):
    raise Exception("ERROR: In train with output, test data must be a file not a directory")

# make test data
if config['training_source_path'] != config['test_source_file']:
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
    test_target_signal_padded = np.concatenate([np.zeros(previous_size), test_target_signal, np.zeros(future_size + test_mod)]).astype(default_float)
    test_source_signal_padded = np.concatenate([np.zeros(previous_size), test_source_signal, np.zeros(future_size + test_mod)]).astype(default_float)
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
loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
train_loss =tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

# load model
if load_check_point_name != "":
    model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_path(), load_check_point_name))
    model.load_optimizer_state(optimizer, '{}/checkpoint/{}'.format(cf.load_path(), load_check_point_name), 'optimizer', model.trainable_variables)
    saved_epoch = int(load_check_point_name.split('_')[-1])
else:
    cf.clear_plot_file('{}/{}'.format(cf.load_path(), config['plot_file']))
    saved_epoch = 0

# train function
@tf.function
def train_step(x, y):
    y_true = tf.squeeze(y)
    if len(y_true.shape) == 2:
        start = [0,previous_size]
        size = [-1,current_size]
    elif len(y_true.shape) == 1:
        start = [previous_size]
        size = [current_size]
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_object(tf.slice(y_true, start, size), tf.slice(y_pred, start, size))*2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

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


# train and test run
for epoch in range(saved_epoch, saved_epoch + epochs):
    i = 0
    start = time.time()
    for x, y in train_dataset:
        print("\rTrain : epoch {}/{}, training {}/{}".format(epoch + 1, saved_epoch+epochs, i + 1, math.ceil(num_of_total_frame / batch_size)), end='')
        train_step(x, y)
        i += 1
    print(" | loss : {}".format(train_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))
    train_loss.reset_states()

    result = []
    result_noise = []
    i = 0
    sample = 0
    start = time.time()
    while sample < test_size_of_source:
        print("\rTest : epoch {}/{}, training {}/{}".format(epoch + 1, saved_epoch+epochs, i + 1, math.ceil(test_size_of_source/current_size)), end='')
        y_pred = test_step(test_source_signal_padded[sample:sample+previous_size+current_size+future_size],
                           test_target_signal_padded[sample:sample+previous_size+current_size+future_size])
        y_pred = np.array(y_pred, dtype=default_float)
        b_pred = np.array(test_source_signal_padded[sample:sample+previous_size+current_size+future_size], dtype=default_float)-y_pred
        y_pred = y_pred.tolist()
        b_pred = b_pred.tolist()
        result.extend(y_pred[previous_size:previous_size+current_size])
        result_noise.extend(b_pred[previous_size:previous_size+current_size])
        sample += current_size
        i += 1
    print(" | loss : {}".format(test_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

    # save checkpoint
    cf.createFolder("{}/checkpoint/{}_{}".format(cf.load_path(), save_check_point_name, epoch+1))
    model.save_weights('{}/checkpoint/{}_{}/data.ckpt'.format(cf.load_path(), save_check_point_name, epoch+1))
    model.save_optimizer_state(optimizer, '{}/checkpoint/{}_{}'.format(cf.load_path(), save_check_point_name, epoch + 1), 'optimizer')
    cf.write_plot_file('{}/{}'.format(cf.load_path(), config['plot_file']), epoch+1, train_loss.result())

    # save output
    cf.createFolder("{}/train_result".format(cf.load_path()))
    wav.write_wav(result[:len(result)-test_mod], "{}/train_result/result{}.wav".format(cf.load_path(), epoch + 1), test_source_sample_rate)
    wav.write_wav(result_noise[:len(result_noise)-test_mod], "{}/train_result/result_noise{}.wav".format(cf.load_path(), epoch + 1), test_source_sample_rate)

    test_loss.reset_states()