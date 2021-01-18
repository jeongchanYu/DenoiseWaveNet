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

previous_size = config['previous_size']
current_size = config['current_size']
future_size = config['future_size']
receptive_size = previous_size + current_size + future_size
shift_size = config['shift_size']
batch_size = config['batch_size']
epochs = config['epochs']

# read train data file
target_signal, target_sample_rate = wav.read_wav(config['training_target_file'])
source_signal, source_sample_rate = wav.read_wav(config['training_source_file'])
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

# make model
model = wavenet.DenoiseWaveNet(config['dilation'], config['relu_alpha'], config['default_float'])

# load model
if config['load_check_point_name'] != "":
    model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_path(), config['load_check_point_name']))

loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
train_loss = tf.keras.metrics.Mean(name='train_loss')

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
        loss = loss_object(tf.slice(y_true, start, size), tf.slice(y_pred, start, size), 2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

# train run
for epoch in range(epochs):
    i = 0
    start = time.time()
    for x, y in train_dataset:
        print("\rTrain : epoch {}/{}, frame {}/{}".format(epoch + 1, epochs, i + 1, math.ceil(number_of_frames / batch_size)), end='')
        train_step(x, y)
        i += 1
    print(" | loss : {}".format(train_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

    cf.createFolder("{}/checkpoint/{}_{}".format(cf.load_path(), config['save_check_point_name'], epoch+1))
    model.save_weights('{}/checkpoint/{}_{}/data.ckpt'.format(cf.load_path(), config['save_check_point_name'], epoch+1))
