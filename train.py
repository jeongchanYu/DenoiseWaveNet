import tensorflow as tf
import denoisewavenet as wavenet
import numpy as np
import wav
import json
import customfunction as cf
import time
import datetime
import math

# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

PREVIOUS_SIZE = config['previous_size']
CURRENT_SIZE = config['current_size']
FUTURE_SIZE = config['future_size']
RECEPTIVE_SIZE = PREVIOUS_SIZE + CURRENT_SIZE + FUTURE_SIZE
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']

# read train data file
target_signal, target_sample_rate = wav.read_wav(config['training_target_file'])
source_signal, source_sample_rate = wav.read_wav(config['training_source_file'])
target_signal = np.array(target_signal)
source_signal = np.array(source_signal)
size_of_target = target_signal.size
size_of_source = source_signal.size

# source & target file incorrect
if size_of_source != size_of_target:
    raise Exception("ERROR: Input output size mismatch")
if size_of_source < CURRENT_SIZE:
    raise Exception("ERROR: Input file length is too small")

# padding
target_signal_padded = np.concatenate([np.zeros(PREVIOUS_SIZE), target_signal, np.zeros(FUTURE_SIZE)]).astype('float32')
source_signal_padded = np.concatenate([np.zeros(PREVIOUS_SIZE), source_signal, np.zeros(FUTURE_SIZE)]).astype('float32')

# make dataset
x_signal, y_signal = [], []
number_of_frames = size_of_source-CURRENT_SIZE+1
for i in range(number_of_frames):
    x_signal.append(source_signal_padded[i:i + RECEPTIVE_SIZE])
    y_signal.append(target_signal_padded[i:i + RECEPTIVE_SIZE])
train_dataset = tf.data.Dataset.from_tensor_slices((x_signal, y_signal)).batch(BATCH_SIZE)


# prevent GPU overflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.InteractiveSession(config=config)

# make model
model = wavenet.DenoiseWaveNet(config['dilation'], config['relu_alpha'], config['default_float'])

# load model
if config['load_check_point_name'] != "":
    model.load_weights('{}/checkpoint/{}/checkpoint.ckpt'.format(cf.load_path(), config['load_check_point_name']))

loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
train_loss = tf.keras.metrics.MeanAbsoluteError(name='train_loss')

# train function
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_object(y, y_pred, 2)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(y, y_pred, 2)

# train run
for epoch in range(EPOCHS):
    i = 0
    start = time.time()
    for x, y in train_dataset:
        print("\rTrain : epoch {}/{}, frame {}/{}".format(epoch + 1, EPOCHS, i + 1, math.ceil(number_of_frames/BATCH_SIZE)), end='')
        train_step(x, y)
        i += 1
    print(" | loss : {}".format(train_loss.result()), " | 경과시간 :", datetime.timedelta(seconds=time.time() - start))

    cf.createFolder("{}/checkpoint/{}_{}".format(cf.load_path(), config['save_check_point_name'], epoch+1))
    model.save_weights('{}/checkpoint/{}_{}/data.ckpt'.format(cf.load_path(), config['save_check_point_name'], epoch+1))
