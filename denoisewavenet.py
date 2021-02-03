import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import Model
import numpy as np
import os

class DenoiseWaveNet(Model):
    def __init__(self, dilation, relu_alpha=0.0, default_float='float32'):
        super(DenoiseWaveNet, self).__init__()

        tf.keras.backend.set_floatx(default_float)
        self.relu = lambda x: tf.keras.activations.relu(x, alpha=relu_alpha)
        self.dilation = dilation

        self.conv_input = Conv1D(128, 3, padding='same')
        self.conv_gated_tanh = [Conv1D(128, 3, padding='same', dilation_rate=d, activation='tanh') for d in self.dilation]
        self.conv_gated_sigmoid = [Conv1D(128, 3, padding='same', dilation_rate=d, activation='sigmoid') for d in self.dilation]
        self.conv_residual = [Conv1D(128, 1) for l in self.dilation[:-1]]
        self.conv_skip = [Conv1D(128, 1) for l in self.dilation]
        self.conv_out1 = Conv1D(2048, 3, padding='same', activation=self.relu)
        self.conv_out2 = Conv1D(256, 3, padding='same', activation=self.relu)
        self.conv_proj = Conv1D(1, 1, activation='tanh')

    def call(self, x):
        temp_x = x
        if len(temp_x.shape) == 2:
            temp_x = tf.reshape(temp_x, [temp_x.shape[0], temp_x.shape[1], -1])
        elif len(temp_x.shape) == 1:
            temp_x = tf.reshape(temp_x, [1, temp_x.shape[0], 1])

        temp_x = self.conv_input(temp_x)
        for i in range(len(self.dilation)):
            dilated_x = self.conv_gated_tanh[i](temp_x) * self.conv_gated_sigmoid[i](temp_x)
            if i != len(self.dilation)-1:
                temp_x += self.conv_residual[i](dilated_x)
            if i == 0:
                skip_output = self.conv_skip[i](dilated_x)
            else:
                skip_output += self.conv_skip[i](dilated_x)

        skip_output = self.relu(skip_output)
        output1 = self.conv_out1(skip_output)
        output2 = self.conv_out2(output1)
        output2 = self.conv_proj(output2)
        output2 = tf.squeeze(output2)
        return output2

    def save_optimizer_state(self, optimizer, save_path, save_name):

        # Create folder if it does not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save weights
        np.save(os.path.join(save_path, save_name), optimizer.get_weights())

        return


    def load_optimizer_state(self, optimizer, load_path, load_name):

        opt_weights = np.load(os.path.join(load_path, load_name) + '.npy', allow_pickle=True)

        optimizer.set_weights(opt_weights)

        return
