import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import Model
import numpy as np

class DenoiseWaveNet(Model):
    def __init__(self, dilation, relu_alpha=0.0, default_float='float32'):
        super(DenoiseWaveNet, self).__init__()
        tf.keras.backend.set_floatx(default_float)

        self.relu = tf.keras.activations.relu(alpha=relu_alpha)

        self.dilation = dilation
        self.conv_gated_tanh = [Conv1D(128, 1, padding='same', dilation_rate=d, activation='tanh') for d in self.dilation]
        self.conv_gated_sigmoid = [Conv1D(128, 1, padding='same', dilation_rate=d, activation='sigmoid') for d in self.dilation]
        self.conv_residual = [Conv1D(128, 1) for l in self.dilation[:-1]]
        self.conv_skip = [Conv1D(128, 1) for l in self.dilation]
        self.conv_out1 = Conv1D(2048, 3, padding='same', activation=self.relu)
        self.conv_out2 = Conv1D(256, 3, padding='same',  activation=self.relu)
        self.conv_proj = Conv1D(1, 1, activation='tanh')

    def call(self, x):
        temp_x = x
        temp_x = tf.reshape(temp_x, [temp_x.shape[0], temp_x.shape[1], 1])
        for i in range(len(self.dilation)):
            dilated_x = self.conv_gated_tanh[i](temp_x) * self.conv_gated_sigmoid[i](temp_x)
            if i == 0:
                temp_x = self.conv_residual[i](dilated_x)
                skip_output = self.conv_skip[i](dilated_x)
            else:
                if i != len(self.dilation)-1:
                    temp_x += self.conv_residual[i](dilated_x)
                skip_output += self.conv_skip[i](dilated_x)
        skip_output = tf.keras.backend.relu(skip_output)
        output1 = self.conv_out1(skip_output)
        output2 = self.conv_out2(output1)
        output2 = self.conv_proj(output2)
        output2 = tf.reshape(output2, [x.shape[0], x.shape[1]])
        return output2
