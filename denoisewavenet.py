import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import Model

class DenoiseWaveNet(Model):
    def __init__(self, dilation, relu_alpha=0.0, default_float='float32'):
        super(DenoiseWaveNet, self).__init__()

        tf.keras.backend.set_floatx(default_float)
        self.relu = lambda x: tf.keras.activations.relu(x, alpha=relu_alpha)
        self.dilation = dilation

        self.conv_input = Conv1D(128, 1)
        self.conv_gated_in = [Conv1D(128, 1, padding='same', dilation_rate=d) for d in self.dilation]
        self.conv_gated_out = [Conv1D(128, 1) for l in self.dilation]
        self.conv_out1 = Conv1D(2048, 3, padding='same', activation=self.relu)
        self.conv_out2 = Conv1D(256, 3, padding='same', activation=self.relu)
        self.conv_proj = Conv1D(1, 1, activation='tanh')

    def call(self, x):
        temp_x = x
        if len(temp_x.shape)==2:
            temp_x = tf.reshape(temp_x, [temp_x.shape[0], temp_x.shape[1], -1])
        elif len(temp_x.shape)==1:
            temp_x = tf.reshape(temp_x, [1, temp_x.shape[0], 1])

        temp_x = self.conv_input(temp_x)
        for i in range(len(self.dilation)):
            dilated_x = self.conv_gated_in[i](temp_x)
            dilated_x = tf.keras.activations.sigmoid(dilated_x) * tf.keras.activations.tanh(dilated_x)
            dilated_x = self.conv_gated_out[i](dilated_x)
            temp_x += dilated_x
            if i == 0:
                skip_output = dilated_x
            else:
                skip_output += dilated_x

        skip_output = self.relu(skip_output)
        output1 = self.conv_out1(skip_output)
        output2 = self.conv_out2(output1)
        output2 = self.conv_proj(output2)
        output2 = tf.squeeze(output2)
        return output2
