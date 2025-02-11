import keras
import numpy as np
import tensorflow as tf


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(
            10000,
            (2 * (i // 2)) / tf.cast(d_model, tf.float32)
        )
        print(f'angle shape: {angles.shape}')
        return position * angles

    def positional_encoding(self, position, d_model):
        angle = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )

        angle_rad = np.zeros(angle.shape)
        angle_rad[:, 0::2] = tf.math.sin(angle[:, 0::2])
        angle_rad[:, 1::2] = tf.math.cos(angle[:, 1::2])

        pos_encoding = tf.constant(angle_rad)[tf.newaxis, ...]
        print(pos_encoding.shape)

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs * self.pos_encoding[:, :tf.shape(inputs)[1], :]
