import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from positional_encoding import PositionalEncoding
from attention import scaled_dot_product_attention


sample_pos_encoding = PositionalEncoding(50, 128)
plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim(0, 128)
plt.ylabel('Position')
plt.colorbar()
plt.show()

temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)  # (4, 2)

temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
temp_out, temp_attn = scaled_dot_product_attention(
    temp_q, temp_k, temp_v, None)

print(temp_attn)
print(temp_out)

# p.123
