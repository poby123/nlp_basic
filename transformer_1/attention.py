import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    key_depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(key_depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weight = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weight, value)

    return output, attention_weight
