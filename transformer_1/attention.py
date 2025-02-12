import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    key_depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(key_depth)

    # padding mask
    if mask is not None:
        logits += (mask * -1e9)

    attention_weight = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weight, value)

    return output, attention_weight


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.depth = d_model // self.num_heads

        # WQ, WK, WV에 해당하는 밀집층
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # WO에 해당하는 밀집층
        self.output_dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        # inputs: (batch_size, seq_len, hidden_dim)
        # reshape: (batch_size, seq_len, num_heads, depth)
        inputs = tf.reshape(
            inputs,
            shape=(batch_size, -1, self.num_heads, self.depth)
        )

        # transpose: (batch_size, num_heads, seq_len, depth)
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. WQ, WK, WV을 지나 Q, K, V 얻기
        # (batch_size, seq_len, d_model)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 헤드 나누기
        # (batch_size, num_heads, seq_len, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. 스케일드 닷 프로덕트 어텐션
        # (batch_size, num_heads, seq_len, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(
            query, key, value, mask
        )

        # (batch_size, seq_len, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 4. 헤드 연결하기
        # (batch_size, seq_len, d_model)
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )

        # 5. WO를 지나기
        # (batch_size, seq_len, d_model)
        outputs = self.output_dense(concat_attention)

        return outputs
