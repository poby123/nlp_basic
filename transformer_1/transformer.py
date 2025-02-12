import tensorflow as tf
from attention import MultiHeadAttention
from positional_encoding import PositionalEncoding


def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    '''
        Implement of the encoder layer
    '''
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # Encoder Layer 1 - Self Attention
    attention = MultiHeadAttention(d_model, num_heads, name='attention')({
        # Q = K = V
        'query': inputs, 'key': inputs, 'value': inputs,
        'mask': padding_mask
    })

    # - Dropout + Add & Norm
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        inputs + attention
    )

    # Encoder Layer 2 - FFNN
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # - Dropout + Add & Norm
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        attention + outputs
    )

    return tf.keras.Model(
        inputs=[inputs, padding_mask],
        outputs=outputs,
        name=name
    )


def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='encoder'):
    '''
        Implement of the encoder
    '''
    inputs = tf.keras.Input(shape=(None, ), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embedding = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embedding = PositionalEncoding(vocab_size, d_model)(embedding)
    output = tf.keras.layers.Dropout(rate=dropout)(embedding)

    for i in range(num_layers):
        output = encoder_layer(
            dff=dff,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name=f"encoder_layer_{i}"
        )([output, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask],
        output=output,
        name=name
    )


def decoder_layer(dff, d_model, num_heads, dropout, name='decoder_layer'):
    '''
        Implement of the decoder layer
    '''
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    encoder_outputs = tf.keras.Input(
        shape=(None, d_model),
        name='encoder_outputs'
    )

    # Decoder Layer 1 - Masked Self Attention
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None),
        name="look_ahead_mask"
    )
    masked_self_attention = MultiHeadAttention(d_model, num_heads, name='masked_self_attention')({
        # Q = K = V
        'query': inputs, 'key': inputs, 'value': inputs,
        'mask': look_ahead_mask
    })

    # -- Dropout + Add & Norm
    masked_self_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        masked_self_attention + inputs
    )

    # Decoder Layer 2 - Decoder-Encoder Attention
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    decoder_encoder_attention = MultiHeadAttention(d_model, num_heads, name='decoder_encoder_attention')({
        'query': masked_self_attention, 'key': encoder_outputs, 'value': encoder_outputs,
        'mask': padding_mask
    })

    # -- Dropout + Add & Norm
    decoder_encoder_attention = tf.keras.layers.Dropout(rate=dropout)(
        decoder_encoder_attention
    )
    decoder_encoder_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        masked_self_attention + decoder_encoder_attention
    )

    # Decoder Layer 3 - FFNN
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(
        decoder_encoder_attention
    )
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # -- Dropout + Add & Norm
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        outputs + decoder_encoder_attention
    )

    return tf.keras.Model(
        inputs=[inputs, encoder_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )


def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='decoder'):
    '''
        Implement of the decoder
    '''
    inputs = tf.keras.Input(shape=(None, ), name='inputs')
    encoder_outputs = tf.keras.Input(
        shape=(None, d_model),
        name='encoder_outputs'
    )

    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None),
        name="look_ahead_mask"
    )
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embedding = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embedding = PositionalEncoding(vocab_size, d_model)(embedding)
    output = tf.keras.layers.Dropout(rate=dropout)(embedding)

    for i in range(num_layers):
        output = decoder_layer(
            dff,
            d_model,
            num_heads,
            dropout,
            name=f'decoder_layer_{i}'
        )(inputs=[output, encoder_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, encoder_outputs, look_ahead_mask, padding_mask],
        output=output,
        name=name
    )
