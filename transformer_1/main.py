'''
https://github.com/ukairia777/tensorflow-nlp-tutorial/blob/main/16.%20Transformer%20(Chatbot)/16-1%20~%2016-2.%20transformer_chatbot.ipynb
'''
import os
import re
import urllib.request

import pandas as pd
import sentencepiece as spm
import tensorflow as tf
from custom_schedule import CustomSchedule
from keras.callbacks import ModelCheckpoint
from transformer import transformer
from checkpoint import get_latest_checkpoint

MAX_LENGTH = 40


def load_data(filename: str):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.isfile(filename) == False:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
            filename
        )
    train_data = pd.read_csv(filename)
    print('The number of samples: ', len(train_data))
    print('The number of null: ', train_data.isnull().sum())

    return train_data


def preprocess_sentence(sentence: str):
    sentence = re.sub(r'([?.!,])', r' \1 ', sentence).strip()
    return sentence


def split_data(train_data: pd.DataFrame):
    questions = []
    for sentence in train_data['Q']:
        questions.append(preprocess_sentence(sentence))

    answers = []
    for sentence in train_data['A']:
        answers.append(preprocess_sentence(sentence))

    return questions, answers


def get_tokenizer(filename: str):
    spm.SentencePieceTrainer.Train(
        input=filename,
        model_prefix="tokenizer",
        vocab_size=2**13
    )

    tokenizer = spm.SentencePieceProcessor()
    vocab_file = 'tokenizer.model'
    tokenizer.Load(vocab_file)

    test_encode_string = tokenizer.Encode(questions[20])
    test_decode_string = tokenizer.Decode(test_encode_string)

    print(f'Original sample question: {questions[20]}')
    print(f'Tokenized sample question: {test_encode_string}')
    print(f'Tokenized sample question: {test_decode_string}')

    return tokenizer


def get_special_token(tokenizer: spm.SentencePieceProcessor):
    START_TOKEN = [tokenizer.vocab_size()]
    END_TOKEN = [tokenizer.vocab_size()+1]

    return START_TOKEN, END_TOKEN


def tokenize_and_padding(tokenizer: spm.SentencePieceProcessor, inputs: list[str], outputs: list[str]):
    START_TOKEN, END_TOKEN = get_special_token(tokenizer)
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = START_TOKEN + tokenizer.Encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.Encode(sentence2) + END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs,
        maxlen=MAX_LENGTH,
        padding='post'
    )

    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs,
        maxlen=MAX_LENGTH,
        padding='post'
    )

    return tokenized_inputs, tokenized_outputs


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


def accuracy(y_true, y_pred):
    # (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH-1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def prepare_tensor_dataset(questions, answers):
    BATCH_SIZE = 64
    BUFFER_SIZE = 20_000

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'encoder_inputs': questions,

            # remove the last token for teacher forcing that
            'decoder_inputs': answers[:, :-1]
        },
        {
            # remove the first token for answer sequence
            'outputs': answers[:, 1:]
        }
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def get_model(tokenizer: spm.SentencePieceProcessor):
    tf.keras.backend.clear_session()

    VOCAB_SIZE = tokenizer.vocab_size()+2
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    DFF = 512
    DROPOUT = 0.1

    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )

    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    model.compile(optimizer, loss_function, metrics=[accuracy])
    return model


def evaluate(model: tf.keras.Model, tokenizer: spm.SentencePieceProcessor, sentence: str):
    START_TOKEN, END_TOKEN = get_special_token(tokenizer)

    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.Encode(sentence) + END_TOKEN, axis=0
    )
    output = tf.expand_dims(START_TOKEN, 0)

    for _ in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(model: tf.keras.Model, tokenizer: spm.SentencePieceProcessor, sentence: str):
    prediction = evaluate(model, tokenizer, sentence)
    prediction = prediction.numpy().tolist()

    predicted_sentence = tokenizer.Decode(
        [i for i in prediction if i < tokenizer.vocab_size()]
    )

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


if __name__ == '__main__':
    filename = 'data/ChatBotData.csv'
    train_data = load_data(filename)
    questions, answers = split_data(train_data)
    tokenizer = get_tokenizer(filename)
    questions, answers = tokenize_and_padding(tokenizer, questions, answers)

    print('질문 데이터의 크기(shape) :', questions.shape)
    print('답변 데이터의 크기(shape) :', answers.shape)

    dataset = prepare_tensor_dataset(questions, answers)
    model = get_model(tokenizer)

    checkpoint_path = 'checkpoint/{epoch:04d}.ckpt'
    latest_checkpoint = get_latest_checkpoint(checkpoint_path)
    print(f'latest checkpoint: {latest_checkpoint}')

    model_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
    )

    if latest_checkpoint == None:
        EPOCHS = 50
        model.save_weights(checkpoint_path.format(epoch=0))
        model.fit(dataset, epochs=EPOCHS, callbacks=[model_callback])
    else:
        model.load_weights(latest_checkpoint)

    predict(model, tokenizer, '안녕?')
    predict(model, tokenizer, '이름이 뭐야?')
    predict(model, tokenizer, '오늘이 몇일인지 아니?')
    predict(model, tokenizer, '갤럭시가 좋아? 아이폰이 좋아?')
