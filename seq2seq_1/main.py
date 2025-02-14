import os
import re
import shutil
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import urllib3
from keras.layers import LSTM, Dense, Embedding, Input, Masking
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# 64
NUM_SAMPLES = 33000


def install_data():
    http = urllib3.PoolManager(headers={
        'User-Agent': 'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    })
    URL = 'http://www.manythings.org/anki/fra-eng.zip'
    FILE_NAME = 'fra-eng.zip'
    path = Path(os.path.join(os.getcwd(), 'data'))
    path.parent.mkdir(parents=True, exist_ok=True)
    zip_file_name = path.joinpath(FILE_NAME)

    with http.request('GET', URL, preload_content=False) as r, open(zip_file_name, 'wb') as out_file:
        shutil.copyfileobj(r, out_file)

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(path)


def preprocess_sentence(s: str):
    def to_ascii(s: str):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    s = to_ascii(s)
    s = re.sub(r"([?.!,?`])", r" \1", s)  # 단어와 구두점 사이에 공백추가
    s = re.sub(r"[^a-zA-Z!.?]+", r" ", s)  # 주어진 문자에 해당되지 않는 것은 모두 공백으로 변경
    s = re.sub(r"\s+", " ", s)  # 여러 개의 공백을 하나의 공백으로 변경

    return s


def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []

    with open('data/fra.txt', 'r') as lines:
        for i, line in enumerate(lines):
            source_line, target_line, _ = line.strip().split('\t')

            # source data preprocess
            source_line = [w for w in preprocess_sentence(source_line).split()]

            # target data preprocess
            target_line = preprocess_sentence(target_line)
            target_line_in = [w for w in ("<sos> " + target_line).split()]
            target_line_out = [w for w in (target_line + " <eos>").split()]

            encoder_input.append(source_line)
            decoder_input.append(target_line_in)
            decoder_target.append(target_line_out)

            if i == NUM_SAMPLES - 1:
                break

    return encoder_input, decoder_input, decoder_target


if __name__ == '__main__':
    install_data()
    encoder_input, decoder_input, decoder_target = load_preprocessed_data()
    print('Encoder input:', encoder_input[:5])
    print('Decoder input:', decoder_input[:5])
    print('Decoder target:', decoder_target[:5])

    tokenizer_encoder = Tokenizer(filters="", lower=False)
    tokenizer_encoder.fit_on_texts(encoder_input)
    encoder_input = tokenizer_encoder.texts_to_sequences(encoder_input)
    encoder_input = pad_sequences(encoder_input, padding='post')

    tokenizer_decoder = Tokenizer(filters="", lower=False)
    tokenizer_decoder.fit_on_texts(decoder_input)
    tokenizer_decoder.fit_on_texts(decoder_target)

    decoder_input = tokenizer_decoder.texts_to_sequences(decoder_input)
    decoder_input = pad_sequences(decoder_input, padding='post')

    decoder_target = tokenizer_decoder.texts_to_sequences(decoder_target)
    decoder_target = pad_sequences(decoder_target, padding='post')

    print('Encoder Input Shape: ', encoder_input.shape)
    print('Decoder Input Shape: ', decoder_input.shape)
    print('Decoder Target Shape: ', decoder_target.shape)

    source_vocab_size = len(tokenizer_encoder.word_index) + 1
    target_vocab_size = len(tokenizer_decoder.word_index) + 1

    print(
        f'English vocab size: {source_vocab_size}, french vocab size: {target_vocab_size}')

    source2index = tokenizer_encoder.word_index
    index2source = tokenizer_encoder.index_word
    target2index = tokenizer_decoder.word_index
    index2target = tokenizer_decoder.index_word

    indices = np.arange(encoder_input.shape[0])
    np.random.shuffle(indices)
    print(f'random sequences: {indices}')

    encoder_input = encoder_input[indices]
    decoder_input = decoder_input[indices]
    decoder_target = decoder_target[indices]

    print(encoder_input[30997])
    print(decoder_input[30997])
    print(decoder_target[30997])

    n_of_validation = int(encoder_input.shape[0] * 0.1)
    encoder_input_train = encoder_input[:-n_of_validation]
    decoder_input_train = decoder_input[:-n_of_validation]
    decoder_target_train = decoder_target[:-n_of_validation]

    encoder_input_test = encoder_input[-n_of_validation:]
    decoder_input_test = decoder_input[-n_of_validation:]
    decoder_target_test = decoder_target[-n_of_validation:]

    print(f'train encoder input: {encoder_input_train.shape}')
    print(f'train decoder input: {decoder_input_train.shape}')
    print(f'train decoder target: {decoder_target_train.shape}')

    print(f'test encoder input: {encoder_input_test.shape}')
    print(f'test decoder input: {decoder_input_test.shape}')
    print(f'test decoder target: {decoder_target_test.shape}')

    embedding_dim = 64
    hidden_units = 64

    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(source_vocab_size, embedding_dim)
    encoder_embedding = encoder_embedding(encoder_inputs)
    encoder_masking = Masking(mask_value=0.0)(encoder_embedding)
    encoder_lstm = LSTM(hidden_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_masking)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding_layer = Embedding(target_vocab_size, hidden_units)
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_masking = Masking(mask_value=0.0)(decoder_embedding)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_masking,
        initial_state=encoder_states
    )

    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )

    model.fit(
        x=[encoder_input_train, decoder_input_train],
        y=decoder_target_train,
        validation_data=(
            [encoder_input_test, decoder_input_test],
            decoder_target_test
        ),
        batch_size=128, epochs=50
    )

    # Encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(hidden_units,))
    decoder_state_input_c = Input(shape=(hidden_units,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding2 = decoder_embedding_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        decoder_embedding2,
        initial_state=decoder_state_inputs
    )
    decoder_states2 = [state_h2, state_c2]
    decoder_output2 = decoder_dense(decoder_outputs2)

    decoder_model = Model(
        [decoder_inputs] + decoder_state_inputs,
        [decoder_output2] + decoder_states2
    )

    def decode_sequence(input_sequence: str):
        states_value = encoder_model.predict(input_sequence)

        # <SOS>에 해당하는 정수 생성
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target2index['<sos>']

        stop_condition = False
        decoded_sentence = ''

        # stop_condition이 True가 될 때까지 루프 반복
        # 구현의 간소화를 위해서 이 함수는 배치 크기를 1로 가정합니다.
        while not stop_condition:
            # 이전 시점 상태 states_value를 현 시점의 초기 상태로 사용
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value
            )

            # 예측 결과를 단어로 변환
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = index2target[sampled_token_index]

            # 현재 시점의 예측 단어를 예측 문장에 추가
            decoded_sentence += ' ' + sampled_char

            # <eos>에 도달하거나 정해진 길이를 넘으면 중단.
            if (sampled_char == '<eos>' or len(decoded_sentence) > 50):
                stop_condition = True

            # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
            states_value = [h, c]

        return decoded_sentence

    # 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
    def seq_to_src(input_seq):
        sentence = ''
        for encoded_word in input_seq:
            if encoded_word != 0:
                sentence = sentence + index2source[encoded_word] + ' '
        return sentence

    # 번역문의 정수 시퀀스를 텍스트 시퀀스로 변환
    def seq_to_tar(input_seq):
        sentence = ''
        for encoded_word in input_seq:
            if (encoded_word != 0 and encoded_word != target2index['<sos>'] and encoded_word != target2index['<eos>']):
                sentence = sentence + index2target[encoded_word] + ' '
        return sentence

    for seq_index in [3, 50, 100, 300, 1001]:
        input_seq = encoder_input_train[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)

        print("입 력 문 장 :", seq_to_src(encoder_input_train[seq_index]))
        print("정 답 문 장 :", seq_to_tar(decoder_input_train[seq_index]))
        print("번 역 문 장 :", decoded_sentence[1:-5])
        print("-"*50)

    for seq_index in [3, 50, 100, 300, 1001]:
        input_seq = encoder_input_test[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)

        print("입 력 문 장 :", seq_to_src(encoder_input_test[seq_index]))
        print("번 역 문 장 :", decoded_sentence[1:-5])
        print("-"*50)
