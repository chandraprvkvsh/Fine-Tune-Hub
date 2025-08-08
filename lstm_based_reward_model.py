import pickle
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class LSTMRegressor:
    def __init__(self, n_vocab=20000, max_len=128, embedding_dim=1000, hidden_dims=[500, 250, 50]):
        self.n_vocab = n_vocab
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.tokenizer = None
        self.model = None

    def process_paragraph(self, paragraph):
        paragraph = paragraph.lower()
        paragraph = re.sub(r'\d', '', paragraph)
        paragraph = re.sub(r'<.*?>', '', paragraph)
        paragraph = paragraph.replace('-', ' ')
        paragraph = paragraph.replace('  ', '')
        paragraph = re.sub(r'[^a-zA-Z\s]', '', paragraph)
        paragraph = re.sub(r'\s+', ' ', paragraph).strip()
        return paragraph

    def prepare_data(self, csv_path):
        data_df = pd.read_csv(csv_path, usecols=['title', 'cite_read_boost'])
        data_df.drop_duplicates(inplace=True)
        data_df.dropna(inplace=True)

        clean_titles = ['/start ' + self.process_paragraph(t) + ' /end' for t in data_df['title']]
        read_counts = np.array(data_df['cite_read_boost']).reshape(-1)

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.n_vocab,
            filters='',
            lower=True,
            split=' ',
            oov_token="<UNK>"
        )
        self.tokenizer.fit_on_texts(['/start', '/end'])
        self.tokenizer.fit_on_texts(' '.join(clean_titles).split())

        tokenized_titles = self.tokenizer.texts_to_sequences(clean_titles)
        indexes = [i for i, t in enumerate(tokenized_titles) if len(t) > self.max_len]

        filtered_titles = [t for i, t in enumerate(tokenized_titles) if i not in indexes]
        filtered_reads = np.array([r for i, r in enumerate(read_counts) if i not in indexes])

        filtered_titles_padded = tf.keras.utils.pad_sequences(filtered_titles, maxlen=self.max_len, padding='post')
        x_train, x_test, y_train, y_test = train_test_split(
            filtered_titles_padded, filtered_reads, shuffle=True, test_size=0.25, random_state=23
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(32).prefetch(len(x_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(len(x_test)).batch(32)

        return train_dataset, test_dataset

    def build_model(self):
        inputs = Input(shape=(self.max_len,))
        x = Embedding(input_dim=self.n_vocab, output_dim=self.embedding_dim, mask_zero=True)(inputs)
        x = LSTM(self.hidden_dims[0], activation='tanh', return_sequences=True)(x)
        x = LSTM(self.hidden_dims[1], activation='tanh', return_sequences=True)(x)
        x = LSTM(self.hidden_dims[2], activation='tanh')(x)
        x = Dense(10, activation='linear')(x)
        outputs = Dense(1, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])

    def train(self, train_dataset, test_dataset, epochs=2):
        if self.model is None:
            self.build_model()
        history = self.model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, verbose=1)
        return history

    def save(self, tokenizer_path, model_path, history_path=None):
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        self.model.save(model_path)
        if history_path:
            with open(history_path, 'wb') as f:
                pickle.dump(self.history, f)

    def load(self, tokenizer_path, model_path):
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.model = load_model(model_path)

    def preprocess_input(self, title):
        title = self.process_paragraph(title)
        seq = self.tokenizer.texts_to_sequences([title])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.max_len, padding='post')
        return padded_seq

    def predict(self, title):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before prediction.")
        input_seq = self.preprocess_input(title)
        pred = self.model.predict(input_seq)
        return float(np.abs(pred[0][0]))


if __name__ == '__main__':
    regressor = LSTMRegressor()
    train_ds, test_ds = regressor.prepare_data('<your_dataset_path>.csv')
    regressor.build_model()
    regressor.history = regressor.train(train_ds, test_ds, epochs=2)
    regressor.save('<your_output_dir>/tokenizer.pkl', '<your_output_dir>/model.h5')

    # Inference example:
    # regressor.load('<your_output_dir>/tokenizer.pkl', '<your_output_dir>/model.h5')
    # title = input("Provide the title: ")
    # pred = regressor.predict(title)
    # print(f"Predicted read count: {pred:.4f}")
