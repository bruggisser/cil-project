from nn.neural_network import NeuralNetwork

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM, Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class LSTMNetwork(NeuralNetwork):
    """
    Implementation of LSTM
    """

    def __init__(self, config, logger, data_loader):
        super().__init__(config, logger, data_loader)

    def create_model(self, vocab_size, emb_matrix):
        model = Sequential()
        model.add(
            Embedding(vocab_size, 200, input_length=self.max_len))  # 200-dimensional vector space as set by GloVe dims
        model.add(LSTM(200, return_sequences=True))  # LSTM layer
        model.add(LSTM(200))
        model.add(Dense(400, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # freeze GloVe embeddings
        model.layers[0].set_weights([emb_matrix])
        model.layers[0].trainable = False

        return model

    def train(self, model, x_train, y_train, x_val, y_val):
        opt = Adam(0.00005)

        # Fit model on training data
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        callback = EarlyStopping(monitor="val_loss", patience=3)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), callbacks=[callback])
        model.save(self.modelsPath + self.name() + "model")

    def name(self):
        return "lstm_"
