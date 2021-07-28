from nn.neural_network import NeuralNetwork

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class ConvolutionalNetwork(NeuralNetwork):
    """
    Implementation of Convolutional Network
    """

    def __init__(self, config, logger, data_loader):
        super().__init__(config, logger, data_loader)

    def create_model(self, vocab_size, emb_matrix):
        weight_decay = 0.0001
        model = Sequential()
        model.add(Embedding(vocab_size, 200, input_length=self.max_len))  # 200-dimensional vector space as set by GloVe dims
        model.add(Conv1D(filters=128, kernel_size=5, kernel_regularizer=l2(weight_decay), activation='relu'))  # use 1D conv network to learn the relationships between embeddings
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # freeze GloVe embeddings
        model.layers[0].set_weights([emb_matrix])
        model.layers[0].trainable = False

        return model

    def train(self, model, x_train, y_train, x_val, y_val):
        opt = Adam(0.0001)

        # Fit model on training data
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        callback = EarlyStopping(monitor="val_loss", patience=3)
        model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), callbacks=[callback])
        model.save(self.modelsPath + self.name() + "model")

    def name(self):
        return "cnn_"
