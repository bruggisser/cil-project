from keras_preprocessing.sequence import pad_sequences

from model_type import ModelType


class NeuralNetwork(ModelType):
    """
    Baseclass of neural network
    """

    def __init__(self, config, logger, data_loader):
        self.modelsPath = config.get("models_path")
        self.dataLoader = data_loader
        glove_file = config.get("nn_glove_file")
        x_train, x_val, y_train, y_val, vocab_size, self.max_len = self.dataLoader.load_data()
        emb_matrix = self.dataLoader.create_emb_matrix(
           glove_file,
           vocab_size,
           200)

        self.model = self.create_model(vocab_size, emb_matrix)
        self.model.summary(print_fn=logger.log_info)
        self.train(self.model, x_train, y_train, x_val, y_val)
        loss, acc = self.model.evaluate(x_val, y_val)
        logger.log_metric("validation_accuracy", acc)
        logger.log_metric("validation_loss", loss)

    def validate(self, tweets):
        tokenized = self.dataLoader.tokenizer.texts_to_sequences(tweets)
        padded = pad_sequences(tokenized, maxlen=self.max_len, padding='post')
        return self.model.predict(padded)

    def create_model(self, vocab_size, emb_matrix):
        pass

    def train(self, model, x_train, y_train, x_val, y_val):
        pass

    def name(self):
        return "nn_"
