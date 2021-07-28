# Naive Bayes 
from model_type import ModelType
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


class NaiveBayes(ModelType):
    """
    Implementation of NaiveBayes
    """

    def __init__(self, config, logger, data_loader):
        self.dataLoader = data_loader
        x_train, x_val, y_train, y_val = self.dataLoader.get_training_validation_tweets()

        self.model = self.create_model()
        self.vectorizer = CountVectorizer(max_features=self.config.get("lr_features"))
        x_train = self.vectorizer.fit_transform(x_train)
        x_val = self.vectorizer.transform(x_val)
        self.train(x_train, y_train)

        y_pred = self.model.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        logger.log_metric("validation_accuracy", acc)

    def create_model(self):
        model = MultinomialNB()
        return model

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def validate(self, tweets):
        tweets = self.vectorizer.transform(tweets)

        result = self.model.predict_proba(tweets)
        result = [[x[1]] for x in result]
        return result

    def name(self):
        return "nb_"
