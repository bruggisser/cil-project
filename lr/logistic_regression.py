from model_type import ModelType
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel


class LogisticRegression(ModelType):
    """
    Implementation of Logistic Regression
    """

    def __init__(self, config, logger, data_loader):
        self.dataLoader = data_loader
        self.config = config
        x_train, x_val, y_train, y_val = self.dataLoader.get_training_validation_tweets()

        self.vectorizer = CountVectorizer(max_features=self.config.get("lr_features"))
        x_train = self.vectorizer.fit_transform(x_train)
        x_val = self.vectorizer.transform(x_val)

        self.model = self.create_model()
        self.train(x_train, y_train)

        y_pred = self.model.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        logger.log_metric("validation_accuracy", acc)

    def create_model(self):
        model = LogisticRegressionModel(C=1e5, max_iter=8000)
        return model

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def validate(self, tweets):
        tweets = self.vectorizer.transform(tweets)
        pred = self.model.predict_proba(tweets)
        pred = [[x[1]] for x in pred]
        return pred

    def name(self):
        return "lr_"
