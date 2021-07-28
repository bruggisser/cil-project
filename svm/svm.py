from model_type import ModelType
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


class SVMClassifier(ModelType):
    """
    Implementation of SVM
    """

    def __init__(self, config, logger, data_loader):
        self.dataLoader = data_loader
        x_train, x_val, y_train, y_val = self.dataLoader.get_training_validation_tweets()

        x_train = x_train[:config.get("svm_training_size")]
        y_train = y_train[:config.get("svm_training_size")]

        logger.log_metric("svm_training_size", len(x_train))

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
        x_train = self.vectorizer.fit_transform(x_train)
        x_val = self.vectorizer.transform(x_val)
        self.model = self.create_model()
        self.train(self.model, x_train, y_train)

        y_pred = self.model.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        logger.log_metric("validation_accuracy", acc)

    def train(self, model, x_train, y_train):
        model.fit(x_train, y_train)

    def validate(self, tweets):
        # vectorizer = TfidfVectorizer(sublinear_tf = True, use_idf = True)
        # tweets = vectorizer.fit_transform(tweets)
        tweets = self.vectorizer.transform(tweets)
        pred = self.model.predict_proba(tweets)
        pred = [[x[1]] for x in pred]
        return pred

    def create_model(self):
        model = svm.SVC(kernel='rbf', probability=True)
        return model

    def name(self):
        return "svm_"
