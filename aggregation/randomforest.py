import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class RandomForest:
    """
    Train a random forest ot decide which prediction to take.
    """
    def __init__(self, config, logger, data_loader):
        self.data_loader = data_loader
        self.logger = logger
        self.config = config

        data = pd.read_csv(f"{self.config.get('data_dir')}unified_validation_set.csv")
        x = data[[column for column in self.config.get("models").split(",") if column != "rf" and column != "bert"]]
        y = data["Expectation"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        self.logger.log_metric("random_forest_accuracy", metrics.accuracy_score(y_test, y_pred))

    def validate(self, csv_file):
        csv = pd.read_csv(csv_file)
        return self.model.predict(csv)

    def name(self):
        return "rf_"
