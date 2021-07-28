from aggregation.modelaverage import ModelAverage
from bert.bert_classifier import BertClassifier
from aggregation.modelvote import ModelVote
from lr.logistic_regression import LogisticRegression
from nn.convolutional_network import ConvolutionalNetwork
from aggregation.randomforest import RandomForest
from util.dataloader import DataLoader
from nn.lstm_network import LSTMNetwork
from nb.naive_bayes import NaiveBayes
from svm.svm import SVMClassifier
from util.validator import Validator
import tensorflow as tf
import csv


class ModelLoader:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_loader = DataLoader(self.logger, self.config)  # load data once

    def execute_models(self):
        """
        Execute the models and aggregate the results as defined in config.json.
        """
        with tf.device('/GPU:0'):
            validator = Validator(self.data_loader, self.config)

            selected_models = self.config.get("models").split(",")
            selected_aggregations = self.config.get("aggregations").split(",")
            run_bert = "bert" in selected_models
            selected_models = [m for m in selected_models if m != "bert"]

            for selected_model in selected_models:
                self.logger.start_experiment()
                self.logger.log_info(f"selected model: {selected_model}")
                if selected_model == "lstm":
                    model = LSTMNetwork(self.config, self.logger, self.data_loader)
                elif selected_model == "nb":
                    model = NaiveBayes(self.config, self.logger, self.data_loader)
                elif selected_model == "svm":
                    model = SVMClassifier(self.config, self.logger, self.data_loader)
                elif selected_model == "lr":
                    model = LogisticRegression(self.config, self.logger, self.data_loader)
                elif selected_model == "cnn":
                    model = ConvolutionalNetwork(self.config, self.logger, self.data_loader)
                else:
                    continue

                validator.validate(model)
                validator.create_validation_set(model)

            if len(selected_aggregations) > 0 and len(selected_aggregations[0]) > 0:
                self.logger.start_experiment()
                self.logger.log_info("Aggregation of results")
                self.unify_results(selected_models)
                for aggregation in selected_aggregations:
                    if aggregation == "rf":
                        model = RandomForest(self.config, self.logger, self.data_loader)
                    elif aggregation == "vote":
                        model = ModelVote(self.config, self.logger, self.data_loader)
                    else:
                        model = ModelAverage(self.config, self.logger, self.data_loader)
                    validator.validate_aggregation(model)

            if run_bert:
                model = BertClassifier(self.config, self.logger, self.data_loader)
                validator.validate(model)

    def unify_results(self, selected_models):
        """
        Take the results of all models and write them into two files:

        - unified_validation_set.csv with the validation set
        - unified_result_exact.csv for kaggle

        :param selected_models: list of models used
        """
        train_tweets, val_tweets, y_train, y_val = self.data_loader.get_training_validation_tweets()
        results = [{"Expectation": y_val[i]} for i in range(0, len(y_val))]
        columns = ["Expectation"] + selected_models
        self.write_unified_file(results, selected_models, columns, "validation_set")

        results = [{} for _ in range(0, 10000)]
        columns = selected_models
        self.write_unified_file(results, selected_models, columns, "result_exact")

    def write_unified_file(self, results, selected_models, columns, file_name):
        """
        read data from files and write unified data to file
        :param results: expected result if known for each row, otherwise list of empty dictionaries
        :param selected_models: list of models used
        :param columns: columns to write
        :param file_name: type of files to read and write
        :return:
        """
        data_path = self.config.get("data_dir")
        # for each model add predictions
        for model in selected_models:
            with open(f"{data_path}{model}_{file_name}.csv") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    results[int(row.get("Id"))][model] = row.get("Prediction")

        # write predictions to file
        with open(f"{data_path}unified_{file_name}.csv", "w") as unified_file:
            header = ",".join(columns)
            unified_file.write(f"{header}\n")
            for entry in results:
                row = ",".join(map(lambda r: str(entry[r]), columns))
                unified_file.write(f"{row}\n")
