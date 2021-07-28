import csv


class ModelAverage:
    """
    Take the average of the prediction of each model.
    """
    def __init__(self, config, logger, data_loader):
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.selected_models = [m for m in self.config.get("models").split(",") if m != "bert"]

        correct = 0
        wrong = 0

        with open(f"{self.config.get('data_dir')}unified_validation_set.csv", "r") as csv_file:
            reader = csv.DictReader(csv_file)
            weights = {}
            weighted_sum = 0
            for model in self.selected_models:
                weights[model] = self.config.get(f"vote_tie_breaker")[model]
                weighted_sum += weights[model]
            for row in reader:
                values = [float(row.get(column)) * weights[column] for column in self.selected_models]
                avg = sum(values) / weighted_sum
                result = 1 if avg >= 0.5 else 0
                expectation = int(row.get("Expectation"))
                if result == expectation:
                    correct += 1
                else:
                    wrong += 1

        logger.log_metric("average_aggregation_accuracy", (correct / (correct + wrong)))

    def validate(self, csv_file):
        reader = csv.DictReader(csv_file)
        weights = {}
        weighted_sum = 0
        for model in self.selected_models:
            weights[model] = self.config.get(f"vote_tie_breaker")[model]
            weighted_sum += weights[model]
        result_list = []
        for row in reader:
            values = [float(row.get(column)) * weights[column] for column in self.selected_models]
            avg = sum(values) / weighted_sum
            result = 1 if avg >= 0.5 else 0
            result_list.append(result)
        return result_list

    def name(self):
        return "avg_"
