import numpy as np


class Validator:
    """The validator class is used to validate against the Kaggle dataset and the
    validation split of the training data.
    """

    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config

    def validate(self, model):
        """takes a model and creates two files for the Kaggle validation set. 
        A file with the exact probabilities (for the 'positive' class) and one with the predicted class (1, -1)

        Args:
            model ([ModelType]): The classification model
        """

        data_path = self.config.get("data_dir")
        exact_result_file = open(f"{data_path}{model.name()}result_exact.csv", "w")
        result_file = open(f"{data_path}{model.name()}result.csv", "w")

        # write file headers
        result_file.write("Id,Prediction\n")
        exact_result_file.write("Id,Prediction\n")

        # load the test data / tweets
        tweet_ids = []
        tweets = []
        with open(f'{data_path}test_data.txt', "r") as validation_file:
            for entry in validation_file:
                entry = entry.strip()
                tweet_id, tweet = entry.split(",", 1)
                tweet_ids.append(tweet_id)
                tweets.append(tweet)
        tweets = np.array(tweets)

        # preprocess validation data with the same dataloader
        tweets = self.data_loader.preprocess_tweets(tweets)

        # expected format: [[0.003],[0.724],[0.16] etc.], i.e. list of singletons between 0 and 1
        result = model.validate(tweets)

        for idx in range(len(result)):
            estimate, = result[idx]  # result is array of arrays
            exact_result_file.write(f"{int(tweet_ids[idx]) - 1},{estimate}\n")  # exact value, 0-based index
            estimate = 1 if estimate >= 0.5 else -1
            result_file.write(f"{int(tweet_ids[idx])},{estimate}\n")  # rounded value, 1-based index for kaggle

        # close the files
        exact_result_file.close()
        result_file.close()

    def create_validation_set(self, model):
        """takes a model and creates a file with the validation ground truth and the prediction
        for the validation split of the training data

        Args:
            model ([ModelType]): The classification model
        """

        data_path = self.config.get("data_dir")
        # get the same validation for all models
        train_tweets, val_tweets, y_train, y_val = self.data_loader.get_training_validation_tweets()
        val_tweets = np.array(val_tweets)
        val_tweets = self.data_loader.preprocess_tweets(val_tweets)

        # write the predictions to a csv file
        with open(f"{data_path}{model.name()}validation_set.csv", "w") as validation_file:
            validation_file.write("Id,Expectation,Prediction\n")
            result = model.validate(val_tweets)
            result = [f"{i},{y_val[i]},{entry[0]}\n" for i, entry in enumerate(result)]
            for entry in result:
                validation_file.write(entry)

    def validate_aggregation(self, model):
        """
        uses the aggregations to predict outcomes based on previous outputs from other models and stores result in a
        file.

        :param model: the model, either random forest, majority vote or average
        """
        data_path = self.config.get("data_dir")
        result_file = open(f"{data_path}{model.name()}result.csv", "w")

        # write file headers
        result_file.write("Id,Prediction\n")

        with open(f'{data_path}unified_result_exact.csv', "r") as csv_file:
            result = model.validate(csv_file)

        for idx in range(len(result)):
            estimate = result[idx]  # result is flat array
            estimate = 1 if estimate >= 0.5 else -1
            result_file.write(f"{idx + 1},{estimate}\n")  # rounded value, 1-based index

        # close the file
        result_file.close()
