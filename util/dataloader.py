from string import punctuation
from collections import Counter

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_glove(filename):
    """
    Load pretrained vectors from file 'filename'
    :param filename: the name of teh glove file
    :return: a dictionary with the glove embeddings
    """
    dict_embeddings = {}
    with open(filename, "r+", encoding="utf8") as g:
        for line in g:
            split_line = line.split()
            word = split_line[0]
            emb = np.asarray(split_line[1:], dtype="float32")
            dict_embeddings[word] = emb
    return dict_embeddings


def load_tweets(filename, label, tweets, labels):
    """
    read tweets from file
    :param filename: name of the file
    :param label: whether it's positive or negative (1 or 0)
    :param tweets: the tweets array
    :param labels: the labels array
    """
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
            labels.append(label)


class DataLoader:
    def __init__(self, logger, config):
        self.tweets = None
        self.labels = None
        self.embeddings_matrix = None
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None
        self.vocab_size = None
        self.max_len = None
        self.shuffled_indices = None
        self.logger = logger
        self.config = config
        self.tokenizer = Tokenizer()

    def preprocess_tweets(self, tweets_array, write_log=False):
        """
        remove rare words, remove punctuation and <user>, <url>
        :param tweets_array: the uprocessed tweets
        :param write_log: whether progress should be logged
        :return: an array of preprocessed tweets
        """
        n = tweets_array.shape[0]
        filtered_tweets = []
        word_counts = Counter()
        min_occurrence = self.config.get("word_min_occurrence")
        punctuation_set = set(punctuation)

        result = []
        if write_log:
            self.logger.log_info("Starting preprocessing... This may take a while!")
        modulo = round(n / 20)  # log 5% steps
        for i in range(n):
            if write_log and (i + 1) % modulo == 0:
                progress = round(i / n * 100)
                self.logger.log_metric("preprocessing_process", progress)
            tokens = tweets_array[i].split()
            tokens = [t.lower() for t in tokens]  # lowercasing everything
            # remove punctuation, <user> and <url>
            tokens = [t for t in tokens if
                      not (t in punctuation_set or t == "<user>" or t == "<url>")]
            word_counts.update(tokens)
            tokens = ' '.join(tokens)
            filtered_tweets.append(tokens)

        # do not overfit to tweets that contain a word that is seldom
        for tweet in filtered_tweets:
            tokens = tweet.split()
            tokens = [t for t in tokens if word_counts[t] >= min_occurrence]
            tokens = ' '.join(tokens)
            result.append(tokens)
        return np.array(result)

    def get_training_validation_tweets(self):
        """
        load tweets, split into training and validation set
        :return: training tweets, validation tweets and corresponding labels
        """
        if self.tweets is None:
            tweets = []
            labels = []
            data_path = self.config.get("data_dir")
            pos_file = self.config.get("pos_file")
            neg_file = self.config.get("neg_file")
            load_tweets(data_path + neg_file, 0, tweets, labels)
            load_tweets(data_path + pos_file, 1, tweets, labels)
            # Convert to NumPy array to facilitate indexing
            self.tweets = self.preprocess_tweets(np.array(tweets), True)
            self.labels = np.array(labels)
            # Shuffle training set and split according to config
            self.shuffled_indices = np.random.permutation(len(self.tweets))

        split_idx = int((1 - self.config.get('validation_split')) * len(self.tweets))
        train_indices = self.shuffled_indices[:split_idx]
        val_indices = self.shuffled_indices[split_idx:]
        train_tweets = self.tweets[train_indices]
        val_tweets = self.tweets[val_indices]
        y_train = self.labels[train_indices]
        y_val = self.labels[val_indices]

        return train_tweets, val_tweets, y_train, y_val

    def load_data(self):
        if self.x_train is None:
            train_tweets, val_tweets, y_train, y_val = \
                self.get_training_validation_tweets()
            x_train, x_val, max_len = self.tokenize(train_tweets, val_tweets)

            self.logger.log_metric("dataset_size", len(self.tweets))
            self.x_train = x_train
            self.x_val = x_val
            self.y_train = y_train
            self.y_val = y_val
            self.vocab_size = len(self.tokenizer.word_index) + 1
            self.max_len = max_len
        return self.x_train, self.x_val, self.y_train, self.y_val, self.vocab_size, self.max_len

    def create_emb_matrix(self, embeddings_dict_name, dim1, dim2):
        """
        load embeddings matrix if not already loaded
        :param embeddings_dict_name: name of the embeddings file
        :param dim1: first dimension
        :param dim2: second dimension
        :return: embedding matrix
        """
        if self.embeddings_matrix is None:
            embeddings_dict = load_glove(embeddings_dict_name)
            embeddings_matrix = np.zeros((dim1, dim2))
            for w, i in self.tokenizer.word_index.items():
                emb = embeddings_dict.get(w)
                if emb is not None:
                    embeddings_matrix[i] = emb
            self.embeddings_matrix = embeddings_matrix
        return self.embeddings_matrix

    # Filter out punctuation and stop words

    def tokenize(self, train_tweets, val_tweets):
        """
        Tokenize train_tweets and val_tweets
        :param train_tweets: training tweets
        :param val_tweets: validation tweets
        :return: tokenized train_tweets, tokenized val_tweets, max length of tweet (>= 48)
        """
        self.tokenizer.fit_on_texts(train_tweets)  # maps words to integers
        encoded_tweets = self.tokenizer.texts_to_sequences(train_tweets)  # encodes tweets

        # Pad sequences to the lengths of the longest tweet in the training set
        max_len = max(max([len(s.split()) for s in train_tweets]), 48)  # padding should be at least 48 for NN to work
        x_train = pad_sequences(encoded_tweets, maxlen=max_len, padding='post')

        # Prepare validation set
        encoded_tweets = self.tokenizer.texts_to_sequences(val_tweets)
        x_val = pad_sequences(encoded_tweets, maxlen=max_len, padding='post')

        return x_train, x_val, max_len
