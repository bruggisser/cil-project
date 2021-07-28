class ModelType:
    """Simple interface for all our model types
    """

    def validate(self, tweets):
        """Validation method is called on the validation and kaggle data set and
        is given the cleaned tweets as an input and should return the exact probability of
        a tweet to be 'positive', i.e. = 1 as an array.

        Args:
            tweets [array]: Preprocessed tweets as strings
        Returns:
            [array]: Array of the probabilities of a tweet being 'positive'
        """
        pass

    def name(self):
        """
        Returns:
            [string]: the name of the model
        """
        pass
