from util.config import Config
# comet.ml must be loaded before other libraries
from util.logger import Logger

import tensorflow as tf
import numpy as np

from util.modelloader import ModelLoader


def main():
    # load config
    config = Config()
    # init logger
    logger = Logger(config)
    # set seeds
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    # execute models defined in config.json
    model_loader = ModelLoader(config, logger)
    model_loader.execute_models()


if __name__ == "__main__":
    main()
