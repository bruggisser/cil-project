import json
import os


class Config:
    """
    Read configurations from 'config/config.json' and give access through method 'get'.
    """

    def __init__(self):
        with open('config/config.json') as config_file:
            configs = json.load(config_file)
        self.configurations = configs
        self.RANDOM_SEED = configs['random_seed']
        comet_config_file = configs['comet_config_file']
        self.COMET_LOGGING = os.path.isfile(comet_config_file)
        self.LOG_DIR = configs['log_dir']
        self.LOG_FILE = self.LOG_DIR + "/" + configs['log_file']

        if self.COMET_LOGGING:
            with open(comet_config_file) as comet_file:
                comet_config = json.load(comet_file)
                self.COMET_API_KEY = comet_config['api_key']
                self.COMET_PROJECT_NAME = comet_config['project_name']
                self.COMET_WORKSPACE = comet_config['workspace']

    def get(self, key):
        """
        Get the property from the configuration file identified by 'key'.
        :param key: the identifier of the property
        :return: the property defined by 'key'
        """
        return self.configurations[key]
