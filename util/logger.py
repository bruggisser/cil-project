from comet_ml import Experiment
import logging
import os
import shutil


class Logger:
    """
    Class Logger is used to write output into the log file (as defined in config.json) as well as to comet.ml (as
    defined in comet.json).
    """

    def __init__(self, config):
        self.config = config
        if os.path.exists(config.LOG_DIR) and os.path.isdir(config.LOG_DIR):
            shutil.rmtree(config.LOG_DIR, ignore_errors=True)
        os.mkdir(config.LOG_DIR)
        logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO)
        self.experiment = None

    def start_experiment(self):
        """start a new experiment in comet.ml"""
        if self.config.COMET_LOGGING:
            self.experiment = Experiment(
                api_key=self.config.COMET_API_KEY,
                project_name=self.config.COMET_PROJECT_NAME,
                workspace=self.config.COMET_WORKSPACE
            )

    def set_epoch(self, epoch):
        if self.experiment:
            self.experiment.set_epoch(epoch)
        logging.info("=" * 30)
        logging.info(f"Epoch {epoch + 1}")

    def log_metric(self, key, value):
        r"""log the value of metric with identifier 'key' to 'value'.

        Args:
            key (string): key identifying the metric
            value (any): current value of the metric
        """
        if self.experiment:
            self.experiment.log_metric(key, value)
        logging.info(f"{key}: {value}")

    def log_info(self, value):
        if self.experiment:
            self.experiment.log_text(value)
        logging.info(value)

    def log_error(self, value):
        if self.experiment:
            self.experiment.log_text(value)
        logging.error(value)
