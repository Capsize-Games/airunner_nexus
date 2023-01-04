"""
Wrapper functions for logging
"""
import logging


class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d")
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.DEBUG)
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def debug(self, msg):
        """
        Log info message
        :param msg:
        :return: None
        """
        self.logger.debug(msg)

    def info(self, msg):
        """
        Log info message
        :param msg:
        :return: None
        """
        self.logger.info(msg)

    def warning(self, msg):
        """
        Log warning message
        :param msg:
        :return: None
        """
        self.logger.warning(msg)

    def error(self, msg):
        """
        Log error message
        :param msg:
        :return: None
        """
        self.logger.error(msg)


logger = Logger()
