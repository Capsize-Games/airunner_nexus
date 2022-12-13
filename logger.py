"""
Wrapper functions for logging
"""
import logging
import os
import tempfile


class Logger:
    def __init__(self):
        #path = os.path.join(f'{tempfile.gettempdir()}\Capsize Games\Logs\log.txt')
        path = os.path.join(f'./krita-stable-diffusion.log')
        self.logger = logging.getLogger()
        # make logger show line number
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d")
        # check if path exists
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.file_handler = logging.FileHandler(path)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.formatter)
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.DEBUG)
        self.stream_handler.setFormatter(self.formatter)
        #self.logger.addHandler(self.file_handler)
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
