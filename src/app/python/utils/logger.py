"""
File : logger.py
Description : Class is responsible logging info and errors
Created on : 
Author : Prashant Verma
E-mail :
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from src.app.python.constant.project_constant import Constant as constant #pylint: disable = import-error


class Logger(object):
    """
    This class responsible to Logger Object to write application log
    """

    @staticmethod
    def get_logger():
        """
            This method returns a logger object and configures logging as per
            Parameters defined in configuration file
        Return:
            logger: It returns logger object.
        """
        logger = None
        try:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                logging_file_path = os.path.join(constant.PROJECT_ROOT_DIR, constant.LOGGER_FILE_PATH)
                logger_level = constant.LOGGER_LEVEL
                logging.basicConfig(format='%(asctime)s {%(pathname)s:%(lineno)d} - %(levelname)s - %(message)s', handlers=[
                   RotatingFileHandler(filename=logging_file_path, maxBytes=constant.MAX_BYTES, backupCount=constant.BACKUP_COUNT, encoding="utf-8")])
                logger.setLevel(logger_level)
        except Exception as error:
            logger.error(error)
            raise Exception()
        return logger
