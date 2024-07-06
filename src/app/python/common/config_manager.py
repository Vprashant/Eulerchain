"""
File : config_manager.py
Description : Reads configuration file and create getter functions for each
              parameter in config.
Created on : 
Author :Prashant Verma
E-mail :
"""

import configparser
import os
from src.app.python.constant.project_constant import Constant as constant #pylint: disable = import-error
from src.app.python.constant.global_data import GlobalData

class ConfigManager:
    """
    This class opens the configuration file and
    reads all the configuration provided inside it
    """
    def __new__(self):
        if not hasattr(self, 'instance'):
            self.file_name = os.path.join(constant.PROJECT_ROOT_DIR, constant.CONFIG_FILE_PATH)
            self.parser = configparser.ConfigParser()
            self.parser.optionxform = str
            self.config = self.parser

            if os.path.exists(self.file_name):
                self.config.read(self.file_name,encoding='utf-8')
            else:
                raise Exception('No Config file found!')

            self.instance = super(ConfigManager, self).__new__(self)

        return self.config


class ReadConfigFile:
    """
    This class reads specific configurations based on user requirement
    """

    objConfig = ''

    def __init__(self):
        self.obj_config = ConfigManager()

    def get_config_object(self):
        return self.objConfig

    def get_default_config(self, param):
        default_config = self.obj_config[constant.CONFIG_DEFAULT]
        return default_config[param]

    def get_resource_config(self, param):
        resource_config = self.obj_config[constant.CONFIG_RESOURCES]
        return resource_config[param]
    
    def get_environment_config(self, param):
        env_config = self.obj_config[GlobalData.exec_environment_config]
        return env_config[param]
    
    def get_model_config(self, param):
        model_config = self.obj_config[constant.CONFIG_MODEL]
        return model_config[param]

cfg = ReadConfigFile()

