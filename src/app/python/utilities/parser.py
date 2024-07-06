"""
filename: parser.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData
from typing import Optional
import json
import pandas as pd
class Parser(object):
    def __init__(self) -> None:
        ...

    def get_parse_data(self):
        try:
            lines = GlobalData.llm_response.strip().split('\n')
            header = [x.strip() for x in lines[0].strip('|').split('|')]
            data = [dict(zip(header, (x.strip() for x in line.strip().split('|')))) for line in lines[2:]]
            GlobalData.graph_data = pd.DataFrame(data[0].items())
            print('-dfdfdf-', GlobalData.graph_data)
            # GlobalData.graph_data = pd.DataFrame({'num_legs': [2, 4, 8, 0],
            #                             'num_wings': [2, 0, 0, 0],
            #                             'num_specimen_seen': [10, 2, 1, 8]})
            return constant.SUCCESS_STATUS
        except Exception as e:
            print(f'Exception occured in get_parse_data. {e}')
            return constant.FAILURE_STATUS

data_parser = Parser()