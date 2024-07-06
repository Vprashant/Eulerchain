"""
filename: json_handler.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
from typing import Dict, Any, Optional
import glob
import json
from fuzzywuzzy import fuzz
class JsonHandler(object):
    def __init__(self) -> None:
        ...
    
    def load_json(self, file_name: Optional[str]= None,
              dir_path: Optional[str] = None):
        if not file_name:
            data = []
            if not glob.glob('*.json'):
                print(f'There are not any ".json" exsit')
            else:
                files = [file for file in list(glob.glob('*.json'))]
                for file in files:
                    with open(file, 'r') as fp:
                        json_data = json.load(fp)
                        data.append((file, json_data))
                return data
        else:
            with open(file_name, 'r') as file:
                return json.load(file)
        

    def json_similarity(self, query: Optional[str] = None, data:Optional[str] = None):
        """
        """
        threshold = 80
        highest_ratio= 0
        for file_path, json_data in data: 
            ratio = fuzz.partial_ratio(query, json_data.get('header'))
            if ratio > highest_ratio and ratio >= threshold:        
                return (file_path, ratio)
    

    def is_json_available(self, query: Optional[str] = None):
        """
        """
        json_data = self.load_json()
        filepath, ratio = self.json_similarity(query, json_data)
        table_json = self.load_json(filepath)
        if not table_json:
            return False
        return True, table_json