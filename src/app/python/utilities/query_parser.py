"""
filename: query_parser.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
from src.app.python.constant.project_constant import Constant as constant
from typing import Optional, Type
from src.app.python.constant.global_data import GlobalData 
import re

class QueryParser(object):
    def __init__(self) -> None:
        self.query_template = dict().fromkeys([constant.PEERS_IN_QRY, constant.IS_PEERS_QRY, \
                                               constant.YEAR_QRY, constant.SABIC_IN_QRY, constant.COMP_IN_QRY], None)

    def query_metadata(self, user_query) -> dict:
        print(self.query_template)
        for key in self.query_template.keys():
            if key == constant.PEERS_IN_QRY:
                self.query_template[key] = [q_peer.lower() for q_peer in constant.PEERS_LIST if q_peer.lower() in user_query.lower()]
            elif key == constant.IS_PEERS_QRY:
                self.query_template[key] = any(peer is not None for peer in self.query_template.get(constant.PEERS_IN_QRY, []))
            elif key == constant.YEAR_QRY:
                year_mentions = re.findall(r'\b(?:\d{4}|\d{4}|\d{4})\b', user_query, re.IGNORECASE)
                self.query_template[key] = year_mentions if year_mentions else False
            elif key == constant.SABIC_IN_QRY:
                self.query_template[key] = True if constant.SABIC in user_query.lower() else False
            elif key == constant.COMP_IN_QRY:
                peers_in_query = self.query_template.get(constant.PEERS_IN_QRY)
                sabic_mention = self.query_template.get(constant.SABIC_IN_QRY)
                if peers_in_query is not None:
                    comparision_list = [peer.lower() for peer in peers_in_query]
                    if sabic_mention:
                        comparision_list.append(constant.SABIC.lower())
                    self.query_template[key] = comparision_list
                else:
                    self.query_template[key] = []
        GlobalData.query_template = self.query_template
        print("--------dfdf", GlobalData.query_template)

     
    def validate_query(self, user_query)-> tuple:
        """
        """
        try:
            missing_keywords = [keyword for keyword in constant.REQUIRED_KEYWORDS if \
                                                keyword.lower() not in user_query.lower()]
            if len(missing_keywords) == len(constant.REQUIRED_KEYWORDS):
                suggestions = ", ".join(missing_keywords)
                #return (False, f"The query is missing the following keywords: {suggestions}. Please include them in your query.")
                return (False, f"Please direct your questions related to 'SABIC or its Peers'. Please include them in your query.")
            else:
                return (True, f"The query is valid.")
        except Exception as e:
            print(f'Exception Occured while handling validate query: {e}')

    












