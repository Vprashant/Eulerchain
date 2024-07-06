"""
filename: database_connection.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
from src.app.python.constant.global_data import GlobalData
from src.app.python.constant.project_constant import Constant as constant
import pandas as pd
import sqlite3 as sql_db
from langchain.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from src.app.python.common.config_manager import cfg
from sqlalchemy.exc import OperationalError

from langchain.chains import create_sql_query_chain

class DataBaseConnector(object):
    def __init__(self) -> None:
        self.model_path = str(constant.PROJECT_ROOT_DIR)+str(cfg.get_model_config(constant.MODEL_PATH))
        self.db_schema = "sample_sabic.db"
        self.conn_instance = sql_db.connect(self.db_schema)
        ...

    
    def create_connection(self, data_file):
        try:
            doc = pd.read_csv(data_file, encoding='ISO-8859-1')
            columns_declaration = ', '.join(f"{col} TEXT" for col in doc.columns)
            with self.conn_instance as conn:
                conn.execute(f"CREATE TABLE IF NOT EXISTS sample_data ({columns_declaration})")
                conn.commit()
            return True
        except Exception as e:
            if 'already exists' in str(e):
                return False
            

    def get_query_response(self, query, data_file ):
        try:
            connection_status = self.create_connection(data_file)
            model = GoogleGenerativeAI(model=constant.MODEL_GEMINI_PRO, temperature=0.2)
            db = SQLDatabase.from_uri(f"sqlite:///{self.db_schema}")
            sql_chain = create_sql_query_chain(model, db)
            sql_query = sql_chain.invoke({"question": query}).lstrip('SQLQuery:')
            print('sql_query - ',sql_query)
            GlobalData.sql_response = self.conn_instance.execute(f'{sql_query}').fetchall()
            print(f'sql-response: {GlobalData.sql_response}')
        except OperationalError as e:
            print(f'[EXC]: Operational Excetion Occured: {e}.')
        except SyntaxError as e:
            print(f'[EXC]: Syntax Exception Occured {e}.')

            




    

