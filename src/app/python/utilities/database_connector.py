"""
filename: database_connector.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
from src.app.python.constant.global_data import GlobalData
from src.app.python.constant.project_constant import Constant as constant
from langchain.sql_database import SQLDatabase
from langchain_google_genai import GoogleGenerativeAI
from src.app.python.utilities.db_connection import DatabaseConnect
from langchain.chains import create_sql_query_chain
from sqlalchemy.exc import OperationalError
from src.app.python.common.config_manager import cfg
from sqlalchemy import text
# from src.app.python.utils.state_handler import StateHandler


class DataBaseConnector(object):
    def __init__(self):
        self.db_connector = None
        self.conn = None
        self.dbSession = None
        self.status = False
        self.db_uri = "mysql+pymysql://root:tiger@localhost/sabic_report"
        self.model_path = str(constant.PROJECT_ROOT_DIR) + str(cfg.get_model_config(constant.MODEL_PATH))
        # self.state_instance = StateHandler()

    def GetAll(self, sql_query):
        """ Fetches all data from table.
        """
        try:
            self.__get_dbInstance()
            sql_query_cleaned = sql_query.strip('`').replace("sql\n", "")
            query = text(sql_query_cleaned)
            dbModel = self.dbSession.execute(query).fetchall()
            return dbModel if dbModel is not None else constant.FAILURE_STATUS
        except Exception as e:
            raise Exception("Exception in select query: {0}".format(e))
        finally:
            self.delete_dbInstance()



    def __get_dbInstance(self):
        '''
        Create database connection and session
        '''
        
        self.db_connector = DatabaseConnect()
        self.db_connector.create_connection()
        self.db_connector.create_session()
        self.conn = self.db_connector.get_connection()
        self.dbSession = self.db_connector.get_session()

    def delete_dbInstance(self): 
        '''
        Delete database connection and session
        '''
        self.db_connector.delete_connection()
        self.db_connector.delete_session()
        self.db_connector = None
        self.conn = None
        self.dbSession = None

    def get_query_response(self, query):
        try:
            model = GoogleGenerativeAI(model=constant.MODEL_GEMINI_PRO, temperature=0.2)
            db = SQLDatabase.from_uri(self.db_uri)
            sql_chain = create_sql_query_chain(model, db)
            sql_query = sql_chain.invoke({"question": query}).lstrip('SQLQuery:')
            res = self.GetAll(sql_query)
            self.status = True
            GlobalData.state_handler_instance.add_state(state_id=1, state_name="sql", status=self.status, \
                                     query=query, response=res)
            print("data - --state ", GlobalData.state_handler_instance.get_state(state_id=1))
        except OperationalError as e:
            print(f'[EXC]: Operational Exception Occurred: {e}.')
        except SyntaxError as e:
            print(f'[EXC]: Syntax Exception Occurred: {e}.')
        except Exception as err:
            if "Background on this error at" in str(err):
                pass
            else:
                print(f'[EXC]: Exception Occurred: {err}.')
            
            GlobalData.state_handler_instance.add_state(state_id=1, state_name="sql", status=self.status, \
                                     query=query, response=None)
