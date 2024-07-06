from src.app.python.constant.project_constant import Constant as constant
from src.app.python.common.config_manager import cfg
from sqlalchemy import create_engine, update
from sqlalchemy.orm import scoped_session, sessionmaker
from langchain.docstore.document import Document
class DatabaseConnect:
    """
    To connect with the database
    """

    def __init__(self):
        self.db_uri = "mysql://root:tiger@localhost/sabic_report"

    def create_connection(self):
        try:
            engine = create_engine(self.db_uri, echo=True)
            self.conn = engine.connect()
        except Exception as e:
            raise Exception(e)

    def create_session(self):
        try:
            Session = sessionmaker(bind=self.conn)
            self.db_session = Session()
        except Exception as e:
            raise Exception(e)

    def get_connection(self):
        return self.conn

    def get_session(self):
        return self.db_session

    def delete_connection(self):
        try:
            self.conn.close()
        except Exception as e:
            raise Exception(e)

    def delete_session(self):
        try:
            self.db_session.close()
        except Exception as e:
            raise Exception(e)