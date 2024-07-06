"""
filename: project_constant.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
import os
class Constant(object):
    """
    """


    DOT = '.'
    FORWARD_SLASH = '/'
    PIPE = '|'
    UNDER_SCORE = '_'
    AT_THE_RATE = '@'
    COMMA = ','
    EMPTY_STRING = ''
    COLON = ':'
    SPACE = ' '
    RIGHT_ARROW = '->'
    LEFT_ARROW = '<-'

    GOOGLE_API = "GOOGLE_API_KEY"
    GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
    FAISS_IDX = "faiss_index"
    CONTEXT = "context"
    QUESTION = "question"
    EMPTY_STRING = ''
    HISTORY_TXT = 'history'
    GENERATED_TXT = 'generated'
    PAST_TXT = 'past'
    SUCCESS_STATUS = True
    FAILURE_STATUS = False
    MODEL_GEMINI_PRO = "gemini-pro"
    STUFF_CHAIN = "stuff"
    PROJECT_ROOT_DIR = os.path.abspath(os.curdir)
    CONFIG_FILE_PATH = 'configuration/config.ini'
    CONFIG_RESOURCES = 'RESOURCES'
    CONFIG_MODEL = 'MODEL'
    CONFIG_DEFAULT = 'DEFAULT'
    HEADER_IMAGE_FILE = 'HeaderImageFile'
    FOOTER_IMAGE_FILE = 'FooterImageFile'
    MODEL_PATH = 'ModelPath'
    CONFIG_ENVIRONMENT_DATABASE_URI = 'DatabaseUri'
    SABIC_IDX = "sabic_vector_db"
    PEERS_LIST = ['Linde','LyondellBasell','DOW','DuPont']
    REQUIRED_KEYWORDS = ['DOW','DuPont','GRI', 'LINDE','LYONDELLBASELL' ,'year', 'comparison', 'sabic']
    PEERS_IN_QRY = 'peers_in_query'
    IS_PEERS_QRY = 'is_peers_in_query'
    YEAR_QRY = 'year_mention'
    SABIC_IN_QRY = 'sabic_mention'
    COMP_IN_QRY = 'comparision'
    SABIC = 'sabic'
    PROMPT_GENERATOR = 'prompt_generator'
    RE_RANKING = 're_ranking'
    EVALUATION_METRICS = 'evaluation_metrics'