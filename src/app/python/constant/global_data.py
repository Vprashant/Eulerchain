"""
filename: global_data.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
from src.app.python.constant.project_constant import Constant as constant

class GlobalData(object):
    def __init__(self) -> None:
        ...
    
    exec_environment_config = None
    graph_status = bool
    graph_instance = constant.EMPTY_STRING
    g_type = constant.EMPTY_STRING
    llm_response = constant.EMPTY_STRING
    graph_data = constant.EMPTY_STRING
    graph_type_keys = ['pie', 'table', 'bar']
    response_container = constant.EMPTY_STRING
    document_report = constant.EMPTY_STRING
    tables_present = list()
    sql_response = constant.EMPTY_STRING
    customer_list = ['Linde','LyondellBasell','DOW','DuPont']
    query_template = constant.EMPTY_STRING
    db_instance = constant.EMPTY_STRING
    state_handler_instance = constant.EMPTY_STRING
    vllm_instance = constant.EMPTY_STRING
    source_data = constant.EMPTY_STRING
    gemini_llm = constant.EMPTY_STRING
    gemini_embedding = constant.EMPTY_STRING
    global_prompt = constant.EMPTY_STRING
    activate_llm = constant.EMPTY_STRING
    response_generation_status = constant.EMPTY_STRING
    generated_prompt = constant.EMPTY_STRING
