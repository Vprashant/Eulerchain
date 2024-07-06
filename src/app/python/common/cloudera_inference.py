
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gc
import torch
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.prompt_template import prompt_template
from src.app.python.constant.global_data import GlobalData
from operator import itemgetter
from langchain.chains.llm import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceBgeEmbeddings
import time
from langchain_experimental.agents import create_csv_agent
from langchain.sql_database import SQLDatabase
import sqlite3
from langchain.chains import create_sql_query_chain
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document 
from langchain_core.prompts import ChatPromptTemplate


ROOT_PATH = "/home/cdsw/models/base/" 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def initialization_decorator(func):
    def wrapper(*args, **kwargs):
        args[0].chain_params = {'prompt': constant.EMPTY_STRING, 'input_variables': list()}
        GlobalData.graph_status = constant.FAILURE_STATUS
        args[0].vector_state_status = constant.FAILURE_STATUS
        return func(*args, **kwargs)
    return wrapper

def exception_handling_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f'Exception Occurred: {e}')
            GlobalData.llm_response = "An error occurred while processing the request."
            GlobalData.query_template = dict()
    return wrapper

@dataclass
class Inference:
    chain_params: dict
    vector_state_status: str
    

    @initialization_decorator
    def __init__(self) -> None:
        pass

    @exception_handling_decorator
    def _sql_query_validator(self, query, llm, db):
        try:
            prompt = ChatPromptTemplate.from_messages([("system", prompt_template.SQL_SYSTEM_PROMPT), ("human", "{query}")]).partial(dialect=db.dialect)
            validation_sql_chain =   prompt | llm | StrOutputParser()
            return validation_sql_chain
        except Exception as e:
            print(f'[EXC]: exception occured in query validator')
      
    @exception_handling_decorator
    def get_query_response(self, query, data_file,llm_sql ):
      try:
          conn_instance = sqlite3.connect(data_file)
          db = SQLDatabase.from_uri(f"sqlite:///{data_file}")
          sql_chain = SQLDatabaseChain.from_llm(llm_sql, db, verbose=True)
        #   sql_chain = create_sql_query_chain(llm_sql, db)
        #   final_sql_chain = {"query": sql_chain} | self._sql_query_validator(query, llm_sql, db)
          sql_query = sql_chain.invoke(query)
          print('sql_query - ',sql_query)
          GlobalData.sql_response = Document(page_content = sql_query['result'] if 'result' in sql_query.keys() else None, metedata={
            "source": "Database output"})
          print(f'sql-response: {GlobalData.sql_response}')
          return GlobalData.sql_response
      except Exception as e:
          print(f'[EXC]:  Excetion Occured: {e}.')
      except SyntaxError as e:
          print(f'[EXC]: Syntax Exception Occured {e}.')
           
          

    @exception_handling_decorator
    def user_input(self, user_question):  
        llm_sql = LlamaCpp(
            model_path="/home/cdsw/models/base/Mistral-7B-Instruct-SQL/Mistral-7B-Instruct-SQL-Mistral-7B-Instruct-v0.2-slerp.Q4_K_M.gguf",
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            temperature=0.0,
            verbose=False,
        )
        
        llm = VLLM(model=ROOT_PATH + 'Llama-2-13b-chat-hf',trust_remote_code=True,  
                  max_new_tokens=4096,
                  top_k=10,
                  top_p=0.95,
                  temperature=0.1,
                  )
        sql_response =  self.get_query_response(user_question, '/home/cdsw/SQL_DB/esg.db',llm_sql)
        GlobalData.state_handler_instance.add_state(state_id=1,
                                                         state_name='DATABASE',
                                                         status=True,
                                                         query=user_question, response=sql_response)
        print("create csv - response", GlobalData.state_handler_instance.get_state(state_id=1)['response'])
        parser_dict = GlobalData.query_template
        print("parser_dict --->",parser_dict)
        sabic_dir_path = 'vectorDB/sabic/'
        print("All existing Keys: ", GlobalData.state_handler_instance.states.keys())
        print("SQL Data Extraction", GlobalData.state_handler_instance.get_state(state_id=1))
#        if 'hpde'in sabic_dir_path:
#          vector_db = [sabic_dir_path]
#        print("check staus  ---- SABIC_IN_QRY: {0}, YEAR_QRY:{1},COMP_IN_QRY:{2}, COMP_IN_QRY_LEN{3}".format(parser_dict.get(constant.SABIC_IN_QRY), parser_dict.get(constant.YEAR_QRY), 
#                parser_dict.get(constant.COMP_IN_QRY) ,  parser_dict.get(constant.COMP_IN_QRY)))
#  
        if parser_dict.get(constant.SABIC_IN_QRY) and parser_dict.get(constant.YEAR_QRY) and ('sabic' in parser_dict.get(constant.COMP_IN_QRY) and len(parser_dict.get(constant.COMP_IN_QRY)) == 1):
            target_strings = parser_dict.get(constant.YEAR_QRY)
            vector_dir_lst = os.listdir(sabic_dir_path)
            vector_match = [sublist for sublist in vector_dir_lst if any(target in sublist for target in target_strings)]
            vector_db = [sabic_dir_path + vector_db for vector_db in vector_match]
            print("ppppath --", vector_db)
        elif parser_dict.get(constant.PEERS_IN_QRY) and parser_dict.get(constant.COMP_IN_QRY):
            vector_db = []
            path = 'vectorDB/'
            for vector in parser_dict.get(constant.COMP_IN_QRY):
                if vector == 'sabic':
                    if parser_dict.get(constant.YEAR_QRY):
                        vector_db.append(sabic_dir_path + constant.SABIC.upper() + constant.UNDER_SCORE +
                                         parser_dict.get(constant.YEAR_QRY)[0])
                    else:
                        vector_db.append(sabic_dir_path + constant.SABIC.upper() + constant.UNDER_SCORE + '2022')
                else:
                    vector_db.append(path + 'vendor/' + vector)
            print("--sabic compare vector db --", vector_db)
        else:
            vector_db = [sabic_dir_path + constant.SABIC.upper() + constant.UNDER_SCORE + '2022']
            print("-else--vector Match --", vector_db)

        user_question = user_question + "more than 1200 words with headings and conclusion"
        embeddings = HuggingFaceBgeEmbeddings(model_name= ROOT_PATH + 'all-MiniLM-L6-v2',model_kwargs= {'device': device})  # NOTE: Embedding 
 
        for idx, select_vector in enumerate(vector_db):
            print('select --12 vector', select_vector)
            new_db = FAISS.load_local(select_vector,  embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            print("-- docs", docs)
            self.vector_state_status = constant.SUCCESS_STATUS
            GlobalData.state_handler_instance.add_state(state_id=idx + 2,
                                                         state_name=parser_dict.get(constant.COMP_IN_QRY)[idx] if len(parser_dict.get(constant.COMP_IN_QRY))!=0 else 'Document_Retrieval',
                                                         status=self.vector_state_status,
                                                         query=user_question, response=docs)

        if GlobalData.state_handler_instance.get_state_count() > 0:
            print("inside ---", GlobalData.state_handler_instance.get_state_count())
            try:
                if GlobalData.state_handler_instance.get_state(state_id=1)['status']:
                    additional_params = GlobalData.state_handler_instance.get_state(state_id=1)['response']
                    variables = (prompt_template.MULTI_VECTOR_PROMPT, ['additional_params', constant.CONTEXT, constant.QUESTION])
#                    print("----------Multi Vector Prompt", prompt_template.MULTI_VECTOR_PROMPT, )
                else:
                    variables = (prompt_template.MULTI_VECTOR_PROMPT, ['additional_params',constant.CONTEXT, constant.QUESTION])
            except Exception as e:
                print(f'exception occurred --{e}')

            GlobalData.state_handler_instance.remove_state(state_id=1)
            document_list = [doc for value in GlobalData.state_handler_instance.states.values() for doc in value['response']]
            print("Docssdsdsd ---doc-retriever", document_list)
            print("All existing Keys: ", GlobalData.state_handler_instance.states.keys())
            
#            prompt = PromptTemplate(template=variables[0], input_variables=variables[1])
#            print("prompt----check prompt", prompt)

#            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', llm=llm)
#            qa_chain = LLMChain(llm=llm, prompt=prompt, return_final_only=False, memory=memory)
#            out = qa_chain(
#                inputs={
#                    
#                    "question": itemgetter(user_question),
#                    "context": "\n---\n".join([d.page_content for d in document_list])
#                }
#            )
            prompt = PromptTemplate(template=prompt_template.MULTI_VECTOR_PROMPT, input_variables=['additional_params', 'context', 'question'])
            print('check prompt-------->',prompt)
            retrival= RunnableParallel(
                {"additional_params": lambda x: itemgetter(GlobalData.sql_response), "context": lambda x: "\n---\n".join([d.page_content for d in document_list]),"question": RunnablePassthrough()}
                )
            print('retrival multi vector-------->',retrival)
#            output_parser = StrOutputParser()
            new_chain = retrival | prompt | llm
            out_put = new_chain.invoke(user_question)
#            llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
#            out_put = llm_chain(inputs={
#              "additional_params": itemgetter(GlobalData.sql_response),
#              "question": itemgetter(user_question),
#              "context": "\n---\n".join([d.page_content for d in document_list])
#            })

            print("----out--", out_put)
            #GlobalData.llm_response = out_put["text"]
            GlobalData.llm_response = out_put
        
        else:
            GlobalData.llm_response = "Question is out of context, Please provide another query !!"

chat_infernce = Inference()
