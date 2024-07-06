"""
filename:inference.py
Author: Prashant Verma
email: 
"""
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.prompt_template import prompt_template
from src.app.python.constant.global_data import GlobalData
from src.app.python.utilities.parser import data_parser
# from src.app.python.common.visualization import visualization
from src.app.python.common.report_template import ReportTemplate
from src.app.python.utilities.table_handler import TableHandler
from src.app.python.utilities.json_handler import JsonHandler
from io import BytesIO
from xhtml2pdf import pisa
from src.app.python.utils.state_handler import StateHandler
from langchain.chains import LLMChain



load_dotenv()
os.getenv(constant.GOOGLE_API)
genai.configure(api_key=os.getenv(constant.GOOGLE_API))

class Inference:
    def __init__(self) -> None:
        self.chain_params = {'prompt': constant.EMPTY_STRING, 'input_variables': list()}
        GlobalData.graph_status = constant.FAILURE_STATUS
        self.state_instance = StateHandler()
        self.vector_state_status =  constant.FAILURE_STATUS
              

    def get_pdf_text(self, pdf_docs):
        text=constant.EMPTY_STRING
        for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf)
            for page in pdf_reader.pages:
                text+= page.extract_text()
        return  text

    def create_table_data(self, pdf_docs):
        try:
            for pdf in pdf_docs:
                tbl_handler_status = TableHandler(pdf).json_generator()
                if not tbl_handler_status:
                    print(f'table not found in {pdf}.')
                GlobalData.tables_present.append(pdf)
        except Exception as err:
            print(f'f Exception Occured. {err}')

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model = constant.GOOGLE_EMBEDDING_MODEL)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(constant.FAISS_IDX)

    def get_graph_validation(self, user_question): 
        graph_validation = [i for i in list(map(lambda x: x if x in user_question else None, GlobalData.graph_type_keys)) if i is not None]
        if len(graph_validation) != 0:
            print('----', len(graph_validation))
            return graph_validation
        return constant.FAILURE_STATUS
    
    def get_conversational_chain(self, prompt_t):
        print('prompt -t', prompt_t)
        variables = (prompt_template.GRAPH_PROMPT, [constant.CONTEXT, constant.QUESTION]) if prompt_t is None  else \
                                                                    (prompt_template.DEFAULT_PROMPT, [constant.CONTEXT, constant.QUESTION])
        # variables = (prompt_template.REPORT_PROMPT, [constant.CONTEXT, constant.QUESTION])
        model = ChatGoogleGenerativeAI(model=constant.MODEL_GEMINI_PRO, temperature=0)
        prompt = PromptTemplate(template = variables[0], input_variables = variables[1])
        chain = load_qa_chain(model, chain_type=constant.STUFF_CHAIN, prompt=prompt)
        return chain
    
    
    def html_to_pdf(self, html_content, output_path=None):
        pdf_data = BytesIO()
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_data)
        if not pisa_status.err:
            if output_path:
                with open(output_path, "wb") as pdf_file:
                    pdf_file.write(pdf_data.getvalue())
            return pdf_data.getvalue()
        else:
            raise Exception("Failed to generate PDF: %s" % pisa_status.err)


    def user_input(self, user_question):
        try:
            parser_dict = GlobalData.query_template
            sabic_dir_path = 'vectorDB\\sabic\\'
            
            if parser_dict.get(constant.SABIC_IN_QRY) and parser_dict.get(constant.YEAR_QRY) \
                                                 and not parser_dict.get(constant.COMP_IN_QRY):
                target_strings = parser_dict.get(constant.YEAR_QRY)
                vector_dir_lst = os.listdir(sabic_dir_path)
                vector_match = [sublist for sublist in vector_dir_lst if any(target in sublist for target in target_strings)]
                vector_db = [sabic_dir_path + vector_db for vector_db in vector_match]
                print("ppppath --", vector_db)
            elif parser_dict.get(constant.PEERS_IN_QRY) or parser_dict.get(constant.COMP_IN_QRY):
                # execute when there is a comparision 
                vector_db = []
                path = 'vectorDB\\'
                for vector in parser_dict.get(constant.COMP_IN_QRY):
                    if vector == 'sabic':
                        if parser_dict.get(constant.YEAR_QRY):
                            vector_db.append(sabic_dir_path +constant.SABIC.upper()+constant.UNDER_SCORE+parser_dict.get(constant.YEAR_QRY)[0])
                        else:
                            vector_db.append(sabic_dir_path +constant.SABIC.upper()+constant.UNDER_SCORE+'2022')
                        # vector_db.remove('sabic')
                    else:
                        vector_db.append(path+'vendor\\'+ vector)
                print("--sabic compare vector db --", vector_db)
                # vector_db = [sabic_dir_path + vector_db.upper()+constant.UNDER_SCORE+'2022' if vector_db == 'sabic' \
                #                      else path+'vendor\\'+ vector_db for vector_db in parser_dict.get(constant.COMP_IN_QRY)]
                #print("response -- vector_db", vector_db)
            else: 
                vector_db = [sabic_dir_path +constant.SABIC.upper()+constant.UNDER_SCORE+'2022']
                print("-else--vector Match --",vector_db)

            # select_vector = [constant.FAISS_IDX if customer in user_question else constant.SABIC_IDX for customer in GlobalData.customer_list]
            # print("select _ vector ", select_vector)
            
            user_question = user_question + "more than 1200 words with headings and conclusion"
            embeddings = GoogleGenerativeAIEmbeddings(model = constant.GOOGLE_EMBEDDING_MODEL)

            
            for idx, select_vector in enumerate(vector_db):
                print('select --12 vector', select_vector)
                new_db = FAISS.load_local(select_vector, embeddings)
                docs = new_db.similarity_search(user_question)
                self.vector_state_status = constant.SUCCESS_STATUS
                self.state_instance.add_state(state_id=idx+2, state_name=parser_dict.get(constant.COMP_IN_QRY)[idx], status=self.vector_state_status,\
                                                     query=user_question, response=docs)
                
                # print("---state --data from vectors", self.state_instance.get_state(state_id=idx+2))
            
            chain = self.get_conversational_chain('report' if 'report' in user_question else None)
            response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
            
            if self.state_instance.get_state_count() > 1:
                print("inside ---", self.state_instance.get_state_count())
                doc_retreiver = list()
                # print(self.state_instance.states)
                for idx in self.state_instance.states.keys():
                    if self.state_instance.get_state(state_id=idx)['status']:
                        docs = self.state_instance.get_state(state_id=idx)['response']

                document_list = [doc for value in self.state_instance.states.values() for doc in value['response']]
                print("Docssdsdsd ---doc-retreiver", document_list)
                llm = ChatGoogleGenerativeAI(model=constant.MODEL_GEMINI_PRO, temperature=0)
                # try:
                #     print(self.state_instance.get_state)
                #     if self.state_instance.get_state(state_id=1)['status']:
                #         additional_params = self.state_instance.get_state(state_id=1)['response']
                #         variables = (prompt_template.MULTI_VECTOR_PROMPT, [additional_params, constant.CONTEXT, constant.QUESTION])
                #     else: 
                #         variables = (prompt_template.MULTI_VECTOR_PROMPT, [constant.EMPTY_STRING, constant.CONTEXT, constant.QUESTION])
                # except Exception as e:
                #     print(f'exception occured --{e}')
                variables = (prompt_template.MULTI_VECTOR_PROMPT, [constant.CONTEXT, constant.QUESTION])
                
                prompt = PromptTemplate(template = variables[0], input_variables = variables[1])
                # qa_chain = LLMChain(llm=llm, prompt=prompt, metadata= self.state_instance.get_state(state_id=1) \
                #                         if self.state_instance.get_state(state_id=1)['status'] else None)
                qa_chain = LLMChain(llm=llm, prompt=prompt)
                out = qa_chain(
                    inputs={
                        # "additional_params":additional_params,
                        "question": user_question,
                        "context":"\n---\n".join([d.page_content for d in document_list])
                    }
                )
                # print("--sd-sd-s-d", out["text"])
                GlobalData.llm_response = out["text"]

            if 'report' in user_question:
                # try:
                #     table_status, json_tbl = JsonHandler.is_json_available(user_question)
                # except Exception as e:
                #     raise Exception(f'Exception occured handling table: {e}')
                
                report_template = ReportTemplate(response)
                html_content = report_template.render_template()
                # html_content = ReportTemplate(response, table_status, None).render_template() if not table_status \
                #                                  else ReportTemplate(response, table_status, json_tbl).render_template()
                pdf_data = self.html_to_pdf(html_content)
                GlobalData.document_report = pdf_data
                # GlobalData.llm_response = 'requested report created, please download report by click download button'
                GlobalData.llm_response = response["output_text"]
            else:
                GlobalData.llm_response = response["output_text"]
                
            # if self.get_graph_validation(user_question) and data_parser.get_parse_data():
            #     graph_keywords = {'pie':visualization.is_pie(), 'table': visualization.is_table(), 'bar': visualization.is_bar()}
            #     graph_keywords.get(self.get_graph_validation(user_question)[0])
            #     GlobalData.graph_status = True
            #     GlobalData.g_type = self.get_graph_validation(user_question)[0]
        except Exception as e:
            print(f'Exception Occurred - In user-input function. {e}')
            GlobalData.llm_response = "I don't have '2023' SABIC data and couldn't answer the given question."
            GlobalData.query_template = constant.EMPTY_STRING
           
chat_infernce = Inference()


