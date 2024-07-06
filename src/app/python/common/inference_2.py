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
            sabic_dir_path = 'vector_NEW_ALL\\data\\'

            user_question = user_question + "more than 1200 words with headings and conclusion"
            embeddings = GoogleGenerativeAIEmbeddings(model = constant.GOOGLE_EMBEDDING_MODEL)
            new_db = FAISS.load_local(sabic_dir_path, embeddings)
            docs = new_db.similarity_search(user_question)
           
            chain = self.get_conversational_chain('report' if 'report' in user_question else None)
            response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
            
        
            if 'report' in user_question:
     
                report_template = ReportTemplate(response)
                html_content = report_template.render_template()

                pdf_data = self.html_to_pdf(html_content)
                GlobalData.document_report = pdf_data
                GlobalData.llm_response = response["output_text"]
            else:
                GlobalData.llm_response = response["output_text"]

        except Exception as e:
            print(f'Exception Occurred - In user-input function. {e}')
            GlobalData.llm_response = "I don't have '2023' SABIC data and couldn't answer the given question."
            GlobalData.query_template = constant.EMPTY_STRING
           
chat_infernce = Inference()


