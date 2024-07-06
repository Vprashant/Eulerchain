from dataclasses import dataclass
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.prompt_template import prompt_template
from src.app.python.constant.global_data import GlobalData
from src.app.python.utilities.parser import data_parser
from src.app.python.common.report_template import ReportTemplate
from src.app.python.utilities.table_handler import TableHandler
from src.app.python.utilities.json_handler import JsonHandler
from io import BytesIO
from xhtml2pdf import pisa
from operator import itemgetter
from langchain.chains.llm import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.llms.ollama import Ollama

load_dotenv()
os.getenv(constant.GOOGLE_API)
genai.configure(api_key=os.getenv(constant.GOOGLE_API))

def initialization_decorator(func):
    def wrapper(*args, **kwargs):
        args[0].chain_params = {'prompt': constant.EMPTY_STRING, 'input_variables': list()}
        GlobalData.graph_status = constant.FAILURE_STATUS
        args[0].vector_state_status = constant.FAILURE_STATUS
        args[0].ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)
        return func(*args, **kwargs)
    return wrapper

def exception_handling_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f'Exception Occurred: {e}')
            GlobalData.llm_response = "An error occurred while processing the request."
            GlobalData.query_template = constant.EMPTY_STRING
    return wrapper

@dataclass
class Inference:
    chain_params: dict
    vector_state_status: str
    ollama_llm: Ollama

    @initialization_decorator
    def __init__(self) -> None:
        pass
    
    @exception_handling_decorator
    def get_pdf_text(self, pdf_docs):
        text = constant.EMPTY_STRING
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    @exception_handling_decorator
    def create_table_data(self, pdf_docs):
        try:
            for pdf in pdf_docs:
                tbl_handler_status = TableHandler(pdf).json_generator()
                if not tbl_handler_status:
                    print(f'table not found in {pdf}.')
                GlobalData.tables_present.append(pdf)
        except Exception as err:
            print(f'Exception Occurred: {err}')

    @exception_handling_decorator
    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    @exception_handling_decorator
    def get_vector_store(self, text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model=constant.GOOGLE_EMBEDDING_MODEL)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(constant.FAISS_IDX)

    @exception_handling_decorator
    def get_graph_validation(self, user_question):
        graph_validation = [i for i in list(map(lambda x: x if x in user_question else None, GlobalData.graph_type_keys)) if i is not None]
        if len(graph_validation) != 0:
            print('----', len(graph_validation))
            return graph_validation
        return constant.FAILURE_STATUS

    @exception_handling_decorator
    def get_conversational_chain(self, prompt_t):
        print('prompt -t', prompt_t)
        variables = (prompt_template.GRAPH_PROMPT, [constant.CONTEXT, constant.QUESTION]) if prompt_t is None else \
                    (prompt_template.MULTI_VECTOR_PROMPT, [constant.CONTEXT, constant.QUESTION])
        model = ChatGoogleGenerativeAI(model=constant.MODEL_GEMINI_PRO, temperature=0)
        prompt = PromptTemplate(template=variables[0], input_variables=variables[1])
        chain = load_qa_with_sources_chain(model, chain_type=constant.STUFF_CHAIN, prompt=prompt)
        return chain

    @exception_handling_decorator
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

    @exception_handling_decorator
    def user_input(self, user_question):
        parser_dict = GlobalData.query_template
        sabic_dir_path = 'vectorDB\\sabic\\'
        print("All existing Keys: ", GlobalData.state_handler_instance.states.keys())
        print("SQL Data Extraction", GlobalData.state_handler_instance.get_state(state_id=1))
        if parser_dict.get(constant.SABIC_IN_QRY) and parser_dict.get(constant.YEAR_QRY) and (
                parser_dict.get(constant.COMP_IN_QRY) == 'sabic' and len(parser_dict.get(constant.COMP_IN_QRY) == 1)):
            target_strings = parser_dict.get(constant.YEAR_QRY)
            vector_dir_lst = os.listdir(sabic_dir_path)
            vector_match = [sublist for sublist in vector_dir_lst if any(target in sublist for target in target_strings)]
            vector_db = [sabic_dir_path + vector_db for vector_db in vector_match]
            print("ppppath --", vector_db)
        elif parser_dict.get(constant.PEERS_IN_QRY) and parser_dict.get(constant.COMP_IN_QRY):
            vector_db = []
            path = 'vectorDB\\'
            for vector in parser_dict.get(constant.COMP_IN_QRY):
                if vector == 'sabic':
                    if parser_dict.get(constant.YEAR_QRY):
                        vector_db.append(sabic_dir_path + constant.SABIC.upper() + constant.UNDER_SCORE +
                                         parser_dict.get(constant.YEAR_QRY)[0])
                    else:
                        vector_db.append(sabic_dir_path + constant.SABIC.upper() + constant.UNDER_SCORE + '2022')
                else:
                    vector_db.append(path + 'vendor\\' + vector)
            print("--sabic compare vector db --", vector_db)
        else:
            vector_db = [sabic_dir_path + constant.SABIC.upper() + constant.UNDER_SCORE + '2022']
            print("-else--vector Match --", vector_db)

        user_question = user_question + "more than 1200 words with headings and conclusion"
        embeddings = GoogleGenerativeAIEmbeddings(model=constant.GOOGLE_EMBEDDING_MODEL)

        for idx, select_vector in enumerate(vector_db):
            print('select --12 vector', select_vector)
            new_db = FAISS.load_local(select_vector, embeddings)
            docs = new_db.similarity_search(user_question)
            self.vector_state_status = constant.SUCCESS_STATUS
            GlobalData.state_handler_instance.add_state(state_id=idx + 2,
                                                         state_name=parser_dict.get(constant.COMP_IN_QRY)[idx],
                                                         status=self.vector_state_status,
                                                         query=user_question, response=docs)

        if GlobalData.state_handler_instance.get_state_count() > 1:
            print("inside ---", GlobalData.state_handler_instance.get_state_count())
            try:
                if GlobalData.state_handler_instance.get_state(state_id=1)['status']:
                    additional_params = GlobalData.state_handler_instance.get_state(state_id=1)['response']
                    variables = (prompt_template.MULTI_VECTOR_PROMPT, ["additional_params", constant.CONTEXT, constant.QUESTION])
                    print("----------Multi Vector Prompt", prompt_template.MULTI_VECTOR_PROMPT, )
                else:
                    additional_params = ""
                    variables = (prompt_template.MULTI_VECTOR_PROMPT, [constant.EMPTY_STRING, constant.CONTEXT, constant.QUESTION])
            except Exception as e:
                print(f'exception occurred --{e}')

            GlobalData.state_handler_instance.remove_state(state_id=1)
            document_list = [doc for value in GlobalData.state_handler_instance.states.values() for doc in
                             value['response']]
            print("Docssdsdsd ---doc-retriever", document_list)
            llm = ChatGoogleGenerativeAI(model=constant.MODEL_GEMINI_PRO, temperature=0)

            print("All existing Keys: ", GlobalData.state_handler_instance.states.keys())

            prompt = PromptTemplate(template=variables[0], input_variables=variables[1])
            print("prompt----check prompt", prompt)

            from langchain.chains import RetrievalQA
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', llm=llm)
            qa_chain = LLMChain(llm=llm, prompt=prompt, return_final_only=False)
            out = qa_chain(
                inputs={
                    "additional_params": itemgetter(additional_params),
                    "question": itemgetter(
                        user_question + str(additional_params) + "if give use this value as an output of database table use this data to create report"),
                    "context": "\n---\n".join([d.page_content for d in document_list])
                }
            )

            print("----out--", out)
            GlobalData.llm_response = out["text"]
        if 'report' in user_question:
            response = out["text"]
            report_template = ReportTemplate(response)
            html_content = report_template.render_template()
            pdf_data = self.html_to_pdf(html_content)
            GlobalData.document_report = pdf_data
            GlobalData.llm_response = out["text"]
        else:
            GlobalData.llm_response = out["text"]

chat_infernce = Inference()

