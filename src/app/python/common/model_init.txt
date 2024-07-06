import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData

load_dotenv()
os.getenv(constant.GOOGLE_API)
genai.configure(api_key=os.getenv(constant.GOOGLE_API))

class Initialization:
    def __init__(self) -> None:
        self.model_status = constant.FAILURE_STATUS

    def initiate_service(self):
        try:
            GlobalData.gemini_llm = ChatGoogleGenerativeAI(model=constant.MODEL_GEMINI_PRO, temperature=0)
            GlobalData.gemini_embedding = GoogleGenerativeAIEmbeddings(model = constant.GOOGLE_EMBEDDING_MODEL)
            self.model_status = constant.SUCCESS_STATUS
            return self.model_status
        except Exception as e:
            print(f"Exception occurs in service initialization {e}")
            return self.model_status

