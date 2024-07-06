"""
filename: llm_error_handler.py
Author: Prashant Verma
email: prashantv@sabic.com
"""

from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData
from src.app.python.constant.prompt_template import prompt_template
from typing import Optional, Type
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional


class LLMValidator:
    def __init__(self,
                 response: Optional[str] = None,
                 context: Optional[list] = None,
                 question: Optional[str] = None) -> None:
        
        self.response = response
        self.context = context
        self.question = question
    
    def validator(self) -> str:
        prompt_text = prompt_template.RESPONSE_ERROR_VALIDATOR
        print("===df==", prompt_text)
        input_variables = ["question", "context", "response"]
        retrieval = {"question" : RunnablePassthrough(), "context": RunnablePassthrough(), "response": RunnablePassthrough()}
        input_data = {
            "question": self.question,
            "context": "\n---\n".join([d.page_content for d in self.context]),
            "response": self.response
        }
        prompt_template_obj = PromptTemplate(input_variables=input_variables, template=prompt_text)
        retrieval_obj = RunnableParallel(retrieval)
        prompt_chain = retrieval_obj | prompt_template_obj | GlobalData.gemini_llm | StrOutputParser()
        output = prompt_chain.invoke(input_data)
        return output 


class LLMErrorHandler(LLMValidator):
    def __init__(self, 
                 response: Optional[str] = None,
                 context: Optional[list] = None,
                 question: Optional[str] = None):
        super().__init__(response=response, context=context, question=question)
        self.error_codes = {
            'logical_error': {
                'code': 1001,
                'description': 'Logical inconsistency in the response.'
            },
            'reasoning_error': {
                'code': 1002,
                'description': 'Incorrect reasoning or flawed argumentation.'
            },
            'miscellaneous_error': {
                'code': 1003,
                'description': 'Other types of errors not categorized as logical or reasoning errors.'
            }
        }
        self.responses = []

    def add_response(self, response_id):
        validatore_response = self.validator()
        print("#@######", validatore_response)
        error_type, explanation = validatore_response['error_type'], validatore_response['explanation']
        if error_type and error_type not in self.error_codes:
            raise ValueError(f"Invalid error type: {error_type}")
        response = {
            'response_id': response_id,
            'response_text': self.response,
            'error_type': error_type, 
            'explanation': explanation,
            'status': 'validated' if not error_type else 'error_detected'
        }
        self.responses.append(response)

    def validate_response(self, response_text):
        if "logical" in response_text:
            return 'logical_error', "Detected a logical inconsistency."
        elif "reason" in response_text:
            return 'reasoning_error', "Detected incorrect reasoning."
        else:
            return None, None

    def process_responses(self):
        for response in self.responses:
            if response['status'] == 'validated':
                error_type, explanation = self.validate_response(response['response_text'])
                if error_type:
                    response['error_type'] = error_type
                    response['explanation'] = explanation
                    response['status'] = 'error_detected'

    def categorize_errors(self):
        categorized_errors = {
            'logical_errors': [],
            'reasoning_errors': [],
            'miscellaneous_errors': []
        }
        for response in self.responses:
            if response['status'] == 'error_detected':
                if response['error_type'] == 'logical_error':
                    categorized_errors['logical_errors'].append(response)
                elif response['error_type'] == 'reasoning_error':
                    categorized_errors['reasoning_errors'].append(response)
                else:
                    categorized_errors['miscellaneous_errors'].append(response)
        return categorized_errors

    def get_error_descriptions(self):
        return {key: value['description'] for key, value in self.error_codes.items()}

    def get_response(self, response_id):
        for response in self.responses:
            if response['response_id'] == response_id:
                return response
        raise ValueError(f"Response with ID '{response_id}' not found.")

    def remove_response(self, response_id):
        self.responses = [response for response in self.responses if response['response_id'] != response_id]

# error_handler = LLMErrorHandler()
# error_handler.add_response(response_id=1, response_text="This is a logical error example.")
# error_handler.add_response(response_id=2, response_text="This response shows reasoning error.")
# error_handler.process_responses()
# categorized_errors = error_handler.categorize_errors()
# print(categorized_errors)
