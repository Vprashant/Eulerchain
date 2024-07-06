"""
filename: prompt_generator.py
Author: Prashant Verma
email: prashantv@sabic.com
"""

from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData
from src.app.python.constant.prompt_template import prompt_template
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import PromptTemplate
import re


class PromptGeneration(BaseModel):
    response: str
    question: str
    attempts: int = Field(default=2)
    prompt_score: float = Field(default=0.0, init=False)
    prompt: str = Field(default=prompt_template.MULTI_VECTOR_PROMPT, init=False)
    attempt_counter: int = Field(default=0, init=False)
   
    def score_formatter(self, string: str) -> float:
        matches = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", string)
        return float(matches[0]) if matches else 0.0
    
    def prompt_analyzer(self, create_prompt: bool) -> dict:
        try:
            if create_prompt:
                prompt_text = prompt_template.PROMPT_GENERATION_REQUEST
                input_variables = ["question"]
                retrieval = {"question": RunnablePassthrough()}
                input_data = {"question": self.question}
            else:
                prompt_text = prompt_template.PROMPT_VALIDATION_REQUEST
                input_variables = ["question", "prompt", "response"]
                retrieval = {
                    "question": RunnablePassthrough(),
                    "prompt": RunnablePassthrough(),
                    "response": RunnablePassthrough(),
                }
                input_data = {
                    "question": self.question,
                    "prompt": self.prompt,
                    "response": self.response
                }

            prompt_template_obj = PromptTemplate(input_variables=input_variables, template=prompt_text)
            retrieval_obj = RunnableParallel(retrieval)
            prompt_chain = retrieval_obj | prompt_template_obj | GlobalData.gemini_llm | StrOutputParser()
            output = prompt_chain.invoke(input_data)
            print("output--", output)
            if output:
                return {"score": self.score_formatter(output)} if "score" in output else {"prompt": output}
            else:
                print("Warning: Output is None.")
                return {}
                
        except Exception as e:
            print(f"Exception occurred inside prompt analyzing. {e}")
            return {}

    def _get_updated_prompt(self):
        print("In _get_updated_prompt")
        prompt_instance = self.prompt_analyzer(create_prompt=False)
        print("prompt_instance", prompt_instance)
        if isinstance(prompt_instance.get("score"), float) and prompt_instance["score"] >= 0.7:
            self.prompt_score = prompt_instance["score"]
            print("self ---1 prompt score", self.prompt_score)
            self.maintain_state()
            return constant.SUCCESS_STATUS
        else:
            while self.attempt_counter < self.attempts:
                new_prompt_instance = self.prompt_analyzer(create_prompt=True)
                if "prompt" in new_prompt_instance:
                    self.prompt = new_prompt_instance["prompt"]
                prompt_instance = self.prompt_analyzer(create_prompt=False)
                self.attempt_counter += 1 
                self.prompt_score = prompt_instance.get("score", self.prompt_score)  

                if isinstance(prompt_instance.get("score"), float) and prompt_instance["score"] >= 0.7:
                    self.prompt_score = prompt_instance["score"]
                    self.maintain_state()
                    return constant.SUCCESS_STATUS
                else:
                    self.maintain_state()
            return constant.FAILURE_STATUS


    def maintain_state(self):
        current_state = max(list(GlobalData.state_handler_instance.states.keys()))
        state_exists = any(data.get('state_name') == constant.PROMPT_GENERATOR for data in GlobalData.state_handler_instance.get_all_states())
        
        if not state_exists:   
            print("Inside maintin state --", self.attempt_counter + 1, self.prompt_score)  
            GlobalData.state_handler_instance.add_state(
                state_id=current_state + 1,
                state_name=constant.PROMPT_GENERATOR,
                status=constant.SUCCESS_STATUS,
                query=self.question,
                response=None,
                score=self.prompt_score,  # Pass score directly
                iterations=self.attempt_counter + 1  # Pass iterations directly
            )
        else:
            prompt_data_template = """\n Context: {context} \n. Question: {question} \n. 
                                    Additional Params: {additional_params} \n. Answer: """
            
            GlobalData.state_handler_instance.update_state(
                state_name=constant.PROMPT_GENERATOR, 
                status=constant.SUCCESS_STATUS,
                optional_params={
                    'score': self.prompt_score, 
                    'iterations': self.attempt_counter, 
                    'data': f"{self.prompt}" + prompt_data_template.format(context="", question=self.question, additional_params="")
                }
            )



































































































    # def maintain_state(self):
    #     current_state = max(list(GlobalData.state_handler_instance.states.keys()))
    #     if not (any(data.get('state_name') == constant.PROMPT_GENERATOR \
    #                                       for data in GlobalData.state_handler_instance.get_all_states())): 
    #         print("Inside maintin state --", self.attempt_counter + 1, self.prompt_score)  
    #         GlobalData.state_handler_instance.add_state(
    #             state_id= current_state + 1,
    #             state_name=constant.PROMPT_GENERATOR,
    #             status=constant.SUCCESS_STATUS,
    #             query=self.question,
    #             response=None,
    #             optional_params={"score": self.prompt_score, 'iterations': self.attempt_counter + 1}
    #         )
    #     else:
    #         prompt_data_template = """\n Context: {context} \n. Question: {question} \n. 
    #                                 Additional Params: {additional_params} \n. Answer: """
            
    #         GlobalData.state_handler_instance.update_state(state_name = constant.PROMPT_GENERATOR, status=constant.SUCCESS_STATUS, \
    #                                                        optional_params={'score': self.prompt_score, 'iterations': self.attempt_counter, 'data': f"{self.prompt}" + prompt_data_template})






















































































































 # def _get_updated_prompt(self):
    #     print("In _get_updated_prompt")
    #     prompt_instance = self.prompt_analyzer(create_prompt=False)
    #     print("prompt_instance", prompt_instance)
    #     if isinstance(prompt_instance.get("score"), float) and prompt_instance["score"] >= 0.7:
    #         self.prompt_score = prompt_instance["score"]
    #         self.maintain_state()
    #         return constant.SUCCESS_STATUS
    #     else:
    #         while self.attempt_counter < self.attempts:
    #             new_prompt_instance = self.prompt_analyzer(create_prompt=True)
    #             if "prompt" in new_prompt_instance:
    #                 self.prompt = new_prompt_instance["prompt"]
    #             prompt_instance = self.prompt_analyzer(create_prompt=False)
    #             self.attempt_counter += 1 
    #             self.prompt_score = prompt_instance.get("score", self.prompt_score)  

    #             if isinstance(prompt_instance.get("score"), float) and prompt_instance["score"] >= 0.7:
    #                 self.prompt_score = prompt_instance["score"]
    #                 self.maintain_state()
    #                 return constant.SUCCESS_STATUS
    #             else:
    #                 self.maintain_state()
    #         return constant.FAILURE_STATUS

    # def maintain_state(self):
    #     current_state = max(list(GlobalData.state_handler_instance.states.keys()))
    #     state_exists = any(data.get('state_name') == constant.PROMPT_GENERATOR for data in GlobalData.state_handler_instance.get_all_states())
        
    #     if not state_exists:   
    #         GlobalData.state_handler_instance.add_state(
    #             state_id=current_state + 1,
    #             state_name=constant.PROMPT_GENERATOR,
    #             status=constant.SUCCESS_STATUS,
    #             query=self.question,
    #             response=None,
    #             optional_params={"score": self.prompt_score, 'iterations': self.attempt_counter + 1}
    #         )
    #     else:
    #         prompt_data_template = """\n Context: {context} \n. Question: {question} \n. 
    #                                 Additional Params: {additional_params} \n. Answer: """
            
    #         GlobalData.state_handler_instance.update_state(
    #             state_name=constant.PROMPT_GENERATOR, 
    #             status=constant.SUCCESS_STATUS,
    #             optional_params={
    #                 'score': self.prompt_score, 
    #                 'iterations': self.attempt_counter, 
    #                 'data': f"{self.prompt}" + prompt_data_template.format(context="", question=self.question, additional_params="")
    #             }
    #         )

