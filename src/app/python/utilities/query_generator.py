"""
filename: query_generator.py
Author: Prashant Verma
email: prashantv@sabic.com
"""
from __future__ import annotations
from typing import Tuple, List, Dict
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData
from pydantic import BaseModel, Field
import re
from src.app.python.constant.prompt_template import prompt_template
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import PromptTemplate
import numpy as np

class QueryGenerator(BaseModel):
    attempts: int = Field(default=2)
    base_question: str = None
    attempt_counter: int = Field(default=0, init=False)
    query_weighted_score: List[int] = Field(default_factory=list, init=False)
    retrivals: List = Field(default_factory=list)
    k_indices: int = Field(default=3, init=False)
    query_counter: int = Field(default=0, init=False)
    reranked_docs:List = Field(default_factory=list, init=False)
    sorted_docs :List = Field(default_factory=list, init=False)
    query_acceptance_response: Dict = Field(default_factory=lambda: {'status': False, 'generated_query': ''})
    create_query: bool = Field(default=False, description="handles query acceptance for checking request are for create query or not \
                                                    and also helps to handle states.")
    generated_query: str=  Field(default=None, init= False)


    def is_query_acceptable(self) -> dict:
        
        if not self.create_query:
            # print("inside --p query acceptable", [score for score, _ in self.sorted_docs if score > 0.3], (len([score for score, _ in self.sorted_docs if score > 0.5]) >= self.k_indices * 0.6))
            return {'status': bool(len([score for score, _ in self.sorted_docs if score > 0.3]) >= self.k_indices * 0.6),
                    'generated_query': self.base_question
                    }
        return {'status': bool(len([score for score, _ in self.sorted_docs if score > 0.3]) >= self.k_indices * 0.6),
                    'generated_query': self.reranked_docs[np.argmax([score for score, _ in self.reranked_docs ])][1] 
                    }
                            

    def cross_encoder(self, new_retrivals=None) -> List:
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")

            retrivals = new_retrivals if new_retrivals is not None else self.retrivals
            pairs = [[self.base_question, doc.page_content] for doc in retrivals]
            scores = cross_encoder.predict(pairs)
            scored_docs = list(zip(scores, retrivals))
            sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
            reranked_docs = [doc for _, doc in sorted_docs][:self.k_indices]
            print("re_ranked_document_list", reranked_docs)
            self.query_weighted_score = [score for score, _ in sorted_docs]
            print("self query weighted score --", self.query_weighted_score)
            return reranked_docs, sorted_docs

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Module not found. Please install sentence_transformers.")
        except Exception as e:
            print(f"[EXC]: Exception occurred inside cross encoder. {e}")

    def invoke_prompt_chain(self, base_question: str, retrivals: List) -> List[str]:
        prompt_text = prompt_template.QUERY_GENERATOR
        input_variables = ["question", "context"]
        retrieval = {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
        input_data = {
            "question": base_question,
            "context": "\n---\n".join([d.page_content for d in retrivals])
        }
        prompt_template_obj = PromptTemplate(input_variables=input_variables, template=prompt_text)
        retrieval_obj = RunnableParallel(retrieval)
        prompt_chain = retrieval_obj | prompt_template_obj | GlobalData.gemini_llm | StrOutputParser()
        queries = prompt_chain.invoke(input_data)
        return queries

    def query_analyzer(self, create_query: bool) -> dict:
        try:
            self.create_query = create_query
            if self.create_query:
                while self.attempts > self.query_counter:
                    self.query_counter += 1
                    new_queries = self.invoke_prompt_chain(self.base_question, self.retrivals)
                    docs = [GlobalData.retriever.get_relevant_documents(query) for query in new_queries]
                    unique_contents = set(doc.page_content for sublist in docs for doc in sublist)
                    unique_docs = [doc for sublist in docs for doc in sublist if doc.page_content in unique_contents]
                    self.reranked_docs, self.sorted_docs = self.cross_encoder(new_retrivals=unique_docs)
                    self.query_acceptance_response = self.is_query_acceptable() 
                    self.maintain_state()
                    if self.query_acceptance_response['status']:
                        self.generated_query = self.query_acceptance_response['generated_query']
                        return constant.SUCCESS_STATUS, self.reranked_docs
                    return constant.FAILURE_STATUS, self.reranked_docs 
            else:
                self.reranked_docs, self.sorted_docs = self.cross_encoder(new_retrivals=None)
                self.query_acceptance_response = self.is_query_acceptable()  
                self.maintain_state()
                print("---generate query --", self.query_acceptance_response )
                print("----", GlobalData.state_handler_instance.get_state_by_name(constant.RE_RANKING))
                if self.query_acceptance_response['status']:
                    self.generated_query = self.query_acceptance_response['generated_query']
                    return constant.SUCCESS_STATUS, self.reranked_docs
                return constant.FAILURE_STATUS, self.reranked_docs
        except Exception as e:
            print(f"[EXC]: Exception occurred in query analyzer. {e}")

    def maintain_state(self):
        current_state = max(list(GlobalData.state_handler_instance.states.keys()))
        state_exists = any(data.get('state_name') == constant.RE_RANKING for data in GlobalData.state_handler_instance.get_all_states())        
        if not state_exists:   
            print("Inside maintin state --", self.attempt_counter + 1, self.query_weighted_score)  
            GlobalData.state_handler_instance.add_state(
                state_id=current_state + 1,
                state_name=constant.RE_RANKING,
                status=self.query_acceptance_response['status'],
                query=self.base_question,
                response=None,
                score=self.query_weighted_score, 
                iterations=self.attempt_counter + 1  
            )
        else: 
            GlobalData.state_handler_instance.update_state(
                state_name=constant.RE_RANKING, 
                status=self.query_acceptance_response['status'],
                optional_params={
                    'score': self.query_weighted_score, 
                    'iterations': self.attempt_counter, 
                    'data': self.generated_query
                }
            )

if __name__ == "__main__":
    retrivals = [...]  
    base_question = "tell me about sabic 2022 report summary."

    query_generator = QueryGenerator(base_question=base_question, retrivals=retrivals)
    query_generator.query_analyzer(create_query=True)












































































































    # def maintain_state(self):
    #     current_state = max(list(GlobalData.state_handler_instance.states.keys()))
    #     if not (any(data.get('state_name') == constant.RE_RANKING \
    #                                       for data in GlobalData.state_handler_instance.get_all_states())): 
    #         GlobalData.state_handler_instance.add_state(
    #             state_id=current_state + 1,
    #             state_name= constant.RE_RANKING,
    #             status=constant.SUCCESS_STATUS,
    #             query=self.base_question,
    #             response=None,
    #             optional_params={"score": self.query_weighted_score, 'iterations': self.query_counter + 1, 'data': base_question}
    #         )
    #     else:
    #         GlobalData.state_handler_instance.update_state(state_name = constant.RE_RANKING, status=constant.SUCCESS_STATUS, \
    #                                                        optional_params={'score': self.query_weighted_score, 'iterations': self.query_counter +1, 'data': self.generated_query})