"""
filename:inference.py
Author: Prashant Verma
email: 
"""

import os
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.prompt_template import prompt_template
from src.app.python.constant.global_data import GlobalData
from src.app.python.common.report_template import ReportTemplate
from operator import itemgetter
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from src.app.python.utils.evaluation_metrices import evaluate_faithfulness, evaluate_answer_relevancy, evaluate_context_recall, evaluate_context_relevancy
from src.app.python.utils.re_ranking import ReRankingCrossEmbedding
from src.app.python.utilities.prompt_generator import PromptGeneration
from src.app.python.utilities.query_generator import QueryGenerator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from src.app.python.utils.llm_error_handler import LLMErrorHandler


class Inference:
    def __init__(self) -> None:
        self.chain_params = {'prompt': constant.EMPTY_STRING, 'input_variables': list()}
        GlobalData.graph_status = constant.FAILURE_STATUS
        self.vector_state_status =  constant.FAILURE_STATUS
        self.failure_state = list()
        self.response_generation_status = constant.FAILURE_STATUS
        self.document_list = list()
        # GlobalData.activate_recurrent_process = constant.FAILURE_STATUS
        
        
    def _get_performance_statistics(self):
        performance_stats = GlobalData.state_handler_instance.get_state_performance_stats()
        for state in performance_stats:
            if all(x < 70 for x in state['optional_params']['score']):
                self.failure_state.append(state['state_name'])
        
    def evaluation_metices_state_handling(self, **scores_args):
        """
        """
        current_state = GlobalData.state_handler_instance.get_state_count()
        if not GlobalData.state_handler_instance.get_state_by_name(constant.EVALUATION_METRICS):
            print("inside maintain state -- ", bool(GlobalData.state_handler_instance.get_state_by_name(constant.EVALUATION_METRICS)))
            GlobalData.state_handler_instance.add_state(
                state_id=current_state + 1,
                state_name=constant.EVALUATION_METRICS,
                status=constant.SUCCESS_STATUS,
                query=self.user_question,
                response=None,
                optional_params={"score": scores_args, 'iterations': self.attempt_counter + 1}
            )
        else:    
            GlobalData.state_handler_instance.update_state(state_name = constant.EVALUATION_METRICS, status=constant.SUCCESS_STATUS, \
                                                            optional_params={'score': scores_args, 'iterations': self.attempt_counter})
            

    def llm_execution(self, retrivals: list):
        
        prompt_text = prompt_template.MULTI_VECTOR_PROMPT if len(GlobalData.generated_prompt) ==0 else GlobalData.generated_prompt
        print("--" * 80 + "\n", prompt_text)
        input_variables = ["question", "context", "additional_params"]
        retrieval = {"question": RunnablePassthrough(), "context": RunnablePassthrough(), "additional_params": RunnableParallel()}
        input_data = {
            "question": self.user_question,
            "context": "\n---\n".join([d.page_content for d in retrivals]),
            "additional_params": self.additional_params
        }
        prompt_template_obj = PromptTemplate(input_variables=input_variables, template=prompt_text)
        retrieval_obj = RunnableParallel(retrieval)
        prompt_chain = retrieval_obj | prompt_template_obj | GlobalData.gemini_llm | StrOutputParser()
        response = prompt_chain.invoke(input_data)
        return response

    def check_all_state_status(self, state_name):
        states = GlobalData.state_handler_instance.get_state_performance_stats()
        print("Inside -- check_state_prompt_status_", states)
        for state in states:
            if state.get('state_name') == state_name:
                optional_params = state.get('optional_params', {})
                score = optional_params.get('score', [])
                iterations = optional_params.get('iterations')

                if (state_name == constant.PROMPT_GENERATOR) and  \
                          ((iterations == 2 and all(s == 0.0 for s in score)) or (iterations ==1 and score >=0.7)):
                    print("Condition met: Perform the necessary action")
                    return False
                else:
                    return state.get('status') # Handling Re-Ranking staus if true or not
        return True

    def user_input(self, user_question):
        try:
            parser_dict = GlobalData.query_template
            sabic_dir_path = 'vectorDB\\sabic_new\\'
            
            if parser_dict.get(constant.SABIC_IN_QRY) and parser_dict.get(constant.YEAR_QRY) and \
                            ('sabic' in parser_dict.get(constant.COMP_IN_QRY) and len(parser_dict.get(constant.COMP_IN_QRY)) == 1):
                
                target_strings = parser_dict.get(constant.YEAR_QRY)
                vector_dir_lst = os.listdir(sabic_dir_path)
                vector_match = [sublist for sublist in vector_dir_lst if any(target in sublist for target in target_strings)]
                vector_db = [sabic_dir_path + vector_db for vector_db in vector_match]
    
            elif parser_dict.get(constant.PEERS_IN_QRY) and parser_dict.get(constant.COMP_IN_QRY):
        
                vector_db = []
                path = 'vectorDB\\'
                for vector in parser_dict.get(constant.COMP_IN_QRY):
                    if vector == 'sabic':
                        if parser_dict.get(constant.YEAR_QRY):
                            vector_db.append(sabic_dir_path +constant.SABIC.upper()+constant.UNDER_SCORE+parser_dict.get(constant.YEAR_QRY)[0])
                        else:
                            vector_db.append(sabic_dir_path +constant.SABIC.upper()+constant.UNDER_SCORE+'2022')
                    else:
                        vector_db.append(path+'vendor\\'+ vector)
                print("--sabic compare vector db --", vector_db)
            
            else: 
                vector_db = [sabic_dir_path +constant.SABIC.upper()+constant.UNDER_SCORE+'2022']
                print("-else--vector Match --",vector_db)
            
            self.user_question = user_question + "more than 1200 words with headings and conclusion"
            print("all states ----", GlobalData.state_handler_instance.states)
            for idx, select_vector in enumerate(vector_db):
                print('select --12 vector', select_vector)

                new_db = FAISS.load_local(select_vector, GlobalData.gemini_embedding)
                docs = new_db.similarity_search(user_question)
                self.vector_state_status = constant.SUCCESS_STATUS
                GlobalData.state_handler_instance.add_state(state_id=idx+2, state_name=parser_dict.get(constant.COMP_IN_QRY)[idx], status=self.vector_state_status,\
                                                     query=user_question, response=docs)
              
            
            if GlobalData.state_handler_instance.get_state_count() > 1:
                print("inside ---", GlobalData.state_handler_instance.get_state_count())
            
                
                try:
                    if GlobalData.state_handler_instance.get_state(state_id=1)['status']:
                        self.additional_params = GlobalData.state_handler_instance.get_state(state_id=1)['response']
                        variables = (prompt_template.MULTI_VECTOR_PROMPT, ["additional_params", constant.CONTEXT, constant.QUESTION])
                        print("----------Multi Vector Prompt", prompt_template.MULTI_VECTOR_PROMPT,)
                    else: 
                        variables = (prompt_template.MULTI_VECTOR_PROMPT, [constant.EMPTY_STRING, constant.CONTEXT, constant.QUESTION])
                        self.additional_params = constant.EMPTY_STRING
                except Exception as e:
                    print(f'exception occured --{e}')
                
                GlobalData.state_handler_instance.remove_state(state_id=1)
                self.document_list = [doc for value in GlobalData.state_handler_instance.states.values() for doc in value['response']]
                print("Docssdsdsd ---doc-retreiver", self.document_list)

                retrivers = new_db.as_retriever()
                # re_ranker = ReRankingCrossEmbedding(user_question, retrivers)._get_reranking(document_list)
                query_analyzer, self.document_list = QueryGenerator(base_question=user_question, retrivals=self.document_list).query_analyzer(create_query=False)
                
                print("Re--ranking, query_analyzer", self.document_list, query_analyzer)

                # if re_ranker:
                #     currect_state = GlobalData.state_handler_instance.get_state_count()
                #     GlobalData.state_handler_instance.add_state(state_id= currect_state +1, state_name="re_ranking", status=constant.SUCCESS_STATUS,\
                #                                      query=user_question, response=docs)
                    
                print("All exsiting Keys: ", GlobalData.state_handler_instance.states.keys())
                prompt = PromptTemplate(template = variables[0], input_variables = variables[1])
                print("prompt----check prompt", prompt)   
                print("length of documnet list", len(self.document_list))

                # response_context = "\n---\n".join([d.page_content for d in document_list]) 
                # qa_chain = LLMChain(llm=GlobalData.gemini_llm, prompt=prompt)
                # out = qa_chain(
                #     inputs={
                #         "additional_params":itemgetter(self.additional_params),
                #         "question": itemgetter(user_question),
                #         "context": "\n---\n".join([d.page_content for d in document_list]) 
                #     }
                # )
                
                context_data = self.document_list
                print("context_data ---", context_data)
                response_out = self.llm_execution(self.document_list)

                print("context_data ---", response_out)
                faith_scr = evaluate_faithfulness(context_data, response_out)
                ans_scr = evaluate_answer_relevancy(context_data, response_out)
                context_scr = evaluate_context_recall(context_data, response_out)
                context_rel_scr = evaluate_context_relevancy(context_data, response_out)
                print("--12, ans_scr", faith_scr, ans_scr, context_scr, context_rel_scr)

                # if all(score < 0.7 for score in (faith_scr, ans_scr, context_scr, context_rel_scr)):
                #     GlobalData.activate_recurrent_process = constant.SUCCESS_STATUS
                print("002032300", bool(all(score < 0.7 for score in (faith_scr, ans_scr, context_scr, context_rel_scr))))
                if not all(score > 0.7 for score in (faith_scr, ans_scr, context_scr, context_rel_scr)):
                    print("Inside -- condition -", faith_scr, ans_scr, context_scr, context_rel_scr)
                    status = PromptGeneration(
                                response= response_out,
                                question=user_question,
                                attempts=2
                        )._get_updated_prompt()
                print("all dict --", GlobalData.state_handler_instance.states)
                # print("--check---", GlobalData.state_handler_instance.get_state_performance_stats())
                        # self.llm_execution(document_list)
                prompt_status = self.check_all_state_status(state_name=constant.PROMPT_GENERATOR)
                if not prompt_status:
                    ...
                    error_handler = LLMErrorHandler(response=response_out, context=self.document_list, question=self.user_question)
                    print(" fail check prompt_status",prompt_status)
                    if self.check_all_state_status(state_name=constant.RE_RANKING):
                        print(f"[INFO]: re-ranking status: {constant.SUCCESS_STATUS}.")
                        error_handler.add_response(response_id=1)
                    else:
                        query_analyzer, self.document_list = QueryGenerator(base_question=user_question, retrivals=self.document_list).query_analyzer(create_query=False)

                    #LLMErrorHandler().
                else:
                    print("Pass ", prompt_status)




                GlobalData.llm_response = response_out
               
            else:
                GlobalData.llm_response =  response_out

        except Exception as e:
            print(f'Exception Occurred - In user-input function. {e}')
            GlobalData.llm_response = "I don't have '2023' SABIC data and couldn't answer the given question."
            GlobalData.query_template = constant.EMPTY_STRING
           
chat_infernce = Inference()


