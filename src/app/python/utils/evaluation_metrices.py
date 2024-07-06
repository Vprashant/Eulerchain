"""
filename: evaluation_metrices.py
Author: Prashant Verma
email: prashantv@sabic.com
"""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def evaluate_faithfulness(context, response):
    faithfulness_prompt = PromptTemplate(
        input_variables=["context", "response"],
        template=(
            "Context: {context}\n"
            "Response: {response}\n"
            "Evaluate the faithfulness of the response to the context on a scale from 0 to 1. be very precise give me only the score as output. do return any explination.\n"
            "Score:"
        )
    )

    retrival= RunnableParallel({"context": lambda x: "\n---\n".join([d.page_content for d in context]), "response": RunnablePassthrough()})
    faithfulness_chain = retrival | faithfulness_prompt | GlobalData.gemini_llm | StrOutputParser()
  
    score = faithfulness_chain.invoke(response)
    print("Faithful cscore", score)
    return float(score)

def evaluate_context_recall(context, response):
    context_recall_prompt = PromptTemplate(
        input_variables=["context", "response"],
        template=(
            "Context: {context}\n"
            "Response: {response}\n"
            "Evaluate how well the response recalls the context on a scale from 0 to 1. be very precise give me only the score as output. do return any explination.\n"
            "Score:"
        )
    )
    retrival= RunnableParallel({"context": lambda x: "\n---\n".join([d.page_content for d in context]), "response": RunnablePassthrough()})
    context_recall_chain = retrival | context_recall_prompt | GlobalData.gemini_llm | StrOutputParser()
    score = context_recall_chain.invoke(response)
    return float(score)

def evaluate_answer_relevancy(context, response):
    answer_relevancy_prompt = PromptTemplate(
        input_variables=["context", "response"],
        template=(
            "Context: {context}\n"
            "Response: {response}\n"
            "Evaluate the relevancy of the response to the question on a scale from 0 to 1. be very precise give me only the score as output. do return any explination.\n"
            "Score:"
        )
    )
    retrival= RunnableParallel({"context": lambda x: "\n---\n".join([d.page_content for d in context]), "response": RunnablePassthrough()})
    answer_relevancy_chain = retrival | answer_relevancy_prompt | GlobalData.gemini_llm | StrOutputParser()
    score = answer_relevancy_chain.invoke(response)
    return float(score)

def evaluate_context_relevancy(context, response):
    context_relevancy_prompt = PromptTemplate(
        input_variables=["context", "response"],
        template=(
            "Context: {context}\n"
            "Response: {response}\n"
            "Evaluate the relevancy of the response to the context on a scale from 0 to 1. be very precise give me only the score as output. do return any explination.\n"
            "Score:"
        )
    )
    retrival= RunnableParallel({"context": lambda x: "\n---\n".join([d.page_content for d in context]), "response": RunnablePassthrough()})
    context_relevancy_chain = retrival | context_relevancy_prompt | GlobalData.gemini_llm | StrOutputParser()
    score = context_relevancy_chain.invoke( response)
    return float(score)





