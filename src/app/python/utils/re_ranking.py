"""
filename: re_ranking.py
Author: Prashant Verma
email: prashantv@sabic.com
"""
import umap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.app.python.constant.global_data import GlobalData
from src.app.python.constant.project_constant import Constant as constant
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import torch


    
class ReRankingCrossEmbedding:
    def __init__(self, query, retriever, show_global_embedding=False, k=5) -> None:
        self.status = constant.FAILURE_STATUS
        self.ROOT_PATH = "/home/cdsw/models/base/" 
        self.query = query
        self.retriever = retriever
        self.show_global_embedding = show_global_embedding
        self.output_parser = StrOutputParser()
        self.k = k
        self.re_ranking_status = constant.FAILURE_STATUS
        self.reranking_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from a vector
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search.
            Provide these alternative questions separated by newlines. Only provide the query, no numbering.
            be very precise to give only question in list.
            Original question: {question}""",
        )

        try:
            from sentence_transformers import CrossEncoder
            # , "cross-encoder/nli-deberta-v3-large"
            self.cross_encoder = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Module not found. Please install sentence_transformers.")
        
        try:
            from langchain_community.document_transformers import (
                LongContextReorder)
            self.reordering = LongContextReorder()
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Module not found. Please install sentence_transformers.")
#        self.umap_transformer = umap.UMAP(random_state=0, transform_seed=0).fit(self.vectors)

    
#    def _get_umap_embed(self):
#        umap_embeddings = np.array([self.umap_transformer.transform([vector])[0] for vector in tqdm(self.vectors)])
#        return umap_embeddings
#
#    def calc_global_embeddings(self):
#        q_embedding = self.embeddings.embed_query(self.query)
#        docs = self.retriever.get_relevant_documents(self.query)
#        page_contents = [doc.page_content for doc in docs]
#        vectors_content_vectors = self.embeddings.embed_documents(page_contents)
#        query_embeddings = self.embeddings.umap_transform(q_embedding)
#        retrieved_embeddings = self.embeddings.umap_transform(vectors_content_vectors)
#
#        global_embeddings = self._get_umap_embed()
#
#        plt.figure()
#        plt.scatter(global_embeddings[:, 0], global_embeddings[:, 1], s=10, color='gray')
#        plt.scatter(query_embeddings[:, 0], query_embeddings[:, 1], s=150, marker='X', color='r')
#        plt.scatter(retrieved_embeddings[:, 0], retrieved_embeddings[:, 1], s=50, facecolors='none', edgecolors='g')
#        plt.gca().set_aspect('equal', 'datalim')
#        plt.title(f'{self.query}')
#        plt.axis('off')
#        plt.show()

    def _get_unique_retrivals(self, context):  
        retrival= RunnableParallel({"context": lambda x: "\n---\n".join([d.page_content for d in context]), "question": RunnablePassthrough()})
        retrival_chain = retrival | self.reranking_prompt | GlobalData.gemini_llm | self.output_parser
        quaries = retrival_chain.invoke(self.query)
        print("quaries", quaries)
        docs = [self.retriever.get_relevant_documents(query) for query in quaries]
        unique_contents = set(doc.page_content for sublist in docs for doc in sublist)
        unique_docs = [doc for sublist in docs for doc in sublist if doc.page_content in unique_contents]
        unique_contents = list(unique_contents)
        print("unique_contents", unique_contents)
        return unique_docs

    def _get_cross_embeddings(self, retrivals):
        pairs = [[self.query, doc.page_content] for doc in retrivals]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = list(zip(scores, retrivals))
        print("re_ranked scored_docs",scored_docs)
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for _, doc in sorted_docs][0: self.k]
        print("re_ranked_document_list", reranked_docs)
        return reranked_docs

    def _get_reranking(self, context):
#        if self.show_global_embedding:
#            self.calc_global_embeddings()
        # queries = self._get_unique_retrivals(context)
        # retrivals = self._get_cross_embeddings(queries)
        
        self.re_ranking_status = constant.SUCCESS_STATUS

        return self.re_ranking_status
