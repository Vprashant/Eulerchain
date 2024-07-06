import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain.docstore.document import Document
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ['GOOGLE_API_KEY'] = "AIzaSyCY3eJIcFCMjp6lirBFA4CIJ7dzlUXASdg"
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class KnowledgeGraph:
    def __init__(self, model_name="gemini-pro", neo4j_uri="bolt://localhost:7687", neo4j_username="neo4j", neo4j_password="password"):
        self.llm = ChatGoogleGenerativeAI(temperature=0, model_name=model_name)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        os.environ["NEO4J_URI"] = neo4j_uri
        os.environ["NEO4J_USERNAME"] = neo4j_username
        os.environ["NEO4J_PASSWORD"] = neo4j_password
        self.graph = Neo4jGraph()

    def create_graph_from_document(self, text):
        documents = [Document(page_content=text)]
        graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        self.graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    def query_graph(self, query):
        cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=ChatGoogleGenerativeAI(temperature=0, model="gemini-pro"),
            qa_llm=ChatGoogleGenerativeAI(temperature=0, model="gemini-pro"),
            validate_cypher=True,
            verbose=True
        )
        result = cypher_chain.run(query)
        return result
    
if __name__ == "__main__":
    kG = KnowledgeGraph()
    kG.create_graph_from_document(text="")
    