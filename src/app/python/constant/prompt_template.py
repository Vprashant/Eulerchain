"""
filename: prompt_template.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""

class PromptTemplate(object):
    def __init__(self) -> None:
        ...

    SABIC_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

    SQL_SYSTEM_PROMPT =  """Double check the user's {dialect} query for common mistakes, including:
                                    - Using NOT IN with NULL values
                                    - Using UNION when UNION ALL should have been used
                                    - Using BETWEEN for exclusive ranges
                                    - Data type mismatch in predicates
                                    - Properly quoting identifiers
                                    - Using the correct number of arguments for functions
                                    - Casting to the correct data type
                                    - Using the proper columns for joins

                                    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

                                    Output the final SQL query only."""

    DEFAULT_PROMPT = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    
    GRAPH_PROMPT = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer and extarct data to plot the 
        proper graph and return all data for graph in json structure format also extract the specific columns and data points. \n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    
    REPORT_PROMPT = """
        Answer the question as detailed as possible with heading and conclusion and Always try to write in more than 500 words if there is not sufficent info try to add relevent information from the paragraphs and try para phrase in proffesional way, 
        if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer and always try to extract taular data if present and convert it into json strcuture format also extract the specific columns and data points. \n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    
    SQL_PROMPT = """Based on the table schema below, write a SQL query that would answer the user's question:
                {context}
                Question: {question}
                SQL Query:"""
    
    RESEARCH_PROMPT = """
        --------
        {research_summary}
        --------
        Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
        The report should focus on the answer to the question, should be well structured, informative, \
        in depth, with facts and numbers if available and a minimum of 1,200 words.

        You should strive to write the report as long as you can using all relevant and necessary information provided.
        You must write the report with markdown syntax.
        You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
        Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
        You must write the report in apa format.
        Please do your best. \n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    MULTI_VECTOR_PROMPT = """
        Use the following pieces of context and additional context is given as SQL response always use SQL response to generate the response 
        to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        use both data and try to make a comparision report and The report should focus on the answer to the question, should be well structured, informative, \
        in depth, with facts and numbers if available and a minimum of 1,200 words.

    
        Please do your best. \n\n
        Context:\n {context}\n
        SQL Response:\n {additional_params}\n
        Question: \n{question} ?\n

        Answer:
        """
    
  

    
    PROMPT_VALIDATION_REQUEST = """
        Question: \n {question} \n
        Prompt: \n {prompt} \n 
        Response: \n {response} \n

        Evaluate the provided question, prompt, and generated response. Analyze whether the provided prompt is a good match for the response in relation to the question on a scale from 0 to 1. \ 
        Be very precise and give only the score as output write in format of output 'score is: 0.7'. Do not return any explanation.

        Chain of Thought:
        1. Understand the question and the information it seeks.
        2. Review the prompt and determine if it effectively addresses the question.
        3. Assess the response to see if it accurately and comprehensively answers the question based on the prompt.
        4. Rate the overall match between the question, prompt, and response.

        Score:
        """


    PROMPT_GENERATION_REQUEST = """
        Question: \n {question} \n

        Generate a detailed, structured prompt based on the user's question to help obtain an accurate and non-hallucinatory response. The prompt should be comprehensive, 
        inspired by the user's query, and organized to support thorough summarization and reporting. It should include the following components:
        - Title
        - Information Section
        - Detailed Subsections
        - Conclusion
        - References

        Chain of Thought:
        1. Identify the core requirement of the user's question.
        2. Design a title that encapsulates the main theme of the query.
        3. Develop an introductory information section that provides an overview of the topic.
        4. Break down the topic into detailed subsections to ensure clarity and depth of the response.
        5. Conclude with a summary that encapsulates the key findings.
        6. Include references to support the content, if available.

        Prompt:
        """

    
    RESPONSE_ERROR_VALIDATOR = """
       I have tested a generated response on prompt tuning after failing to create a proper prompt within the provided attempts. I am sending you a user question, the retrieved context, and the generated response.
        Always remember that I am forwarding this information only when it fails in the following categories: faithfulness, answer relevance, context recall, and context relevancy. Additionally, 
        it should be noted that this information is sent when the prompt generation scores lower than expected.
        Please deeply analyze the user question, context, and generated response, and tell me the hallucination category for the generated response. Based on the given input, categorize the response errors. 
        Response error categories should be given in the format: error_type: " ", explanation: "" (as a dictionary).

        Question: \n {question} \n
        Context: \n {context} \n
        Response: \n {response} \n

        Error Categories:
        1. Reasoning Error: Errors in the logical reasoning applied in the response.
        2. Logical Error: Errors in the logical structure or conclusions in the response.
        3. Instruction Error: Errors where the response does not follow the given instructions.
        4. Context Retrieval Error: Errors in retrieving or applying the provided context.
        5. Response Faithfulness Error: Errors in the truthfulness or accuracy of the response.
        6. Linguistic Error: Errors in language use, grammar, or syntax.
        7. Miscellaneous Error: Errors that do not fall into the above categories.
        
        Chain of Thought:
        Identify the error within the response after understanding the context.
        Map the identified errors to the given categories in a recursive manner.
        Collect all matching error categories. If the error does not fit any specific category, return it as a miscellaneous error.
        Error Categories:
                """
    
    QUERY_GENERATOR = """

        You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines. Only provide the query, no numbering.
        be very precise to give only question in list.
        Original question: {question}

        """

prompt_template = PromptTemplate()