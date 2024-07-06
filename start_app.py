"""
filename: new_start_app.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
import streamlit as st
from streamlit_chat import message
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.common.navigation import Navigation
#from src.app.python.common.inference import chat_infernce
#from src.app.python.common.inference_new_1 import chat_infernce
from src.app.python.common.new_inference import chat_infernce
from src.app.python.constant.global_data import GlobalData
#from st_aggrid import AgGrid
#from src.app.python.utilities.database_connection import DataBaseConnector
from src.app.python.utilities.database_connector import DataBaseConnector
from src.app.python.utilities.query_parser import QueryParser
from src.app.python.utils.state_handler import StateHandler
from src.app.python.common.model_init import Initialization
import sys



def main():
    try:
        print(f"[INFO] Model Initializing....")
        status = Initialization().initiate_service()
        if not status:
            print(f'Model Failed to Load, Please check Model decleration file. {sys.exit()}')
        print(f"[INFO] Model Initialize Successfully....")
        
        st.set_page_config(page_title= "SABIC- Assistant", layout="wide")
        st.header("SABIC-Chat Assistant")
        Navigation.navigation_bar()
              
        GlobalData.response_container = st.container()
        container = st.container()

        if constant.HISTORY_TXT not in st.session_state:
            st.session_state[constant.HISTORY_TXT] = []
        if constant.GENERATED_TXT not in st.session_state:
            st.session_state[constant.GENERATED_TXT] = ["Hello, Welcome to SABIC assistant ! \n Ask me anything about from pdf" " 🤗"]
        if constant.PAST_TXT not in st.session_state:
            st.session_state[constant.PAST_TXT] = ["Hey ! 👋"]
        with container:
            with st.form(key='my_form', clear_on_submit=True):     
                input = st.text_area("Query:", placeholder="Ask Questions PDF data here (:", key='input')
                submit_button = st.form_submit_button(label='Send Query')
                
            if submit_button and input:
                GlobalData.state_handler_instance = StateHandler()
                DataBaseConnector().get_query_response(input) # change this line n- (input, './data/report.csv')
                query_analysis = QueryParser()
                status, validation_response = query_analysis.validate_query(input)
             
                if status:
                    st.markdown(f"<span style='color:green'>{validation_response}</span>", unsafe_allow_html=True)
                    query_analysis.query_metadata(input)
                    chat_infernce.user_input(input)
                    
                    st.session_state[constant.PAST_TXT].append(input)
                    st.session_state[constant.GENERATED_TXT].append(GlobalData.llm_response)
                else:
                    st.markdown(f"<span style='color:red; font-weight:bold'>{validation_response}</span>", unsafe_allow_html=True)
                
        if st.session_state[constant.GENERATED_TXT]:
            with GlobalData.response_container:
                for i in range(len(st.session_state[constant.GENERATED_TXT])):
                    message(st.session_state[constant.PAST_TXT][i], is_user=True, key=str(i) + '_user', logo='https://raw.githubusercontent.com/Vprashant/Artificial-Inteligence/master/ESG%20pfp.png') 
                    message(st.session_state[constant.GENERATED_TXT][i], key=str(i), logo='https://raw.githubusercontent.com/Vprashant/Artificial-Inteligence/master/ESG%20Robot%20pfp.png')
                    if GlobalData.graph_status:
                        #AgGrid(GlobalData.graph_data)
                        pass

        if GlobalData.document_report:
            download_button_key = "download_button_" + str(len(st.session_state[constant.GENERATED_TXT]))
            st.download_button(
                label="Download PDF Report",
                data=GlobalData.document_report,
                file_name='report.pdf',
                mime='application/pdf',
                key=download_button_key  
            )

    except Exception as e:
        print(f'Exception Occured in main infernce. {e}')

if __name__ == "__main__":
    main()
