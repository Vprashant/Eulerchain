from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from st_aggrid import AgGrid

class Visualization(object):
    def __init__(self, 
                 ) -> None:
        self.g_status = False
    
    def is_pie(self):
        df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
        df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries' 
        return px.pie(df, values='pop', names='country', title='Population of European continent')
       

    def is_table(self):
        try:           
            with GlobalData.response_container:
                st.table(GlobalData.graph_data)
         
        except Exception as e:
            st.write('Exception might occured due to unsufficent information thats \n leads to create Graphs, Kindly provide a complete information !')
            st.write(GlobalData.graph_data)
    
    def is_bar(self):
        try:
            with GlobalData.response_container:
                st.bar_chart(GlobalData.graph_data)
        except Exception as e:
            st.write('Exception might occured due to unsufficent information thats \n leads to create Graphs, Kindly provide a complete information !')
            st.write(GlobalData.graph_data)
    
visualization = Visualization()


    