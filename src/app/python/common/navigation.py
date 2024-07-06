"""
filename:navigation.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
import streamlit as st
from PIL import Image

class Navigation:
    def __init__(self) -> None:
        pass

    def add_logo(logo_path, width, height):
        """Read and return a resized logo"""
        logo = Image.open(logo_path)
        modified_logo = logo.resize((width, height))
        return modified_logo
    
    def navigation_bar():
        with st.container():
            selected = option_menu(
                menu_title=None,
                options=["Home", "Upload", "Analytics", 'Settings', 'Contact'],
                icons=['house', 'cloud-upload', "graph-up-arrow", 'gear', 'phone'],
                menu_icon="cast",
                orientation="horizontal",
                styles={
                    "nav-link": {
                        "text-align": "left",
                        "--hover-color": "#eee",
                    }
                }
            )
            if selected == "Analytics":
                switch_page("Analytics")
            if selected == "Contact":
                switch_page("Contact")
            if selected == "Settings":
                switch_page("Settings")