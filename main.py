import streamlit as st

from streamlit_option_menu import option_menu


st.set_page_config(page_title="Gen AI use cases", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

import upload_pdf_index, sql_generator, chat_v2


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Select App', # Heath Wiser
                options=['Upload PDF Files', 'EDA with LLMs', 'Bedrock-Chatbot'], #
                icons=['cloud-upload', 'file-earmark-bar-graph', 'robot'], # 
                menu_icon='robot',
                default_index=0,
                styles={
                        "container": {"padding": "6!important", "background-color": "#FFFCEF"},
                        "icon": {"color": "WHITE", "font-size": "20px"}, 
                        "nav-link": {"font-size": "14px", "text-align": "top", "margin":"0px", "--hover-color": "#eee"},
                        "nav-link-selected": {"background-color": "#FB790B"},
                    }
            )
        
        if app == "Upload PDF Files":
            upload_pdf_index.app()
        if app == "EDA with LLMs":
            sql_generator.app()
        if app == "Bedrock-Chatbot":
            chat_v2.app()

    run()            