import streamlit as st
import pandas as pd
import numpy as np
from numpy import random
import random
from Utilities.functions import *
from st_aggrid import AgGrid
# from st_aggrid.grid_options_builder import GridOptionsBuilder

import os
import requests

# from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import ElasticVectorSearch, Weaviate, FAISS
from langchain.vectorstores import Pinecone as lc_pinecone


# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI


from pyathena import connect
from pyathena.pandas.util import as_pandas
from pyathena.pandas.cursor import PandasCursor


from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec
use_serverless = os.environ.get("USE_SERVERLESS", "True").lower() == "true"

# environment
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import boto3
import json

import warnings
warnings.filterwarnings('ignore')

from Utilities.functions import *


# ################################################################### Streamlit #########################################################

# st.set_page_config(page_title="EDA with Text", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
# st.set_page_config(layout='wide')


@st.cache_resource
def boto_s3_client():
    s3_client = boto3.client('s3',
                         aws_access_key_id=os.environ['ATHENA_ACCESS_KEY'],
                         aws_secret_access_key=os.environ['ATHENA_SECRET_KEY'],
                         region_name= 'us-east-1')
    return s3_client

@st.cache_resource
def load_database_map_key(boto_client=boto_s3_client()):
    s3_json_map = s3_read_object_json(bucket = 'sql-index-prompt-store', 
                                  key = "project_key_map/database_key_map.json",
                                  client= boto_client)
    return s3_json_map

database_dict_key = load_database_map_key()
database_selector = [key for key in database_dict_key.keys()]



@st.cache_resource
def load_pinecone_serverless_db():
    # initialize connection to pinecone (get API key at app.pc.io)
    api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    environment = os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'
    # configure client
    pc = Pinecone(api_key=api_key)

    # Serverless Picecone Specification
    if use_serverless:
        spec = ServerlessSpec(cloud='aws', region='us-west-2')
    else:
        spec = PodSpec(environment=environment)
    return pc , spec

pc_object, specs  = load_pinecone_serverless_db()

def llm_embeddings(model_= 'text-embedding-ada-002'):
    embeddings = OpenAIEmbeddings(model=model_,
                                  openai_api_key= os.environ["OPENAI_API_KEY"])
    return embeddings

load_embeddings = llm_embeddings()

# openai==0.28
# gpt-3.5-turbo-instruct
@st.cache_resource
def qa_chain(model = "gpt-4", token_size = 500, output_size = 1):
    chain = load_qa_chain(ChatOpenAI(model_name=model,
                                 max_tokens=token_size, 
                                 openai_api_key= os.environ["OPENAI_API_KEY"],
                                 n=output_size,
                                 temperature =0.5), chain_type="stuff")
    return chain

chain_ = qa_chain()


@st.cache_resource
def pandas_cursor():
    pandas_cursor = connect(aws_access_key_id=  os.environ['ATHENA_ACCESS_KEY'],
                            aws_secret_access_key=os.environ['ATHENA_SECRET_KEY'],
                            s3_staging_dir='s3://athena-query-general-2024/pyathena_output/',
                            work_group = 'primary',
                            region_name="us-east-1").cursor()
    return pandas_cursor

pd_cursor = pandas_cursor()






def app():
    st.markdown("<h2 style='text-align: center; color: #00004d;'> EDA with LLMs</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([999,1,1])

    m = st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                            background-color: rgb(223, 223, 235);
                        }
                        </style>""", unsafe_allow_html=True)

    st.markdown("""
                        <style>
                        [data-testid="stAppViewContainer"]{
                            background-color: #ffffff
                        }
                        </style>""", unsafe_allow_html=True)


    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


    with col1:
        st.markdown("""<h7 style='text-align: center; color: #00004d;'>The project commences with the seamless integration of cutting-edge LLMs into the EDA process. 
                    These models possess exceptional natural language understanding, enabling users to articulate analytical queries in a conversational manner. 
                    Subsequently, the LLM converts these prompts into precise SQL queries, automating the interaction with the underlying database.</h7>"""
                ,unsafe_allow_html=True)
        
    st.markdown("""<h4 style='text-align: left; color: #800000;'> Auto Generate Sample Query: </h4>""",unsafe_allow_html=True)

    # Setting up condition only to run when submitted
    try:
        # check if the key exists in session state
        _ = st.session_state.question_generator
    except AttributeError:
        # otherwise set it to false
        st.session_state.question_generator = False


    # Form 1 

    with st.form(key="input_parameters_1"):
        col1_, col2_ = st.columns([1,1])
        with col1_:
            _option_ = st.selectbox('**Select Dataset**', database_selector)
            
        submitted_1 = st.form_submit_button('**Generate sample Question**')

    if submitted_1 or st.session_state['question_generator']:
        with col2_:
            question_bank = s3_read_object_json(bucket = 'sql-index-prompt-store', 
                                            key = database_dict_key[_option_]["sample_ques_key"],
                                            client= boto_s3_client())
            sample_int = int(np.random.randint(len(question_bank["questions"]), size=1))   # random.randint(len(question_bank["questions"]))
            ques = question_bank["questions"][sample_int]
            med_script= st.markdown("""
                                    
                                    <h6>Query to copy Below text Input</h6>

                                    ```
                                    {0}""".format(ques),unsafe_allow_html=True)
            st.session_state['question_generator'] = True


    st.markdown("""<h4 style='text-align: left; color: #003366;'> Input field:</h4>""",unsafe_allow_html=True)
    # st.write(f"{np.random.randint(100, size=1)}")

    # Setting up condition only to run when submitted
    try:
        # check if the key exists in session state
        _ = st.session_state.multi_selector
    except AttributeError:
        # otherwise set it to false
        st.session_state.multi_selector = False



    # username input at column 1

    with st.form(key="input_parameters_2"):
        col1, col2 = st.columns([1,1])
        with col1:
            option = st.selectbox('**Select Dataset**', database_selector)
        with col2:
            med_script= st.text_input("""Enter Query""")
        # generate_sample = st.button("Submit")
        submitted = st.form_submit_button("Submit")


    if submitted or st.session_state['multi_selector']:
        # print(".......................................................Submitted................................................................")

        # create doc search object
        index_name =  database_dict_key[option]['pinecone_index']
        docsearch = lc_pinecone.from_existing_index(index_name, load_embeddings.embed_query)

        # Vector store object
        index = pc_object.Index(index_name)
        vector_store = lc_pinecone(index, load_embeddings.embed_query, "text")

        # print(".......................................................Vector setup complete................................................................")

        # ques = "Identify all patients with their latest recorded date who have a BMI greater than 30"
        query = f"""answer ONLY in valid Presto SQL query having 'database'.'table_name' format and with column names starting with `SELECT` for given query: '{med_script}'"""
        docs = vector_store.similarity_search(query)
        results = chain_.run(input_documents=docs, question=query)
        st.markdown("""```
                            {0}""".format(results))
        # sql_quer= sqlparse.format(results.split("```sql")[-1], reindent=True)
        try:
            cursor_execute = pd_cursor.execute(results)
            result_df = as_pandas(cursor_execute)
            st.dataframe(result_df, use_container_width=True)
        except:
            st.error("Ugh 😟,  Error has occured please try again")
        st.session_state['multi_selector'] = True


