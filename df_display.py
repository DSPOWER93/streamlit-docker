import os
from pinecone import Pinecone

import streamlit as st
import pandas as pd

from Utilities.helper_func import (
    upload_df_glue_w_index,
    retrieve_from_pinecone
)

from config.env_config import env_config

st.set_page_config(page_title="Gen AI use cases", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)



@st.cache_resource
def load_pinecone_index_names():
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    return pc.list_indexes()

load_pc_index_names= load_pinecone_index_names().names()


# Streamlit app layout
st.header('Upload Data')

css = '''load_pc_index_names
<style>
    [data-testid='stFileUploader'] {
        width: max-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
    }

</style>
'''
st.markdown(css, unsafe_allow_html=True)


# Create a form
with st.form("data_upload_form"):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        # Dropdown to select data format
        file_format = st.selectbox('Select data format', ['csv', 'parquet'])
    with col2:
        # Index Name
        _index_name_= st.text_input("""Dataset Name""")
    with col3:
        # File uploader
        uploaded_file = st.file_uploader('Upload your data file', type=[file_format])
    # Submit button (this becomes the form submit button)
    submit = st.form_submit_button('Submit')



# Handle form submission
if submit and uploaded_file is not None:
    try:
        if file_format == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_format == 'parquet':
            df = pd.read_parquet(uploaded_file)

        with st.spinner('Uploading...'):
        
            upload_df_glue_w_index(
            df=df,
            database=env_config["athena_configs"]["database"],
            s3_path=env_config["athena_configs"]["s3_path"],
            region_name=env_config["athena_configs"]["region"],
            unique_df_values=10,
            llm_max_output_tokens=3500,
            pc_index_name=_index_name_,
            chunk_size=3000,
            chunk_overlap=10,
            dimension=1536,
            )
            
            # Display success message
            st.success(f"Data successfully Uploaded!")
                    
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

elif submit and uploaded_file is None:
    st.warning("Please upload a file before submitting.")


st.header('Analysis on Dataset')

# Create a form
with st.form("data_retrive_form_2"):
    _col1_, _col2_ = st.columns([1,1])
    with _col1_:
        # Dropdown to select data format
        _option_ = st.selectbox('**Select Dataset**', load_pc_index_names)
    with _col2_:
        # Index Name
        index_query= st.text_input("""Enter your queries""")
    
    # Submit button (this becomes the form submit button)
    submitted = st.form_submit_button('Submit')

if submitted:
    st.write(_option_)