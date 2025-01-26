

import streamlit as st


import os
import pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import Pinecone as lc_pinecone
from langchain.text_splitter import CharacterTextSplitter
use_serverless = os.environ.get("USE_SERVERLESS", "True").lower() == "true"
from PyPDF2 import PdfReader
import boto3

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.



# @st.cache_resource
# def load_credentials():
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


## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime",
                     aws_access_key_id=os.environ['MASTER_ACCESS_KEY'],
                     aws_secret_access_key=os.environ['MASTER_SECRET_KEY'],
                     region_name= 'us-east-1')
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)



def create_n_upload_pinecone_serverless(chunk_text, index_name, embeddings=bedrock_embeddings ,upload_embeddings = True):
  # creating an index
  if len([a['name'] for a in pc.list_indexes() if a['name'] == index_name]) == 0:
      print(f'Creating index {index_name} ...')
      pc.create_index(index_name,
                      dimension=1536,
                      metric='cosine',
                      spec=spec)
      print('Done!')
  else:
      print(f'Index {index_name} already exists.')

  # Extract vector count
  index = pc.Index(index_name)
  index_dict = index.describe_index_stats()


  if index_dict['total_vector_count'] > 0:
      st.error(f'Index {index_name} not uploaded as it already has {index_dict["total_vector_count"]} vectors.')
  else:
    if upload_embeddings:
        st.success(f"Uploading embeddings under {index_name}")
        vector_store = lc_pinecone.from_texts(chunk_text, embeddings.embed_query, index_name=index_name)
    else:
        st.error("Embeddings not uploaded")



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# st.subheader("Upload file to Cloud")
# uploaded_pdf = st.file_uploader("Select .pdf file to upload",accept_multiple_files=True)
# button_ = st.button("submit")


def app():

    css = '''
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

    st.subheader("Select File to Upload")
    with st.form(key="input_parameters_2"):
        col1, col2 = st.columns([1,1])
        with col1:
            upload_pdf = st.file_uploader("Select a .pdf file",accept_multiple_files=True)
        with col2:
            _index_name_= st.text_input("""Index Name""")    
        submitted = st.form_submit_button("Submit")

    if submitted:
        with st.spinner('Uploading...'):
            # print(uploaded_pdf, type(uploaded_pdf))
            read_pdf = get_pdf_text(upload_pdf)
            chunk_text = get_text_chunks(read_pdf)
            create_n_upload_pinecone_serverless(chunk_text= chunk_text, 
                                                index_name= _index_name_, 
                                                embeddings=bedrock_embeddings ,
                                                upload_embeddings = True)
            st.success('Request Submitted')






