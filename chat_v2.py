
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

from langchain.chains import ConversationChain

# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate

import boto3

from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import streamlit as st
from streamlit_chat import message


import os
import pinecone
from pinecone import Pinecone
from Utilities.chatbot_func import (
     get_conversation_string,
     bedrock_model_dict,
     get_response_llm,
     bedrock_query_refiner,
     bedrock_embeddings_pinecone_vector_store )



from Utilities.configs import (
     chatbot_configs,
     bedrock_model_name_dict
     )

import json
# from botocore.exceptions import ClientError # type: ignore


# Pinecone client.
@st.cache_resource
def pinecone_client():
    
    # initialize connection to pinecone (get API key at app.pc.io)
    lc_api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    environment = os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'
    
    if lc_api_key == 'PINECONE_API_KEY':
            raise Exception("PINECONE_API_KEY not found")
        
    # configure client
    pc = Pinecone(api_key=lc_api_key)

    return pc

# aws bedrock client.
@st.cache_resource
def aws_bedrock_client():
    bedrock_client=boto3.client(service_name="bedrock-runtime",
                                aws_access_key_id=os.environ['MASTER_ACCESS_KEY'],
                                aws_secret_access_key=os.environ['MASTER_SECRET_KEY'],
                                region_name= 'us-east-1')
    
    return bedrock_client



def app():
    index_list = [index["name"] for index in pinecone_client().list_indexes()]
    model_list = [model for model in bedrock_model_dict.keys()]

    # aws_embedding= "amazon.titan-embed-text-v1"
    # from utils import *
    st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Please select Index name and LLM Model to begin Conversation"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="")

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

    col1, col2 = st.columns([1,1])
    with col1:
        index_name = st.selectbox('**Select Index**', index_list)
    with col2:
        model_name = st.selectbox('**Select Model**', model_list)

    # container for chat history
    response_container = st.container()
    # container for text box
    textcontainer = st.container()

    try:
        embedding_model 
    except:
        embedding_model = chatbot_configs["default_embedding_model"]

    with textcontainer:
        llm= bedrock_model_dict[model_name](_token_ = chatbot_configs["query_refine_token"],
                                            bedrock_client = aws_bedrock_client()) 
        query = st.text_input("Type your message: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                # st.code(conversation_string)
                refined_query = bedrock_query_refiner(
                        context = conversation_string
                    ,query= query
                    ,model_id = bedrock_model_name_dict[model_name]
                    ,token_size = 64
                    ,bedrock_client=aws_bedrock_client())
                # st.subheader("Refined Query:")
                # st.write(refined_query)
                response = get_response_llm(
                    llm=llm,
                    vectorstore_=bedrock_embeddings_pinecone_vector_store(model_id=embedding_model, 
                                                                            bedrock_client= aws_bedrock_client(), 
                                                                            index_name= index_name,
                                                                            pinecone_client= pinecone_client()),
                    query=refined_query)
                # context = find_match(refined_query)
                # print(context)  
                # response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 
    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')