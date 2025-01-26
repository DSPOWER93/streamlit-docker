import boto3
import json
import os
import streamlit as st
from botocore.exceptions import ClientError # type: ignore
from langchain.llms.bedrock import Bedrock

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import Pinecone as lc_pinecone


########################################################################################################################################



## Bedrock Clients
def bedrock_query_refiner(context,query,model_id,token_size,bedrock_client):
  promt_ = f"""Given the following user query and conversation log, refine the query to be most relevant to the conversation.
                 \n\nCONVERSATION LOG: \n"{context}"\n\n Query: {query}\n\n answer only in one sentence. Refined query:"""
  # Format the request payload using the model's native structure.
  native_request = {
      "prompt": promt_,
      "max_gen_len": token_size,
      "temperature": 0.2,
      "top_p": 0.95
      }
  # Convert the native request to JSON.
  request = json.dumps(native_request)


  try:
      # Invoke the model with the request.
      response = bedrock_client.invoke_model(modelId=model_id, body=request)

  except (ClientError, Exception) as e:
      print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
      exit(1)

  # Decode the response body.
  model_response = json.loads(response["body"].read())

  # Extract and print the response text.
  response_text = model_response["generation"]
  return response_text



def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n" # to avoid the first response as first response is already loaded default by system
    return conversation_string



def get_mistral_8x7_llm(_token_,bedrock_client):
    ##create the Anthropic Model
    llm=Bedrock(model_id="mistral.mixtral-8x7b-instruct-v0:1",client=bedrock_client,
                model_kwargs={'max_tokens':_token_,
                              "temperature": 0.5,
                              "top_p": 0.95})
    return llm


def get_llama3_large_llm(_token_,bedrock_client):
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock_client,
                model_kwargs={'max_gen_len':_token_,
                              "temperature": 0.5,
                              "top_p": 0.95})
    return llm


# bedrock model Dict 
bedrock_model_dict = {
    'llama3_70b': get_llama3_large_llm,
    'mistral_8x7':get_mistral_8x7_llm

}


########################################################################################################################################

prompt_template = """[/INST] Use the following pieces of context to provide a concise answer to the question at the end but use atleast summarize with
maximum size of 250 words explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer. <context> {context} </context> Question: {question} Answer: [/INST]"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm,vectorstore_,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT})
    # print(qa)
    answer=qa({"query":query})
    # answer=qa(query)
    return answer['result']

########################################################################################################################################


def bedrock_embeddings_pinecone_vector_store(model_id, bedrock_client, index_name, pinecone_client):
    # Actiate bedrock embeddings
    embeddings = BedrockEmbeddings(model_id=model_id,client=bedrock_client)    
    index = pinecone_client.Index(index_name)
    vector_store = lc_pinecone(index, embeddings.embed_query, "text")
    return vector_store