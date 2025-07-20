import json
import boto3
import os
import awswrangler as wr
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_pinecone import PineconeVectorStore  # Updated import
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI as lc_OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

def write_data_to_glue_db(df,
                          table_name,
                          database,
                          s3_path,
                          region_name):
    """
    Save a pandas DataFrame as a Parquet dataset in a specified S3 location and register/update it as a table in AWS Glue Catalog.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be saved.
    table_name : str
        The name of the Glue table to create or overwrite.
    database : str
        The Glue database where the table will be registered.
    s3_path : str
        The S3 bucket path where the Parquet dataset will be stored (e.g., 's3://my-bucket/data/').
    region_name : str
        The AWS region where the Glue Catalog and S3 bucket reside.

    Raises
    ------
    EnvironmentErrorLangchainPinecone
        If the required AWS credentials ('MASTER_ACCESS_KEY' and 'MASTER_SECRET_KEY') are not found in the environment variables.
    Exception
        Propagates any exception raised by boto3 or awswrangler during the write operation.

    Example
    -------
    >>> write_data_to_glue_db(
            df=my_dataframe,
            table_name="my_table",
            database="my_database",
            s3_path="s3://my-bucket/data/",
            region_name="us-east-1"
        )
    """
    access_key = os.getenv("MASTER_ACCESS_KEY")
    secret_key = os.getenv("MASTER_SECRET_KEY")
    if not access_key or not secret_key:
        raise EnvironmentError("AWS credentials not found in environment variables: 'MASTER_ACCESS_KEY' and/or 'MASTER_SECRET_KEY'.")

    # Create boto3 session with credentials
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name
    )
    # Save as dataset with Glue Catalog metadata using the session
    wr.s3.to_parquet(
        df=df,
        path=os.path.join(s3_path, table_name),
        dataset=True,
        database=database,
        table=table_name,
        mode="overwrite",
        boto3_session=session  # Pass the session with credentials
    )


def read_athena_table(sql_query,env_configs):
    """
    Execute SQL query on AWS Athena and return results as a pandas DataFrame.
    
    This function creates an authenticated AWS session using credentials from environment
    variables and executes the provided SQL query against an Athena database. The database
    and region configuration are retrieved from the env_configs parameter.
    
    Args:
        sql_query (str): The SQL query to execute against the Athena database.
            Can include queries spanning multiple databases using fully qualified names.
        env_configs (dict): Configuration dictionary containing Athena settings.
            Must include:
            - athena_configs (dict): Nested dictionary with:
                - database (str): Target database name for query execution
                - region (str): AWS region where Athena service is located
    
    Returns:
        pandas.DataFrame: Query results as a pandas DataFrame. Returns empty DataFrame
            if query produces no results.
    
    Raises:
        EnvironmentError: If required AWS credentials ('MASTER_ACCESS_KEY' or 
            'MASTER_SECRET_KEY') are not found in environment variables.
        KeyError: If required configuration keys are missing from env_configs.
        ClientError: If AWS authentication fails or insufficient permissions.
        Exception: For other AWS Athena query execution errors (timeout, syntax errors, etc.).
    
    Environment Variables Required:
        MASTER_ACCESS_KEY (str): AWS access key ID for authentication
        MASTER_SECRET_KEY (str): AWS secret access key for authentication
    
    Example:
        >>> env_config = {
        ...     "athena_configs": {
        ...         "database": "my_database",
        ...         "region": "us-east-1"
        ...     }
        ... }
        >>> query = "SELECT * FROM table_name LIMIT 10"
        >>> df = read_athena_table(query, env_config)
        >>> print(df.shape)
        (10, 5)
    
    Note:
        - Ensure AWS credentials have appropriate permissions for Athena query execution
        - Query costs are based on data scanned; use LIMIT clauses when appropriate
        - For cross-database queries, use fully qualified table names (database.table)
    """
    access_key = os.getenv("MASTER_ACCESS_KEY")
    secret_key = os.getenv("MASTER_SECRET_KEY")

    if not access_key or not secret_key:
        raise EnvironmentError("AWS credentials not found in environment variables: 'MASTER_ACCESS_KEY' and/or 'MASTER_SECRET_KEY'.")

    # Create boto3 session with credentials
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=env_configs["athena_configs"]["region"]
    )
    
    # Read data using SQL query
    df_load = wr.athena.read_sql_query(
        sql= sql_query,
        database=env_configs["athena_configs"]["database"],
        boto3_session=session
    )

    return df_load


def distinct_values_dict(df, n_values=10):
    result = {}
    for col in df.columns:
        distinct_vals = df[col].dropna().unique()[:n_values]
        result[col] = distinct_vals.tolist()
    return result


def generate_response_o4_mini(prompt, think=False, effort="medium", max_output_tokens=4000):
    """
    Generate response using OpenAI's o4-mini reasoning model via Responses API
    Tested and corrected based on official API structure
    """
    client = OpenAI()
    
    if think:
        # Enable reasoning mode with summary
        response = client.responses.create(
            model="o4-mini",
            reasoning={
                "effort": effort,
                "summary": "auto"  # This generates reasoning summaries
            },
            max_output_tokens=max_output_tokens,
            input=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract text content and reasoning from response.output
        text_content = ""
        reasoning_summary = ""
        
        # Iterate through output items to find text and reasoning
        for item in response.output:
            if getattr(item, "type", None) == "text":
                text_content = getattr(item, "text", "")
            elif getattr(item, "type", None) == "reasoning":
                reasoning_summary = getattr(item, "summary", "")
        
        return {
            "response": text_content,
            "thoughts": reasoning_summary
        }
    else:
        # Direct response without reasoning
        response = client.responses.create(
            model="o4-mini",
            max_output_tokens=max_output_tokens,
            input=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract text content from response.output
        for item in response.output:
            if getattr(item, "type", None) == "text":
                return getattr(item, "text", "No text output found")
        
        # Fallback: check for output_text attribute
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        
        return response # "No valid output returned by the model."

# Alternative simpler version for testing
def generate_response_o4_mini_simple(prompt, max_output_tokens=1000):
    """
    Simplified version for testing - uses direct output_text if available
    """
    client = OpenAI()
    
    response = client.responses.create(
        model="o4-mini",
        max_output_tokens=max_output_tokens,
        input=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    # Check for direct output_text first (as shown in DataCamp example)
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()
    
    # Otherwise iterate through output items
    for item in response.output:
        if getattr(item, "type", None) == "text":
            return getattr(item, "text", "")
    
    return "No valid output found"


def create_data_dict_prompt(json_string):
    
    dict_prompt= """You are required to data dictionary on basis of dict passed as an input. the doct contains column names and its unique values in form of
                        '{"column_name": [value1,value2 etc..]}' output to be generated as should be only in python dictionary only in specific mentioned format:                     
                        {"objective": Objective about the dataset,
                         "Columns": {"col1":{
                                            "desc": Description about the column based on its unique values,
                                            "column_data_type": column data type,
                                            "sample_values": ["value 1", "value 2", "value 3"] # total of upto three sample values
                                            }
                                    }
                        } dict values are as follows:: """+ json_string

    return dict_prompt



def upload_to_pinecone(text,
                       index_name,
                       chunk_size,
                       chunk_overlap,
                       dimension=1536,
                       metric='cosine',
                       model="OpenAI",
                       embedding_model="text-embedding-3-small"):
    """
    Upload text corpus to Pinecone serverless vector database with text chunking and embeddings.
    
    Args:
        text (str): Text corpus to upload and vectorize
        index_name (str): Name of the Pinecone index to create/use
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Character overlap between consecutive chunks
        dimension (int, optional): Vector embedding dimension. Defaults to 1536.
        metric (str, optional): Distance metric for similarity search. Defaults to 'cosine'.
        model (str, optional): Embedding model provider. Defaults to "OpenAI".
        embedding_model (str, optional): Specific embedding model name. Defaults to "text-embedding-3-small".
    
    Returns:
        PineconeVectorStore: Vector store object for the uploaded documents.
    
    Environment Variables:
        PINECONE_API_KEY: Required for Pinecone authentication
        PINECONE_ENVIRONMENT: AWS region for serverless index (defaults to us-east-1)
    """
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
            )
        )
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    chunks = text_splitter.split_text(text)
    
    if model=="OpenAI":
        # Initialize embeddings (compatible with o3-mini)
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            dimensions=dimension
    )
    
    # Create vector store and upload documents
    vector_store = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    print(f"Successfully uploaded {len(chunks)} chunks to Pinecone index '{index_name}'")
    return vector_store




def retrieve_from_pinecone(
        query,
        index_name, 
        k=4,
        model_name="o3-mini",
        embedding_model="text-embedding-3-small"):
    
    """
    Retrieve and answer questions using RAG (Retrieval-Augmented Generation) from Pinecone vector store.
    
    Args:
        query (str): Question or query to answer
        index_name (str): Name of the existing Pinecone index to search
        k (int, optional): Number of similar documents to retrieve. Defaults to 4.
        model_name (str, optional): ChatOpenAI model for answer generation. Defaults to "o3-mini".
        embedding_model (str, optional): OpenAI embedding model for query vectorization. Defaults to "text-embedding-3-small".
    
    Returns:
        str: Generated answer based on retrieved context, limited to 3 sentences.
    
    Note:
        Requires existing Pinecone index with embedded documents and OpenAI API access.
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        dimensions=1536
    )

    # Connect to existing vector store
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Use ChatOpenAI for chat models
    llm = ChatOpenAI(model=model_name)

    # Create prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the chain using the new pattern
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Use invoke instead of run
    response = rag_chain.invoke({"input": query})

    return response["answer"]


def upload_df_glue_w_index(
        df,
        database,
        s3_path,
        region_name,
        unique_df_values,
        llm_max_output_tokens,
        pc_index_name,
        chunk_size,
        chunk_overlap,
        dimension=1536,
        model="OpenAI",
        embedding_model="text-embedding-3-small"       
):
    """
    Upload DataFrame to AWS Glue, generate LLM-powered data dictionary, and store in Pinecone vector database.
    
    Combines three operations: (1) saves DataFrame as Parquet to S3 and registers in Glue Catalog, 
    (2) generates structured data dictionary using OpenAI o4-mini model, (3) uploads dictionary to Pinecone for RAG queries.
    
    Args:
        df (pandas.DataFrame): DataFrame to upload and analyze
        table_name (str): Glue table name
        database (str): Glue database name  
        s3_path (str): S3 bucket path for Parquet storage
        region_name (str): AWS region
        unique_df_values (int): Number of unique values per column to analyze
        llm_max_output_tokens (int): Token limit for LLM data dictionary generation
        pc_index_name (str): Pinecone index name for vector storage
        chunk_size (int): Text chunk size for vector embedding
        chunk_overlap (int): Character overlap between chunks
        dimension (int, optional): Vector embedding dimension. Defaults to 1536.
        model (str, optional): Embedding model provider. Defaults to "OpenAI".
        embedding_model (str, optional): Embedding model name. Defaults to "text-embedding-3-small".
    
    Returns:
        PineconeVectorStore: Vector store containing the generated data dictionary.
    
    Environment Variables:
        MASTER_ACCESS_KEY, MASTER_SECRET_KEY: AWS credentials
        PINECONE_API_KEY: Pinecone authentication
        OPENAI_API_KEY: OpenAI API access
    """

    pc_table_name= pc_index_name.replace("-","_")

    write_data_to_glue_db(df= df,
                        table_name=  pc_table_name,
                        database= database,
                        s3_path= s3_path ,
                        region_name=region_name)
    

    dist_values_dict = distinct_values_dict(df=df,
                                            n_values=unique_df_values)
    dict_string= json.dumps(dist_values_dict)
    final_dict_prompt = create_data_dict_prompt(dict_string)

    # Test with appropriate token limit
    result = generate_response_o4_mini(
        prompt=final_dict_prompt,
        think=False,
        max_output_tokens=llm_max_output_tokens  # Increased from 50
        )
    
    # print(result)
    
    final_input_text = """
                    ### Database Schema
                    This text contains data dictionary for DB.table name as  {db}.{table_name} .
                    table dict has following information: Objectives, columns which has sub heading names as >> "desc", "column_data_type", "sample_values", 
                    the details of dict as follows:
                    {result}
                    """.format(db= database ,
                               table_name= pc_table_name,
                               result=result
                               )
    
    upload_to_pinecone(text=final_input_text,
                        index_name= pc_index_name ,
                        chunk_size= chunk_size ,
                        chunk_overlap=chunk_overlap,
                        dimension= dimension,
                        metric='cosine',
                        model=model,
                        embedding_model= embedding_model
                        )
    

def retrieve_from_pinecone(query, index_name, k=4):
    """
    Retrieve and query data from Pinecone serverless vector database.
    
    Args:
        query (str): The query/question to search for
        index_name (str): Name of the Pinecone index
        k (int): Number of similar documents to retrieve
    
    Returns:
        str: The answer generated by the QA chain
    """
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    # Connect to existing vector store
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # Use ChatOpenAI instead of OpenAI for o3-mini
    llm = ChatOpenAI(model_name="o3-mini")
    
    # Create prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.keep the answer concise"
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the chain using the new pattern
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Use invoke instead of run
    response = rag_chain.invoke({"input": query})
    
    return response["answer"]
