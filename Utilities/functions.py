

import pandas as pd
import boto3
import json
import os

################################################################### convert dict to df ########################################################################
def convert_dict_to_df(dict_):
  master_key = []; sub_key = []; sub_key_codes = []; sub_key_descriptions = [];
  for _key_1 in dict_:
    for _key_2 in dict_[_key_1]:
      try:
        master_key.append(_key_1)
      except:
        master_key.append('')
      try:
        sub_key.append(_key_2)
      except:
        sub_key.append('')
      try:
        sub_key_codes.append(dict_[_key_1][_key_2]['CODE'])
      except:
        sub_key_codes.append('')
      try:
        sub_key_descriptions.append(dict_[_key_1][_key_2]['DESCRIPTION'])
      except:
        sub_key_descriptions.append('')

  return_df = pd.DataFrame({
      'CODE_FAMILY': master_key,
      'CODE_SUB_FAMILY': sub_key,
      'CODES': sub_key_codes,
      'DESCRIPTION': sub_key_descriptions
      })
  return return_df



# SQL S3 connector
def s3_read_object(bucket, key , client):
    obj = client.get_object(Bucket=bucket, Key=key)
    obj_read = obj['Body'].read().decode("utf-8").replace('\r', '').replace('\t', '')
    return obj_read

def s3_read_object_json(bucket, key , client):
    obj = client.get_object(Bucket=bucket, Key=key)
    obj_read = obj['Body'].read().decode("utf-8")
    obj_read = json.loads(obj_read)
    return obj_read

def sql_prompt_generator(db_name, table_name, table_desc):
    prompt = f"""
              ### Database Schema
              This query will run on a database name '{db_name}' whose schema is represented below:
              There are few instructions in generating queries:
              1. double quotation that is "" should not be used in giving input values.
              table schema has three information : column name, data type, and column description. create table syntax is as follows:
              CREATE TABLE '{table_name}' as {table_desc}
              Presto SQL query needs to be run on following schema."""
    return prompt