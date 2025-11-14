from functools import lru_cache
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from langchain_qdrant import FastEmbedSparse


HF_TOKEN=os.getenv("HF_TOKEN")

@lru_cache()
def get_dense_embedding():
    _dense =  HuggingFaceEndpointEmbeddings(
    provider="hf-inference",
  huggingfacehub_api_token=HF_TOKEN,
  model="BAAI/bge-large-en-v1.5"
)
    return _dense
def get_sparse_embedding():
    _sparse   = FastEmbedSparse(model_name="Qdrant/bm25")
    return _sparse
    
