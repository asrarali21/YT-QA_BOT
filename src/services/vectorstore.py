from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache
from .loader import load_youtube_video
from .embedding import get_dense_embedding , get_sparse_embedding
from langchain_qdrant import QdrantVectorStore , RetrievalMode
from qdrant_client import QdrantClient



def spilt_docs_youtube(docs_text):
 split_docs = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    )
 return split_docs.split_documents(docs_text)

def get_qdrant_client():
    return QdrantClient("localhost", port=6333)

@lru_cache()
def create_youtube_vector_store(url , collection_name = "youtube-vector"):
    docs = load_youtube_video(url)
    chunks = spilt_docs_youtube(docs)


    dense = get_dense_embedding()
    sparse = get_sparse_embedding()

    vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=dense,
    collection_name=collection_name,
    sparse_embedding=sparse,
    retrieval_mode = RetrievalMode.HYBRID
)
    return vector_store


def get_vector_store(url: str, collection_name: str = "youtube-vector"):
   return create_youtube_vector_store(url, collection_name)



 



