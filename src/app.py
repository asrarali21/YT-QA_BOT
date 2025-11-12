from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import  FastEmbedSparse , QdrantVectorStore , RetrievalMode
from qdrant_client import QdrantClient , models
from langchain_qdrant import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from qdrant_client.http.models import Distance , SparseVectorParams , VectorParams
from dotenv import load_dotenv
from  langchain_community.retrievers import BM25Retriever  
from langchain.retrievers import EnsembleRetriever 

print(langchain.__version__)


load_dotenv()

GOOGLE_APIKEY=os.getenv("GOOGLE_API_KEY")



loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=xg0hNUWVIgQ",
     
)
docs_text = loader.load()

print(len(docs_text))




split_docs = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
)


docs_chunks =  split_docs.split_documents(docs_text)
print(len(docs_chunks))



dense_embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_APIKEY
)


client = QdrantClient(
    "localhost", port=6333
)

collection_name = "youtube-vector"

# client.create_collection(
#     collection_name=collection_name,
#      vectors_config=VectorParams(
#          size=768 , distance=models.Distance.COSINE,
#      ),
#      sparse_vectors_config={
#          "sparse" : models.SparseVectorParams(
#              modifier=models.Modifier.IDF
#          )
#      }

# )

# sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")

vectorStore = QdrantVectorStore.from_documents(
    documents=docs_chunks,
    embedding=dense_embedding,
    collection_name=collection_name,
    retrieval_mode = RetrievalMode.HYBRID
)


print("✅ Vector store created with hybrid search")


dense_retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
bm25_ret = BM25Retriever.from_documents(docs_chunks)
bm25_ret.k = 6
hybrid_ret = EnsembleRetriever(retrievers=[dense_ret, bm25_ret], weights=[0.6, 0.4])


template = """Answer the question based on the following context from a YouTube video:

Context:
{context}

Question: {question}

Answer:"""


prompt = ChatPromptTemplate.from_template(template) 



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
   google_api_key=GOOGLE_APIKEY,
   temperature=0
)

rag_chain = (
    {
        "context" : retriever ,
        "question" : RunnablePassthrough()
    }
    |prompt
    |llm
    |StrOutputParser()
)


question = "What is the main topic discussed in this video?"
print(f"\n❓ Question: {question}")
print(f"\n💬 Answer: {rag_chain.invoke(question)}")















