from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import  FastEmbedSparse , QdrantVectorStore , RetrievalMode
from qdrant_client import QdrantClient , models 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

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


HF_TOKEN=os.getenv("HF_TOKEN")


dense_embedding = HuggingFaceEndpointEmbeddings(
    provider="hf-inference",
  huggingfacehub_api_token=HF_TOKEN,
  model="BAAI/bge-large-en-v1.5"
)
print("✅ Dense embeddings initialized")


sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
print("✅ Sparse embeddings initialized")



client = QdrantClient(
    "localhost", port=6333
)

collection_name = "youtube-vector"


print("\n🏗️ Creating vector store with hybrid search...")

vectorStore = QdrantVectorStore.from_documents(
    documents=docs_chunks,
    embedding=dense_embedding,
    collection_name=collection_name,
    sparse_embedding=sparse_embedding,
    retrieval_mode = RetrievalMode.HYBRID
)



print("✅ Vector store created with hybrid search")


retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)



template = """Answer the question based on the following context from a YouTube video:

Context:
{context}

Question: {question}

Answer:"""


prompt = ChatPromptTemplate.from_template(template) 

GROQ_APIKEY=os.getenv("GROQ_API_KEY")

print("✅ API keys loaded")

def format_docs(docs):
    """Combine all retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
   api_key=GROQ_APIKEY
)

rag_chain = (
    {
        "context" : retriever | format_docs,
        "question" : RunnablePassthrough()
    }
    |prompt
    |llm
    |StrOutputParser()
)


question = "what exactly the person is talking about core ai and applied ai"
print(f"\n❓ Question: {question}")
print(f"\n💬 Answer: {rag_chain.invoke(question)}")
