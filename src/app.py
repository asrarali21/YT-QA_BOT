from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


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



embedding = GoogleGenerativeAIEmbeddings(
   
)











