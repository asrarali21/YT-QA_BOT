from .vectorstore import get_vector_store

def get_retriever():
    vs = get_vector_store()
    retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)