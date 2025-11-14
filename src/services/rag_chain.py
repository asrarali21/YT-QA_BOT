from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

def get_rag_chain(retriever):
    template = """Answer the question based on the following context from a YouTube video:

Context:
{context}

Question: {question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    GROQ_APIKEY = os.getenv("GROQ_API_KEY")

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
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def answer_question(question , retriever):
    chain = get_rag_chain(retriever)
    return chain.invoke(question)