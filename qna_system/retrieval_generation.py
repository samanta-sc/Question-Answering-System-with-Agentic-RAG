from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable
import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from qna_system.ingest import get_vector_store

load_dotenv()
## load the GROQ API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')

@traceable(project_name="PDF Agentic chatbot")
def retrieval_chain():
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Do not make up any information.
    Keep the answer concise and to the point.
    For each fact in your answer, mention the source file it came from.
    For example: "The sky is blue (source: weather_report.pdf)."

    <context>
    {context}
    </context>

    Question: {input}
    """
    )

    llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.3-70b-versatile")

    document_chain=create_stuff_documents_chain(llm,prompt)
    vectorstore = get_vector_store()
    retriever=vectorstore.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
        
    return llm, retrieval_chain