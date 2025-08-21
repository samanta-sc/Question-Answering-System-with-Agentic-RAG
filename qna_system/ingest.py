import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
from qna_system.data_converter import read_doc, get_text_chunks

load_dotenv()
## load the GOOGLE API KEY 
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')


def get_vector_store():
    docs = read_doc('../documents/')

    text_chunks = ""
    for doc in docs:
        text_chunk = get_text_chunks(doc)
        text_chunks += text_chunk

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    vstore = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="qna_system",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            # namespace=ASTRA_DB_KEYSPACE,
    )

    # Convert the list of strings to a list of Document objects
    list_of_documents = [Document(page_content=s) for s in text_chunks]
    vstore.add_documents(list_of_documents)

    return vstore

#     # Storing in session state
#     st.session_state["vectorstore"] = vstore