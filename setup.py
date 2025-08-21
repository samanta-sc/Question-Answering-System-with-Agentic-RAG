from setuptools import find_packages, setup

setup(
    name="QnA-Chatbot",
    version="0.0.1",
    author="Samanta",
    author_email="farjanakabirsamanta85@gmail.com",
    packages=find_packages(),
    install_requires=['chromadb','langchain ','langchain-groq','PyPDF2','langchain_huggingface','python-dotenv','streamlit']
)