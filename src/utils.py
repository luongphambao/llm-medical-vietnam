import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from bs4 import BeautifulSoup
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

def load_corpus(corpus_path):
    """load corpus from a directory of text files"""
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(corpus_path, loader_cls = TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    for doc in docs:
        soup = BeautifulSoup(doc.page_content, 'html.parser')
        doc.page_content = soup.get_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return docs,texts
def get_text_from_html_file(html_path):
    with open(html_path, "r") as f:
        html_text = f.read()
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text()
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(raw_text)

    return chunks