from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
from dotenv import load_dotenv
from src.utils import *
from src.llm import LLM
from src.embedding import Embedding
from src.search import Searching
load_dotenv(".env")
api_key =os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
from langchain_openai import ChatOpenAI
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
corpus_path = 'corpus/'
docs,texts = load_corpus(corpus_path)
print("Loaded corpus")
splits =texts 
embedding = Embedding(model_name="BAAI/bge-m3", device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge")
#embedding  = Embedding(model_name="google", device='cpu', cache_dir="cache/", persist_directory="chroma_db_google")
#embedding =   Embedding(model_name="openai", device='cpu', cache_dir="cache/", persist_directory="chroma_db_openai",openai_api_key=openai_api_key)
print(embedding.model_name)
vectordb = embedding.load_embedding()
print("Loaded embedding")
search = Searching(1,1,vectordb,splits)
print("Loaded search")
model_name ="ontocord/vistral"
model = LLM(model_name=model_name,ollama_use=True)
print("Loaded model")
def RAG(question):
    """RAG for human message"""
    docs = search.hybrid_search(question)
    context = search.get_context(docs)
    question = preprocess_question(question)
    prompt = model.preprocess_prompt(question=question,context= context)
    answer = model.generate(prompt)
    return answer   

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    respone = RAG(message)
    return respone
gr.ChatInterface(predict).launch(share = True)