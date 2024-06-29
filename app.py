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
from langchain_openai import ChatOpenAI
load_dotenv(".env")
api_key =os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
def preproces_context(context:list):
    """Preprocess context"""
    context = " ".join(context)
    context = context.replace("\n","")
    #replace  ** with ""
    context = context.replace("*","")

    return context


client = OpenAI(api_key=api_key)
#drop down menu llm model gradio
model_list= ["gpt-3.5-turbo","gpt-3.5-turbo-davinci","gpt-3.5-turbo-curie","gpt-3.5-turbo-babbage","gpt-3.5-turbo-ada","gpt-3.5-turbo-codex","gpt-3.5-turbo-cushman","gpt-3.5-turbo-davinci-codex","gpt-3.5-turbo-davinci-cushman","gpt-3.5-turbo-davinci-codex-cushman"]

corpus_path = 'corpus_summarize/'
docs,texts = load_corpus(corpus_path)
print("Loaded corpus")
splits =texts 
embedding = Embedding(model_name="BAAI/bge-m3", device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge_v4")
#embedding  = Embedding(model_name="google", device='cpu', cache_dir="cache/", persist_directory="chroma_db_google")
#embedding =   Embedding(model_name="openai", device='cpu', cache_dir="cache/", persist_directory="chroma_db_openai",openai_api_key=openai_api_key)
print(embedding.model_name)
vectordb = embedding.load_embedding()
print("Loaded embedding")
search = Searching(1,1,vectordb,splits)
print("Loaded search")
model = LLM(google_api_key=google_api_key)
# model_name ="ontocord/vistral"
# model = LLM(model_name=model_name,ollama_use=True)
print("Loaded model")
def RAG(question):
    """RAG for human message"""
    docs = search.hybrid_search(question)
    context = search.get_context(docs)
    #print(context)
    context = preproces_context(context)
    
    question = preprocess_question(question)
    prompt = model.preprocess_prompt(question=question,context= context)
    answer = model.generate(prompt)
    respone = f"{answer} \n\n\n Tài liệu tham khảo:\n\n {context}"
    return respone   

def predict(message, history):
    history_langchain_format = []
    # for human, ai in history:
    #     history_langchain_format.append(HumanMessage(content=human))
    #     history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    respone = RAG(message)
    #print(respone)
    return respone
gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(height=300),
        title="Tư vấn y tế",
        theme="soft",
        examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],).launch(share = True)