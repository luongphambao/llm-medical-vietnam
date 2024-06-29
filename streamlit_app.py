import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
) 
from openai import OpenAI
import gradio as gr
import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
from src.utils import *
from src.llm import LLM
from src.embedding import Embedding
from src.search import Searching
from langchain_openai import ChatOpenAI



# embedding = Embedding(model_name="BAAI/bge-m3", device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge_v4")
# #embedding  = Embedding(model_name="google", device='cpu', cache_dir="cache/", persist_directory="chroma_db_google")
# #embedding =   Embedding(model_name="openai", device='cpu', cache_dir="cache/", persist_directory="chroma_db_openai",openai_api_key=openai_api_key)
# print(embedding.model_name)
# vectordb = embedding.load_embedding()
# print("Loaded embedding")
# search = Searching(1,1,vectordb,splits)
# print("Loaded search")
# model = LLM(google_api_key=google_api_key)

def preproces_context(context:list):
    """Preprocess context"""
    context = " ".join(context)
    context = context.replace("\n","")
    #replace  ** with ""
    context = context.replace("*","")

    return context
def RAG(question,search,docs,model):
    """RAG for human message
    Args:
        question (str): question
        search (Searching): searching object
        docs (list): list of docs
        model (LLM): LLM model"""
    docs = search.hybrid_search(question)
    context = search.get_context(docs)
    #print(context)
    context = preproces_context(context)
    
    question = preprocess_question(question)
    prompt = model.preprocess_prompt(question=question,context= context)
    answer = model.generate(prompt)
    respone = f"{answer} \n\n\n Tài liệu tham khảo:\n\n {context}"
    return respone   
def main():
    load_dotenv(".env")
    corpus_path = 'corpus_summarize/'
    docs,texts = load_corpus(corpus_path)
    print("Loaded corpus")
    splits =texts 
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')

    st.set_page_config(page_title="Medical Question Answering", page_icon=":books:")
    st.header("Chat with Medical Assistant Bot")
    embedding = Embedding(model_name="BAAI/bge-m3", device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge_v4")
    vectordb = embedding.load_embedding()
    search = Searching(1,1,vectordb,splits)
    model = LLM(google_api_key=google_api_key)
    print(embedding, model, search)
    print("Loaded model")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I'm a medical assistant. How can I help you today?"),
        ]
    print("Sao nó run lại ta")
    with st.sidebar:

        st.subheader("Selecting Embedding Model")  
        embedding_model = st.selectbox("Select the Embedding Model", ["BAAI/bge-m3", "google", "openai"])
        llm_model = st.selectbox("Select the Generation Model", ["google", "openai","ontocord/vistral"])
        search_type = st.selectbox("Select the Search Type", ["hybrid", "vector", "bm25"])
        process_button = st.button("Process")
        #uploaded = pdf_docs
        
        if process_button:
            print("Processing...")
            #loading
            
            embedding = Embedding(model_name=embedding_model, device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge_v4")
            vectordb = embedding.load_embedding()
            search = Searching(1,1,vectordb,splits)
            model = LLM(google_api_key=google_api_key)
            print(embedding, model, search)
            print("Loaded model")
            #st
            #show loading embeding model
            
        #     # if not uploaded:
        #     #     st.warning("Please upload PDFs first.")
        #     # else:
        #     #     with st.spinner("Processing..."):
        #     #         st.session_state.vector_store, st.session_state.bm25_retriever = get_vectorstore_and_BM25(pdf_docs)

        # if st.button("Summarize"):

        #     if "vector_store" in st.session_state:
        #         with st.spinner("Summarizing"):

        #             if "chunks" not in st.session_state:
        #                 st.session_state.chunks = get_text_chunks(get_pdf_text(pdf_docs))

        #             llm = OpenAI(api_key=openai_api_key)
        #             results_queue = queue.Queue()  

        #             threads = []
        #             for chunk in st.session_state.chunks:
        #                 thread = threading.Thread(target=summarize_chunk, args=(chunk, results_queue, llm))
        #                 thread.start()
        #                 threads.append(thread)

        #             for thread in threads:
        #                 thread.join()

        #             summarize_chunks = []
        #             while not results_queue.empty():
        #                 summarize_chunks.append(results_queue.get())

        #         context = build_final_context(summarize_chunks)
        #         message = [{"role": "user", "content": Prompts.final_ans_prompt(context)}]

        #         final_response = llm.chat.completions.create(model='gpt-3.5-turbo',
        #             messages= message,
        #             temperature=0.1,
        #             max_tokens=4096)
                
        #         st.write(final_response.choices[0].message.content)

        #     else:
        #         if not uploaded:
        #             st.warning("Please upload PDFs first.")
        #         else:
        #             st.warning("Please process the PDFs.")


    user_query = st.chat_input("Type your mesage here ...")

    if user_query is not None and user_query != "":
        # Check if embedding model,llm,search is loaded
        print(embedding, model, search)
        if embedding is None or model is None or search is None:
            st.warning("Please select the embedding model, llm model and search type first.")
            # Get response from the model based on user's message
        #response = get_response(user_query, st.session_state.bm25_retriever)
        # Append user's message and response to the chat history
        else:
            response = RAG(user_query,search,docs,model)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))


    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,  HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


if __name__ == '__main__':
    
    main()