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
import re
import random
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
def process_single_row(row):
    question = row["question"].strip()
    list_answer = [str(row["option_1"]), str(row["option_2"]), str(row["option_3"]), str(row["option_4"]),
                   str(row["option_5"]), str(row["option_6"])]
    tmp_ans = []
    for c, a in zip(["A", "B", "C", "D", "E", "F"], list_answer):
        if a in ["nan", "", "NaN"]:
            continue
        if a.startswith(c):
            tmp_ans.append(a)
            continue
        tmp_ans.append(f"{c} {a}")
    answer_choices = "\n".join(tmp_ans)
    return question, answer_choices, len(tmp_ans)

def preprocess_question(question):
    question = question.replace("?.", "?")
    if question[-1] != "?":
        question += "?"
    type_predict = 0
    if "là gì" in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
        type_predict = 1
    if "cách gì" in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
        type_predict = 1
    if 'bệnh gì' in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
        type_predict = 1
    if 'bệnh nào' in question.lower() and not any(word in question.lower() for word in ["các", "những"]):
        type_predict = 1
    if 'bệnh lý nào' in question.lower():
        type_predict = 1
    if 'bao nhiêu' in question.lower():
        type_predict = 1
    if 'nhất' in question.lower():
        type_predict = 1
    if 'định nghĩa' in question.lower():
        type_predict = 1
    if "đúng hay sai" in question.lower():
        type_predict = 2
    if "có phải" in question.lower():
        type_predict = 2
    if "hay không" in question.lower():
        type_predict = 2
    if type_predict == 1 or type_predict == 2:
        question += " (仅选择 1 个正确答案。)"
    else:
        question += " (您必须选择 2 个或更多答案。)"
    return question
def extract_letters(line):
    matches = re.findall(r'\b([A-F])\b', line)
    return matches
def process_output(output, num_ans):
    res = ""
    MAP_ANS = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f"}
    # output = re.split(r"[.,、]", output)
    # output = [c.strip().lower() for c in output if len(c)]
    output = [c.lower() for c in extract_letters(output)]
    print("OUTPUT: ", output)
    for i in range(num_ans):
        if MAP_ANS[i] in output:
            res += "1"
        else:
            res += "0"
    if all(c == "0" for c in res):
        # Randomly select an index to change to 1
        index_to_change = random.randint(0, num_ans - 1)
        res = res[:index_to_change] + "1" + res[index_to_change + 1:]
    print(res)
    return res