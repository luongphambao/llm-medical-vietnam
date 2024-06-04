import os
from getpass import getpass
import time
# langchain packages
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
path = 'corpus/'
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(path, loader_cls = TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()
from bs4 import BeautifulSoup
for doc in docs:
    soup = BeautifulSoup(doc.page_content, 'html.parser')
    doc.page_content = soup.get_text()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3
snippet = """Phương pháp phòng ngừa bệnh viêm cầu thận cấp tính nào được khuyến khích thực hiện?.
A. Sử dụng thuốc kháng sinh
B. Sử dụng thuốc giảm đau
C. Tăng cường hệ miễn dịch và giảm thiểu tối đa nguy cơ bị các bệnh viêm nhiễm
"""
docs = bm25_retriever.get_relevant_documents(snippet)
import ollama

model_id ="ontocord/vistral"
prompt_template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết, nếu có nhiều hơn 1 đáp án đúng thì liệt kê các đáp án đúng.
{context}
Câu hỏi: {question}
Trả lời:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
#get all doc to context
context = ""
for doc in docs:
    context += doc.page_content + "\n"

prompt = prompt_template.format_map({"context": context, "question": snippet})
#prompt = open("prompt.txt").read()x``
print(prompt)
respone = ollama.generate(model=model_id, prompt=prompt,options={"temperature":0.5})
print(respone)
print(respone["response"])
