from utils import get_text_from_html_file, get_text_chunks
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os 
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(".env")
openai_api_key = os.getenv('OPENAI_API_KEY')
corpus_path = 'corpus2/'
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(corpus_path, loader_cls = TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()
from bs4 import BeautifulSoup
for doc in docs:
    soup = BeautifulSoup(doc.page_content, 'html.parser')
    doc.page_content = soup.get_text()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
#splits = []
splits =texts 

print (f"Your {len(docs)} documents have been split into {len(texts)} chunks")
#embedding = OpenAIEmbeddings(openai_api_key)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-m3"
device = 'cuda:0'
model_kwargs = {'device': device}
cache_dir = "cache/"
encode_kwargs = {'normalize_embeddings': True}
embed_model = SentenceTransformerEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs={'batch_size': 16, "normalize_embeddings": True, "device": device },
    cache_folder=cache_dir
)
# vectordb = Chroma.from_documents(documents=splits, embedding=embed_model,persist_directory="chroma_db_bge")
# vectordb.persist()
persist_directory = "chroma_db_bge"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
print("Done!")
query ="Các triệu chứng của bệnh cúm là gì?"
k=3
# results = vectordb.similarity_search(query=query, k=k)
# print(results)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
#retriever.k = k
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 2 
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5])
ensemble_docs = ensemble_retriever.get_relevant_documents(query)
prompt_template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết.
{context}
Câu hỏi: {question}
Trả lời:"""

question ="""Các đối tượng có nguy cơ cao mắc bệnh cúm là ai?.,
A. Chỉ những người già,
B. Bất cứ ai,
C. Chỉ những người trẻ tuổi,
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
llm = ChatOpenAI(temperature=0.1,openai_api_key=openai_api_key)
result=llm.predict(text=PROMPT.format_prompt(
    context=ensemble_docs,
    question=question
).text)
print(result)