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
from dotenv import load_dotenv
load_dotenv(".env")
openai_api_key = os.getenv('OPENAI_API_KEY')
corpus_path = 'corpus/'
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
vectordb = Chroma.from_documents(documents=splits, embedding=embed_model,persist_directory="chroma_db_bge")
vectordb.persist()
print("Done!")
query ="Các triệu chứng của bệnh cúm là gì?"
k=3
results = vectordb.similarity_search(query=query, k=k)
print(results)