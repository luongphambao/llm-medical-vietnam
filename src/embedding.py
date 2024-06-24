
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os 
import torch
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv
try:
    from src.utils import get_text_from_html_file, get_text_chunks,load_corpus
except:
    from utils import get_text_from_html_file, get_text_chunks,load_corpus
load_dotenv(".env")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
class Embedding:
    def __init__(self, model_name=None, device="cpu", cache_dir=None, persist_directory="openai",openai_api_key=None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key
        print(self.model_name)
        if self.model_name == "openai" and openai_api_key is not None:
            self.embed_model = OpenAIEmbeddings(api_key=openai_api_key)
        elif self.model_name == "google":
            self.embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        else:
            self.embed_model = SentenceTransformerEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'batch_size': 16, "normalize_embeddings": True, "device": device},
                cache_folder=cache_dir
            )
        print(self.embed_model)
    def create_embedding(self,splits):
        vectordb = Chroma.from_documents(documents=splits, embedding=self.embed_model,persist_directory=self.persist_directory)
        vectordb.persist()
        return vectordb
    def add_embedding(self, vectordb, splits):
        vectordb.add_documents(splits)
        vectordb.persist()
        return vectordb
    def load_embedding(self):
        #vectordb = Chroma.load(self.persist_directory)
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embed_model)
        return vectordb
    def similarity_search(self, vectordb, query, k):
        results = vectordb.similarity_search(query=query, k=k)
        return results
def main():
    from dotenv import load_dotenv
    load_dotenv(".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    corpus_path = 'corpus_summarize'
    docs,splits = load_corpus(corpus_path)
    #Embedding(model_name="BAAI/bge-m3", device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge")
    model_name = "BAAI/bge-m3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = "cache/"
    persist_directory = "chroma_db_bge_v3"
    embedding = Embedding(model_name, device, cache_dir, persist_directory)
    vectordb = embedding.create_embedding(splits)
    query ="Các triệu chứng của bệnh cúm là gì?"
    k=3
    results = embedding.similarity_search(vectordb, query, k)
    print(results)
if __name__ == "__main__":
    main()
