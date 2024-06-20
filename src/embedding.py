
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
from utils import get_text_from_html_file, get_text_chunks,load_corpus

class Embedding:
    def __init__(self, model_name=None, device="cpu", cache_dir=None, persist_directory="openai",openai_api_key=None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.persist_directory = persist_directory
        if openai_api_key:
            self.embed_model = OpenAIEmbeddings(api_key=openai_api_key)
        else:
            self.embed_model = SentenceTransformerEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'batch_size': 16, "normalize_embeddings": True, "device": device},
                cache_folder=cache_dir
            )
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
    corpus_path = 'corpus'
    docs,splits = load_corpus(corpus_path)
    model_name = "openai"
    device = 'cpu'
    cache_dir = "cache/"
    persist_directory = "chroma_db_openai"
    embedding = Embedding(model_name, device, cache_dir, persist_directory,openai_api_key)
    vectordb = embedding.create_embedding(splits)
    query ="Các triệu chứng của bệnh cúm là gì?"
    k=3
    results = embedding.similarity_search(vectordb, query, k)
    print(results)
if __name__ == "__main__":
    main()
