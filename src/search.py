
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
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
load_dotenv(".env")
openai_api_key = os.getenv('OPENAI_API_KEY')
import ollama
try:  
    from src.utils import get_text_from_html_file, get_text_chunks,load_corpus
    from src.embedding import Embedding
except:
    from utils import get_text_from_html_file, get_text_chunks,load_corpus
    from embedding import Embedding
model_id ="ontocord/vistral"

class Searching:
    def __init__(self,k1,k2,vectordb,splits):
        self.k1 = k1
        self.k2 = k2
        self.retriever = vectordb.as_retriever(search_kwargs={"k": k1})
        self.bm25_retriever = BM25Retriever.from_documents(splits)
        self.bm25_retriever.k = k2
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.retriever], weights=[0.7, 0.3])
    def hybrid_search(self,query):
        ensemble_docs = self.ensemble_retriever.get_relevant_documents(query)
        return ensemble_docs
    def bm25_search(self,query):
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        return bm25_docs
    def vector_search(self,query):
        vector_docs = self.retriever.get_relevant_documents(query)
        return vector_docs
    def get_context(self,docs):
        context = []
        for doc in docs:
            context.append(doc.page_content)
        return context
# def main():
#     corpus_path = 'corpus/'
#     docs,texts = load_corpus(corpus_path)
#     print("Loaded corpus")
#     splits =texts 
#     embedding = Embedding(model_name="google", device='cpu', cache_dir="cache/", persist_directory="chroma_db_google")
#     vectordb = embedding.load_embedding()
#     print("Loaded embedding")
#     search = Searching(1,1,vectordb,splits)
#     print("Loaded search")
#     query ="Triệu chứng bệnh guts"
#     vertor_result_docs = search.vector_search(query)
#     print(vertor_result_docs[0].page_content)
# if __name__ ==__main__()
#     main()