import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import json
from torch import cuda
from pyvi.ViTokenizer import tokenize
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model_id = 'BAAI/bge-m3'

#device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
device = 'cpu'
cache_dir = "cache/"

DB_SAVE_NAME = f"endoc_{embed_model_id.split('/')[-1].replace('.','-')}"
DOCUMENT_DIR = "processed/"

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'batch_size': 1, "normalize_embeddings": True, "device": device},
    cache_folder=cache_dir
)
print("Load embeddings model successfully!")

# docs = [
#     "this is one document",
#     "and another document"
# ]

# embeddings = embed_model.embed_documents(docs)

# print(
#     f"We have {len(embeddings)} doc embeddings, each with "
#     f"a dimensionality of {len(embeddings[0])}."
# )
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

faiss_docs = []
for filename in sorted(os.listdir(DOCUMENT_DIR)):
    filepath = os.path.join(DOCUMENT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        file_data = json.load(f)
        
        faiss_docs.append(Document(
            page_content=file_data["content"],
            metadata={"filename": filename, "path": filepath, "title": file_data["title"], "category": file_data["category"], "abstract": file_data["abstract"], "subsections": file_data["subsections"]}
        ))

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'batch_size': 1, "normalize_embeddings": True, "device": device },
    cache_folder=cache_dir
)

db = FAISS.from_documents(
    documents=faiss_docs,
    embedding=embed_model,
)
db.save_local(DB_SAVE_NAME)
from langchain.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

device = 'cpu'
embed_model = SentenceTransformerEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'batch_size': 1, "normalize_embeddings": True, "device": device },
    cache_folder=cache_dir
)

loaded_db = FAISS.load_local(
    DB_SAVE_NAME,
    embeddings=embed_model
)
query = """
Đâu là triệu chứng của bệnh van tim?
A. Khó thở
B. Tăng cân nhanh chóng
C. Vàng da
D. Rụng tóc
"""

result = loaded_db.similarity_search(query=query, k=1)
print(result)
context = result[0].page_content
with open("context.txt", "w") as f:
    f.write(context)
print(context)