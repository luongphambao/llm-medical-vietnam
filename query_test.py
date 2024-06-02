from langchain.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
device = 'cpu'
embed_model_id = 'BAAI/bge-m3'
cache_dir = "cache/"
embed_model = SentenceTransformerEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'batch_size': 16, "normalize_embeddings": True, "device": device },
    cache_folder=cache_dir
)
DB_SAVE_NAME = f"endoc_{embed_model_id.split('/')[-1].replace('.','-')}"

loaded_db = FAISS.load_local(
    DB_SAVE_NAME,
    embeddings=embed_model
)
# level3_1,Hương đang mang thai và lo lắng mình có thể gặp phải rau tiền đạo. Hương có thể kiểm tra phát hiện bệnh này từ tuần thứ mấy của thai kỳ?,A. Tuần 10,B.Tuần 20,C. Tuần 30,D. Tuần 40,,
question = """Một người khỏe mạnh ăn mào gà, không bị xuất hiện các triệu chứng bất thường ở vùng bụng, đùi, bên trong hậu môn. Người đó có phải đã mắc bệnh sùi mào gà không?
A. Có
B. Không
"""

prompt_template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết.
{context}
Câu hỏi: {question}
Trả lời:"""
# query = """
# Đâu là triệu chứng của bệnh van tim?
# A. Khó thở
# B. Tăng cân nhanh chóng
# C. Vàng da
# D. Rụng tóc
# """

result = loaded_db.similarity_search(query=question, k=1)
print(result)
context = result[0].page_content
#system_prompt = """"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
import os

#load_dotenv()
with open("context.txt", "w") as f:
    f.write(context)
openai_api_key=os.getenv('OPENAI_API_KEY', 'sk-proj-aQI7CfOG3adLxGpE4zfKT3BlbkFJh3fnuL9nEPhU9HH3T2tj')
llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key)
result=llm.predict(text=PROMPT.format_prompt(
    context=context,
    question=question
).text)
print(result)