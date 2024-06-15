import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
# prompt_template = """Trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết.
# Câu hỏi: {question}
# Trả lời:"""

prompt = PromptTemplate.from_template("Trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết.\nCâu hỏi: {question}\nTrả lời:")

chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

if __name__=="__main__":
    question = """Bệnh nhân Dũng được chuẩn đoán bị viêm gan kéo dài khoảng 4 tháng. Hỏi bệnh nhân Dũng có phải bị viêm gan mãn tính hay không?
    A. Có
    B. Không
    """
    resp = chain.run(question)
    print(resp)