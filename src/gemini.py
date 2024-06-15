import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv(".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# prompt_template = """Trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết.
# Câu hỏi: {question}
# Trả lời:"""

#prompt = PromptTemplate.from_template("Trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết.\nCâu hỏi: {question}\nTrả lời:")

#chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

if __name__=="__main__":
    question = """Thủ đô của Việt Nam là gì?
    """
    resp = llm.invoke(question)
    print(resp.content)
    # resp = chain.run(question)
    # print(resp)