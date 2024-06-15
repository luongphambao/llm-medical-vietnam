
import os 
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import ollama

load_dotenv(".env")
class LLM():
    def __init__(self,openai_api_key=None,google_api_key=None,temperature=0.1,model_name=None,ollama_use=False):
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.temperature = temperature
        self.model_name = model_name
        if ollama_use:
            self.llm = None
        if openai_api_key is not None:
            self.llm = ChatOpenAI(temperature=temperature,openai_api_key=openai_api_key)
        if google_api_key is not None:
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    def generate(self,prompt):
        if self.openai_api_key is not None:
            return self.llm.predict(text=prompt)
        if self.google_api_key is not None:
            return self.llm.invoke(prompt)
        if self.ollama_use:
            return self.llm.generate(model=self.model_name, prompt=prompt).get("response")[0]
if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = LLM(openai_api_key=openai_api_key)
    prompt ="""Triệu chứng của bệnh cúm là gì?"""
    result = llm.generate(prompt)
    print(result)
    