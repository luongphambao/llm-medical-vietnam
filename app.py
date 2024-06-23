from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
import os
load_dotenv(".env")
api_key =os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
from langchain_openai import ChatOpenAI
default = """Bạn là một phiên bản ảo của Trấn Thành. Trấn Thành là một MC nổi tiếng với tính tình hài hước, và được người dân Việt Nam yêu quý.
Một số điều lưu ý về Trấn Thành như sau:
- Trấn Thành ưu tiên sự riêng tư, đừng hỏi những thứ quá cá nhân về Trấn Thành.
- Trấn Thành thích được nói về những bài học cuộc sống cho mọi người học hỏi."""

your_prompt = """YOUR_PROMPT"""
chat = ChatOpenAI(temperature=0, openai_api_key=api_key)
def predict(message, history):
    history_list = []
    history_list.append(SystemMessage(content=default))
    for human, assistant in history:
        history_list.append(HumanMessage(content=human))
        history_list.append(AIMessage(content=assistant))
    history_list.append(HumanMessage(content=message))

    partial_message = ''
    for chunk in chat.stream(history_list):
      if chunk.content is not None:
        partial_message = partial_message + chunk.content
        yield partial_message


gr.ChatInterface(predict).launch(share=True)
