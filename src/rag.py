import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils import *
from data import Medical_Data
from llm import LLM
from prompt import *
from metrics import Metrics
from embedding import Embedding
from search import Searching
from langchain_core.prompts import PromptTemplate

load_dotenv(".env")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def rag():
    prompt = hub.pull("rlm/rag-prompt")
    template = USER_MESSAGE_WITH_CONTEXT_VER_3
    custom_rag_prompt = PromptTemplate(template)
    rag_chain =(
        {
            "context": retriever |format_docs,
            "question": RunnablePassthrough(),
            "choices": RunnablePassthrough(),
        }
        
    )
    return rag_chain
def main():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    corpus_path = 'corpus/'
    docs,texts = load_corpus(corpus_path)
    print("Loaded corpus")
    splits =texts 
    embedding = Embedding(model_name="BAAI/bge-m3", device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge")
    vectordb = embedding.load_embedding()
    print("Loaded embedding")
    search = Searching(5,5,vectordb,splits)
    print("Loaded search")
    df = pd.read_csv("data/public_test.csv")
    result = {"id": [], "answer": []}
    model = LLM(google_api_key=google_api_key)
    #model = LLM(openai_api_key=openai_api_key)
    #model_name ="bdx0/vietcuna"
    #model = LLM(model_name=model_name,ollama_use=True)
    print("Loaded model")
    for index, row in tqdm(df.iterrows()):
        result["id"].append(row["id"].strip())
        question, choices, num_choices = process_single_row(row)
        docs = search.hybrid_search(question)
        context = search.get_context(docs)
        question = preprocess_question(question)
        rag_chain = rag(question,choices,context,question)
        
        # prompt = model.preprocess_prompt(question, choices, context)
        # with open("prompt_sample.txt", "w") as f:
        #     f.write(prompt)
        # answer = model.generate(prompt)
        # output_json = process_output(answer, num_choices)
        # print(output_json)
        # result["answer"].append(output_json)
        break
    newdf = pd.DataFrame(result, dtype=str)
    newdf.to_csv("submit_vietcuna.csv", index=False)
# query ="Đàn ông có thể bị mắc ung thư vú không?"
# vertor_result_docs = search.hybrid_search(query)
# print(vertor_result_docs[0].page_content)
if __name__ == "__main__":
    main()