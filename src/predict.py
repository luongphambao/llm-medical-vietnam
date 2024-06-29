import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
from dotenv import load_dotenv
try:
    from src.cod import get_summarize
    from src.utils import *
    from src.data import Medical_Data
    from src.llm import LLM
    from src.metrics import Metrics
    from src.embedding import Embedding
    from src.search import Searching
except:
    from cod import get_summarize
    from utils import *
    from data import Medical_Data
    from llm import LLM
    from metrics import Metrics
    from embedding import Embedding
    from search import Searching


load_dotenv(".env")

def main():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    corpus_path = 'corpus_summarize/'
    docs,texts = load_corpus(corpus_path)
    print("Loaded corpus")
    splits =texts 
    embedding = Embedding(model_name="BAAI/bge-m3", device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge_v2")
    #embedding  = Embedding(model_name="google", device='cpu', cache_dir="cache/", persist_directory="chroma_db_google")
    #embedding =   Embedding(model_name="openai", device='cpu', cache_dir="cache/", persist_directory="chroma_db_openai",openai_api_key=openai_api_key)
    print(embedding.model_name)
    vectordb = embedding.load_embedding()
    print("Loaded embedding")
    search = Searching(2,2,vectordb,splits)
    print("Loaded search")
    df = pd.read_csv("data/public_test.csv")
    result = {"id": [], "answer": []}
    model = LLM(google_api_key=google_api_key)
    #model = LLM(openai_api_key=openai_api_key)
    #model_name ="ontocord/vistral"
    model_name ="mrjacktung/phogpt-4b-chat-gguf"
    model = LLM(model_name=model_name,ollama_use=True)
    print("Loaded model")
    for index, row in tqdm(df.iterrows()):
        result["id"].append(row["id"].strip())
        # try:
            
        question, choices, num_choices = process_single_row(row)
        docs = search.hybrid_search(question)
        context = search.get_context(docs)
        #context = get_summarize(context)
        #print("Summarized context",context)
        question = preprocess_question(question)
        prompt = model.preprocess_prompt(question, choices, context)
        # with open("prompt_sample.txt", "w") as f:
        #     f.write(prompt)
        answer = model.generate(prompt)
        output_json = process_output(answer, num_choices)
        #print(output_json)
        result["answer"].append(output_json)
        #result_path
        # #except:
        #     print("Error at index: ", index)
        #     ##result["id"].append(row["id"].strip())
        #     value = "0"*num_choices
        #     result["answer"].append(value)
        #     continue
        #break
    newdf = pd.DataFrame(result, dtype=str)
    newdf.to_csv("submit_phogpt.csv", index=False)
# query ="Đàn ông có thể bị mắc ung thư vú không?"
# vertor_result_docs = search.hybrid_search(query)
# print(vertor_result_docs[0].page_content)
if __name__ == "__main__":
    main()