import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
from dotenv import load_dotenv
from utils import *
from data import Medical_Data
from llm import LLM
from metrics import Metrics
from embedding import Embedding
from search import Searching

load_dotenv(".env")

def main():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    corpus_path = 'corpus/'
    docs,texts = load_corpus(corpus_path)
    print("Loaded corpus")
    splits =texts 
    embedding = Embedding(model_name="BAAI/bge-m3", device='cpu', cache_dir="cache/", persist_directory="chroma_db_bge")
    vectordb = embedding.load_embedding()
    print("Loaded embedding")
    search = Searching(1,1,vectordb,splits)
    print("Loaded search")
    df = pd.read_csv("data/public_test.csv")
    result = {"id": [], "answer": []}
    model = LLM(google_api_key=google_api_key)
    print("Loaded model")
    for index, row in tqdm(df.iterrows()):
        result["id"].append(row["id"].strip())
        question, choices, num_choices = process_single_row(row)
        docs = search.hybrid_search(question)
        context = search.get_context(docs)
        question = preprocess_question(question)
        prompt = model.preprocess_prompt(question, choices, context)
        answer = model.generate(prompt)
        output_json = process_output(answer, num_choices)
        print(output_json)
        result["answer"].append(output_json)
    newdf = pd.DataFrame(result, dtype=str)
    newdf.to_csv("submit.csv", index=False)
# query ="Đàn ông có thể bị mắc ung thư vú không?"
# vertor_result_docs = search.hybrid_search(query)
# print(vertor_result_docs[0].page_content)
if __name__ == "__main__":
    main()