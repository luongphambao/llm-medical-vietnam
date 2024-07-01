from FlagEmbedding import FlagReranker
from langchain.retrievers import ContextualCompressionRetriever, CohereRagRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv(".env")
import os
import cohere
cohere_api_key = os.getenv("COHERE_API_KEY")
print("cohere_api_key:",cohere_api_key)
class Rerank():
    def __init__(self,model_name="BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        if self.model_name =="cohere":
            self.reranker = cohere.Client(cohere_api_key)
        else:   
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    def rerank(self, query, passage_list):
        """
        query: str
        passage_list: List[str]
        Rank the passages based on the query"""
        if self.model_name == "cohere":
            reranker_results = self.reranker.rerank(query=query, documents=passage_list, top_n=3, model="rerank-multilingual-v3.0").results
            relevance_score = [result.relevance_score for result in reranker_results]
            print("relevance_score:",relevance_score)
            reranked_passages = [passage_list[result.index] for result in reranker_results]
            return reranked_passages


        else:
            reranked_score = self.reranker.compute_score([[query,passage] for passage in passage_list],normalize=True)
            print("reranked_score:",reranked_score)
            #sort the passages based on the reranked_score
            reranked_passages = [passage for _,passage in sorted(zip(reranked_score,passage_list),reverse=True)]
            return reranked_passages

if __name__ == "__main__":
    #rerank = Rerank()
    co = cohere.Client(cohere_api_key)
    query = "Thủ đô"
    passage_list = ["trung tâm","capital","kinh đô"]
    rerank = Rerank(model_name="cohere")
    reranked_passages = rerank.rerank(query, passage_list)
    print("reranked_passages:",reranked_passages)
    # #reranked_passages = rerank.rerank(query, passage_list)
    # rerank_results = co.rerank(query=query, documents=passage_list, top_n=3, model="rerank-multilingual-v3.0").results
    # print("rerank_results:",rerank_results)
    # relevance_score = [result.relevance_score for result in rerank_results]
    # print("relevance_score:",relevance_score)
    # #print("rerank_results:",rerank_results)
        
