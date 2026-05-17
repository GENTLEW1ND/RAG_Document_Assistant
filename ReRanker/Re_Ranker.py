from langchain_cohere import CohereRerank
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=".env")

def reranker(retrieved_chunks, query):

    reranker = CohereRerank(model="rerank-english-v3.0")

    docs_text = [doc.page_content for doc in retrieved_chunks]

    results = reranker.client.rerank(
        model="rerank-english-v3.0", 
        query=query,
        documents=docs_text,
        top_n=5
    )

    reranked_docs = [retrieved_chunks[r.index] for r in results.results]

    return reranked_docs