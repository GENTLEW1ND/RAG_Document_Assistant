from Ingestion.LoadingandPartitioning import document_partitioning
from Ingestion.ChunkingByTitle import chunking_by_title
from Ingestion.Unstructured_chunking import summarize_chunks
from Ingestion.ConvertToEmbeddings import convert_to_embeddings
from Retrival.MultiQuery import generate_query_variations
from Retrival.DenseAndSparse import two_retrievers
from Retrival.Hybrid import reciprocal_rank_fusion
from Retrival.History import ask_question
from ReRanker.Re_Ranker import reranker
from Llm.LlmGenerator import generator



def setup_pipeline(document_path):
    # Load document
    doc = document_partitioning(document_path)

    # Chunking
    title_chunk_doc = chunking_by_title(doc)

    # Summarization
    sum_chunks = summarize_chunks(title_chunk_doc)

    # Embeddings
    vec_chunks = convert_to_embeddings(sum_chunks)

    return vec_chunks, sum_chunks



def run_pipeline(query, vec_chunks, sum_chunks):

    # Query rewrite
    restructured_query = ask_question(query)

    # Multi-query
    query_variations = generate_query_variations(restructured_query)

    # Retrieval
    vec_chunk, key_chunks = two_retrievers(vec_chunks, sum_chunks, query_variations)

    # RRF
    chunk_list = vec_chunk + key_chunks
    retrieved_chunk = reciprocal_rank_fusion(chunk_list)

    # Top-K
    top_k_chunk = [doc for doc, _ in retrieved_chunk[:10]]

    # Rerank
    reranked_chunks = reranker(top_k_chunk, query)

    # Final Answer
    result = generator(query, reranked_chunks)

    return result, reranked_chunks




# #Enter the document for partition
# doc = document_partitioning("Document/hybrid-search-effectively-combining-keywords-and-semantic-2yhnqxcsxi.pdf")

# #Initialy chunk the doc using title 
# title_chunk_doc = chunking_by_title(doc)

# #Summarize chunks as it contains Images and tables
# sum_chunks = summarize_chunks(title_chunk_doc)

# #Convert the summarized chunks to vector embeddings
# vec_chunks = convert_to_embeddings(sum_chunks)

# #Restructure query using the query history
# restructured_query = ask_question(query)

# #Generate Multiple queries for a single query 
# query_variations = generate_query_variations(restructured_query)

# #Retrival using Dense and Sparse ie Vector Search and Keyword search respectively
# vec_chunk, key_chunks = two_retrievers(vec_chunks, sum_chunks, query_variations)

# chunk_list = vec_chunk + key_chunks
# retrived_chunk = reciprocal_rank_fusion(chunk_list)

# top_k_chunk = [doc for doc, _ in retrived_chunk[:10]]

# #Retrival using Hybrid search specifically ensembleRetrival
# # hybrid_chunks = hybrid(vec_chunk, key_chunks)


# #Rerank the retrived chunks from the hybrid chunks
# reranked_chunks = reranker(top_k_chunk, query)

# #Generate the Answer using LLM 
# result = generator(query, reranked_chunks)
