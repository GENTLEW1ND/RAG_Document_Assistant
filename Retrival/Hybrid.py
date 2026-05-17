from langchain_classic.retrievers.ensemble import EnsembleRetriever
from collections import defaultdict

# def hybrid(vec_retriever, bm25_retriever):
#     print("Setting up Hybrid Retriever...")
#     hybrid_retriever = EnsembleRetriever(
#         retrievers = [vec_retriever, bm25_retriever],
#         weights=[0.5, 0.5] #equal weights to but vector and keyword search
#     )
#     print("Setup complete!!")

#     test_query = "purchase cost 7.5 billion" 

#     retrieved_chunks = hybrid_retriever.invoke(test_query)
#     # for i, doc in enumerate(retrieved_chunks, 1):
#     #     print(f"{i}. {doc.page_content}")
#     print(len(retrieved_chunks))
    
#     return retrieved_chunks


#Reciprocal Rank Fusion
def reciprocal_rank_fusion(chunk_list, k=20, verbose=True):
    if verbose:
        print("*"*50)
        print("Applying Reciprocal Rank Fusion")
        print("*"*50)
    
    #Data Structures for RRF calculation
    rrf_scores = defaultdict(float)
    unique_chunk_objects = {}
    
    chunk_id_map={}
    chunk_no = 1
       
    for chunk_id, chunks in enumerate(chunk_list, 1): 
        
        for position, chunk in enumerate(chunks, 1):
            chunk_content = chunk.page_content
            
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_no}"
                chunk_no += 1
                
            chunk_id = chunk_id_map[chunk_content]
            
            unique_chunk_objects[chunk_content] = chunk
            
            position_score = 1/(k+position)
            
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print("Completed ranking the chunks!")
                
    
        #Sorting the rrf list in ascending or decending order
    sorting_chunk = sorted(
        [(unique_chunk_objects[chunk_content], scores) for chunk_content, scores in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
        
    return sorting_chunk
                    

