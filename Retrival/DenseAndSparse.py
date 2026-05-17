from langchain_community.retrievers import BM25Retriever




def two_retrievers(vec_doc, sparse_doc, query_variations, k=10):
    """
    vec_doc: vectorstore (for dense retrieval)
    sparse_doc: list of Document objects (for BM25)
    query_variations: list of query strings
    """
    
    # Dense retriever (vector search)
    dense_retriever = vec_doc.as_retriever(search_kwargs={"k": k})
    
    # Sparse retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(sparse_doc)
    bm25_retriever.k = 10

    dense_results = []   # list of lists: one list per query variation
    sparse_results = []

    print("=" * 70)
    for i, query in enumerate(query_variations, 1):
        print(f"\nQuery {i}/{len(query_variations)}: {query}")

        # Get results for this query
        dense_docs = dense_retriever.invoke(query)
        sparse_docs = bm25_retriever.invoke(query)

        dense_results.append(dense_docs)
        sparse_results.append(sparse_docs)

        print(f"  → Dense  retrieved: {len(dense_docs)} docs")
        print(f"  → Sparse retrieved: {len(sparse_docs)} docs")

        # Optional: print only first 2 docs to avoid spam
        for j, doc in enumerate(dense_docs[:2], 1):
            print(f"    Dense {j}: {doc.page_content[:120].replace(chr(10),' ')}...")

    print("\n" + "=" * 70)
    print("Multi-Query Retrieval Completed")
    print(f"Collected {len(dense_results)} query variations for dense and sparse")

    return dense_results, sparse_results