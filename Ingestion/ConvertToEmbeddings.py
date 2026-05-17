from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def convert_to_embeddings(chunks, persistant_directory="db/chroma.db"):
    """Convert the chunks into vector representation and store it in chroma.db"""
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    embeddings = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persistant_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    
    print("Vector database creation completed!")
    
    return embeddings
