from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List
from langchain_core.documents import Document


def generator(
    original_query: str, 
    reranked_docs: List[Document],
    model: str = "gpt-4o",
    temperature: float = 0.0
) -> str:
    """
    Generates final answer using reranked documents.
    """
    
    if not reranked_docs:
        return "I don't have enough information to answer this question based on the provided documents."
    
    # Better document formatting
    documents_text = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content.strip()}" 
        for i, doc in enumerate(reranked_docs)
    ])

    prompt = f"""You are a helpful, accurate, and concise assistant.

Answer the following question using **only** the information provided in the documents below.
If the documents do not contain enough information to answer the question, clearly say so.

Question: {original_query}

Documents:
{documents_text}

Instructions:
- Be clear, direct, and well-structured
- Use only information present in the documents
- If you are unsure or information is missing, say "Based on the provided documents, I don't have sufficient information..."
- Do not make up information
"""

    llm = ChatOpenAI(model=model, temperature=temperature)

    messages = [
        SystemMessage(content="You are a helpful, truthful, and accurate assistant."),
        HumanMessage(content=prompt)
    ]

    final_result = llm.invoke(messages)
    
    print("\n" + "="*60)
    print("Generated Response:")
    print("="*60)
    print(final_result.content)
    print("="*60)

    return final_result.content