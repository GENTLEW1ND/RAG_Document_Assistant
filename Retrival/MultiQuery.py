from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List

class QueryVariation(BaseModel):
    queries: List[str]


def generate_query_variations(
    original_query: str, 
    num_variations: int = 3,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    verbose: bool = True      
) -> List[str]:
    """
    Generates multiple query variations using LLM for better retrieval coverage.
    """
    
    # Initialize LLM
    llm = ChatOpenAI(model=model, temperature=temperature)
    llm_with_tools = llm.with_structured_output(QueryVariation)
    
    prompt = f"""You are an expert at query rewriting for information retrieval.
    
Generate {num_variations} different variations of the following query.
Each variation should approach the same topic from a slightly different angle 
to help retrieve more diverse and relevant documents.

Original query: {original_query}

Requirements:
- Keep the core meaning intact
- Use different wording and focus areas
- Make them natural search queries
- Do not add unrelated topics

Return only the list of {num_variations} queries.
"""

    response = llm_with_tools.invoke(prompt)
    query_variations = response.queries

    # Only print when verbose=True (default)
    if verbose:
        print(f"Original Query: '{original_query}'")
        print(f"\nGenerated {len(query_variations)} Query Variations:")
        for i, variation in enumerate(query_variations, 1):
            print(f"{i}. {variation}")
        print("\n" + "="*60)

    return query_variations