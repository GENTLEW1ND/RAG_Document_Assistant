from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
import streamlit as st


def ask_question(user_question: str, model: str = "gpt-4o", temperature: float = 0.0):

    print(f"\n----You asked: {user_question}---")

    llm = ChatOpenAI(model=model, temperature=temperature)

    # ✅ Get history from session (NOT global variable)
    chat_history = st.session_state.get("chat_history", [])

    # -----------------------------
    # Step 1: Rewrite query
    # -----------------------------
    if chat_history:
        messages = [
            SystemMessage(
                content="Rewrite the question to be standalone using chat history. Return only the rewritten question."
            )
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = llm.invoke(messages)
        search_question = result.content.strip()

        print(f"Rewritten Query: {search_question}")

    else:
        search_question = user_question

    return search_question