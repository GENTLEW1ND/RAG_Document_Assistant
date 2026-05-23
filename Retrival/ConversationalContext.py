def build_conversation_context(chat_history, limit=4):

    """
    Convert chat history into clean conversational context.
    """

    if not chat_history:
        return ""

    conversation_context = ""

    for msg in chat_history[-limit:]:

        role = "User" if msg.type == "human" else "Assistant"

        conversation_context += f"{role}: {msg.content}\n"

    return conversation_context