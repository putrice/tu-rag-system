from src import llm_client

def compress_context(context: str, query: str, model: str = "gpt-3.5-turbo", provider: str = "openai") -> str:
    
    compression_prompt = f"""
        Read the following context and the user's question carefully. 
        Identify and extract ONLY the sentences from the context that are directly relevant to answering the question.
        If no sentences are relevant, return an empty string.

        Context:
        ---
        {context}
        ---

        User Question: {query}
        Relevant Sentences:
        """
    response, _, _ = llm_client.get_llm_response(compression_prompt, provider, model)
    
    if response:
        return response.strip()
    else:
        print("Failed to compress context.")
        return context