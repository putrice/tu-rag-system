from . import data_loader, vector_store, llm_client
import argparse

def run_query(query: str, doc_path: str, provider: str = "openai", model: str = "gpt-3.5-turbo"):
    """
    Runs a query against the vector store and retrieves relevant documents.

    Args:
        query (str): The search query.
        doc_path (str): Path to the document to be processed.

    Returns:
        list: List of relevant documents.
    """
    # Load and chunk the document
    chunks = data_loader.load_and_chunk_document(doc_path)
    if not chunks:
        print("No chunks were created from the document.")
        return []
    # Build the vector store with the chunks
    vector_store.build_vector_store(chunks)
    
    # Search the vector store for relevant documents
    results = vector_store.search_vector_store(query)
    context = "\n---\n".join(results)

    prompt_template = f"""
    Based on the following context, answer the question:

    Context:
    {context}

    Question: {query}
    
    Answer:
    """
    response, prompt_tokens, completion_tokens = llm_client.get_llm_response(prompt_template, provider, model)
    if response:
        print(f"Response: {response}")
        print(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}")
    else:
        print("No response received from the LLM.") 

    return response, prompt_tokens, completion_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a query against the vector store.")
    parser.add_argument("query", type=str, help="The search query to run.")
    parser.add_argument("--provider", type=str, default="openai", choices=['openai', 'ollama'], help="LLM provider (openai or ollama).")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM model to use.")
    
    args = parser.parse_args()
    DOCUMENT_PATH = "data/google-10k.pdf"
    
    response, prompt_tokens, completion_tokens = run_query(args.query, DOCUMENT_PATH, args.provider, args.model)
    if response:
        print(f"Final Response: {response}")
    else:
        print("No response generated.")