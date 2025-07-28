from . import data_loader, vector_store
from .modules import reranker, compressor, semantic_cache, router
import argparse

def run_query_v3(query: str, doc_path: str, provider: str = "openai", model: str = "gpt-3.5-turbo"):
    
    cached_answer = semantic_cache.check_cache(query)
    if cached_answer:
        return cached_answer, 0, 0
    
    chunks = data_loader.load_and_chunk_document(doc_path)
    if not chunks:
        return "Failed to process documents", 0, 0
    vector_store.build_vector_store(chunks)

    initial_retrieval = vector_store.search_vector_store(query, top_k=10)
    reracked_docs = reranker.rerank_documents(query, initial_retrieval, top_k=3)
    context = "\n---\n".join(reracked_docs)
    compressed_context = compressor.compress_context(context, query, model, provider)

    if not compressed_context.strip():
        compressed_context = context

    final_answer, input_tokens, output_tokens, model = router.route_query(query, compressed_context)
    if final_answer:
        print(f"Response: {final_answer}")
        print(f"Prompt Tokens: {input_tokens}, Completion Tokens: {output_tokens}")
        semantic_cache.add_to_cache(query, final_answer)

    return final_answer, input_tokens, output_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a query against the vector store.")
    parser.add_argument("query", type=str, help="The search query to run.")
    parser.add_argument("--provider", type=str, default="openai", choices=['openai', 'ollama'], help="LLM provider (openai or ollama).")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM model to use.")
    
    args = parser.parse_args()
    DOCUMENT_PATH = "data/google-10k.pdf"
    
    response = run_query_v3(args.query, DOCUMENT_PATH, args.provider, args.model)
    if response:
        print(f"Final Response: {response}")
    else:
        print("No response generated.")
