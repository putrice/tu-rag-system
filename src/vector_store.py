import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_system_v1")

def build_vector_store(chunks: list):
    """
    Builds a vector store from the provided text chunks.

    Args:
        chunks (list): List of text chunks to be added to the vector store.
    """
    print("Building vector store...")
    collection.add(
        embeddings=embedding_model.encode(chunks).tolist(),
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Vector store built successfully with provided chunks.")

def search_vector_store(query: str, top_k: int = 5):
    """
    Searches the vector store for the most relevant documents based on the query.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to return.

    Returns:
        list: List of documents that match the query.
    """
    embedding = embedding_model.encode(query, convert_to_tensor=True)
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k
    )
    return results['documents'][0] if results['documents'] else []