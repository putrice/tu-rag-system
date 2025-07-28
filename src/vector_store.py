import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_system_v1")

def build_vector_store(chunks: list):

    print("Building vector store...")
    collection.add(
        embeddings=embedding_model.encode(chunks).tolist(),
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Vector store built successfully with provided chunks.")

def search_vector_store(query: str, top_k: int = 5):

    embedding = embedding_model.encode(query, convert_to_tensor=True)
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k
    )
    return results['documents'][0] if results['documents'] else []