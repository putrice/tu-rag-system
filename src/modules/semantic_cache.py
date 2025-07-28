import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

cache_client = chromadb.PersistentClient(path="./cache_db")
cache_collection = cache_client.get_or_create_collection(name="semantic_cache_v1")

SESSION_CACHE = {}

def add_to_cache(query: str, answer: str):
   
    query_embedding = embedding_model.encode([query]).tolist()[0]
    cache_collection.add(
        embeddings=[query_embedding],
        metadatas=[{"answer": answer}],
        documents=[query],
        ids=[query]
    )
    SESSION_CACHE[query] = answer

def check_cache(query: str, threshold: float = 0.8):
    
    if query in SESSION_CACHE:
        return SESSION_CACHE[query]
    
    if cache_collection.count() == 0:
        return None

    query_embedding = embedding_model.encode([query]).tolist()
    results = cache_collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )

    distances = results['distances'][0][0]
    similarity = 1 - distances  # Convert distance to similarity

    if similarity >= threshold:
        cached_answer = results['metadatas'][0][0]['answer']
        SESSION_CACHE[query] = cached_answer
        return cached_answer
    
    return None

