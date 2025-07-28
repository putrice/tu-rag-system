from sentence_transformers.cross_encoder import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query: str, documents: list, top_k: int = 5):
    
    if not documents:
        return []

    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[0], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]