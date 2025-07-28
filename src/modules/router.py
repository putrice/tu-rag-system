import json
from src import llm_client

FIRST_PROVIDER = "ollama"
FIRST_MODEL = "llama3.1"
SECOND_PROVIDER = "openai"
SECOND_MODEL = "gpt-3.5-turbo"

def route_query(query: str, context: str):
    
    router_prompt = f"""
    You are an expert Q&A system. Your task is to answer the user's question based *only* on the provided context.
    Analyze the question and the context, then return a JSON object with two keys:
    1. "answer": Your best attempt at an answer.
    2. "confidence": A score of "high" if you are confident the answer is in the context, or "low" if you are unsure or the answer is not present.

    Context:
    ---
    {context}
    ---
    User Question: {query}

    JSON Response:
    """

    first_response_str, _, _ = llm_client.get_llm_response(router_prompt, FIRST_PROVIDER, FIRST_MODEL)

    try:
        parsed_response = json.loads(first_response_str)
        confidence = parsed_response.get("confidence", "low").lower()
        answer = parsed_response.get("answer", "")
        print(f"Answer: {answer}, Confidence: {confidence}")

        if confidence == "high":
            print(f"Using first model ({FIRST_PROVIDER}/{FIRST_MODEL}) for query: {query}")
            return answer, 0, 0, FIRST_MODEL
        
        return answer, 0, 0, FIRST_MODEL
    except (json.JSONDecodeError, KeyError):
        answer = first_response_str

    return None, 0, 0, FIRST_MODEL
    # final_prompt = f"""
    # Based on the following context, please answer the question.

    # Context:
    # {context}

    # Question: {query}

    # Answer:
    # """
    # final_answer, input_tokens, output_tokens = llm_client.get_llm_response(
    #     prompt=final_prompt,
    #     provider=SECOND_PROVIDER,
    #     model=SECOND_MODEL
    # )

    # return final_answer, input_tokens, output_tokens, SECOND_MODEL