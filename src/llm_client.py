import os
import ollama
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm_response(prompt: str, provider: str = "openai", model: str = "gpt-3.5-turbo") -> str:
    """
    Sends a prompt to the OpenAI API and returns the response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The model to use for the request.

    Returns:
        str: The response from the LLM.
    """
    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            return response.choices[0].message.content.strip(), response.usage.prompt_tokens, response.usage.completion_tokens
        except Exception as e:
            print(f"Error communicating with OpenAI API: {e}")
            return None, 0, 0
    elif provider == "ollama":
        try:

            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            input_tokens = response.get('prompt_eval_count', 0)
            output_tokens = response.get('eval_count', 0)
            return response['message']['content'].strip(), input_tokens, output_tokens
        except Exception as e:
            print(f"Error communicating with Ollama API: {e}")
            return None, 0, 0
    else:
        print(f"Unsupported provider: {provider}. Please use 'openai' or 'ollama'.")
        return None, 0, 0