import json
import csv
import time
from src.main_v1 import run_query

def calculate_cost(input_token, output_token, cost_per_million_tokens=0.0004):
    """
    Calculate the cost based on input and output tokens.
    
    Args:
        input_token (int): Number of input tokens.
        output_token (int): Number of output tokens.
        cost_per_million_tokens (float): Cost per million tokens.
        
    Returns:
        float: Total cost for the tokens used.
    """
    cost_per_input = 5.00 / 1_000_000
    cost_per_output = 15.00 / 1_000_000
    cost = (input_token * cost_per_input) + (output_token * cost_per_output)
    return cost

def run_benchmark():
    DOCUMENT_PATH = "data/google-10k.pdf"

    with open("data/questions.json", "r") as f:
        queries = json.load(f)

    results = []
    
    for item in queries:
        query = item["query"]
        provider = "ollama"
        model = "llama3.1"
        print(f"Running query: {query}" )

        start_time = time.time()
        response, prompt_tokens, completion_tokens = run_query(query, DOCUMENT_PATH, provider, model)
        end_time = time.time()

        latency = end_time - start_time
        cost = calculate_cost(prompt_tokens, completion_tokens)
        results.append({
            "query_id": item["id"],
            "provider": provider,
            "model": model,
            "query": query,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency": round(latency, 2),
            "api_cost": round(cost, 6),
            "quality": item.get("quality", "N/A")
        })
        print(f"Completed query {item['id']} in {latency:.2f} seconds with cost ${cost:.6f}")

    # Save results to CSV
    with open("data/benchmark_results.csv", "w", newline='') as csvfile:
        fieldnames = ["query_id", "provider", "model", "query", "response", "prompt_tokens", "completion_tokens", "latency", "api_cost", "quality"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print("Benchmark results saved to data/benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()