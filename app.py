import streamlit as st
import time
import pandas as pd

from src.main_v1 import run_query
from src.main_v2 import run_query_v2
from src.main_v3 import run_query_v3

from src.modules import security_pattern, security

st.set_page_config(
    page_title="RAG System Comparison",
    page_icon="📊",
    layout="wide"
)

def calculate_cost(input_tokens, output_tokens, model="gpt-4o"):
    """Estimates cost for OpenAI models."""
    if "gpt" not in model:
        return 0.0
    
    cost_per_input = 5.00 / 1_000_000
    cost_per_output = 15.00 / 1_000_000
    return (input_tokens * cost_per_input) + (output_tokens * cost_per_output)

st.title("📊 RAG Pipeline Comparison Dashboard")
st.markdown("""
This application runs the same query through three different RAG pipeline architectures, allowing for a direct comparison of their performance, cost, and answer quality.
- **Phase 1 (Baseline)
- **Phase 2 (Optimized Retrieval)
- **Phase 3 (Cost Control)
""")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("These settings apply to all pipelines where applicable.")
    
    final_provider = st.selectbox("Final Model Provider", ["openai", "ollama"], index=0)
    final_model = st.text_input("Final Model Name", "gpt-4o")
    
    st.markdown("---")
    st.info("The Phase 3 router uses a hardcoded 'first' model (`ollama/llama3.1`) for its initial check.")

query = st.text_input("Enter your query:", placeholder="e.g., What are the primary risk factors?")

if st.button("Run Comparison"):
    if not query:
        st.warning("Please enter a query.")
    elif security_pattern.is_prompt_injection(query):
        st.error("Error: Your query contains language that could be interpreted as a prompt injection attempt. Please rephrase your question.")
    elif security.is_content_toxic(query):
        st.error("Error: Your query contains language that could be interpreted as a prompt injection attempt. Please rephrase your question.")
    else:
        results = []
        
        with st.spinner("Running Phase 1 (Baseline)..."):
            start_time = time.time()
            print(f"Running query: {query} with model {final_model} from provider {final_provider}")
            response, prompt_tokens, completion_tokens = run_query(query, "data/nvidia-10k.pdf", final_provider, final_model)
            latency = time.time() - start_time
            results.append({
                "Phase": "Phase 1: Baseline",
                "Answer": response,
                "Input Tokens": prompt_tokens,
                "Output Tokens": completion_tokens,
                "Latency (s)": f"{latency:.2f}",
                "Model Used": final_model,
                "Est. Cost ($)": calculate_cost(prompt_tokens, completion_tokens, final_model)
            })

        with st.spinner("Running Phase 2 (Optimized Retrieval)..."):
            start_time = time.time()
            response, prompt_tokens, completion_tokens = run_query_v2(query, "data/nvidia-10k.pdf", final_provider, final_model)
            latency = time.time() - start_time
            results.append({
                "Phase": "Phase 2: Optimized",
                "Answer": response,
                "Input Tokens": prompt_tokens,
                "Output Tokens": completion_tokens,
                "Latency (s)": f"{latency:.2f}",
                "Model Used": final_model,
                "Est. Cost ($)": calculate_cost(prompt_tokens, completion_tokens, final_model)
            })
            
        with st.spinner("Running Phase 3 (Caching & Routing)..."):
            start_time = time.time()
            response, prompt_tokens, completion_tokens = run_query_v3(query, "data/nvidia-10k.pdf", final_provider, final_model)
            latency = time.time() - start_time
            results.append({
                "Phase": "Phase 3: Cost Control",
                "Answer": response,
                "Input Tokens": prompt_tokens,
                "Output Tokens": completion_tokens,
                "Latency (s)": f"{latency:.2f}",
                "Model Used": final_model,
                "Est. Cost ($)": calculate_cost(prompt_tokens, completion_tokens, final_model)
            })

        st.subheader("Comparison Results")
        
        col1, col2, col3 = st.columns(3)
        
        for i, res in enumerate(results):
            with [col1, col2, col3][i]:
                st.markdown(f"### {res['Phase']}")
                st.metric(label="Model Used", value=res['Model Used'].upper())
                st.metric(label="Latency", value=f"{res['Latency (s)']}s")
                st.metric(label="Input Tokens", value=res['Input Tokens'])
                st.metric(label="Est. Cost", value=f"${res['Est. Cost ($)']:.6f}")
                
                with st.expander("View Answer"):
                    st.markdown(res['Answer'])
