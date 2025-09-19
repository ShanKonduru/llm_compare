import ollama
import threading
import time
import streamlit as st
import psutil
import torch  # Only if you use CUDA

import subprocess

def get_local_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models_output = result.stdout.strip().splitlines()

        # Example output parsing:
        # Assuming output looks like:
        # model_name  size
        # llama3:latest  4.9 GB
        # llama2:latest  3.8 GB
        # Parse accordingly:
        models = []
        for line in models_output:
            parts = line.split()
            if parts:
                model_name = parts[0]
                models.append(model_name)
        return models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

# Usage:
models_list = get_local_models()

def run_model_thread(model_name, prompt, results, index):
    """
    Runs a single Ollama model inference in a separate thread with additional metrics.
    """
    try:
        # Measure memory before model load/inference
        mem_before = psutil.virtual_memory().used / (1024 ** 2)  # in MB
        gpu_mem_before = None
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / (1024 ** 2)  # in MB

        # Time to load the model if needed - depends on Ollama's API; skipping as singleton
        start_load_time = time.time()
        # Assuming models are loaded and cached; no explicit load call
        load_time = time.time() - start_load_time

        # Run inference
        start_time = time.time()
        response = ollama.generate(model=model_name, prompt=prompt)
        response_time = time.time() - start_time

        # Measure memory after inference
        mem_after = psutil.virtual_memory().used / (1024 ** 2)
        mem_consumed = mem_after - mem_before
        # For GPU
        gpu_mem_after = None
        if torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_mem_used = gpu_mem_after - gpu_mem_before
        else:
            gpu_mem_used = None

        # Count tokens in response
        response_text = response['response']
        token_count = len(response_text.split())
        # Calculate throughput (tokens/sec)
        tokens_per_sec = token_count / response_time if response_time else None

        results[index] = {
            "model": model_name,
            "load_time": round(load_time, 2),
            "response_time": round(response_time, 2),
            "token_count": token_count,
            "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
            "memory_used_MB": round(mem_consumed, 2),
            "gpu_memory_used_MB": round(gpu_mem_used, 2) if gpu_mem_used else "N/A",
            "response": response_text
        }
    except Exception as e:
        results[index] = {
            "model": model_name,
            "load_time": None,
            "response_time": None,
            "token_count": None,
            "tokens_per_sec": None,
            "memory_used_MB": None,
            "gpu_memory_used_MB": None,
            "response": f"Error: {e}"
        }

st.set_page_config(page_title="LLM Response Time & Metrics", layout="wide")
st.title("Enhanced LLM Performance Metrics Comparison with Ollama")

# User input prompt
prompt = st.text_area("Enter your prompt:", value="What is the capital of France?", height=150)

models = models_list # ["llama3:latest", "llama2:latest", "gemma3:latest", "mistral:latest"]
selected_models = st.multiselect("Select models to test:", options=models, default=models_list)

if st.button("Run Tests", use_container_width=True):
    if not selected_models:
        st.warning("Please select at least one model to test.")
    else:
        results = [None] * len(selected_models)
        threads = []

        with st.spinner("Running models... Please wait."):
            for i, model_name in enumerate(selected_models):
                t = threading.Thread(target=run_model_thread, args=(model_name, prompt, results, i))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        st.write("### Detailed Results")
        # Display in a dataframe for better readability
        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df)

        st.success("All tests completed!")
