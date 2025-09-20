import re
import ollama
import threading
import time
import streamlit as st
import psutil
import torch
import pandas as pd
import subprocess

# Define a global list to store model results
model_results = []
# Create a lock for thread-safe access to the results list
results_lock = threading.Lock()

def get_local_models():
    """Fetches a list of local Ollama models and handles potential errors."""
    # Create a client instance, explicitly setting the host to the default Ollama address
    client = ollama.Client(host='http://localhost:11434')
    
    try:
        filtered_models = []
        models_data = client.list()
        
        include_keywords = ["llama", "chat", "gpt", "mistral", "gemma"]
        exclude_keywords = ["embed", "vector", "embedding"]

        for model in models_data.get('models', []):
            model_name = model.get('model')
            if model_name and isinstance(model_name, str):
                name_lower = model_name.lower()
                if any(keyword in name_lower for keyword in include_keywords) and \
                   not any(keyword in name_lower for keyword in exclude_keywords):
                    filtered_models.append(model_name)
        return filtered_models
    except ollama.ResponseError as e:
        st.error(f"Failed to connect to Ollama: {e}")
        return []

def run_model_thread(model_name, prompt, thread_id):
    """
    Runs a single Ollama model, streams the response, and captures metrics.
    
    This function now uses the official Ollama Python client, which provides 
    direct access to detailed metrics like total duration, load duration, and 
    token counts from the API response, making it more reliable than
    parsing stdout with regex.
    """
    try:
        # Initial memory usage before the model run
        process = psutil.Process()
        initial_mem_mb = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()
        full_response = ""
        
        stream = ollama.generate(
            model=model_name,
            prompt=prompt,
            stream=True,
            options={'temperature': 0.0}
        )

        for chunk in stream:
            full_response += chunk.get('response', '')
            # Check for metrics at the end of the stream
            if chunk.get('done', False):
                metrics = chunk.get('metrics', {})
                total_duration_ms = metrics.get('total_duration', 0)
                load_duration_ms = metrics.get('load_duration', 0)
                eval_count = metrics.get('eval_count', 0)
                eval_duration_ms = metrics.get('eval_duration', 0)
                
                # Calculate tokens per second
                tokens_per_sec = (eval_count / (eval_duration_ms / 1000)) if eval_duration_ms > 0 else 0
                
                # Final memory usage after the model run
                final_mem_mb = process.memory_info().rss / (1024 * 1024)
                
                with results_lock:
                    model_results.append({
                        'Model': model_name,
                        'Response': full_response,
                        'Response Time': total_duration_ms / 1000,
                        'Load Time': load_duration_ms / 1000,
                        'Token Count': eval_count,
                        'Tokens/Sec': tokens_per_sec,
                        'Memory Usage (MB)': final_mem_mb - initial_mem_mb,
                        # GPU memory is not directly available from the standard client, 
                        # but you can add it if you're using a specific library like `gpustat`
                        'GPU Memory Used (MB)': 'N/A'
                    })
                st.session_state[f'thread_{thread_id}_done'] = True
    except Exception as e:
        st.error(f"Error running model {model_name}: {e}")
        with results_lock:
            model_results.append({
                'Model': model_name,
                'Response': 'Error',
                'Response Time': 'N/A',
                'Load Time': 'N/A',
                'Token Count': 'N/A',
                'Tokens/Sec': 'N/A',
                'Memory Usage (MB)': 'N/A',
                'GPU Memory Used (MB)': 'N/A'
            })
        st.session_state[f'thread_{thread_id}_done'] = True

# Streamlit app layout
st.title("Ollama Model Performance Benchmarking")
st.markdown("This tool helps you benchmark the performance of different Ollama models on a given prompt.")
st.set_page_config(layout="wide")
# User inputs
prompt = st.text_area("Enter a prompt:", "What is the capital of France?")
st.markdown("---")
models_list = get_local_models()
if models_list:
    selected_models = st.multiselect("Select models to test:", models_list, default=models_list)
else:
    st.warning("No local Ollama models found. Please ensure Ollama is running and you have models installed.")
    selected_models = []

# Action button
if st.button("Run Benchmark") and selected_models:
    model_results.clear()
    
    threads = []
    with st.spinner("Running models... Please wait."):
        # Create and start a thread for each selected model
        for i, model_name in enumerate(selected_models):
            t = threading.Thread(
                target=run_model_thread, 
                args=(model_name, prompt, i)
            )
            threads.append(t)
            t.start()
            
        # Wait for all threads to complete
        for t in threads:
            t.join()

    # Display results in a DataFrame
    if model_results:
        results_df = pd.DataFrame(model_results)
        st.markdown("---")
        st.subheader("Benchmark Results")
        st.dataframe(results_df)