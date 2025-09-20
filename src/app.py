import re
import ollama
import threading
import time
import streamlit as st
import psutil
import pandas as pd
import subprocess

# Attempt to import a GPU monitoring library (pynvml for NVIDIA GPUs)
try:
    from pynvml.pynvml import *
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    
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

import time
import psutil
import subprocess

def get_system_metrics():
    # Get memory usage
    mem = psutil.virtual_memory()
    memory_used_mb = mem.used / (1024 * 1024)
    # Get GPU memory usage (requires nvidia-smi)
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        gpu_memory_used_mb = float(gpu_info.decode().strip())
    except Exception:
        gpu_memory_used_mb = None
    return memory_used_mb, gpu_memory_used_mb

import subprocess

def get_gpu_memory_usage(process_name="ollama.exe"):
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])
        lines = output.decode().splitlines()
        for line in lines:
            pid, proc_name, mem_used = [item.strip() for item in line.split(',')]
            if proc_name == process_name:
                return float(mem_used.replace(' MiB', ''))
        return None
    except Exception:
        return None

def get_response_and_metrics(client, prompt):
    start_time = time.perf_counter()
    response = client.chat(prompt)  # Or relevant method
    end_time = time.perf_counter()

    response_time = end_time - start_time
    load_time = response_time  # Adjust if load time differs from response time

    token_count = len(response['tokens'])  # Assuming response includes token info
    tokens_per_sec = token_count / response_time if response_time > 0 else 0

    mem_usage, gpu_mem = get_system_metrics()

    metrics = {
        'Response Time': response_time,
        'Load Time': load_time,
        'Token Count': token_count,
        'Tokens/Sec': tokens_per_sec,
        'Memory Usage (MB)': mem_usage,
        'GPU Memory Used (MB)': gpu_mem
    }
    return response['text'], metrics

def run_model_thread(model_name, prompt, thread_id):
    """
    Runs a single Ollama model, streams the response, and captures metrics.
    """
    try:
        # Initial CPU memory usage before the model run
        process = psutil.Process()
        initial_mem_mb = process.memory_info().rss / (1024 * 1024)
        
        # Initial GPU memory usage (if NVML is available)
        initial_gpu_mem = 'N/A'
        if NVML_AVAILABLE:
            try:
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(handle)
                initial_gpu_mem = info.used / (1024 * 1024)
            except NVMLError as err:
                initial_gpu_mem = 'N/A'

        full_response = ""
        
        # Use a more complex prompt to ensure a longer response with more tokens
        complex_prompt = f"Elaborate on the topic: '{prompt}'. Provide a detailed, paragraph-long response."

        stream = ollama.generate(
            model=model_name,
            prompt=complex_prompt,
            stream=True,
            options={'temperature': 0.0}
        )
        st.write(f"stream: {stream}")

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
                
                # Final CPU memory usage after the model run
                final_mem_mb = process.memory_info().rss / (1024 * 1024)
                
                # Final GPU memory usage
                final_gpu_mem = 'N/A'
                if NVML_AVAILABLE:
                    try:
                        info = nvmlDeviceGetMemoryInfo(handle)
                        final_gpu_mem = info.used / (1024 * 1024)
                        nvmlShutdown()
                    except NVMLError:
                        final_gpu_mem = 'N/A'
                
                with results_lock:
                    model_results.append({
                        'Model': model_name,
                        'Response': full_response,
                        'Response Time': f"{total_duration_ms / 1000:.2f}s",
                        'Load Time': f"{load_duration_ms / 1000:.2f}s",
                        'Token Count': eval_count,
                        'Tokens/Sec': f"{tokens_per_sec:.2f}",
                        'Memory Usage (MB)': f"{final_mem_mb - initial_mem_mb:.2f}",
                        'GPU Memory Used (MB)': f"{final_gpu_mem - initial_gpu_mem:.2f}" if isinstance(final_gpu_mem, (int, float)) and isinstance(initial_gpu_mem, (int, float)) else 'N/A'
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
st.set_page_config(layout="wide")
st.title("Ollama Model Performance Benchmarking")
st.markdown("This tool helps you benchmark the performance of different Ollama models on a given prompt.")
st.markdown("---")

# User inputs
prompt = st.text_area("Enter a prompt:", "What is the capital of France?")
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