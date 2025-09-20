import re
import ollama
import threading
import time
import streamlit as st
import psutil
import torch  # Only if you use CUDA
import pandas as pd

import subprocess

def get_local_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models_output = result.stdout.strip().splitlines()

        # Define keywords to include or exclude
        include_keywords = ["llama", "chat", "gpt", "mistral", "gemma"]
        exclude_keywords = ["embed", "vector", "embedding"]

        models = []
        for line in models_output:
            parts = line.split()
            if parts:
                model_name = parts[0]
                name_lower = model_name.lower()
                # Include if it contains any include_keyword AND does not contain exclude_keyword
                if any(keyword in name_lower for keyword in include_keywords) and not any(keyword in name_lower for keyword in exclude_keywords):
                    models.append(model_name)
        return models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def get_all_local_models():
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

# This function will be executed by each thread.
# It uses the subprocess library to run a model and capture its output, including memory usage.
def run_model_thread(model_name, prompt, results, index):
    """
    Runs a single Ollama model inference in a separate thread using subprocess.

    This function captures both response time and memory usage from the command-line output.
    
    Args:
        model_name (str): The name of the Ollama model.
        prompt (str): The prompt for the model.
        results (list): A shared list to store the results from each thread.
        index (int): The index in the results list to store this thread's output.
    """
    command = ["ollama", "run", model_name, "--verbose", prompt]
    try:
        start_time = time.time()
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        elapsed_time = time.time() - start_time
        
        # For debugging, print the raw stdout and stderr to the console
        print(f"--- Raw Output for {model_name} ---")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print("---------------------------------")
        
        output = result.stdout.strip()
        
        # Search for a memory usage pattern in both stdout and stderr
        memory_usage = "N/A"
        
        # Check stderr first
        if result.stderr:
            memory_match = re.search(r"memory usage: (\d+\.?\d*)\s*(GB|MB)", result.stderr)
            if memory_match:
                value = float(memory_match.group(1))
                unit = memory_match.group(2)
                memory_usage = f"{value} {unit}"

        # If not found in stderr, check stdout
        if memory_usage == "N/A" and result.stdout:
            memory_match = re.search(r"memory usage: (\d+\.?\d*)\s*(GB|MB)", result.stdout)
            if memory_match:
                value = float(memory_match.group(1))
                unit = memory_match.group(2)
                memory_usage = f"{value} {unit}"


        results[index] = {
            "model": model_name,
            "response_time": round(elapsed_time, 2),
            "memory_usage": memory_usage,
            "response": output
        }

    except subprocess.CalledProcessError as e:
        results[index] = {
            "model": model_name,
            "response_time": None,
            "memory_usage": "Error",
            "response": f"Command Error: {e.stderr}"
        }
    except Exception as e:
        results[index] = {
            "model": model_name,
            "response_time": None,
            "memory_usage": "Error",
            "response": f"Error: {e}"
        }

st.set_page_config(page_title="LLM Comparison", layout="wide")

st.title("LLM Response Time and Memory Comparison with Ollama")

# UI for user input
prompt = st.text_area(
    "Enter your prompt:", 
    value="What is the capital of France?", 
    height=150
)

models = models_list
selected_models = st.multiselect(
    "Select models to test:", 
    options=models, 
    default=models_list
)

if st.button("Run Tests", use_container_width=True):
    if not selected_models:
        st.warning("Please select at least one model to test.")
    else:
        results = [None] * len(selected_models)
        threads = []

        # Use a spinner to indicate that the tests are running
        with st.spinner("Running models... Please wait."):
            # Create and start a thread for each selected model
            for i, model_name in enumerate(selected_models):
                t = threading.Thread(
                    target=run_model_thread, 
                    args=(model_name, prompt, results, i)
                )
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

        # Display the results in a cleaner table format
        st.write("### Results")
        
        headers = ["Model", "Response Time (s)", "Memory Usage", "Response"]
        data = []
        for res in results:
            if res:
                data.append([
                    res['model'],
                    f"{res['response_time']}s" if res['response_time'] is not None else "N/A",
                    res['memory_usage'],
                    res['response']
                ])

        # Create a pandas DataFrame to properly display the headers
        df = pd.DataFrame(data, columns=headers)
        st.dataframe(df, use_container_width=True)
        
        st.success("Tests complete!")