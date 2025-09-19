import ollama
import threading
import time
import streamlit as st
import subprocess

# This function will be executed by each thread.
# It uses the ollama library to run a model and store the result in a shared list.
def run_model_thread(model_name, prompt, results, index):
    """
    Runs a single Ollama model inference in a separate thread.

    Args:
        model_name (str): The name of the Ollama model.
        prompt (str): The prompt for the model.
        results (list): A shared list to store the results from each thread.
        index (int): The index in the results list to store this thread's output.
    """
    try:
        start_time = time.time()
        # Use the ollama library's generate method directly
        response = ollama.generate(model=model_name, prompt=prompt)
        elapsed_time = time.time() - start_time
        
        # Store the relevant data in the shared results list at the correct index
        results[index] = {
            "model": model_name,
            "response_time": round(elapsed_time, 2),
            "response": response['response'] # Extract the actual response text
        }
    except Exception as e:
        # Handle exceptions gracefully and store an error message
        results[index] = {
            "model": model_name,
            "response_time": None,
            "response": f"Error: {e}"
        }

st.set_page_config(page_title="LLM Comparison", layout="wide")

st.title("LLM Response Time Comparison with Ollama")

# UI for user input
prompt = st.text_area(
    "Enter your prompt:", 
    value="What is the capital of France?", 
    height=150
)

models = ["llama3:latest", "llama2:latest", "gemma:latest", "mistral:latest"]
selected_models = st.multiselect(
    "Select models to test:", 
    options=models, 
    default=["llama3:latest", "gemma:latest"]
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

        # Display the results
        st.write("### Results")
        col1, col2 = st.columns(2)
        for res in results:
            if res:
                col1.write(f"**Model:** {res['model']}")
                col1.write(f"Response Time: {res['response_time']} seconds")
                col2.info(f"Response: {res['response']}")
            else:
                st.error("An error occurred. Check the console for details.")
        st.success("Tests complete!")
