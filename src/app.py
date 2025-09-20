# To run this app, you need to have Streamlit, Ollama, pynvml, and psutil installed.
# You also need a running Ollama server with at least one model pulled.
#
# Installation:
# pip install streamlit ollama pandas pynvml psutil
#
# To run the app:
# streamlit run app.py

import streamlit as st
import ollama
import time
import psutil
import sys
import pandas as pd
import plotly.express as px

# It's good practice to wrap pynvml imports in a try-except block
# as it's specific to NVIDIA GPUs and might not be available.
try:
    import pynvml
    PNVML_AVAILABLE = True
except ImportError:
    PNVML_AVAILABLE = False
except Exception as e:
    st.warning(f"Warning: pynvml could not be imported. GPU metrics will not be collected. Error: {e}")
    PNVML_AVAILABLE = False


class OllamaMetrics:
    """
    A utility class for collecting system metrics related to the Ollama process.
    """
    @staticmethod
    def get_ollama_gpu_mem_usage() -> float:
        """
        Finds the Ollama process and returns its GPU memory usage in MB.
        Returns 0.0 if not found or on error.
        
        This function requires pynvml and an NVIDIA GPU.
        """
        if not PNVML_AVAILABLE:
            return 0.0

        try:
            pynvml.nvmlInit()
            
            # Get the PID of the Ollama process
            ollama_process_pid = None
            for proc in psutil.process_iter(['name', 'pid']):
                if 'ollama' in proc.info['name'].lower():
                    ollama_process_pid = proc.info['pid']
                    break
            
            if not ollama_process_pid:
                return 0.0

            # Iterate through all GPU devices
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # Find the process on the GPU that matches the Ollama PID
                for proc_info in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if proc_info.pid == ollama_process_pid:
                        # Memory is in bytes, convert to MB
                        return proc_info.usedGpuMemory / (1024 * 1024)
            
        except pynvml.NVMLError as e:
            st.error(f"pynvml error: {e}")
            return 0.0
        except Exception as e:
            st.error(f"An unexpected error occurred in get_ollama_gpu_mem_usage: {e}")
            return 0.0
        finally:
            if PNVML_AVAILABLE:
                pynvml.nvmlShutdown()

        return 0.0


class OllamaLLMBenchmark:
    """
    A class to run a benchmark on local Ollama LLMs, collecting performance
    and resource usage metrics.
    """
    def __init__(self):
        """
        Initializes the Ollama client.
        """
        try:
            self.client = ollama.Client()
        except Exception as e:
            st.error(f"Error initializing Ollama client. Is Ollama running? Error: {e}")
            st.stop()

    @st.cache_data(show_spinner=False)
    def _get_local_llms(_self) -> list[str]:
        """
        Gets a list of locally available Ollama model names.
        Cached to avoid repeated API calls.
        """
        try:
            models = _self.client.list()['models']
            return [model['model'] for model in models]
        except Exception as e:
            st.error(f"Error getting list of local models. Error: {e}")
            return []

    def _ask_llm(self, model_name: str, question: str) -> dict:
        """
        Sends a question to a local Ollama LLM and returns the response
        along with a detailed metrics object.
        """
        # Find the Ollama process to get RAM usage
        ollama_process = None
        for proc in psutil.process_iter(['name', 'pid']):
            if 'ollama' in proc.info['name'].lower():
                ollama_process = proc
                break
        
        start_time = time.time()
        try:
            response = self.client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': question}],
                stream=False
            )
            end_time = time.time()

            mem_usage_mb = 0.0
            if ollama_process:
                try:
                    mem_usage_bytes = ollama_process.memory_info().rss
                    mem_usage_mb = mem_usage_bytes / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            gpu_mem_used_mb = OllamaMetrics.get_ollama_gpu_mem_usage()

            metrics = {
                'model': model_name,
                'response_text': response['message']['content'],
                'metrics': {
                    'total_duration_s': response.get('total_duration', 0) / 1e9,
                    'load_duration_s': response.get('load_duration', 0) / 1e9,
                    'eval_duration_s': response.get('eval_duration', 0) / 1e9,
                    'eval_count': response.get('eval_count', 0),
                    'mem_usage_mb': mem_usage_mb,
                    'gpu_mem_used_mb': gpu_mem_used_mb,
                    'request_time_s': end_time - start_time,
                }
            }
            return metrics

        except Exception as e:
            return {
                'model': model_name,
                'response_text': f"An error occurred: {e}",
                'metrics': {
                    'total_duration_s': 0.0, 'load_duration_s': 0.0, 'eval_duration_s': 0.0,
                    'eval_count': 0, 'mem_usage_mb': 0.0, 'gpu_mem_used_mb': 0.0,
                    'request_time_s': 0.0
                }
            }
    
    def run_benchmark(self, model_names: list[str], prompts: list[str]) -> list[dict]:
        """
        Runs the full benchmark process and returns the results.
        This method is now a generator to update the progress bar.
        """
        all_metrics = []
        total_steps = len(model_names) * len(prompts)
        step = 0
        
        progress_bar = st.progress(0, text=f"Benchmarking... 0%")
        status_text = st.empty()
        
        for model_name in model_names:
            for prompt in prompts:
                status_text.text(f"Querying {model_name} with prompt: '{prompt[:30]}...'")
                result = self._ask_llm(model_name, prompt)
                all_metrics.append(result)
                step += 1
                percent_complete = int(step / total_steps * 100)
                progress_bar.progress(percent_complete, text=f"Benchmarking... {percent_complete}%")
        
        progress_bar.progress(100, text="Benchmark Complete!")
        status_text.success("All queries finished!")
        
        return all_metrics


def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(layout="wide")
    st.title("Ollama Local LLM Benchmark App 🦙")

    # Initialize the benchmark runner and get available models
    benchmark_runner = OllamaLLMBenchmark()
    local_llms = benchmark_runner._get_local_llms()

    # Create the UI elements
    st.header("1. Configuration")
    if not local_llms:
        st.warning("No local Ollama models found. Please ensure Ollama is running and you have pulled at least one model (e.g., 'ollama pull llama2').")
        st.stop()

    selected_models = st.multiselect(
        "Select one or more Ollama models to benchmark:",
        options=local_llms,
        default=local_llms
    )

    prompts = st.text_area(
        "Enter one or more prompts to test (one per line):",
        value="What is the capital of France?\nExplain the concept of quantum entanglement.\nWrite a short poem about the sea."
    )
    prompts_list = [p.strip() for p in prompts.split('\n') if p.strip()]

    st.markdown("---")
    
    if st.button("Generate Performance Metrics", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model to benchmark.")
            return
        if not prompts_list:
            st.error("Please enter at least one prompt.")
            return

        with st.spinner("Starting benchmark..."):
            all_metrics = benchmark_runner.run_benchmark(selected_models, prompts_list)
            
        st.markdown("---")
        st.header("2. Benchmark Results")
        
        if all_metrics:
            # Prepare data for display
            results_df = pd.DataFrame([
                {'Model': res['model'], **res['metrics']}
                for res in all_metrics if 'metrics' in res
            ])
            
            # Group by model to get average metrics
            avg_metrics_df = results_df.groupby('Model').mean().reset_index()

            st.subheader("Average Metrics by Model")
            st.dataframe(avg_metrics_df, use_container_width=True)

            st.subheader("Graphical Representation")
            
            # Create a collapsible section for the charts
            with st.expander("View Charts", expanded=True):
                # Bar chart for Total Duration
                st.write("#### Average Total Duration")
                fig_total_duration = px.bar(
                    avg_metrics_df,
                    x='Model',
                    y='total_duration_s',
                    title='Average Total Request Duration per Model',
                    labels={'total_duration_s': 'Time (seconds)'},
                    color='Model'
                )
                st.plotly_chart(fig_total_duration, use_container_width=True)

                # Bar chart for GPU Memory Usage
                if PNVML_AVAILABLE:
                    st.write("#### Average GPU Memory Usage")
                    fig_gpu_mem = px.bar(
                        avg_metrics_df,
                        x='Model',
                        y='gpu_mem_used_mb',
                        title='Average GPU Memory Used per Model',
                        labels={'gpu_mem_used_mb': 'Memory (MB)'},
                        color='Model'
                    )
                    st.plotly_chart(fig_gpu_mem, use_container_width=True)

                # Bar chart for RAM Usage
                st.write("#### Average RAM Usage")
                fig_ram_mem = px.bar(
                    avg_metrics_df,
                    x='Model',
                    y='mem_usage_mb',
                    title='Average RAM Usage per Model',
                    labels={'mem_usage_mb': 'Memory (MB)'},
                    color='Model'
                )
                st.plotly_chart(fig_ram_mem, use_container_width=True)

if __name__ == "__main__":
    main()
