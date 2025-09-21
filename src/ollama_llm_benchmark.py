# To install the required libraries:
# pip install ollama psutil
# Note: pynvml is handled in ollama_metrics.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ollama_metrics import OllamaMetrics


import ollama
import time
import psutil


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
            raise RuntimeError(f"Error initializing Ollama client. Is Ollama running? Error: {e}")

    def _get_local_llms(self) -> list[str]:
        """
        Gets a list of locally available Ollama model names.
        """
        try:
            models = self.client.list()["models"]
            return [model["model"] for model in models]
        except Exception as e:
            raise RuntimeError(f"Error getting list of local models. Error: {e}")

    def _ask_llm(self, model_name: str, question: str) -> dict:
        """
        Sends a question to a local Ollama LLM and returns the response
        along with a detailed metrics object.
        """
        # Find the Ollama process to get RAM usage
        ollama_process = None
        for proc in psutil.process_iter(["name", "pid"]):
            if "ollama" in proc.info["name"].lower():
                ollama_process = proc
                break

        start_time = time.time()
        try:
            response = self.client.chat(
                model=model_name,
                messages=[{"role": "user", "content": question}],
                stream=False,
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
                "model": model_name,
                "response_text": response["message"]["content"],
                "metrics": {
                    "total_duration_s": response.get("total_duration", 0) / 1e9,
                    "load_duration_s": response.get("load_duration", 0) / 1e9,
                    "eval_duration_s": response.get("eval_duration", 0) / 1e9,
                    "eval_count": response.get("eval_count", 0),
                    "mem_usage_mb": mem_usage_mb,
                    "gpu_mem_used_mb": gpu_mem_used_mb,
                    "request_time_s": end_time - start_time,
                },
            }
            return metrics

        except Exception as e:
            return {
                "model": model_name,
                "response_text": f"An error occurred: {e}",
                "metrics": {
                    "total_duration_s": 0.0,
                    "load_duration_s": 0.0,
                    "eval_duration_s": 0.0,
                    "eval_count": 0,
                    "mem_usage_mb": 0.0,
                    "gpu_mem_used_mb": 0.0,
                    "request_time_s": 0.0,
                },
            }

    def run_benchmark(self, model_names: list[str], prompts: list[str], 
                     progress_callback=None, status_callback=None) -> list[dict]:
        """
        Runs the full benchmark process and returns the results.
        
        Args:
            model_names: List of model names to benchmark
            prompts: List of prompts to test with each model
            progress_callback: Optional callback function that receives (current_step, total_steps, percentage)
            status_callback: Optional callback function that receives status messages
        """
        all_metrics = []
        total_steps = len(model_names) * len(prompts)
        step = 0

        for model_name in model_names:
            for prompt in prompts:
                if status_callback:
                    status_callback(f"Querying {model_name} with prompt: '{prompt[:30]}...'")
                else:
                    print(f"Querying {model_name} with prompt: '{prompt[:30]}...'")
                
                result = self._ask_llm(model_name, prompt)
                all_metrics.append(result)
                step += 1
                percent_complete = int(step / total_steps * 100)
                
                if progress_callback:
                    progress_callback(step, total_steps, percent_complete)
                else:
                    print(f"Progress: {percent_complete}% ({step}/{total_steps})")

        if status_callback:
            status_callback("All queries finished!")
        else:
            print("Benchmark Complete! All queries finished!")

        return all_metrics
