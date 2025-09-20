# To install the required libraries:
# pip install ollama pynvml psutil

import ollama
import time
import psutil
import sys

# It's good practice to wrap pynvml imports in a try-except block
# as it's specific to NVIDIA GPUs and might not be available.
try:
    import pynvml
    PNVML_AVAILABLE = True
except ImportError:
    PNVML_AVAILABLE = False
except Exception as e:
    print(f"Warning: pynvml could not be imported. GPU metrics will not be collected. Error: {e}")
    PNVML_AVAILABLE = False


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
        print(f"pynvml error: {e}")
        return 0.0
    finally:
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
            print(f"Error initializing Ollama client. Is Ollama running? Error: {e}")
            sys.exit(1)

    def _get_local_llms(self) -> list[str]:
        """
        Gets a list of locally available Ollama model names.
        """
        try:
            models = self.client.list()['models']
            return [model['model'] for model in models]
        except Exception as e:
            print(f"Error getting list of local models. Error: {e}")
            return []

    def _ask_llm(self, model_name: str, question: str) -> dict:
        """
        Sends a question to a local Ollama LLM and returns the response
        along with a detailed metrics object.
        """
        print(f"  - Querying '{model_name}' with prompt: '{question[:40]}...'")
        
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
            
            gpu_mem_used_mb = get_ollama_gpu_mem_usage()

            metrics = {
                'response_text': response['message']['content'],
                'metrics': {
                    'total_duration': response.get('total_duration', 0) / 1e9,
                    'load_duration': response.get('load_duration', 0) / 1e9,
                    'eval_duration': response.get('eval_duration', 0) / 1e9,
                    'eval_count': response.get('eval_count', 0),
                    'mem_usage_mb': mem_usage_mb,
                    'gpu_mem_used_mb': gpu_mem_used_mb,
                    'request_time': end_time - start_time,
                }
            }
            return metrics

        except Exception as e:
            return {
                'response_text': f"An error occurred: {e}",
                'metrics': {
                    'total_duration': 0.0, 'load_duration': 0.0, 'eval_duration': 0.0,
                    'eval_count': 0, 'mem_usage_mb': 0.0, 'gpu_mem_used_mb': 0.0,
                    'request_time': 0.0
                }
            }

    def _calculate_average_metrics(self, all_metrics: list) -> dict:
        """
        Calculates the average of all collected metrics.
        """
        if not all_metrics:
            return {}

        avg_metrics = {
            'total_duration': 0.0,
            'load_duration': 0.0,
            'eval_duration': 0.0,
            'eval_count': 0,
            'mem_usage_mb': 0.0,
            'gpu_mem_used_mb': 0.0,
            'request_time': 0.0,
        }
        
        num_runs = len(all_metrics)
        for result in all_metrics:
            if 'metrics' in result:
                metrics = result['metrics']
                for key in avg_metrics.keys():
                    avg_metrics[key] += metrics.get(key, 0)
        
        for key in avg_metrics.keys():
            avg_metrics[key] /= num_runs
        
        return avg_metrics

    def run_benchmark(self, prompts: list[str]) -> None:
        """
        Runs the full benchmark process and prints the results.
        """
        model_names = self._get_local_llms()
        if not model_names:
            print("No local Ollama models found. Please pull a model (e.g., 'ollama pull llama2') and try again.")
            return

        all_metrics = []
        print(f"Starting benchmark for {len(model_names)} models and {len(prompts)} prompts...")
        print("-" * 50)
        
        for model_name in model_names:
            print(f"Benchmarking model: {model_name}")
            for prompt in prompts:
                result = self._ask_llm(model_name, prompt)
                all_metrics.append(result)
            print("-" * 50)

        # Calculate and display average metrics
        avg_metrics = self._calculate_average_metrics(all_metrics)
        
        if avg_metrics:
            print("\nBenchmark Complete - Average Metrics Across All Models & Prompts:")
            for key, value in avg_metrics.items():
                if 'duration' in key or 'time' in key:
                    print(f"  - {key.replace('_', ' ').title()}: {value:.2f} seconds")
                elif 'mb' in key:
                    print(f"  - {key.replace('_', ' ').title()}: {value:.2f} MB")
                else:
                    print(f"  - {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print("No metrics collected. Please check your Ollama installation and ensure models are running.")


# --- Main Execution Block ---
if __name__ == "__main__":
    prompts_to_test = [
        "Why is the sky blue?",
        'Explain the concept of quantum entanglement.',
        'What is the capital of France?',
        'Write a short poem about the sea.',
        'Summarize the plot of "To Kill a Mockingbird".',
        'What are the benefits of regular exercise?',
        'How does photosynthesis work in plants?',
        'What is the significance of the theory of relativity?',
        'Describe the process of human digestion.',
        'What are the main causes of climate change?',
        'Explain the difference between mitosis and meiosis.',
        'List the first 100 prime numbers.',
    ]

    benchmark_runner = OllamaLLMBenchmark()
    benchmark_runner.run_benchmark(prompts_to_test)
