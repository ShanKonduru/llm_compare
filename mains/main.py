import time
import psutil
import pynvml
import ollama

# Initialize NVML for GPU metrics
pynvml.nvmlInit()

def get_gpu_memory_used():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / (1024 * 1024)  # MiB

def get_process_memory(pid):
    try:
        p = psutil.Process(pid)
        return p.memory_info().rss / (1024 * 1024)  # MB
    except psutil.NoSuchProcess:
        return None

def measure_response_with_metrics(prompt, model_name):
    # Instantiate client for the specific model
    client = ollama.Client(host='http://localhost:11434', model=model_name)
    current_pid = psutil.Process().pid
    mem_before = get_process_memory(current_pid)

    # Start timing
    start_time = time.perf_counter()

    # Generate stream of responses
    response_iter = client.generate(prompt)  # no stream param needed here

    response_text = ""
    total_tokens = 0

    for chunk in response_iter:
        # chunk properties depend on the SDK's streaming output
        # For example: chunk['text'], chunk['tokens']
        response_text += chunk.get('text', '')
        total_tokens += len(chunk.get('tokens', []))
        # Optionally: measure duration of chunks if available
    end_time = time.perf_counter()

    response_time = end_time - start_time
    mem_after = get_process_memory(current_pid)
    gpu_used = get_gpu_memory_used()

    mem_delta = (mem_after - mem_before) if mem_before is not None and mem_after is not None else None

    return {
        'response_text': response_text,
        'response_time': response_time,
        'token_count': total_tokens,
        'mem_usage_mb': mem_delta,
        'gpu_mem_used_mb': gpu_used
    }


import ollama

def ask_local_llm(model_name: str, question: str) -> str:
    """
    Sends a question to a local Ollama LLM and returns the response.

    Args:
        model_name: The name of the LLM model to use (e.g., 'llama3').
        question: The question to ask the model.

    Returns:
        The text response from the LLM.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': question,
                },
            ],
        )
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

import ollama
import time

def ask_local_llm_with_metrics(model_name: str, question: str) -> dict:
    """
    Sends a question to a local Ollama LLM, gets a response, and calculates metrics.

    Args:
        model_name: The name of the LLM model to use (e.g., 'llama3').
        question: The question to ask the model.

    Returns:
        A dictionary containing the response text and a metrics object.
    """
    start_time = time.time()
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': question,
                },
            ],
            stream=False  # Ensure the full response is returned at once
        )
        end_time = time.time()

        # The ollama response object contains metrics under 'eval_count', 'prompt_eval_count', etc.
        # These are usually returned in nanoseconds, so we convert them to seconds or milliseconds.
        response_time_ns = response.get('total_duration', 0)
        response_time_seconds = response_time_ns / 1_000_000_000 if response_time_ns > 0 else (end_time - start_time)

        metrics = {
            'response_text': response['message']['content'],
            'metrics': {
                'response_time': response_time_seconds,
                'token_count': response.get('prompt_eval_count', 0) + response.get('eval_count', 0),
                # Note: Direct memory and GPU metrics are not returned by the `ollama` Python library.
                # These typically require system-level monitoring tools or the Ollama server's
                # own internal logging/metrics endpoint (if available and enabled).
                # We'll use dummy values or report 0 for these for this function.
                'mem_usage_mb': 0.0,
                'gpu_mem_used_mb': 0.0,
            }
        }
        return metrics

    except Exception as e:
        return {
            'response_text': f"An error occurred: {e}",
            'metrics': {
                'response_time': time.time() - start_time,
                'token_count': 0,
                'mem_usage_mb': 0.0,
                'gpu_mem_used_mb': 0.0,
            }
        }

import ollama

def ask_local_llm_with_full_metrics(model_name: str, question: str) -> dict:
    """
    Sends a question to a local Ollama LLM and returns the response
    along with a detailed metrics object.

    Args:
        model_name: The name of the LLM model to use (e.g., 'llama3').
        question: The question to ask the model.

    Returns:
        A dictionary containing the response text and a metrics object.
        The metrics object includes durations and token counts.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': question,
                },
            ],
            stream=False  # Crucial to get the full response object with metrics
        )

        metrics = {
            'response_text': response['message']['content'],
            'metrics': {
                # Convert nanoseconds to seconds for readability
                'total_duration': response.get('total_duration', 0) / 1e9,
                'load_duration': response.get('load_duration', 0) / 1e9,
                'eval_duration': response.get('eval_duration', 0) / 1e9,
                'eval_count': response.get('eval_count', 0),
                # Memory and GPU metrics are not provided in this API response.
                'mem_usage_mb': 0.0,
                'gpu_mem_used_mb': 0.0,
            }
        }
        return metrics

    except Exception as e:
        return {
            'response_text': f"An error occurred: {e}",
            'metrics': {
                'total_duration': 0.0,
                'load_duration': 0.0,
                'eval_duration': 0.0,
                'eval_count': 0,
                'mem_usage_mb': 0.0,
                'gpu_mem_used_mb': 0.0,
            }
        }

# To install the library
# pip install nvidia-ml-py
import pynvml
import ollama
import time
import psutil

def get_ollama_gpu_mem_usage() -> float:
    """
    Finds the Ollama process and returns its GPU memory usage in MB.
    Returns 0.0 if not found or on error.
    """
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
        
    except pynvml.NVMLError:
        # Handle cases where the NVIDIA driver or GPU is not available
        pass
    finally:
        pynvml.nvmlShutdown()

    return 0.0

# Integrate this function into your main code
def ask_local_llm_with_full_metrics_gpu(model_name: str, question: str) -> dict:
    """
    Sends a question to a local Ollama LLM and returns the response
    along with a detailed metrics object, including both RAM and GPU memory usage.
    """
    # Find the Ollama process to get RAM usage
    ollama_process = None
    for proc in psutil.process_iter(['name', 'pid']):
        if 'ollama' in proc.info['name'].lower():
            ollama_process = proc
            break
    
    start_time = time.time()
    try:
        response = ollama.chat(
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
            }
        }
        return metrics

    except Exception as e:
        return {
            'response_text': f"An error occurred: {e}",
            'metrics': {
                'total_duration': 0.0, 'load_duration': 0.0, 'eval_duration': 0.0,
                'eval_count': 0, 'mem_usage_mb': 0.0, 'gpu_mem_used_mb': 0.0,
            }
        }
        
# Usage
if __name__ == "__main__":
    model_name = "llama2"  # or your selected model
    prompts = [
        "Why sky is blue",
        'Explain the concept of quantum entanglement.'
        'What is the capital of France?',
        'Write a short poem about the sea.',
        'Summarize the plot of "To Kill a Mockingbird".',
        'What are the benefits of regular exercise?',
        'How does photosynthesis work in plants?',
        'What is the significance of the theory of relativity?',
        'Describe the process of human digestion.',
        'What are the main causes of climate change?',
        'Explain the difference between mitosis and meiosis.'
        'List the first 100 prime numbers.'
    ]

    for prompt in prompts:
        # Option #1: Basic usage
        response = ask_local_llm(model_name, prompt)
        print(response)

        # Option #2: With basic metrics
        result = ask_local_llm_with_metrics(model_name, prompt)
        print("Response:\n", result['response_text'])
        print("\nMetrics:")
        print(f"Response Time: {result['metrics']['response_time']:.3f} seconds")
        print(f"Token Count: {result['metrics']['token_count']}")
        print(f"Memory used by process: {result['metrics']['mem_usage_mb']:.2f} MB")
        print(f"GPU Memory Used: {result['metrics']['gpu_mem_used_mb']:.2f} MB")

        # Option #3: With full metrics
        result = ask_local_llm_with_full_metrics(model_name, prompt)
        print("Response:\n", result['response_text'])
        print("\nMetrics:")
        print(f"Total Duration: {result['metrics']['total_duration']:.3f} seconds")
        print(f"Load Duration: {result['metrics']['load_duration']:.3f} seconds")
        print(f"Evaluation Duration: {result['metrics']['eval_duration']:.3f} seconds")
        print(f"Evaluation Token Count: {result['metrics']['eval_count']}")
        print(f"Memory used by process: {result['metrics']['mem_usage_mb']:.2f} MB")
        print(f"GPU Memory Used: {result['metrics']['gpu_mem_used_mb']:.2f} MB")

        # Option #4: With full metrics including GPU memory usage
        result = ask_local_llm_with_full_metrics_gpu(model_name, prompt)
        print("Response:\n", result['response_text'])
        print("\nMetrics:")
        print(f"Total Duration: {result['metrics']['total_duration']:.3f} seconds")
        print(f"Load Duration: {result['metrics']['load_duration']:.3f} seconds")
        print(f"Evaluation Duration: {result['metrics']['eval_duration']:.3f} seconds")
        print(f"Evaluation Token Count: {result['metrics']['eval_count']}")
        print(f"Memory used by process: {result['metrics']['mem_usage_mb']:.2f} MB")
        print(f"GPU Memory Used: {result['metrics']['gpu_mem_used_mb']:.2f} MB")

        # Option #5 not working due to lack of streaming metrics in ollama library
        # metrics = measure_response_with_metrics(prompt, model_name)
        # print("Response:\n", metrics['response_text'])
        # print("\nMetrics:")
        # print(f"Response Time: {metrics['response_time']:.3f} seconds")
        # print(f"Token Count: {metrics['token_count']}")
        # print(f"Memory used by process: {metrics['mem_usage_mb']:.2f} MB")
        # print(f"GPU Memory Used: {metrics['gpu_mem_used_mb']:.2f} MB")
