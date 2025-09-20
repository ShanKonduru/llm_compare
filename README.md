# Ollama Local LLM Benchmark

## Description
This repository contains a Python script designed to benchmark the performance and resource consumption of large language models (LLMs) running locally via the Ollama server. The script provides a structured approach to evaluate multiple models against a set of prompts and then consolidates the results into average metrics.

### Internals and Architecture
The core functionality is encapsulated within the OllamaLLMBenchmark class, promoting code organization and reusability. The process is broken down into the following key steps:

Model Discovery: The benchmark begins by communicating with the local Ollama server to get a list of all available models. This is handled by the _get_local_llms method, which uses the ollama.Client().list() API.

Querying with Metrics: For each model and for every prompt in the predefined list, the _ask_llm method is invoked. This method is the heart of the metric collection. It uses the ollama.Client().chat function to query the LLM.

### Resource Measurement:

Duration: The time taken for each request is measured using Python's standard time library, specifically time.time(), to capture the request_time. The Ollama response itself provides more detailed timings, such as total_duration, load_duration, and eval_duration, which are also recorded.

RAM Usage: The psutil library is used to find the running Ollama process. Once found, psutil.Process.memory_info().rss is called to capture the current RAM usage (Resident Set Size).

GPU Usage: The get_ollama_gpu_mem_usage function uses pynvml, a Python wrapper for NVIDIA's NVML library. This allows the script to directly query the NVIDIA GPU driver for memory usage by the Ollama process. A try-except block gracefully handles environments without an NVIDIA GPU or the necessary library.

Aggregation: After running all the queries, the _calculate_average_metrics method processes the collected data. It iterates through all the individual results and computes the average value for each metric (e.g., average total_duration, average mem_usage_mb), providing a concise summary of the models' performance.

### Uses and Applications
This benchmark script is a valuable tool for anyone working with local LLMs, particularly for:

Model Selection: Objectively compare the performance of different LLMs on your specific hardware.

Resource Monitoring: Understand the system resources (RAM, GPU) required by different models and query loads.

Performance Optimization: Identify bottlenecks in the LLM pipeline, such as slow model loading or evaluation, to help inform decisions about hardware upgrades or model quantization.

Reproducible Testing: Run a standardized test suite to ensure consistent performance over time, which is crucial for development and integration.

### Imported Libraries
The script relies on the following Python libraries to function:

ollama: The official Python library for interacting with a local Ollama server. It provides functions to list models and send prompts.

time: A standard Python library used for measuring the time elapsed during LLM queries to calculate performance metrics.

psutil: A cross-platform library for retrieving system and process information. It is used here to get real-time memory usage of the Ollama process.

pynvml: A Python wrapper for the NVIDIA Management Library (NVML), which is essential for collecting detailed GPU-specific metrics such as used VRAM.

sys: A standard Python library used to handle system-specific parameters and functions, primarily used here for clean script termination in case of a critical error.

## 

While running this code run the following command to see actual mem usage
```dos
for /l %g in () do @(ollama ps & timeout /t 2)
```

## Installation


1.  **Initialize git (Windows):**
    Run the `000_init.bat` file.

2.  **Create a virtual environment (Windows):**
    Run the `001_env.bat` file.

3.  **Activate the virtual environment (Windows):**
    Run the `002_activate.bat` file.

4.  **Install dependencies:**
    Run the `003_setup.bat` file. This will install all the packages listed in `requirements.txt`.

5.  **Deactivate the virtual environment (Windows):**
    Run the `005_deactivate.bat` file.

## Usage

1.  **Run the main application (Windows):**
    Run the `004_run.bat` file.

    [Provide instructions on how to use your application.]

## Batch Files (Windows)

This project includes the following batch files to help with common development tasks on Windows:

* `000_init.bat`: Initialized git and also usn and pwd config setup also done.
* `001_env.bat`: Creates a virtual environment named `venv`.
* `002_activate.bat`: Activates the `venv` virtual environment.
* `003_setup.bat`: Installs the Python packages listed in `requirements.txt` using `pip`.
* `004_run.bat`: Executes the main Python script (`main.py`).
* `005_run_test.bat`: Executes the pytest  scripts (`test_main.py`).
* `008_deactivate.bat`: Deactivates the currently active virtual environment.

## Contributing

[Explain how others can contribute to your project.]

## License

[Specify the project license, if any.]
