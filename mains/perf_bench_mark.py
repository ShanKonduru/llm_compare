# --- Main Execution Block ---
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ollama_llm_benchmark import OllamaLLMBenchmark

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
    benchmark_runner.run_benchmark(model_names=benchmark_runner._get_local_llms(), prompts=prompts_to_test)
