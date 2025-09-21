#!/usr/bin/env python3
"""
Example usage of the standalone OllamaLLMBenchmark class.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from ollama_llm_benchmark import OllamaLLMBenchmark


def main():
    """Example usage of the standalone OllamaLLMBenchmark class."""
    
    # Example 1: Basic usage with default print output
    try:
        benchmark = OllamaLLMBenchmark()
        
        # Get available models
        models = benchmark._get_local_llms()
        print(f"Available models: {models}")
        
        # Define test prompts
        prompts = [
            "What is Python?",
            "Explain machine learning in simple terms."
        ]
        
        # Run benchmark with first model (if available)
        if models:
            results = benchmark.run_benchmark([models[0]], prompts)
            
            # Print results
            for result in results:
                print(f"\nModel: {result['model']}")
                print(f"Response: {result['response_text'][:100]}...")
                print(f"Metrics: {result['metrics']}")
        else:
            print("No models available for benchmarking.")
            
    except RuntimeError as e:
        print(f"Error: {e}")
        return


def example_with_callbacks():
    """Example using custom progress and status callbacks."""
    
    def progress_callback(current, total, percentage):
        print(f"Custom Progress: [{current}/{total}] {percentage}%")
    
    def status_callback(message):
        print(f"Status: {message}")
    
    try:
        benchmark = OllamaLLMBenchmark()
        models = benchmark._get_local_llms()
        
        if models:
            prompts = ["Hello, how are you?"]
            results = benchmark.run_benchmark(
                [models[0]], 
                prompts,
                progress_callback=progress_callback,
                status_callback=status_callback
            )
            
            print(f"Benchmark completed with {len(results)} results.")
        else:
            print("No models available for benchmarking.")
            
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=== Basic Usage Example ===")
    main()
    
    print("\n=== Callback Usage Example ===")
    example_with_callbacks()