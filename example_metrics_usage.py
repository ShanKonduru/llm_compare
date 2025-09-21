#!/usr/bin/env python3
"""
Example usage of the standalone OllamaMetrics class.
"""

import sys
import os
import logging

# Configure logging to see error messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from ollama_metrics import OllamaMetrics


def main():
    """Example usage of the standalone OllamaMetrics class."""
    
    print("=== OllamaMetrics Standalone Example ===")
    
    # Test GPU memory usage
    print("Checking Ollama GPU memory usage...")
    gpu_mem_usage = OllamaMetrics.get_ollama_gpu_mem_usage()
    
    if gpu_mem_usage > 0:
        print(f"Ollama GPU memory usage: {gpu_mem_usage:.2f} MB")
    else:
        print("No GPU memory usage detected (possible reasons:")
        print("  - Ollama is not running")
        print("  - No NVIDIA GPU available")
        print("  - pynvml is not installed")
        print("  - Ollama is not using GPU acceleration")
    
    print("\nExample completed!")


def advanced_example():
    """Advanced example showing how to use the metrics in a monitoring loop."""
    import time
    
    print("\n=== Advanced Monitoring Example ===")
    print("Monitoring Ollama GPU usage for 30 seconds...")
    print("(Start Ollama and run some queries to see usage)")
    
    for i in range(6):  # Monitor for 30 seconds (6 x 5-second intervals)
        gpu_usage = OllamaMetrics.get_ollama_gpu_mem_usage()
        timestamp = time.strftime("%H:%M:%S")
        
        if gpu_usage > 0:
            print(f"[{timestamp}] GPU Memory: {gpu_usage:.2f} MB")
        else:
            print(f"[{timestamp}] GPU Memory: No usage detected")
        
        if i < 5:  # Don't sleep on the last iteration
            time.sleep(5)
    
    print("Monitoring completed!")


if __name__ == "__main__":
    main()
    advanced_example()