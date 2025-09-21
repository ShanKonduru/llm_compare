# To install the required libraries:
# pip install pynvml psutil

import psutil
import sys
import logging

# It's good practice to wrap pynvml imports in a try-except block
# as it's specific to NVIDIA GPUs and might not be available.
try:
    import pynvml

    PNVML_AVAILABLE = True
except ImportError:
    PNVML_AVAILABLE = False
except Exception as e:
    print(
        f"Warning: pynvml could not be imported. GPU metrics will not be collected. Error: {e}"
    )
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
            for proc in psutil.process_iter(["name", "pid"]):
                if "ollama" in proc.info["name"].lower():
                    ollama_process_pid = proc.info["pid"]
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
            logging.error(f"pynvml error: {e}")
            return 0.0
        except Exception as e:
            logging.error(f"An unexpected error occurred in get_ollama_gpu_mem_usage: {e}")
            return 0.0
        finally:
            if PNVML_AVAILABLE:
                pynvml.nvmlShutdown()

        return 0.0
