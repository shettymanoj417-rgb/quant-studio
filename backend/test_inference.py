import os
import time
import statistics


def measure_file_loading(model_path, runs=5):
    """
    Generic performance measurement.

    Measures how long the model file takes to be read from disk.
    This works for different model formats without assuming
    a particular model architecture.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    timings = []

    for _ in range(runs):
        start = time.perf_counter()

        with open(model_path, "rb") as f:
            while f.read(1024 * 1024):
                pass

        end = time.perf_counter()

        timings.append((end - start) * 1000)

    average_ms = statistics.mean(timings)

    return round(average_ms, 2)


def measure_model_performance(model_path):
    """
    Generic benchmark for any model file.

    Returns the average file loading time in milliseconds.
    """

    try:
        performance_ms = measure_file_loading(model_path)

        return {
            "status": "success",
            "performance_ms": performance_ms,
            "message": "Performance measured successfully."
        }

    except Exception as error:

        return {
            "status": "error",
            "performance_ms": None,
            "message": str(error)
        }