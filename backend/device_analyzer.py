import os
import platform
import shutil


class DeviceAnalyzer:
    """
    Detects the resources available on the current device
    and provides a target-device compatibility foundation.
    """

    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.processor = platform.processor()

    # ==========================================================
    # ANALYZE CURRENT DEVICE
    # ==========================================================

    def analyze_current_device(self):

        memory_total = self._get_total_memory()
        storage_total = self._get_storage()

        cpu_cores = os.cpu_count() or 1

        gpu_info = self._get_gpu_info()

        return {
            "device_type": self._detect_device_type(),

            "operating_system": self.system,

            "architecture": self.machine,

            "processor": self.processor,

            "cpu_cores": cpu_cores,

            "ram": {
                "total_bytes": memory_total,
                "total_gb": self._bytes_to_gb(memory_total)
            },

            "storage": {
                "total_bytes": storage_total,
                "total_gb": self._bytes_to_gb(storage_total)
            },

            "gpu": gpu_info
        }

    # ==========================================================
    # TARGET DEVICE ANALYSIS
    # ==========================================================

    def analyze_target_device(
        self,
        ram_gb,
        storage_gb,
        cpu_architecture="unknown",
        gpu_type="none",
        runtime="onnxruntime"
    ):

        try:
            ram_gb = float(ram_gb)
            storage_gb = float(storage_gb)
        except (TypeError, ValueError):

            raise ValueError(
                "RAM and storage must be numeric values."
            )

        if ram_gb <= 0:
            raise ValueError(
                "RAM must be greater than zero."
            )

        if storage_gb <= 0:
            raise ValueError(
                "Storage must be greater than zero."
            )

        return {

            "ram_gb": ram_gb,

            "storage_gb": storage_gb,

            "cpu_architecture":
                str(cpu_architecture),

            "gpu_type":
                str(gpu_type),

            "runtime":
                str(runtime),

            "constraints": {

                "max_model_size_gb":
                    round(
                        storage_gb * 0.80,
                        2
                    ),

                "max_runtime_memory_gb":
                    round(
                        ram_gb * 0.70,
                        2
                    )
            }
        }

    # ==========================================================
    # DEVICE COMPATIBILITY
    # ==========================================================

    def check_model_compatibility(
        self,
        model_size_bytes,
        estimated_memory_gb,
        target_device
    ):

        model_size_gb = (
            model_size_bytes /
            (1024 ** 3)
        )

        available_storage = float(
            target_device["storage_gb"]
        )

        available_ram = float(
            target_device["ram_gb"]
        )

        max_model_size = (
            available_storage * 0.80
        )

        max_runtime_memory = (
            available_ram * 0.70
        )

        storage_pass = (
            model_size_gb <= max_model_size
        )

        memory_pass = (
            estimated_memory_gb
            <= max_runtime_memory
        )

        if storage_pass and memory_pass:

            status = "compatible"

        elif storage_pass or memory_pass:

            status = "warning"

        else:

            status = "not_compatible"

        score = self._calculate_score(
            storage_pass,
            memory_pass
        )

        return {

            "status": status,

            "score": score,

            "model_size_gb":
                round(model_size_gb, 3),

            "estimated_memory_gb":
                round(
                    float(estimated_memory_gb),
                    3
                ),

            "available_storage_gb":
                available_storage,

            "available_ram_gb":
                available_ram,

            "checks": {

                "storage": {
                    "passed": storage_pass,
                    "message":
                        (
                            "Model fits available storage."
                            if storage_pass
                            else
                            "Model exceeds safe storage limit."
                        )
                },

                "ram": {
                    "passed": memory_pass,
                    "message":
                        (
                            "Estimated runtime memory fits RAM."
                            if memory_pass
                            else
                            "Estimated runtime memory exceeds safe RAM."
                        )
                }
            }
        }

    # ==========================================================
    # COMPATIBILITY SCORE
    # ==========================================================

    def _calculate_score(
        self,
        storage_pass,
        memory_pass
    ):

        score = 0

        if storage_pass:
            score += 50

        if memory_pass:
            score += 50

        return score

    # ==========================================================
    # TOTAL MEMORY
    # ==========================================================

    def _get_total_memory(self):

        try:

            import psutil

            return int(
                psutil.virtual_memory().total
            )

        except ImportError:

            return 0

    # ==========================================================
    # STORAGE
    # ==========================================================

    def _get_storage(self):

        try:

            total, used, free = shutil.disk_usage(
                os.getcwd()
            )

            return int(total)

        except Exception:

            return 0

    # ==========================================================
    # GPU
    # ==========================================================

    def _get_gpu_info(self):

        gpu = {

            "available": False,

            "name": None,

            "memory_gb": None,

            "backend": None
        }

        # ------------------------------------------------------
        # NVIDIA / CUDA
        # ------------------------------------------------------

        try:

            import torch

            if torch.cuda.is_available():

                gpu["available"] = True

                gpu["backend"] = "CUDA"

                gpu["name"] = (
                    torch.cuda.get_device_name(0)
                )

                memory_bytes = (
                    torch.cuda.get_device_properties(0)
                    .total_memory
                )

                gpu["memory_gb"] = round(
                    memory_bytes /
                    (1024 ** 3),
                    2
                )

                return gpu

        except Exception:

            pass

        return gpu

    # ==========================================================
    # DEVICE TYPE
    # ==========================================================

    def _detect_device_type(self):

        machine = (
            self.machine or ""
        ).lower()

        if "aarch64" in machine:
            return "ARM64 Edge Device"

        if "arm64" in machine:
            return "ARM64 Device"

        if "arm" in machine:
            return "ARM Device"

        if "x86_64" in machine:
            return "x86_64 Computer"

        if "amd64" in machine:
            return "x86_64 Computer"

        return "Unknown Device"

    # ==========================================================
    # BYTES → GB
    # ==========================================================

    @staticmethod
    def _bytes_to_gb(value):

        if not value:
            return 0

        return round(
            value /
            (1024 ** 3),
            2
        )


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    analyzer = DeviceAnalyzer()

    print(
        "\n======================================"
    )

    print(
        "DEVICE ANALYZER"
    )

    print(
        "======================================"
    )

    device = (
        analyzer.analyze_current_device()
    )

    print(
        "\nDevice type:",
        device["device_type"]
    )

    print(
        "Operating system:",
        device["operating_system"]
    )

    print(
        "Architecture:",
        device["architecture"]
    )

    print(
        "CPU cores:",
        device["cpu_cores"]
    )

    print(
        "RAM:",
        device["ram"]["total_gb"],
        "GB"
    )

    print(
        "Storage:",
        device["storage"]["total_gb"],
        "GB"
    )

    print(
        "GPU:",
        device["gpu"]
    )

    print(
        "\nDevice analyzer working successfully."
    )