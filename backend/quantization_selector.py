class QuantizationSelector:
    """
    Selects a quantization configuration based on the
    target device resources.

    This module does not perform quantization.
    It decides which precision should be preferred.
    """

    def __init__(self):
        self.precisions = {
            "fp16": {
                "storage_factor": 0.50,
                "memory_factor": 1.20,
                "priority": 3
            },

            "int8": {
                "storage_factor": 0.28,
                "memory_factor": 1.50,
                "priority": 2
            },

            "int4": {
                "storage_factor": 0.16,
                "memory_factor": 1.70,
                "priority": 1
            }
        }

    # ==========================================================
    # MAIN SELECTION FUNCTION
    # ==========================================================

    def select(
        self,
        model_size_bytes,
        target_ram_gb,
        target_storage_gb,
        cpu_architecture="unknown",
        gpu_type="none"
    ):

        if model_size_bytes <= 0:
            raise ValueError(
                "Model size must be greater than zero."
            )

        if target_ram_gb <= 0:
            raise ValueError(
                "Target RAM must be greater than zero."
            )

        if target_storage_gb <= 0:
            raise ValueError(
                "Target storage must be greater than zero."
            )

        model_size_gb = (
            model_size_bytes /
            (1024 ** 3)
        )

        # ------------------------------------------------------
        # SAFE RESOURCE LIMITS
        # ------------------------------------------------------

        safe_ram_gb = (
            target_ram_gb * 0.70
        )

        safe_storage_gb = (
            target_storage_gb * 0.80
        )

        results = {}

        # ------------------------------------------------------
        # TEST EACH PRECISION
        # ------------------------------------------------------

        for precision, settings in self.precisions.items():

            estimated_size_gb = (
                model_size_gb *
                settings["storage_factor"]
            )

            estimated_memory_gb = (
                estimated_size_gb *
                settings["memory_factor"]
            )

            storage_pass = (
                estimated_size_gb <=
                safe_storage_gb
            )

            memory_pass = (
                estimated_memory_gb <=
                safe_ram_gb
            )

            compatible = (
                storage_pass and
                memory_pass
            )

            results[precision] = {

                "estimated_model_size_gb":
                    round(
                        estimated_size_gb,
                        3
                    ),

                "estimated_memory_gb":
                    round(
                        estimated_memory_gb,
                        3
                    ),

                "storage_pass":
                    storage_pass,

                "memory_pass":
                    memory_pass,

                "compatible":
                    compatible
            }

        # ------------------------------------------------------
        # SELECT BEST PRECISION
        # ------------------------------------------------------

        compatible_precisions = [

            precision

            for precision, result
            in results.items()

            if result["compatible"]
        ]

        # Prefer the highest precision that fits.
        # FP16 > INT8 > INT4
        if "fp16" in compatible_precisions:

            recommended = "fp16"

            reason = (
                "FP16 fits the target device and "
                "provides the highest precision "
                "among the compatible configurations."
            )

        elif "int8" in compatible_precisions:

            recommended = "int8"

            reason = (
                "INT8 fits the target device and "
                "provides a good balance between "
                "model size and resource usage."
            )

        elif "int4" in compatible_precisions:

            recommended = "int4"

            reason = (
                "INT4 is recommended because the "
                "target device has limited resources "
                "and lower precision is required."
            )

        else:

            recommended = None

            reason = (
                "None of the supported quantization "
                "configurations is estimated to fit "
                "the target device."
            )

        # ------------------------------------------------------
        # DEVICE-SPECIFIC NOTES
        # ------------------------------------------------------

        notes = []

        architecture = (
            str(cpu_architecture)
            .lower()
        )

        gpu = (
            str(gpu_type)
            .lower()
        )

        if architecture in [
            "arm",
            "arm64"
        ]:

            notes.append(
                "ARM-based target detected. "
                "Verify that the selected runtime "
                "supports the model operators."
            )

        if gpu == "cuda":

            notes.append(
                "CUDA GPU selected. Actual GPU "
                "performance should be verified "
                "with a hardware benchmark."
            )

        if gpu == "none":

            notes.append(
                "CPU-only target selected."
            )

        # ------------------------------------------------------
        # RETURN RESULT
        # ------------------------------------------------------

        return {

            "recommended_quantization":
                recommended,

            "reason":
                reason,

            "model_size_gb":
                round(
                    model_size_gb,
                    3
                ),

            "target_device": {

                "ram_gb":
                    target_ram_gb,

                "storage_gb":
                    target_storage_gb,

                "cpu_architecture":
                    cpu_architecture,

                "gpu_type":
                    gpu_type
            },

            "safe_limits": {

                "ram_gb":
                    round(
                        safe_ram_gb,
                        3
                    ),

                "storage_gb":
                    round(
                        safe_storage_gb,
                        3
                    )
            },

            "configurations":
                results,

            "compatible_configurations":
                compatible_precisions,

            "notes":
                notes
        }


# ==============================================================
# SIMPLE TEST
# ==============================================================

if __name__ == "__main__":

    selector = QuantizationSelector()

    # Example:
    # 8 GB original model
    # Target device:
    # 4 GB RAM
    # 16 GB storage

    model_size_bytes = (
        8 *
        1024 *
        1024 *
        1024
    )

    result = selector.select(

        model_size_bytes=
            model_size_bytes,

        target_ram_gb=4,

        target_storage_gb=16,

        cpu_architecture="arm64",

        gpu_type="none"
    )

    print()
    print(
        "======================================"
    )
    print(
        "QUANTIZATION SELECTOR"
    )
    print(
        "======================================"
    )

    print(
        "\nOriginal model:",
        result["model_size_gb"],
        "GB"
    )

    print(
        "Target RAM:",
        result["target_device"]["ram_gb"],
        "GB"
    )

    print(
        "Target storage:",
        result["target_device"]["storage_gb"],
        "GB"
    )

    print(
        "\nConfigurations:"
    )

    for precision, data in (
        result["configurations"].items()
    ):

        print(
            precision.upper(),
            "->",
            "Model:",
            data["estimated_model_size_gb"],
            "GB,",
            "Memory:",
            data["estimated_memory_gb"],
            "GB,",
            "Compatible:",
            data["compatible"]
        )

    print(
        "\nRecommended:",
        result[
            "recommended_quantization"
        ]
    )

    print(
        "\nReason:",
        result["reason"]
    )

    print(
        "\nQuantization selector working successfully."
    )