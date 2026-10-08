import os


def find_safetensors_files(directory):
    files = []
    for root, directories, filenames in os.walk(directory):
        directories.sort()
        files.extend(
            os.path.join(root, filename)
            for filename in sorted(filenames)
            if filename.lower().endswith(".safetensors")
        )
    return files


class QuantizationConfig:
    """Configuration for model quantization."""

    def __init__(self, quantization_type="int8", device_map="auto"):
        self.quantization_type = quantization_type
        self.device_map = device_map


class ModelQuantizer:
    """
    Model quantizer supporting:
    - SafeTensors INT8
    - ONNX INT8
    """

    def __init__(self, config=None):
        self.config = config or QuantizationConfig()

    # ============================================================
    # MAIN QUANTIZATION FUNCTION
    # ============================================================

    def quantize(self, input_path, output_path):

        if not (os.path.isfile(input_path) or os.path.isdir(input_path)):
            raise ValueError(f"Input model not found: {input_path}")

        quantization_type = self.config.quantization_type.lower()

        if quantization_type != "int8":
            raise ValueError(f"Currently supported quantization: INT8")

        if os.path.isdir(input_path):
            candidates = find_safetensors_files(input_path)
            if not candidates:
                raise ValueError(
                    f"No SafeTensors model files were found in: {input_path}"
                )
            extension = ".safetensors"
        else:
            extension = os.path.splitext(input_path)[1].lower()

        # --------------------------------------------------------
        # SAFETENSORS
        # --------------------------------------------------------

        if extension == ".safetensors":
            return self._quantize_safetensors(input_path, output_path)

        # --------------------------------------------------------
        # ONNX
        # --------------------------------------------------------

        if extension == ".onnx":
            return self._quantize_onnx(input_path, output_path)

        # --------------------------------------------------------
        # PYTORCH
        # --------------------------------------------------------

        if extension in [".pt", ".pth", ".bin"]:
            raise ValueError(
                "PyTorch .pt/.pth/.bin files require their "
                "model architecture before generic INT8 "
                "quantization can be performed. "
                "Use a .safetensors or .onnx model for "
                "automatic quantization."
            )

        # --------------------------------------------------------
        # OTHER
        # --------------------------------------------------------

        raise ValueError(
            f"Unsupported model format: {extension}. "
            f"Supported formats for automatic INT8 "
            f"quantization are .safetensors and .onnx."
        )

    # ============================================================
    # SAFETENSORS INT8
    # ============================================================

    def _quantize_safetensors(self, input_path, output_path):

        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file

        except ImportError as error:
            raise RuntimeError(
                "Missing SafeTensors dependencies. "
                "Run: pip install safetensors torch"
            ) from error

        tensors = {}
        quantized_count = 0
        skipped_count = 0
        metadata = {}

        files_to_quantize = []
        if os.path.isdir(input_path):
            files_to_quantize = find_safetensors_files(input_path)
        else:
            files_to_quantize = [input_path]

        if not files_to_quantize:
            raise ValueError(f"No SafeTensors files found in {input_path}")

        for safe_file in files_to_quantize:
            with safe_open(safe_file, framework="pt", device="cpu") as source:
                source_metadata = source.metadata() or {}
                if not metadata:
                    metadata = dict(source_metadata)
                for name in source.keys():
                    tensor = source.get_tensor(name)
                    if tensor.is_floating_point():
                        tensor = tensor.float()
                        maximum = float(tensor.abs().max().item())
                        if maximum == 0:
                            scale = 1.0
                        else:
                            scale = maximum / 127.0
                        quantized_tensor = (
                            torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
                        )
                        tensors[name] = quantized_tensor
                        tensors[f"{name}.__scale__"] = torch.tensor(
                            [scale], dtype=torch.float32
                        )
                        quantized_count += 1
                    else:
                        tensors[name] = tensor
                        skipped_count += 1

        metadata["quantization"] = "int8-per-tensor"
        metadata["source_format"] = "safetensors"

        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        save_file(tensors, output_path, metadata=metadata)

        return {
            "status": "success",
            "format": "safetensors",
            "quantization": "INT8",
            "model": os.path.basename(input_path),
            "output_model": os.path.basename(output_path),
            "quantized_tensors": quantized_count,
            "unchanged_tensors": skipped_count,
            "message": "SafeTensors model quantized successfully.",
        }

    # ============================================================
    # ONNX INT8
    # ============================================================

    def _quantize_onnx(self, input_path, output_path):

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is not installed. " "Run: pip install onnx onnxruntime"
            ) from error

        # --------------------------------------------------------
        # CREATE OUTPUT DIRECTORY
        # --------------------------------------------------------

        output_directory = os.path.dirname(output_path)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # --------------------------------------------------------
        # ONNX DYNAMIC INT8 QUANTIZATION
        # --------------------------------------------------------

        try:

            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
            )

        except Exception as error:

            raise RuntimeError(
                "ONNX INT8 quantization failed: " f"{str(error)}"
            ) from error

        # --------------------------------------------------------
        # CHECK OUTPUT
        # --------------------------------------------------------

        if not os.path.isfile(output_path):

            raise RuntimeError(
                "ONNX quantization completed " "but output model was not created."
            )

        # --------------------------------------------------------
        # RESULT
        # --------------------------------------------------------

        return {
            "status": "success",
            "format": "onnx",
            "quantization": "INT8",
            "model": os.path.basename(input_path),
            "output_model": os.path.basename(output_path),
            "message": ("ONNX model quantized " "successfully."),
        }
