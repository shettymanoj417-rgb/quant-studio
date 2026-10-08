import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel
from werkzeug.utils import secure_filename

# ============================================================
# PATH CONFIGURATION
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

sys.path.append(BACKEND_DIR)
sys.path.append(PROJECT_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model_detector import detect_model
from model_evaluator import ModelEvaluator
from model_quantizer import ModelQuantizer, QuantizationConfig, find_safetensors_files

from config.config import Config

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../frontend",
    static_url_path="/static",
)

app.config.from_object(Config)


# ============================================================
# CORS
# ============================================================

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ["http://127.0.0.1:8000", "http://localhost:8000", "*"],
            "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": False,
        }
    },
)


# ============================================================
# REQUIRED DIRECTORIES
# ============================================================

UPLOAD_DIR = os.path.join(PROJECT_DIR, app.config["UPLOAD_FOLDER"])

MODELS_DIR = os.path.join(PROJECT_DIR, app.config["MODELS_FOLDER"])

os.makedirs(UPLOAD_DIR, exist_ok=True)

os.makedirs(MODELS_DIR, exist_ok=True)


MODEL_FILE_EXTENSIONS = {
    ".safetensors",
    ".onnx",
    ".pt",
    ".pth",
    ".bin",
    ".model",
}


def get_safe_zip_member_name(name):
    normalized_name = name.replace("\\", "/")
    parts = normalized_name.split("/")

    if (
        normalized_name.startswith("/")
        or any(part in ("", "..") for part in parts if part != "")
        or ":" in parts[0]
    ):
        raise ValueError("The uploaded ZIP contains an unsafe file path.")

    return normalized_name


def extract_model_from_zip(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        candidates = []

        for info in archive.infolist():
            if info.is_dir():
                continue

            member_name = get_safe_zip_member_name(info.filename)
            member_extension = os.path.splitext(member_name)[1].lower()
            if member_extension in MODEL_FILE_EXTENSIONS:
                candidates.append((info, member_name))

        if not candidates:
            raise ValueError(
                "The ZIP must contain a supported model file, such as "
                "model.safetensors or model.onnx."
            )

        preferred = [
            candidate
            for candidate in candidates
            if os.path.basename(candidate[1]).lower()
            in ("model.safetensors", "model.onnx")
        ]
        if len(preferred) == 1:
            model_info, member_name = preferred[0]
            extension = os.path.splitext(member_name)[1].lower().lstrip(".")
            descriptor, model_path = tempfile.mkstemp(
                suffix=f".{extension}", dir=UPLOAD_DIR
            )
            try:
                with os.fdopen(descriptor, "wb") as destination:
                    with archive.open(model_info) as source:
                        shutil.copyfileobj(source, destination)
            except Exception:
                os.remove(model_path)
                raise
            return model_path, member_name, extension

        if len(candidates) == 1:
            model_info, member_name = candidates[0]
            extension = os.path.splitext(member_name)[1].lower().lstrip(".")
            descriptor, model_path = tempfile.mkstemp(
                suffix=f".{extension}", dir=UPLOAD_DIR
            )
            try:
                with os.fdopen(descriptor, "wb") as destination:
                    with archive.open(model_info) as source:
                        shutil.copyfileobj(source, destination)
            except Exception:
                os.remove(model_path)
                raise
            return model_path, member_name, extension

        shard_candidates = []
        for info, member_name in candidates:
            basename = os.path.basename(member_name)
            if basename.lower().endswith(".safetensors") and "-of-" in basename:
                shard_candidates.append((info, member_name))

        if shard_candidates:
            extracted_dir = tempfile.mkdtemp(prefix="sharded_model_", dir=UPLOAD_DIR)
            primary_member = shard_candidates[0][1]
            for info, member_name in shard_candidates:
                destination_path = os.path.join(
                    extracted_dir, os.path.basename(member_name)
                )
                with open(destination_path, "wb") as destination:
                    with archive.open(info) as source:
                        shutil.copyfileobj(source, destination)
            return extracted_dir, primary_member, "safetensors"

        raise ValueError(
            "The ZIP contains multiple model files. Include exactly one "
            "supported model file in the bundle."
        )


def create_download_bundle(
    source_path,
    model_path,
    quantized_path,
    bundle_path,
    model_member=None,
    bundled_source=False,
):
    with zipfile.ZipFile(bundle_path, "w") as bundle:
        if bundled_source:
            if os.path.isdir(source_path):
                quantized_name = "model" + os.path.splitext(quantized_path)[1].lower()
                for root, directories, filenames in os.walk(source_path):
                    directories[:] = [
                        directory for directory in directories if directory != ".cache"
                    ]
                    directories.sort()
                    for filename in sorted(filenames):
                        file_path = os.path.join(root, filename)
                        member_name = get_safe_zip_member_name(
                            os.path.relpath(file_path, source_path)
                        )
                        extension = os.path.splitext(filename)[1].lower()
                        if extension in {".safetensors", ".onnx"}:
                            continue

                        if member_name.endswith(".safetensors.index.json"):
                            with open(file_path, encoding="utf-8") as index_file:
                                index = json.load(index_file)
                            weight_map = index.get("weight_map")
                            if isinstance(weight_map, dict):
                                index["weight_map"] = {
                                    tensor_name: quantized_name
                                    for tensor_name in weight_map
                                }
                                bundle.writestr(
                                    member_name, json.dumps(index, indent=2)
                                )
                                continue

                        bundle.write(file_path, member_name)

                bundle.write(quantized_path, quantized_name)
            else:
                with zipfile.ZipFile(source_path) as source:
                    if model_member is not None:
                        skip_names = {model_member}
                    else:
                        skip_names = set()

                    for info in source.infolist():
                        if info.is_dir():
                            continue

                        member_name = get_safe_zip_member_name(info.filename)
                        if member_name in skip_names:
                            continue

                        copied_info = zipfile.ZipInfo(member_name, info.date_time)
                        copied_info.compress_type = info.compress_type
                        copied_info.external_attr = info.external_attr
                        with source.open(info) as source_file:
                            with bundle.open(copied_info, "w") as destination:
                                shutil.copyfileobj(source_file, destination)

                if model_member is not None:
                    with open(quantized_path, "rb") as quantized_model:
                        with bundle.open(model_member, "w") as destination:
                            shutil.copyfileobj(quantized_model, destination)
                else:
                    archive_name = "model" + os.path.splitext(quantized_path)[1].lower()
                    bundle.write(quantized_path, archive_name)
        else:
            archive_name = "model" + os.path.splitext(model_path)[1].lower()
            bundle.write(quantized_path, archive_name)


# ============================================================
# DISTILBERT PERFORMANCE FUNCTIONS
# ============================================================

DISTILBERT_NAME = "distilbert-base-uncased"

_tokenizer = None
_original_model = None


def get_original_distilbert():

    global _tokenizer
    global _original_model

    if _tokenizer is None:

        _tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_NAME)

    if _original_model is None:

        _original_model = DistilBertModel.from_pretrained(DISTILBERT_NAME)

        _original_model.eval()

    return _tokenizer, _original_model


def get_test_inputs():

    tokenizer, _ = get_original_distilbert()

    text = "This is a test of the AI model."

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    return inputs


def measure_original_inference():

    _, model = get_original_distilbert()

    inputs = get_test_inputs()

    # Warm-up
    with torch.no_grad():
        model(**inputs)

    times = []

    with torch.no_grad():

        for _ in range(5):

            start = time.perf_counter()

            model(**inputs)

            end = time.perf_counter()

            times.append((end - start) * 1000)

    return sum(times) / len(times)


def load_quantized_distilbert(model_path):

    with safe_open(model_path, framework="pt", device="cpu") as file:

        quantized = {key: file.get_tensor(key) for key in file.keys()}

    state_dict = {}

    for name, tensor in quantized.items():

        # Skip scale tensors
        if name.endswith(".__scale__"):
            continue

        # Remove distilbert prefix
        if name.startswith("distilbert."):

            clean_name = name[len("distilbert.") :]
        else:

            clean_name = name

        # Dequantize INT8 tensor
        if tensor.dtype == torch.int8:

            scale_name = name + ".__scale__"

            if scale_name in quantized:

                scale = quantized[scale_name]

                tensor = tensor.float() * scale

        state_dict[clean_name] = tensor

    config = DistilBertConfig()

    model = DistilBertModel(config)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.eval()

    return model, missing, unexpected


def measure_quantized_distilbert(model_path):

    model, missing, unexpected = load_quantized_distilbert(model_path)

    inputs = get_test_inputs()

    # Warm-up
    with torch.no_grad():
        model(**inputs)

    times = []

    with torch.no_grad():

        for _ in range(5):

            start = time.perf_counter()

            model(**inputs)

            end = time.perf_counter()

            times.append((end - start) * 1000)

    average_time = sum(times) / len(times)

    return (average_time, missing, unexpected)


def measure_distilbert_performance(original_path, quantized_path):

    try:

        before_time = measure_original_inference()

        after_time, missing, unexpected = measure_quantized_distilbert(quantized_path)

        result = {
            "before": round(before_time, 2),
            "after": round(after_time, 2),
            "unit": "ms",
            "missing_tensors": len(missing),
            "unexpected_tensors": len(unexpected),
        }

        if before_time > 0:

            result["change_percent"] = round(
                ((after_time - before_time) / before_time) * 100, 2
            )

        return result

    except Exception as error:

        print("DistilBERT performance measurement failed:", error)

        return {
            "before": "Not measured",
            "after": "Not measured",
            "unit": "ms",
            "error": str(error),
        }


# ============================================================
# ONNX PERFORMANCE FUNCTIONS
# ============================================================


def create_onnx_inputs(session):
    """
    Create safe dummy inputs based on the ONNX graph.

    This allows us to benchmark many ONNX models without
    assuming they are DistilBERT.
    """

    import numpy as np

    inputs = {}

    for input_info in session.get_inputs():

        name = input_info.name
        input_type = input_info.type
        shape = input_info.shape

        actual_shape = []

        for dimension in shape:

            if isinstance(dimension, int) and dimension > 0:
                actual_shape.append(dimension)

            else:
                # Dynamic dimension
                actual_shape.append(1)

        if len(actual_shape) == 0:
            actual_shape = [1]

        # Integer inputs
        if "int64" in input_type:

            # Most NLP ONNX models use int64 token IDs
            inputs[name] = np.ones(actual_shape, dtype=np.int64)

        elif "int32" in input_type:

            inputs[name] = np.ones(actual_shape, dtype=np.int32)

        elif "float16" in input_type:

            inputs[name] = np.ones(actual_shape, dtype=np.float16)

        else:

            inputs[name] = np.ones(actual_shape, dtype=np.float32)

    return inputs


def measure_onnx_inference(model_path):

    try:

        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        inputs = create_onnx_inputs(session)

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        session.run(None, inputs)

        times = []

        # ----------------------------------------------------
        # Measure
        # ----------------------------------------------------

        for _ in range(5):

            start = time.perf_counter()

            session.run(None, inputs)

            end = time.perf_counter()

            times.append((end - start) * 1000)

        average_time = sum(times) / len(times)

        return {
            "time": round(average_time, 2),
            "unit": "ms",
            "inputs": len(session.get_inputs()),
            "outputs": len(session.get_outputs()),
        }

    except Exception as error:

        print("ONNX inference measurement failed:", error)

        return {"time": "Not measured", "unit": "ms", "error": str(error)}


def measure_onnx_performance(original_path, quantized_path):

    try:

        before = measure_onnx_inference(original_path)

        after = measure_onnx_inference(quantized_path)

        before_time = before.get("time")

        after_time = after.get("time")

        result = {
            "before": before_time,
            "after": after_time,
            "unit": "ms",
            "before_details": before,
            "after_details": after,
        }

        if (
            isinstance(before_time, (int, float))
            and isinstance(after_time, (int, float))
            and before_time > 0
        ):

            result["change_percent"] = round(
                ((after_time - before_time) / before_time) * 100, 2
            )

        else:

            result["change_percent"] = None

        return result

    except Exception as error:

        return {
            "before": "Not measured",
            "after": "Not measured",
            "unit": "ms",
            "error": str(error),
        }


# ============================================================
# GENERIC MODEL DETECTION FALLBACK
# ============================================================


def get_model_detection(model_path, extension):
    model_file_path = model_path
    if os.path.isdir(model_path):
        candidates = find_safetensors_files(model_path)
        if not candidates:
            model_file_path = model_path
        else:
            model_file_path = candidates[0]

    try:
        detection = detect_model(model_file_path)

    except Exception as error:
        detection = {
            "status": "error",
            "model_type": "Unknown",
            "architecture": "Unknown",
            "tensor_count": 0,
            "error": str(error),
        }

    if extension == "onnx":
        if detection.get("model_type") in [None, "", "Unknown"]:
            detection["model_type"] = "ONNX model"

        if detection.get("architecture") in [
            None,
            "",
            "Unknown",
            "Unknown architecture",
        ]:
            detection["architecture"] = "ONNX / Unknown architecture"

        detection["format"] = "ONNX"
    else:
        detection["format"] = extension

    return detection


# ============================================================
# FRONTEND
# ============================================================


@app.get("/")
def index():

    return send_from_directory(os.path.join(PROJECT_DIR, "frontend"), "index.html")


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/api/health")
def health_check():

    return jsonify({"status": "healthy"})


# ============================================================
# DEVICE INFORMATION
# ============================================================


@app.get("/api/device-info")
def device_info():

    cuda_available = False

    try:

        cuda_available = torch.cuda.is_available()

    except Exception:
        pass

    memory_total = None

    try:

        import psutil

        memory_total = psutil.virtual_memory().total

    except Exception:
        pass

    return jsonify({"cuda_available": cuda_available, "memory_total": memory_total})


# ============================================================
# MODEL UPLOAD API
# ============================================================


@app.post("/api/upload-model")
def upload_model():

    model_file = request.files.get("model")

    if model_file is None or not model_file.filename:

        return jsonify({"error": "A model file is required"}), 400

    filename = secure_filename(model_file.filename)

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in app.config["ALLOWED_EXTENSIONS"]:

        return jsonify({"error": f"Unsupported model file type: .{extension}"}), 400

    upload_path = os.path.join(PROJECT_DIR, app.config["UPLOAD_FOLDER"], filename)

    try:

        model_file.save(upload_path)

        file_size = os.path.getsize(upload_path)

        detection = get_model_detection(upload_path, extension)

        print(f"Model uploaded: {filename}")

        return jsonify(
            {
                "status": "success",
                "filename": filename,
                "extension": extension,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "detection": detection,
                "message": "Model uploaded successfully.",
            }
        )

    except Exception as error:

        return jsonify({"error": str(error)}), 500


# ============================================================
# MODEL DETECTION API
# ============================================================


@app.post("/api/detect-model")
def detect_uploaded_model():

    model_file = request.files.get("model")

    if model_file is None or not model_file.filename:

        return jsonify({"error": "A model file is required"}), 400

    filename = secure_filename(model_file.filename)

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in app.config["ALLOWED_EXTENSIONS"]:

        return jsonify({"error": f"Unsupported model file type: .{extension}"}), 400

    upload_path = os.path.join(PROJECT_DIR, app.config["UPLOAD_FOLDER"], filename)

    try:

        model_file.save(upload_path)

        result = get_model_detection(upload_path, extension)

        return jsonify(result)

    except Exception as error:

        return jsonify({"error": str(error)}), 500


def evaluate_quantized_model(
    original_path, quantized_path, extension, model_type, device_map, dataset=None
):
    """Run ModelEvaluator for original and quantized models to measure accuracy before and after."""

    try:
        evaluator = ModelEvaluator(
            device=device_map if device_map in ["cpu", "cuda"] else "cpu"
        )

        comparison = evaluator.evaluate_comparison(
            original_path=original_path,
            quantized_path=quantized_path,
            dataset=dataset,
            model_type=model_type,
        )

        return comparison

    except Exception as error:
        print("Model evaluation failed:", error)
        return {
            "measured": False,
            "before": "Not measured",
            "after": "Not measured",
            "difference": None,
            "reason": str(error),
            "details": [],
        }


# ============================================================
# QUANTIZATION API
# ============================================================


@app.post("/api/quantize")
def quantize_model():

    # --------------------------------------------------------
    # Get model source
    # --------------------------------------------------------

    model_source = request.form.get("model_source", "upload").lower()
    model_file = request.files.get("model")
    model_member = None
    hub_snapshot = None

    if model_source == "huggingface":
        repo_id = request.form.get("model_id", "").strip()
        if not repo_id:
            return (
                jsonify({"error": "A Hugging Face model repository ID is required"}),
                400,
            )

        upload_directory = os.path.join(PROJECT_DIR, app.config["UPLOAD_FOLDER"])
        os.makedirs(upload_directory, exist_ok=True)
        hub_snapshot = tempfile.mkdtemp(
            prefix="huggingface_model_", dir=upload_directory
        )
        try:
            hub_snapshot = snapshot_download(
                repo_id=repo_id,
                revision=request.form.get("revision") or None,
                token=app.config.get("HF_TOKEN"),
                local_dir=hub_snapshot,
                allow_patterns=[
                    "*.safetensors",
                    "**/*.safetensors",
                    "*.onnx",
                    "**/*.onnx",
                    "*.json",
                    "**/*.json",
                    "*.txt",
                    "**/*.txt",
                    "*.model",
                    "**/*.model",
                    "*.spm",
                    "**/*.spm",
                    "*.vocab",
                    "**/*.vocab",
                    "*.merges",
                    "**/*.merges",
                    "*.tiktoken",
                    "**/*.tiktoken",
                    "*.py",
                    "**/*.py",
                    "*.yaml",
                    "**/*.yaml",
                    "*.yml",
                    "**/*.yml",
                ],
            )
        except Exception as error:
            shutil.rmtree(hub_snapshot, ignore_errors=True)
            return (
                jsonify(
                    {
                        "error": f"Could not download Hugging Face model '{repo_id}': {error}"
                    }
                ),
                400,
            )

        safetensors_files = find_safetensors_files(hub_snapshot)
        onnx_files = []
        for root, _, filenames in os.walk(hub_snapshot):
            for entry in filenames:
                full_path = os.path.join(root, entry)
                if entry.lower().endswith(".onnx"):
                    onnx_files.append(full_path)

        if safetensors_files:
            extension = "safetensors"
            model_input_path = hub_snapshot
        elif onnx_files:
            extension = "onnx"
            preferred_onnx = next(
                (
                    path
                    for path in onnx_files
                    if os.path.basename(path).lower() == "model.onnx"
                ),
                sorted(onnx_files)[0],
            )
            model_input_path = preferred_onnx
        else:
            shutil.rmtree(hub_snapshot, ignore_errors=True)
            return (
                jsonify(
                    {
                        "error": "The Hugging Face repository has no supported .safetensors or .onnx model weights."
                    }
                ),
                400,
            )

        filename = secure_filename(repo_id.replace("/", "-")) or "huggingface-model"
        upload_path = hub_snapshot
        bundled_source = True
    elif model_source == "upload":
        if model_file is None or not model_file.filename:
            return jsonify({"error": "A model file is required"}), 400

        filename = secure_filename(model_file.filename)
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in app.config["ALLOWED_EXTENSIONS"]:
            return jsonify({"error": f"Unsupported model file type: .{extension}"}), 400

        upload_directory = os.path.join(PROJECT_DIR, app.config["UPLOAD_FOLDER"])
        os.makedirs(upload_directory, exist_ok=True)
        upload_path = os.path.join(upload_directory, filename)
        try:
            model_file.save(upload_path)
        except Exception as error:
            return (
                jsonify({"error": "Could not save uploaded model: " + str(error)}),
                500,
            )

        model_input_path = upload_path
        bundled_source = extension == "zip"
    else:
        return jsonify({"error": "Unsupported model source"}), 400

    if bundled_source:
        if model_source == "upload":
            try:
                model_input_path, model_member, extension = extract_model_from_zip(
                    upload_path
                )
            except (ValueError, zipfile.BadZipFile, OSError) as error:
                return jsonify({"error": str(error)}), 400

    # --------------------------------------------------------
    # Detect model
    # --------------------------------------------------------

    detection_path = model_input_path
    if os.path.isdir(model_input_path):
        shard_candidates = find_safetensors_files(model_input_path)
        if shard_candidates:
            detection_path = shard_candidates[0]

    detection = get_model_detection(detection_path, extension)

    print()
    print("======================================")
    print("MODEL INFORMATION")
    print("======================================")
    print("Filename:", filename)
    print("Format:", extension)
    print("Model type:", detection.get("model_type", "Unknown"))
    print("Architecture:", detection.get("architecture", "Unknown"))
    print("======================================")

    # --------------------------------------------------------
    # Quantization configuration
    # --------------------------------------------------------

    quantization_type = request.form.get(
        "quantization", app.config["DEFAULT_QUANTIZATION"]
    )

    device_map = request.form.get("device_map", app.config["DEVICE_MAP"])

    quantization_config = QuantizationConfig(
        quantization_type=quantization_type, device_map=device_map
    )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    base_name = os.path.splitext(filename)[0]

    if os.path.isdir(model_input_path):
        output_extension = ".safetensors"
    else:
        output_extension = os.path.splitext(model_input_path)[1]

    output_filename = (
        f"{base_name}." f"{quantization_type}." f"quantized" f"{output_extension}"
    )

    output_path = os.path.join(
        PROJECT_DIR, app.config["MODELS_FOLDER"], output_filename
    )

    # --------------------------------------------------------
    # Perform quantization
    # --------------------------------------------------------

    print()
    print("======================================")
    print("STARTING QUANTIZATION")
    print("======================================")
    print("Quantization:", quantization_type)
    print("Input:", filename)
    print("Output:", output_filename)
    print("======================================")

    try:

        result = ModelQuantizer(quantization_config).quantize(
            model_input_path, output_path
        )

    except Exception as error:

        print("Quantization failed:", error)

        if bundled_source and model_source == "huggingface":
            shutil.rmtree(hub_snapshot, ignore_errors=True)
        elif bundled_source:
            if os.path.isdir(model_input_path):
                shutil.rmtree(model_input_path)
            else:
                os.remove(model_input_path)

        return (
            jsonify(
                {
                    "error": str(error),
                    "model": filename,
                    "format": extension,
                    "detection": detection,
                }
            ),
            400,
        )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    if not os.path.isfile(output_path):

        if bundled_source and model_source == "huggingface":
            shutil.rmtree(hub_snapshot, ignore_errors=True)
        elif bundled_source:
            if os.path.isdir(model_input_path):
                shutil.rmtree(model_input_path)
            else:
                os.remove(model_input_path)

        return (
            jsonify(
                {
                    "error": "Quantization completed but "
                    "the output model was not created."
                }
            ),
            500,
        )

    # --------------------------------------------------------
    # Calculate sizes
    # --------------------------------------------------------

    if model_source == "huggingface":
        input_size = 0
        for root, directories, files in os.walk(hub_snapshot):
            directories[:] = [
                directory for directory in directories if directory != ".cache"
            ]
            for file_name in files:
                input_size += os.path.getsize(os.path.join(root, file_name))
    elif os.path.isdir(model_input_path):
        input_size = 0
        for root, _, files in os.walk(model_input_path):
            for file_name in files:
                input_size += os.path.getsize(os.path.join(root, file_name))
    else:
        input_size = os.path.getsize(model_input_path)

    output_size = os.path.getsize(output_path)

    if input_size > 0:

        reduction_percent = ((input_size - output_size) / input_size) * 100

    else:

        reduction_percent = 0

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    performance = {
        "before": "Not measured",
        "after": "Not measured",
        "unit": "ms",
        "change_percent": None,
    }

    # --------------------------------------------------------
    # SafeTensors DistilBERT
    # --------------------------------------------------------

    if extension == "safetensors" and detection.get("model_type") == "DistilBERT":

        print("\nMeasuring DistilBERT performance...")

        performance = measure_distilbert_performance(model_input_path, output_path)

    # --------------------------------------------------------
    # ONNX
    # --------------------------------------------------------

    elif extension == "onnx":

        print("\nMeasuring ONNX performance...")

        performance = measure_onnx_performance(model_input_path, output_path)

    # --------------------------------------------------------
    # Other model formats
    # --------------------------------------------------------

    else:

        performance = {
            "before": "Not measured",
            "after": "Not measured",
            "unit": "ms",
            "change_percent": None,
            "reason": "Architecture-specific benchmark "
            "is not configured for this model yet.",
        }

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Accuracy Measurement (Before vs After)
    # --------------------------------------------------------
    dataset_file = request.files.get("dataset")
    dataset_data = None
    if dataset_file and dataset_file.filename:
        try:
            dataset_data = dataset_file.read()
        except Exception as err:
            print("Failed to read dataset file:", err)
    elif request.form.get("dataset"):
        dataset_data = request.form.get("dataset")

    accuracy = evaluate_quantized_model(
        original_path=detection_path,
        quantized_path=output_path,
        extension=extension,
        model_type=detection.get("model_type", "Unknown"),
        device_map=device_map,
        dataset=dataset_data,
    )

    bundle_filename = f"{base_name}.{quantization_type}.quantized.zip"
    bundle_path = os.path.join(
        PROJECT_DIR, app.config["MODELS_FOLDER"], bundle_filename
    )

    try:
        create_download_bundle(
            source_path=upload_path,
            model_path=model_input_path,
            quantized_path=output_path,
            bundle_path=bundle_path,
            model_member=model_member,
            bundled_source=bundled_source,
        )
    except Exception as error:
        if bundled_source and model_source == "huggingface":
            shutil.rmtree(hub_snapshot, ignore_errors=True)
        elif bundled_source:
            if os.path.isdir(model_input_path):
                shutil.rmtree(model_input_path)
            else:
                os.remove(model_input_path)
        return jsonify({"error": f"Could not create model ZIP: {error}"}), 500

    if bundled_source and model_source == "huggingface":
        shutil.rmtree(hub_snapshot, ignore_errors=True)
    elif bundled_source:
        if os.path.isdir(model_input_path):
            shutil.rmtree(model_input_path)
        else:
            os.remove(model_input_path)

    # --------------------------------------------------------
    # Add result information
    # --------------------------------------------------------

    result["output_file"] = bundle_filename
    result["quantized_model_file"] = output_filename

    result["download_url"] = f"/api/download/{bundle_filename}"

    result["model_detection"] = {
        "model_type": detection.get("model_type", "Unknown"),
        "architecture": detection.get("architecture", "Unknown"),
        "format": extension,
        "tensor_count": detection.get("tensor_count", 0),
        "detection_method": detection.get("detection_method", "Unknown"),
    }

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    result["metrics"] = {
        "size": {
            "before_bytes": input_size,
            "after_bytes": output_size,
            "reduction_percent": round(reduction_percent, 2),
        },
        "performance": performance,
        "accuracy": accuracy,
        "inference_validation": {
            "missing_tensors": performance.get("missing_tensors", None),
            "unexpected_tensors": performance.get("unexpected_tensors", None),
        },
    }

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("======================================")
    print("QUANTIZATION COMPLETE")
    print("======================================")
    print("Model:", filename)
    print("Format:", extension)
    print("Original:", round(input_size / (1024 * 1024), 2), "MB")
    print("Quantized:", round(output_size / (1024 * 1024), 2), "MB")
    print("Reduction:", round(reduction_percent, 2), "%")
    print("Performance before:", performance.get("before", "Not measured"))
    print("Performance after:", performance.get("after", "Not measured"))
    print("Output:", output_filename)
    print("======================================")

    return jsonify(result)


# ============================================================
# DOWNLOAD QUANTIZED MODEL
# ============================================================


@app.get("/api/download/<path:filename>")
def download_model(filename):

    safe_filename = secure_filename(filename)

    models_directory = os.path.join(PROJECT_DIR, app.config["MODELS_FOLDER"])

    model_path = os.path.join(models_directory, safe_filename)

    if not os.path.isfile(model_path):

        return jsonify({"error": "Quantized model not found"}), 404

    return send_file(model_path, as_attachment=True, download_name=safe_filename)


# ============================================================
# EVALUATE ACCURACY API (STANDALONE)
# ============================================================


@app.post("/api/evaluate-accuracy")
def evaluate_accuracy_route():
    """
    Endpoint to evaluate and compare accuracy between an original model
    and a quantized model on a benchmark or custom dataset.
    """
    try:
        # Check files or existing model names
        orig_file = request.files.get("original_model")
        quant_file = request.files.get("quantized_model")
        dataset_file = request.files.get("dataset")

        orig_filename = request.form.get("original_model_filename")
        quant_filename = request.form.get("quantized_model_filename")

        orig_path = None
        quant_path = None

        if orig_file and orig_file.filename:
            safe_orig = secure_filename(orig_file.filename)
            orig_path = os.path.join(
                PROJECT_DIR, app.config["UPLOAD_FOLDER"], safe_orig
            )
            orig_file.save(orig_path)
        elif orig_filename:
            for folder in [
                app.config["UPLOAD_FOLDER"],
                app.config["MODELS_FOLDER"],
                "backend/accuracy_model",
            ]:
                candidate = os.path.join(PROJECT_DIR, folder, orig_filename)
                if os.path.isfile(candidate):
                    orig_path = candidate
                    break

        if quant_file and quant_file.filename:
            safe_quant = secure_filename(quant_file.filename)
            quant_path = os.path.join(
                PROJECT_DIR, app.config["MODELS_FOLDER"], safe_quant
            )
            quant_file.save(quant_path)
        elif quant_filename:
            for folder in [
                app.config["MODELS_FOLDER"],
                app.config["UPLOAD_FOLDER"],
                "backend/accuracy_model",
            ]:
                candidate = os.path.join(PROJECT_DIR, folder, quant_filename)
                if os.path.isfile(candidate):
                    quant_path = candidate
                    break

        # Defaults if still not found: check accuracy_model directory
        if not orig_path:
            candidate = os.path.join(
                PROJECT_DIR, "backend", "accuracy_model", "model.safetensors"
            )
            if os.path.isfile(candidate):
                orig_path = candidate
        if not quant_path:
            candidate = os.path.join(
                PROJECT_DIR,
                "backend",
                "accuracy_model",
                "model.int8.quantized.safetensors",
            )
            if os.path.isfile(candidate):
                quant_path = candidate

        if not orig_path or not os.path.isfile(orig_path):
            return (
                jsonify(
                    {
                        "error": "Original model not found. Provide original_model file or filename."
                    }
                ),
                400,
            )
        if not quant_path or not os.path.isfile(quant_path):
            return (
                jsonify(
                    {
                        "error": "Quantized model not found. Provide quantized_model file or filename."
                    }
                ),
                400,
            )

        dataset_data = None
        if dataset_file and dataset_file.filename:
            dataset_data = dataset_file.read()
        elif request.form.get("dataset"):
            dataset_data = request.form.get("dataset")

        evaluator = ModelEvaluator(device=request.form.get("device_map", "cpu"))
        comparison = evaluator.evaluate_comparison(
            original_path=orig_path, quantized_path=quant_path, dataset=dataset_data
        )

        return jsonify(
            {
                "status": "success",
                "original_model": os.path.basename(orig_path),
                "quantized_model": os.path.basename(quant_path),
                "accuracy_comparison": comparison,
            }
        )

    except Exception as error:
        return jsonify({"error": str(error)}), 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
