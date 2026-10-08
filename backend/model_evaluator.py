import os
import time
import json
import csv
import io


# ==============================================================
# DEFAULT BENCHMARK DATASET (SST-2 Sentiment Classification)
# ==============================================================

DEFAULT_BENCHMARK_DATASET = [
    {"text": "I really enjoyed this movie.", "label": 1},
    {"text": "This movie was excellent.", "label": 1},
    {"text": "I hated this movie.", "label": 0},
    {"text": "The movie was terrible.", "label": 0},
    {"text": "The product is amazing.", "label": 1},
    {"text": "This was a very bad experience.", "label": 0},
    {"text": "I am very happy with the service.", "label": 1},
    {"text": "I would never recommend this product.", "label": 0},
    {"text": "The quality is excellent.", "label": 1},
    {"text": "This is disappointing.", "label": 0},
    {"text": "Outstanding performance and great value.", "label": 1},
    {"text": "Completely defective and stopped working immediately.", "label": 0},
    {"text": "Highly satisfied with the purchase and support.", "label": 1},
    {"text": "Worst customer service I have ever encountered.", "label": 0},
    {"text": "Super fast, reliable, and exceeded my expectations.", "label": 1},
    {"text": "Poor quality material, broke on the first use.", "label": 0},
    {"text": "The design is elegant, clean and works wonderfully.", "label": 1},
    {"text": "Frustrating experience and total waste of money.", "label": 0},
    {"text": "Impressive accuracy and outstanding results.", "label": 1},
    {"text": "Not worth the price, very subpar performance.", "label": 0}
]


def parse_dataset(dataset_input):
    """
    Parse an evaluation dataset from various formats:
    - List of dicts: [{'text': ..., 'label': ...}]
    - JSON string / JSON file
    - CSV string / CSV file
    - Line-separated TXT with tab/comma
    Returns (samples_list, source_name).
    """
    if not dataset_input:
        return DEFAULT_BENCHMARK_DATASET, "Built-in SST-2 Benchmark"

    # Already a list of dicts
    if isinstance(dataset_input, list):
        normalized = []
        for item in dataset_input:
            if isinstance(item, dict):
                text = item.get("text") or item.get("sentence") or item.get("input") or item.get("content")
                label = item.get("label") if "label" in item else item.get("target")
                if text is not None and label is not None:
                    try:
                        normalized.append({"text": str(text), "label": int(label)})
                    except (ValueError, TypeError):
                        normalized.append({"text": str(text), "label": label})
        if normalized:
            return normalized, f"Custom Dataset ({len(normalized)} samples)"
        return DEFAULT_BENCHMARK_DATASET, "Built-in SST-2 Benchmark"

    # If dataset_input is a filepath
    if isinstance(dataset_input, str) and os.path.isfile(dataset_input):
        filename = os.path.basename(dataset_input)
        try:
            with open(dataset_input, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            samples, _ = parse_dataset_content(content, filename)
            if samples:
                return samples, f"{filename} ({len(samples)} samples)"
        except Exception as err:
            print(f"Error reading dataset file {dataset_input}: {err}")
        return DEFAULT_BENCHMARK_DATASET, "Built-in SST-2 Benchmark"

    # If dataset_input is raw string or bytes
    if isinstance(dataset_input, (str, bytes)):
        if isinstance(dataset_input, bytes):
            dataset_input = dataset_input.decode("utf-8", errors="ignore")
        samples, src = parse_dataset_content(dataset_input, "Uploaded Dataset")
        if samples:
            return samples, f"{src} ({len(samples)} samples)"

    # Fallback
    return DEFAULT_BENCHMARK_DATASET, "Built-in SST-2 Benchmark"


def parse_dataset_content(content_str, default_name="Uploaded Dataset"):
    """Parse string content as JSON, CSV, or TXT."""
    content_str = content_str.strip()
    if not content_str:
        return [], default_name

    # Try JSON
    if content_str.startswith("[") or content_str.startswith("{"):
        try:
            data = json.loads(content_str)
            if isinstance(data, dict):
                data = data.get("samples") or data.get("data") or data.get("test") or []
            if isinstance(data, list):
                normalized = []
                for item in data:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("sentence") or item.get("input") or item.get("content")
                        label = item.get("label") if "label" in item else item.get("target")
                        if text is not None and label is not None:
                            try:
                                normalized.append({"text": str(text), "label": int(label)})
                            except (ValueError, TypeError):
                                normalized.append({"text": str(text), "label": label})
                if normalized:
                    return normalized, default_name
        except Exception:
            pass

    # Try CSV
    try:
        reader = csv.reader(io.StringIO(content_str))
        rows = list(reader)
        if len(rows) > 1:
            header = [h.strip().lower() for h in rows[0]]
            text_idx = -1
            label_idx = -1
            for i, col in enumerate(header):
                if col in ["text", "sentence", "input", "review", "content", "prompt"]:
                    text_idx = i
                elif col in ["label", "target", "class", "sentiment", "output"]:
                    label_idx = i

            # If header found
            if text_idx != -1 and label_idx != -1:
                normalized = []
                for row in rows[1:]:
                    if len(row) > max(text_idx, label_idx):
                        try:
                            normalized.append({
                                "text": str(row[text_idx]),
                                "label": int(row[label_idx])
                            })
                        except (ValueError, TypeError):
                            pass
                if normalized:
                    return normalized, default_name

            # If no clear header, assume first column text, second column label
            if len(rows[0]) >= 2:
                normalized = []
                start_row = 1 if not rows[0][1].strip().isdigit() else 0
                for row in rows[start_row:]:
                    if len(row) >= 2:
                        try:
                            normalized.append({
                                "text": str(row[0]),
                                "label": int(row[1])
                            })
                        except (ValueError, TypeError):
                            pass
                if normalized:
                    return normalized, default_name
    except Exception:
        pass

    return [], default_name


class ModelEvaluator:
    """
    Evaluation engine for models before and after quantization.

    Supports:
    - SafeTensors Sequence Classification & Transformers models
    - ONNX models
    - Built-in benchmark datasets & custom uploaded datasets
    """

    def __init__(self, device="cpu"):
        self.device = device
        self._cached_models = {}

    # ==========================================================
    # RESOURCE RESOLUTION HELPER
    # ==========================================================

    def _find_model_resources(self, model_path):
        """
        Locates the config.json, tokenizer, and architecture
        associated with a model path.
        """
        model_dir = os.path.dirname(os.path.abspath(model_path))
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(backend_dir)

        # 1. Look in the same directory as the model
        if os.path.isfile(os.path.join(model_dir, "config.json")):
            return model_dir

        # 2. Look in backend/accuracy_model
        acc_dir_backend = os.path.join(backend_dir, "accuracy_model")
        if os.path.isfile(os.path.join(acc_dir_backend, "config.json")):
            return acc_dir_backend

        # 3. Look in project root accuracy_model
        acc_dir_proj = os.path.join(project_dir, "backend", "accuracy_model")
        if os.path.isfile(os.path.join(acc_dir_proj, "config.json")):
            return acc_dir_proj

        acc_dir_root = os.path.join(project_dir, "accuracy_model")
        if os.path.isfile(os.path.join(acc_dir_root, "config.json")):
            return acc_dir_root

        # 4. Fallback to standard HuggingFace Hub name
        return "distilbert-base-uncased-finetuned-sst-2-english"

    # ==========================================================
    # MAIN EVALUATION FUNCTION (Single Model)
    # ==========================================================

    def evaluate(
        self,
        model_path,
        dataset=None,
        model_type=None
    ):
        if not os.path.isfile(model_path):
            raise ValueError(
                f"Model not found: {model_path}"
            )

        # Normalize dataset or use benchmark
        samples, dataset_name = parse_dataset(dataset)

        extension = os.path.splitext(
            model_path
        )[1].lower()

        # ------------------------------------------------------
        # ONNX
        # ------------------------------------------------------
        if extension == ".onnx":
            return self._evaluate_onnx(
                model_path,
                samples
            )

        # ------------------------------------------------------
        # SAFETENSORS
        # ------------------------------------------------------
        if extension == ".safetensors":
            return self._evaluate_safetensors(
                model_path,
                samples,
                model_type
            )

        # ------------------------------------------------------
        # UNSUPPORTED
        # ------------------------------------------------------
        return {
            "status": "success",
            "performance": {
                "measured": False,
                "latency_ms": None
            },
            "accuracy": {
                "measured": False,
                "value": None,
                "reason": (
                    f"No evaluation engine is "
                    f"available for {extension} models."
                )
            }
        }

    # ==========================================================
    # COMPARISON EVALUATION (Before vs After Quantization)
    # ==========================================================

    def evaluate_comparison(
        self,
        original_path,
        quantized_path,
        dataset=None,
        model_type=None
    ):
        """
        Evaluates both original and quantized models on the same dataset
        and produces a direct accuracy and performance comparison.
        """
        if not os.path.isfile(original_path):
            raise ValueError(f"Original model not found: {original_path}")
        if not os.path.isfile(quantized_path):
            raise ValueError(f"Quantized model not found: {quantized_path}")

        samples, dataset_name = parse_dataset(dataset)

        extension = os.path.splitext(original_path)[1].lower()

        # ------------------------------------------------------
        # SafeTensors Comparison
        # ------------------------------------------------------
        if extension == ".safetensors":
            return self._compare_safetensors(
                original_path,
                quantized_path,
                samples,
                dataset_name,
                model_type
            )

        # ------------------------------------------------------
        # ONNX Comparison
        # ------------------------------------------------------
        if extension == ".onnx":
            return self._compare_onnx(
                original_path,
                quantized_path,
                samples,
                dataset_name
            )

        return {
            "measured": False,
            "before": "Not measured",
            "after": "Not measured",
            "difference": None,
            "reason": f"Evaluation not supported for format: {extension}"
        }

    # ==========================================================
    # SAFETENSORS COMPARISON
    # ==========================================================

    def _compare_safetensors(
        self,
        original_path,
        quantized_path,
        samples,
        dataset_name,
        model_type=None
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from safetensors import safe_open
        except ImportError as err:
            return {
                "measured": False,
                "before": "Not measured",
                "after": "Not measured",
                "difference": None,
                "reason": f"Missing PyTorch or Transformers: {err}"
            }

        target_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        resource_dir = self._find_model_resources(original_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(resource_dir)
        except Exception as err:
            return {
                "measured": False,
                "before": "Not measured",
                "after": "Not measured",
                "difference": None,
                "reason": f"Could not load tokenizer for evaluation: {err}"
            }

        # 1. Load Original Model
        try:
            original_model = AutoModelForSequenceClassification.from_pretrained(resource_dir)
            with safe_open(original_path, framework="pt", device="cpu") as f:
                orig_tensors = {k: f.get_tensor(k) for k in f.keys()}
            state_orig = {}
            for name, tensor in orig_tensors.items():
                clean = name[len("distilbert."):] if name.startswith("distilbert.") else name
                state_orig[clean] = tensor
            original_model.load_state_dict(state_orig, strict=False)
            original_model.to(target_device)
            original_model.eval()
        except Exception as err:
            return {
                "measured": False,
                "before": "Not measured",
                "after": "Not measured",
                "difference": None,
                "reason": f"Could not load original SafeTensors model: {err}"
            }

        # 2. Load Quantized Model
        try:
            quantized_model = AutoModelForSequenceClassification.from_pretrained(resource_dir)
            with safe_open(quantized_path, framework="pt", device="cpu") as f:
                quant_tensors = {k: f.get_tensor(k) for k in f.keys()}
            state_quant = {}
            for name, tensor in quant_tensors.items():
                if name.endswith(".__scale__"):
                    continue
                clean = name[len("distilbert."):] if name.startswith("distilbert.") else name
                if tensor.dtype == torch.int8:
                    scale_name = name + ".__scale__"
                    if scale_name in quant_tensors:
                        tensor = tensor.float() * quant_tensors[scale_name]
                state_quant[clean] = tensor
            quantized_model.load_state_dict(state_quant, strict=False)
            quantized_model.to(target_device)
            quantized_model.eval()
        except Exception as err:
            return {
                "measured": False,
                "before": "Not measured",
                "after": "Not measured",
                "difference": None,
                "reason": f"Could not load quantized SafeTensors model: {err}"
            }

        # 3. Evaluate both on samples
        correct_orig = 0
        correct_quant = 0
        matching_preds = 0
        total = len(samples)
        sample_details = []

        with torch.no_grad():
            for sample in samples:
                text = sample.get("text", "")
                expected = sample.get("label")

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

                out_orig = original_model(**inputs)
                out_quant = quantized_model(**inputs)

                pred_orig = torch.argmax(out_orig.logits, dim=-1).item()
                pred_quant = torch.argmax(out_quant.logits, dim=-1).item()

                is_orig_correct = (pred_orig == expected)
                is_quant_correct = (pred_quant == expected)

                if is_orig_correct:
                    correct_orig += 1
                if is_quant_correct:
                    correct_quant += 1
                if pred_orig == pred_quant:
                    matching_preds += 1

                if len(sample_details) < 10:
                    sample_details.append({
                        "text": text,
                        "expected": expected,
                        "predicted_before": pred_orig,
                        "predicted_after": pred_quant,
                        "match": (pred_orig == pred_quant)
                    })

        acc_orig = round((correct_orig / total) * 100, 2)
        acc_quant = round((correct_quant / total) * 100, 2)
        diff = round(acc_quant - acc_orig, 2)
        agreement = round((matching_preds / total) * 100, 2)

        return {
            "measured": True,
            "before": acc_orig,
            "after": acc_quant,
            "difference": diff,
            "unit": "%",
            "total_samples": total,
            "correct_before": correct_orig,
            "correct_after": correct_quant,
            "agreement": agreement,
            "dataset_source": dataset_name,
            "details": sample_details,
            "reason": (
                f"Evaluated on {total} samples ({dataset_name}). "
                f"Prediction agreement: {agreement}%."
            )
        }

    # ==========================================================
    # SAFETENSORS EVALUATION (Single Model)
    # ==========================================================

    def _evaluate_safetensors(
        self,
        model_path,
        dataset=None,
        model_type=None
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "PyTorch and SafeTensors are required. "
                "Run: pip install torch safetensors transformers"
            ) from error

        target_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        resource_dir = self._find_model_resources(model_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(resource_dir)
            model = AutoModelForSequenceClassification.from_pretrained(resource_dir)

            with safe_open(model_path, framework="pt", device="cpu") as f:
                tensors = {k: f.get_tensor(k) for k in f.keys()}

            state_dict = {}
            for name, tensor in tensors.items():
                if name.endswith(".__scale__"):
                    continue
                clean = name[len("distilbert."):] if name.startswith("distilbert.") else name
                if tensor.dtype == torch.int8:
                    scale_name = name + ".__scale__"
                    if scale_name in tensors:
                        tensor = tensor.float() * tensors[scale_name]
                state_dict[clean] = tensor

            model.load_state_dict(state_dict, strict=False)
            model.to(target_device)
            model.eval()

            samples, dataset_name = parse_dataset(dataset)
            correct = 0
            latencies = []

            with torch.no_grad():
                for sample in samples:
                    text = sample.get("text", "")
                    expected = sample.get("label")
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    inputs = {k: v.to(target_device) for k, v in inputs.items()}

                    start = time.perf_counter()
                    out = model(**inputs)
                    end = time.perf_counter()

                    latencies.append((end - start) * 1000)
                    pred = torch.argmax(out.logits, dim=-1).item()
                    if pred == expected:
                        correct += 1

            total = len(samples)
            acc = round((correct / total) * 100, 2)
            avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

            return {
                "status": "success",
                "performance": {
                    "measured": avg_latency is not None,
                    "latency_ms": avg_latency
                },
                "accuracy": {
                    "measured": True,
                    "value": acc,
                    "total_samples": total,
                    "correct": correct,
                    "dataset_source": dataset_name,
                    "reason": f"Evaluated on {total} samples ({dataset_name})."
                },
                "model_info": {
                    "tensor_count": len(tensors),
                    "device": target_device
                }
            }

        except Exception as error:
            # Fallback to metadata-only if inference cannot run
            return {
                "status": "success",
                "performance": {
                    "measured": False,
                    "latency_ms": None
                },
                "accuracy": {
                    "measured": False,
                    "value": None,
                    "reason": f"SafeTensors evaluation error: {error}"
                },
                "error": str(error)
            }

    # ==========================================================
    # ONNX COMPARISON
    # ==========================================================

    def _compare_onnx(
        self,
        original_path,
        quantized_path,
        samples,
        dataset_name
    ):
        try:
            orig_eval = self._evaluate_onnx(original_path, samples)
            quant_eval = self._evaluate_onnx(quantized_path, samples)

            acc_orig = orig_eval.get("accuracy", {}).get("value")
            acc_quant = quant_eval.get("accuracy", {}).get("value")

            measured = (acc_orig is not None and acc_quant is not None)
            diff = round(acc_quant - acc_orig, 2) if measured else None

            return {
                "measured": measured,
                "before": acc_orig if acc_orig is not None else "Not measured",
                "after": acc_quant if acc_quant is not None else "Not measured",
                "difference": diff,
                "unit": "%",
                "total_samples": len(samples),
                "dataset_source": dataset_name,
                "reason": (
                    f"Evaluated on {len(samples)} samples ({dataset_name})."
                    if measured else "ONNX accuracy evaluation incomplete."
                )
            }
        except Exception as error:
            return {
                "measured": False,
                "before": "Not measured",
                "after": "Not measured",
                "difference": None,
                "reason": f"ONNX comparison error: {error}"
            }

    # ==========================================================
    # ONNX EVALUATION (Single Model)
    # ==========================================================

    def _evaluate_onnx(
        self,
        model_path,
        dataset=None
    ):
        try:
            import onnxruntime as ort
            import numpy as np
        except ImportError as error:
            return {
                "status": "error",
                "performance": {"measured": False, "latency_ms": None},
                "accuracy": {
                    "measured": False,
                    "value": None,
                    "reason": "onnxruntime is not installed. Run: pip install onnxruntime"
                }
            }

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            session = ort.InferenceSession(model_path, providers=providers)
        except Exception as err:
            return {
                "status": "error",
                "performance": {"measured": False, "latency_ms": None},
                "accuracy": {"measured": False, "value": None, "reason": str(err)}
            }

        inputs_info = session.get_inputs()
        outputs_info = session.get_outputs()

        samples, dataset_name = parse_dataset(dataset)

        # Check if model takes token inputs (input_ids)
        input_names = [item.name for item in inputs_info]
        needs_tokenizer = any("input_ids" in name for name in input_names)

        tokenizer = None
        if needs_tokenizer:
            try:
                from transformers import AutoTokenizer
                resource_dir = self._find_model_resources(model_path)
                tokenizer = AutoTokenizer.from_pretrained(resource_dir)
            except Exception:
                pass

        latencies = []
        correct = 0
        total = 0

        for sample in samples:
            feed_dict = {}

            if tokenizer and isinstance(sample, dict) and "text" in sample:
                encoded = tokenizer(
                    sample["text"],
                    return_tensors="np",
                    padding=True,
                    truncation=True
                )
                for name in input_names:
                    if name in encoded:
                        feed_dict[name] = encoded[name]
                    elif "attention_mask" in name and "attention_mask" in encoded:
                        feed_dict[name] = encoded["attention_mask"]

            elif isinstance(sample, dict):
                for k, v in sample.items():
                    if k != "label" and k in input_names:
                        feed_dict[k] = np.asarray(v)

            if not feed_dict:
                continue

            try:
                start = time.perf_counter()
                result = session.run(None, feed_dict)
                end = time.perf_counter()

                latencies.append((end - start) * 1000)

                if result and "label" in sample:
                    prediction = np.asarray(result[0])
                    if prediction.ndim > 1:
                        predicted_label = int(np.argmax(prediction, axis=-1).reshape(-1)[0])
                    else:
                        predicted_label = int(np.argmax(prediction))

                    if predicted_label == int(sample["label"]):
                        correct += 1
                    total += 1
            except Exception:
                continue

        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None
        accuracy = round((correct / total) * 100, 2) if total > 0 else None

        return {
            "status": "success",
            "performance": {
                "measured": avg_latency is not None,
                "latency_ms": avg_latency
            },
            "accuracy": {
                "measured": accuracy is not None,
                "value": accuracy,
                "reason": (
                    f"Evaluated on {total} samples ({dataset_name})."
                    if accuracy is not None
                    else "Could not run ONNX accuracy with given inputs."
                )
            },
            "model_info": {
                "inputs": input_names,
                "outputs": [item.name for item in outputs_info],
                "providers": session.get_providers()
            }
        }


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print("ModelEvaluator loaded successfully.")