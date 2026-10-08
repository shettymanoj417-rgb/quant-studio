import os
import json

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


# ============================================================
# MODEL DETECTOR
# ============================================================

def detect_model(model_path):

    filename = os.path.basename(model_path)
    extension = os.path.splitext(filename)[1].lower()

    size_bytes = os.path.getsize(model_path)
    size_mb = round(
        size_bytes / (1024 * 1024),
        2
    )

    model_type = "Unknown"
    architecture = "Unknown"
    tensor_count = 0
    detection_method = "filename"


    # ========================================================
    # HELPER
    # ========================================================

    def detect_from_text(text):

        text = str(text).lower()

        # ----------------------------------------------------
        # Transformer models
        # ----------------------------------------------------

        if "distilbert" in text:
            return (
                "DistilBERT",
                "Transformer / DistilBERT"
            )

        if "roberta" in text:
            return (
                "RoBERTa",
                "Transformer / RoBERTa"
            )

        if "albert" in text:
            return (
                "ALBERT",
                "Transformer / ALBERT"
            )

        if "deberta" in text:
            return (
                "DeBERTa",
                "Transformer / DeBERTa"
            )

        if "electra" in text:
            return (
                "ELECTRA",
                "Transformer / ELECTRA"
            )

        if "bert" in text:
            return (
                "BERT",
                "Transformer / BERT"
            )

        if "mistral" in text:
            return (
                "Mistral",
                "Transformer / Mistral"
            )

        if "llama" in text:
            return (
                "LLaMA",
                "Transformer / LLaMA"
            )

        if (
            "model.layers" in text
            or "self_attn.q_proj" in text
            or "q_proj" in text
        ):
            return (
                "LLaMA-compatible",
                "Transformer / LLaMA-compatible"
            )

        if "gemma" in text:
            return (
                "Gemma",
                "Transformer / Gemma"
            )

        if "phi" in text:
            return (
                "Phi",
                "Transformer / Phi"
            )

        if "qwen" in text:
            return (
                "Qwen",
                "Transformer / Qwen"
            )

        if "t5" in text:
            return (
                "T5",
                "Transformer / T5"
            )

        if "bart" in text:
            return (
                "BART",
                "Transformer / BART"
            )

        if (
            "transformer.h" in text
            or "attn.c_attn" in text
            or "gpt" in text
        ):
            return (
                "GPT-compatible",
                "Transformer / GPT"
            )

        # ----------------------------------------------------
        # Computer vision models
        # ----------------------------------------------------

        if "mobilenet" in text:
            return (
                "MobileNet",
                "CNN / MobileNet"
            )

        if "resnet" in text:
            return (
                "ResNet",
                "CNN / ResNet"
            )

        if "efficientnet" in text:
            return (
                "EfficientNet",
                "CNN / EfficientNet"
            )

        if "convnext" in text:
            return (
                "ConvNeXt",
                "CNN / ConvNeXt"
            )

        if (
            "vit" in text
            or "vision_transformer" in text
        ):
            return (
                "Vision Transformer",
                "Vision Transformer / ViT"
            )

        # ----------------------------------------------------
        # Generic architecture detection
        # ----------------------------------------------------

        if (
            "attention" in text
            or "attn" in text
            or "layer_norm" in text
            or "layernorm" in text
        ):
            return (
                "Transformer",
                "Transformer / Generic"
            )

        if (
            "conv" in text
            or "batchnorm" in text
            or "bn" in text
        ):
            return (
                "CNN",
                "Convolutional Neural Network"
            )

        return (
            "Unknown",
            "Unknown architecture"
        )


    # ========================================================
    # SAFETENSORS
    # ========================================================

    if extension == ".safetensors":

        if safe_open is None:

            return {
                "status": "error",
                "filename": filename,
                "extension": extension,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "error": (
                    "safetensors is not installed. "
                    "Run: pip install safetensors"
                )
            }

        try:

            with safe_open(
                model_path,
                framework="pt",
                device="cpu"
            ) as f:

                keys = list(f.keys())

                tensor_count = len(keys)

                # --------------------------------------------
                # Tensor names
                # --------------------------------------------

                key_text = " ".join(
                    keys
                ).lower()

                model_type, architecture = detect_from_text(
                    key_text
                )

                detection_method = "tensor names"

                # --------------------------------------------
                # SafeTensors metadata
                # --------------------------------------------

                metadata = f.metadata() or {}

                metadata_text = " ".join(
                    f"{key} {value}"
                    for key, value in metadata.items()
                ).lower()

                if metadata_text:

                    metadata_type, metadata_arch = (
                        detect_from_text(
                            metadata_text
                        )
                    )

                    if metadata_type != "Unknown":

                        model_type = metadata_type
                        architecture = metadata_arch
                        detection_method = (
                            "SafeTensors metadata"
                        )

        except Exception as error:

            return {
                "status": "error",
                "filename": filename,
                "extension": extension,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "error": (
                    "Could not inspect SafeTensors file: "
                    + str(error)
                )
            }


    # ========================================================
    # PYTORCH
    # ========================================================

    elif extension in [
        ".pt",
        ".pth",
        ".bin"
    ]:

        model_type = "PyTorch model"
        architecture = "PyTorch / Unknown architecture"

        try:

            import torch

            try:

                checkpoint = torch.load(
                    model_path,
                    map_location="cpu",
                    weights_only=True
                )

            except TypeError:

                checkpoint = torch.load(
                    model_path,
                    map_location="cpu"
                )

            # ------------------------------------------------
            # State dictionary
            # ------------------------------------------------

            if isinstance(checkpoint, dict):

                if "state_dict" in checkpoint:

                    checkpoint = checkpoint[
                        "state_dict"
                    ]

                elif "model_state_dict" in checkpoint:

                    checkpoint = checkpoint[
                        "model_state_dict"
                    ]

                keys = list(
                    checkpoint.keys()
                )

                tensor_count = len(keys)

                key_text = " ".join(
                    str(key)
                    for key in keys
                ).lower()

                model_type, architecture = (
                    detect_from_text(
                        key_text
                    )
                )

                detection_method = (
                    "PyTorch tensor names"
                )

        except Exception as error:

            return {
                "status": "error",
                "filename": filename,
                "extension": extension,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "error": (
                    "Could not inspect PyTorch model: "
                    + str(error)
                )
            }


    # ========================================================
    # JSON CONFIGURATION
    # ========================================================

    elif extension == ".json":

        model_type = "Model configuration"
        architecture = "Configuration file"

        try:

            with open(
                model_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            model_type_value = str(
                data.get(
                    "model_type",
                    ""
                )
            )

            architecture_value = str(
                data.get(
                    "architectures",
                    ""
                )
            )

            model_name = str(
                data.get(
                    "_name_or_path",
                    ""
                )
            )

            combined = (
                model_type_value
                + " "
                + architecture_value
                + " "
                + model_name
            )

            detected_type, detected_architecture = (
                detect_from_text(
                    combined
                )
            )

            if detected_type != "Unknown":

                model_type = detected_type
                architecture = detected_architecture
                detection_method = (
                    "model configuration"
                )

        except Exception as error:

            return {
                "status": "error",
                "filename": filename,
                "extension": extension,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "error": (
                    "Could not read JSON configuration: "
                    + str(error)
                )
            }


    # ========================================================
    # ZIP MODEL
    # ========================================================

    elif extension == ".zip":

        model_type = "Model archive"
        architecture = "Compressed model package"
        detection_method = "file extension"


    # ========================================================
    # UNKNOWN FORMAT
    # ========================================================

    else:

        model_type = "Unknown model"
        architecture = "Unknown"
        detection_method = "unsupported format"


    # ========================================================
    # MODEL SUPPORT STATUS
    # ========================================================

    supported_models = [
        "DistilBERT",
        "BERT",
        "RoBERTa",
        "ALBERT",
        "DeBERTa",
        "ELECTRA",
        "Mistral",
        "LLaMA",
        "LLaMA-compatible",
        "Gemma",
        "Phi",
        "Qwen",
        "T5",
        "BART",
        "GPT-compatible",
        "MobileNet",
        "ResNet",
        "EfficientNet",
        "ConvNeXt",
        "Vision Transformer",
        "CNN",
        "Transformer"
    ]

    if model_type in supported_models:
        support_status = "Detected"
    elif model_type == "Unknown":
        support_status = "Unknown"
    else:
        support_status = "Generic"


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "status": "success",
        "filename": filename,
        "extension": extension,
        "model_type": model_type,
        "architecture": architecture,
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "tensor_count": tensor_count,
        "detection_method": detection_method,
        "support_status": support_status,
        "message": "Model detected successfully."
    }