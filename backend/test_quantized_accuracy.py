import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from safetensors import safe_open


# ============================================================
# PATHS
# ============================================================

ORIGINAL_MODEL = "accuracy_model"

QUANTIZED_MODEL = (
    "accuracy_model/model.int8.quantized.safetensors"
)


# ============================================================
# TEST DATA
# ============================================================

texts = [
    "I really enjoyed this movie.",
    "This movie was excellent.",
    "I hated this movie.",
    "The movie was terrible.",
    "The product is amazing.",
    "This was a very bad experience.",
    "I am very happy with the service.",
    "I would never recommend this product.",
    "The quality is excellent.",
    "This is disappointing."
]


labels = [
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    0
]


# ============================================================
# ORIGINAL MODEL ACCURACY
# ============================================================

print("\nLoading original model...")

tokenizer = AutoTokenizer.from_pretrained(
    ORIGINAL_MODEL
)

original_model = (
    AutoModelForSequenceClassification
    .from_pretrained(ORIGINAL_MODEL)
)

original_model.eval()


correct_original = 0


print("\n==============================")
print("ORIGINAL MODEL")
print("==============================")


with torch.no_grad():

    for text, label in zip(
        texts,
        labels
    ):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        output = original_model(
            **inputs
        )

        prediction = torch.argmax(
            output.logits,
            dim=1
        ).item()

        print(
            f"Expected: {label} | "
            f"Predicted: {prediction} | "
            f"{text}"
        )

        if prediction == label:
            correct_original += 1


original_accuracy = (
    correct_original /
    len(labels)
) * 100


print(
    "\nOriginal accuracy:",
    round(original_accuracy, 2),
    "%"
)


# ============================================================
# LOAD QUANTIZED TENSORS
# ============================================================

print("\nLoading quantized model...")

with safe_open(
    QUANTIZED_MODEL,
    framework="pt",
    device="cpu"
) as f:

    quantized = {
        key: f.get_tensor(key)
        for key in f.keys()
    }


# ============================================================
# DEQUANTIZE
# ============================================================

state_dict = {}


for name, tensor in quantized.items():

    # Ignore scale tensors
    if name.endswith(
        ".__scale__"
    ):
        continue


    # Remove distilbert. prefix
    if name.startswith(
        "distilbert."
    ):

        clean_name = name[
            len("distilbert."):]
    else:

        clean_name = name


    # Convert INT8 back to FP32
    if tensor.dtype == torch.int8:

        scale_name = (
            name +
            ".__scale__"
        )

        if scale_name in quantized:

            scale = quantized[
                scale_name
            ]

            tensor = (
                tensor.float() *
                scale
            )


    state_dict[
        clean_name
    ] = tensor


# ============================================================
# CREATE CLASSIFICATION MODEL
# ============================================================

quantized_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        ORIGINAL_MODEL
    )
)


# ============================================================
# LOAD QUANTIZED WEIGHTS
# ============================================================

missing, unexpected = (
    quantized_model.load_state_dict(
        state_dict,
        strict=False
    )
)


print(
    "\nMissing tensors:",
    len(missing)
)

print(
    "Unexpected tensors:",
    len(unexpected)
)


quantized_model.eval()


# ============================================================
# QUANTIZED MODEL ACCURACY
# ============================================================

correct_quantized = 0


print("\n==============================")
print("QUANTIZED MODEL")
print("==============================")


with torch.no_grad():

    for text, label in zip(
        texts,
        labels
    ):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        output = quantized_model(
            **inputs
        )

        prediction = torch.argmax(
            output.logits,
            dim=1
        ).item()

        print(
            f"Expected: {label} | "
            f"Predicted: {prediction} | "
            f"{text}"
        )

        if prediction == label:
            correct_quantized += 1


quantized_accuracy = (
    correct_quantized /
    len(labels)
) * 100


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("======================================")
print("       ACCURACY COMPARISON")
print("======================================")

print(
    "Original accuracy :",
    round(original_accuracy, 2),
    "%"
)

print(
    "Quantized accuracy:",
    round(quantized_accuracy, 2),
    "%"
)

print(
    "Accuracy difference:",
    round(
        quantized_accuracy -
        original_accuracy,
        2
    ),
    "%"
)

print("======================================")