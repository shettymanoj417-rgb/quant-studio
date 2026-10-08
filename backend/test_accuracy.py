import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

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

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

correct = 0

print("\nTesting original model...\n")

with torch.no_grad():

    for text, label in zip(texts, labels):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        output = model(**inputs)

        prediction = torch.argmax(
            output.logits,
            dim=1
        ).item()

        print(
            "Text:",
            text
        )

        print(
            "Expected:",
            label,
            "Predicted:",
            prediction
        )

        if prediction == label:
            correct += 1

accuracy = (
    correct / len(labels)
) * 100

print("\n==============================")
print("ORIGINAL MODEL ACCURACY")
print("==============================")

print(
    "Correct:",
    correct,
    "/",
    len(labels)
)

print(
    "Accuracy:",
    round(accuracy, 2),
    "%"
)