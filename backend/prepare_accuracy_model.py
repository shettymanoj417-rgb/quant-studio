from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print("Downloading classification model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model.save_pretrained(
    "accuracy_model"
)

tokenizer.save_pretrained(
    "accuracy_model"
)

print("Model saved successfully!")
print("Location: backend/accuracy_model")