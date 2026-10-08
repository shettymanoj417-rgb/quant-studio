import time
import torch
from transformers import AutoTokenizer, DistilBertModel

MODEL_NAME = "distilbert-base-uncased"

print("Loading original model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertModel.from_pretrained(MODEL_NAME)

model.eval()

text = "This is a test of the original AI model."

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

# Warm-up
with torch.no_grad():
    model(**inputs)

# Measure inference
times = []

for i in range(10):
    start = time.perf_counter()

    with torch.no_grad():
        output = model(**inputs)

    end = time.perf_counter()
    times.append((end - start) * 1000)

average_time = sum(times) / len(times)

print("\n===== ORIGINAL MODEL =====")
print("Inference successful!")
print("Output shape:", output.last_hidden_state.shape)
print("Average inference time:", round(average_time, 2), "ms")
print("==========================")