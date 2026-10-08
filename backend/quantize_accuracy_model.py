import os

from model_quantizer import ModelQuantizer, QuantizationConfig


INPUT_MODEL = os.path.join(
    "accuracy_model",
    "model.safetensors"
)

OUTPUT_MODEL = os.path.join(
    "accuracy_model",
    "model.int8.quantized.safetensors"
)


print("Loading classification model...")
print("Input:", INPUT_MODEL)


config = QuantizationConfig(
    quantization_type="int8",
    device_map="cpu"
)


print("Starting INT8 quantization...")


quantizer = ModelQuantizer(config)

result = quantizer.quantize(
    INPUT_MODEL,
    OUTPUT_MODEL
)


print("\n==============================")
print("QUANTIZATION COMPLETE")
print("==============================")

print("Output:", OUTPUT_MODEL)
print("Result:", result)


if os.path.exists(OUTPUT_MODEL):

    input_size = os.path.getsize(
        INPUT_MODEL
    )

    output_size = os.path.getsize(
        OUTPUT_MODEL
    )

    reduction = (
        (input_size - output_size)
        / input_size
    ) * 100

    print("\nOriginal size:",
          round(input_size / (1024 * 1024), 2),
          "MB")

    print("Quantized size:",
          round(output_size / (1024 * 1024), 2),
          "MB")

    print("Size reduction:",
          round(reduction, 2),
          "%")

else:

    print("ERROR: Quantized file was not created.")