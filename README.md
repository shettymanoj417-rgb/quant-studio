# Quantization Studio

Quantization Studio is a lightweight Flask application for uploading, detecting, and INT8 quantizing model artifacts, then validating the result with a lightweight accuracy/performance comparison.

It supports SafeTensors and ONNX models, includes ZIP bundle handling for model packages, and produces a downloadable quantized archive.

## Features

- Upload `.safetensors`, `.onnx`, `.pt`, `.pth`, `.bin`, `.model`, and ZIP bundles
- Detect model type and architecture from tensor names and metadata
- Run INT8 quantization for SafeTensors and ONNX models
- Measure basic before/after performance for DistilBERT and ONNX paths
- Compare accuracy using the built-in SST-2 benchmark or a custom dataset
- Download a packaged quantized model bundle

## Project structure

- `backend/app.py` – Flask API and web entry point
- `backend/model_detector.py` – model-type detection logic
- `backend/model_quantizer.py` – INT8 quantization implementation
- `backend/model_evaluator.py` – accuracy and comparison logic
- `config/config.py` – runtime configuration
- `frontend/` – static web frontend assets
- `templates/` – template files
- `uploads/` – uploaded model files
- `quantized_models/` – generated quantized outputs
- `tests/` – backend regression tests

## Requirements

This project expects Python 3.10+ and uses the dependencies in `requirements.txt`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you are using the workspace venv already prepared for the project, you can use:

```bash
./.venv2/Scripts/python.exe -m pip install -r requirements.txt
```

## Configuration

The app reads environment variables through `config/config.py`.

Key defaults:

- `DEBUG=false`
- `HOST=0.0.0.0`
- `PORT=5000`
- `UPLOAD_FOLDER=uploads`
- `MODELS_FOLDER=quantized_models`
- `DEFAULT_QUANTIZATION=int8`
- `DEVICE_MAP=auto`

You can override them in a `.env` file or by exporting environment variables before running the app.

## Running the app

From the project root:

```bash
./.venv2/Scripts/python.exe backend/app.py
```

Then open:

```text
http://localhost:5000
```

## API endpoints

### Health

```http
GET /api/health
```

### Quantize a model

```http
POST /api/quantize
```

Multipart form fields:

- `model_source` – `upload` (default) or `huggingface`
- `model` – uploaded model file or ZIP bundle when `model_source=upload`
- `model_id` – Hugging Face repository ID when `model_source=huggingface`
- `revision` – optional Hugging Face branch, tag, or commit
- `quantization` – defaults to `int8`
- `device_map` – defaults to `auto`
- `dataset` – optional custom dataset file

For Hugging Face downloads, the repository must contain SafeTensors or ONNX
weights. Public repositories can be fetched without additional setup. Set
`HF_TOKEN` in the server environment to access private or gated repositories.

### Download a quantized model

```http
GET /api/download/<filename>
```

### Accuracy evaluation

```http
POST /api/evaluate-accuracy
```

## Typical usage

1. Open the app in the browser.
2. Upload a model file or ZIP package, or select the Hugging Face source and enter a repository ID.
3. Choose the quantization profile (INT8 is the supported profile).
4. Start quantization.
5. Review size reduction, performance, and accuracy metrics.
6. Download the quantized bundle.

## Notes

- The app is optimized for SafeTensors and ONNX model artifacts.
- ZIP bundles should include a single model file or a sharded SafeTensors set plus the supporting model metadata files.
- The built-in benchmark is SST-2; a custom dataset may be supplied for additional validation.
- For a stable local run, debug mode should remain off unless you are intentionally developing and reloading the app.

## Testing

Run the automated tests:

```bash
./.venv2/Scripts/python.exe -m pytest tests/ -q
```

## License

This project is provided under the repository license in `LICENSE`.
