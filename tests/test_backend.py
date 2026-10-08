import json
import os
import zipfile
from io import BytesIO

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from backend.app import app, create_download_bundle, extract_model_from_zip
from backend.model_quantizer import ModelQuantizer


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"


def test_device_info(client):
    """Test device info endpoint"""
    response = client.get("/api/device-info")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "cuda_available" in data
    assert "memory_total" in data


def test_quantize_invalid_request(client):
    """Test quantization with invalid request"""
    response = client.post("/api/quantize", json={}, content_type="application/json")
    assert response.status_code == 400


def test_quantize_from_huggingface_repository(client, monkeypatch, tmp_path):
    downloads = {}

    def fake_snapshot_download(**kwargs):
        downloads.update(kwargs)
        snapshot_dir = kwargs["local_dir"]
        model_snapshot = snapshot_dir
        with open(os.path.join(model_snapshot, "config.json"), "w") as config:
            json.dump({"model_type": "test"}, config)
        with open(
            os.path.join(model_snapshot, "model.safetensors.index.json"), "w"
        ) as index:
            json.dump(
                {"weight_map": {"layer.weight": "model-00001-of-00002.safetensors"}},
                index,
            )
        for shard_name in (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ):
            with open(os.path.join(model_snapshot, shard_name), "wb") as shard:
                shard.write(b"original weights")
        with open(os.path.join(model_snapshot, "tokenizer.json"), "w") as tokenizer:
            tokenizer.write("{}")
        return snapshot_dir

    class FakeQuantizer:
        def __init__(self, config):
            pass

        def quantize(self, input_path, output_path):
            assert os.path.isdir(input_path)
            with open(output_path, "wb") as quantized_model:
                quantized_model.write(b"quantized weights")
            return {"status": "success"}

    monkeypatch.setattr("backend.app.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("backend.app.ModelQuantizer", FakeQuantizer)
    monkeypatch.setattr(
        "backend.app.get_model_detection",
        lambda path, extension: {
            "model_type": "TestModel",
            "architecture": "TestArchitecture",
            "tensor_count": 2,
            "detection_method": "test",
        },
    )
    monkeypatch.setattr(
        "backend.app.evaluate_quantized_model",
        lambda **kwargs: {"status": "ok"},
    )
    monkeypatch.setitem(app.config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setitem(app.config, "MODELS_FOLDER", str(tmp_path / "models"))
    monkeypatch.setitem(app.config, "HF_TOKEN", "test-token")
    os.makedirs(app.config["MODELS_FOLDER"])

    response = client.post(
        "/api/quantize",
        data={
            "model_source": "huggingface",
            "model_id": "org/model",
            "revision": "main",
        },
    )

    assert response.status_code == 200
    assert downloads["repo_id"] == "org/model"
    assert downloads["revision"] == "main"
    assert downloads["token"] == "test-token"
    download = client.get(response.get_json()["download_url"])
    assert download.status_code == 200
    with zipfile.ZipFile(BytesIO(download.data)) as bundle:
        assert "config.json" in bundle.namelist()
        assert "tokenizer.json" in bundle.namelist()
        assert "model.safetensors" in bundle.namelist()
        assert not {
            name
            for name in bundle.namelist()
            if name.endswith(".safetensors") and name != "model.safetensors"
        }
        index = json.loads(bundle.read("model.safetensors.index.json"))
        assert set(index["weight_map"].values()) == {"model.safetensors"}
        assert bundle.read("model.safetensors") == b"quantized weights"
    assert not os.path.exists(downloads["local_dir"])


def test_quantize_huggingface_requires_repository_id(client):
    response = client.post(
        "/api/quantize",
        data={"model_source": "huggingface"},
    )

    assert response.status_code == 400
    assert "repository ID is required" in response.get_json()["error"]


def test_zip_bundle_preserves_model_assets_and_replaces_weights(tmp_path):
    source_path = tmp_path / "source.zip"
    quantized_path = tmp_path / "quantized.safetensors"
    bundle_path = tmp_path / "download.zip"
    required_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "quantization_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    with zipfile.ZipFile(source_path, "w") as source:
        for name in required_files:
            source.writestr(name, b"original model asset")

    with open(quantized_path, "wb") as quantized_model:
        quantized_model.write(b"quantized weights")

    model_path, model_member, _ = extract_model_from_zip(source_path)
    try:
        create_download_bundle(
            source_path=source_path,
            model_path=model_path,
            quantized_path=quantized_path,
            bundle_path=bundle_path,
            model_member=model_member,
            bundled_source=True,
        )
    finally:
        os.remove(model_path)

    with zipfile.ZipFile(bundle_path) as bundle:
        assert set(bundle.namelist()) == set(required_files)
        assert bundle.read("model.safetensors") == b"quantized weights"
        for asset_name in required_files:
            if asset_name != "model.safetensors":
                assert bundle.read(asset_name) == b"original model asset"


def test_quantize_zip_download_contains_required_model_files(
    client, monkeypatch, tmp_path
):
    required_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "quantization_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    source = BytesIO()
    with zipfile.ZipFile(source, "w") as bundle:
        for name in required_files:
            bundle.writestr(name, b"original model asset")
    source.seek(0)

    class FakeQuantizer:
        def __init__(self, config):
            pass

        def quantize(self, input_path, output_path):
            with open(output_path, "wb") as quantized_model:
                quantized_model.write(b"quantized weights")
            return {"status": "success"}

    monkeypatch.setattr("backend.app.ModelQuantizer", FakeQuantizer)
    monkeypatch.setattr(
        "backend.app.get_model_detection",
        lambda path, extension: {
            "model_type": "TestModel",
            "architecture": "TestArchitecture",
            "tensor_count": 1,
            "detection_method": "test",
        },
    )
    monkeypatch.setattr(
        "backend.app.evaluate_quantized_model",
        lambda **kwargs: {"status": "ok"},
    )
    monkeypatch.setitem(app.config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setitem(app.config, "MODELS_FOLDER", str(tmp_path / "models"))
    os.makedirs(app.config["UPLOAD_FOLDER"])
    os.makedirs(app.config["MODELS_FOLDER"])

    response = client.post(
        "/api/quantize",
        data={"model": (source, "model_bundle.zip")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = response.get_json()
    assert result["download_url"].endswith(".zip")

    download = client.get(result["download_url"])
    assert download.status_code == 200
    with zipfile.ZipFile(BytesIO(download.data)) as bundle:
        assert set(bundle.namelist()) == set(required_files)
        assert bundle.read("model.safetensors") == b"quantized weights"


def test_quantize_zip_with_sharded_safetensors_bundle(client, monkeypatch, tmp_path):
    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    source = BytesIO()
    with zipfile.ZipFile(source, "w") as bundle:
        for name in required_files:
            bundle.writestr(name, b"original model asset")
    source.seek(0)

    class FakeQuantizer:
        def __init__(self, config):
            pass

        def quantize(self, input_path, output_path):
            with open(output_path, "wb") as quantized_model:
                quantized_model.write(b"quantized weights")
            return {"status": "success"}

    monkeypatch.setattr("backend.app.ModelQuantizer", FakeQuantizer)
    monkeypatch.setattr(
        "backend.app.get_model_detection",
        lambda path, extension: {
            "model_type": "TestModel",
            "architecture": "TestArchitecture",
            "tensor_count": 2,
            "detection_method": "test",
        },
    )
    monkeypatch.setattr(
        "backend.app.evaluate_quantized_model",
        lambda **kwargs: {"status": "ok"},
    )
    monkeypatch.setitem(app.config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setitem(app.config, "MODELS_FOLDER", str(tmp_path / "models"))
    os.makedirs(app.config["UPLOAD_FOLDER"])
    os.makedirs(app.config["MODELS_FOLDER"])

    response = client.post(
        "/api/quantize",
        data={"model": (source, "model_bundle.zip")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = response.get_json()
    assert result["download_url"].endswith(".zip")

    download = client.get(result["download_url"])
    assert download.status_code == 200
    with zipfile.ZipFile(BytesIO(download.data)) as bundle:
        assert "model-00001-of-00002.safetensors" in bundle.namelist()
        assert bundle.read("model-00001-of-00002.safetensors") == b"quantized weights"


def test_model_quantizer_accepts_sharded_safetensors_directory(tmp_path):
    shard_dir = tmp_path / "model_shards"
    nested_weights = shard_dir / "weights"
    nested_weights.mkdir(parents=True)
    save_file(
        {"layer_a.weight": torch.tensor([1.0, -2.0])},
        str(nested_weights / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"layer_b.weight": torch.tensor([3.0, -4.0])},
        str(nested_weights / "model-00002-of-00002.safetensors"),
    )

    output_path = tmp_path / "out.safetensors"

    result = ModelQuantizer().quantize(str(shard_dir), str(output_path))

    assert result["status"] == "success"
    with safe_open(output_path, framework="pt", device="cpu") as quantized_model:
        assert "layer_a.weight" in quantized_model.keys()
        assert "layer_b.weight" in quantized_model.keys()
