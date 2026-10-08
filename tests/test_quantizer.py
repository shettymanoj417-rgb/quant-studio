import torch
from safetensors import safe_open
from safetensors.torch import save_file

from backend.model_quantizer import ModelQuantizer


def test_quantize_bfloat16_safetensors(tmp_path):
    source_path = tmp_path / "source.safetensors"
    output_path = tmp_path / "quantized.safetensors"
    source_tensor = torch.tensor([0.25, -0.5, 1.0], dtype=torch.bfloat16)

    save_file(
        {
            "weight": source_tensor,
            "counter": torch.tensor([1, 2], dtype=torch.int64),
        },
        str(source_path),
    )

    result = ModelQuantizer().quantize(str(source_path), str(output_path))

    with safe_open(str(output_path), framework="pt", device="cpu") as output:
        assert output.get_tensor("weight").dtype == torch.int8
        assert output.get_tensor("weight.__scale__").dtype == torch.float32
        assert torch.equal(
            output.get_tensor("counter"),
            torch.tensor([1, 2], dtype=torch.int64),
        )

    assert result["quantized_tensors"] == 1
    assert result["unchanged_tensors"] == 1
