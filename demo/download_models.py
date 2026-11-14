"""Download the model files used by preprocessing and recommendation."""

import os
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1] / "cache/models"
SIGLIP_REPO = "google/siglip2-base-patch16-224"
SIGLIP_FILES = (
    "config.json", "preprocessor_config.json", "special_tokens_map.json",
    "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "model.safetensors",
)
NIMA_NAME = "NIMA_VGG16_ava-dc4e8265.pth"
VGG_REPO = "timm/vgg16.tv_in1k"
VGG_REVISION = "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f"


def main():
    hub_cache = MODEL_ROOT / "huggingface/hub"
    os.environ.update(
        HF_HOME=str(hub_cache.parent),
        HF_HUB_CACHE=str(hub_cache),
        HF_HUB_OFFLINE="0",
    )
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(
        SIGLIP_REPO, revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        local_dir=MODEL_ROOT / "siglip2-base-patch16-224",
        allow_patterns=SIGLIP_FILES, token=False, max_workers=2,
    )
    hf_hub_download(
        "chaofengc/IQA-PyTorch-Weights", NIMA_NAME,
        revision="f480f1d8f73dfa374670097cb4098d4a13704165",
        local_dir=MODEL_ROOT / "torch/hub/pyiqa", token=False,
    )

    snapshot = Path(snapshot_download(
        VGG_REPO, revision=VGG_REVISION, cache_dir=hub_cache,
        allow_patterns=["config.json", "model.safetensors"], token=False, max_workers=2,
    ))
    # timm uses the repository's default ref when loading the cached backbone.
    ref = snapshot.parent.parent / "refs/main"
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(VGG_REVISION, encoding="ascii")
    print(f"Model files ready in {MODEL_ROOT}")


if __name__ == "__main__":
    main()
