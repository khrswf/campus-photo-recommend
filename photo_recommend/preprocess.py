"""Build the photo feature cache and optionally inspect text retrieval."""

import argparse
import hashlib
import io
import json
import os
from functools import cached_property, lru_cache
from importlib.metadata import version
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import perf_counter
from zipfile import BadZipFile

import numpy as np
from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
MODEL_CACHE = PROJECT / "cache" / "models"
SIGLIP_DIR = MODEL_CACHE / "siglip2-base-patch16-224"
NIMA_METRIC = "nima-vgg16-ava"
NIMA_FILE = MODEL_CACHE / "torch/hub/pyiqa/NIMA_VGG16_ava-dc4e8265.pth"
EMBEDDING_SIZE = 768


def read_catalog(path):
    catalog = json.loads(path.read_text(encoding="utf-8"))
    photo_ids = [photo["photo_id"] for photo in catalog["photos"]]
    for value in (catalog["dataset_id"], catalog["version"], *photo_ids):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError(f"Invalid catalog identifier: {value!r}")
    if len(set(photo_ids)) != len(photo_ids):
        raise ValueError("Duplicate photo_id in catalog")
    return catalog


def model_signature():
    packages = {
        name: version(name)
        for name in ("torch", "torchvision", "transformers", "pyiqa", "timm", "Pillow")
    }
    vgg_cache = MODEL_CACHE / "huggingface/hub/models--timm--vgg16.tv_in1k"
    vgg_revision = (vgg_cache / "refs/main").read_text(encoding="ascii").strip()
    files = [*sorted(SIGLIP_DIR.iterdir()), NIMA_FILE,
             vgg_cache / "snapshots" / vgg_revision / "model.safetensors"]
    assets = {}
    for path in files:
        if path.suffix in {".json", ".safetensors", ".pth", ".model"}:
            info = path.stat()
            assets[path.relative_to(MODEL_CACHE).as_posix()] = model_file_digest(
                path, info.st_size, info.st_mtime_ns, info.st_ctime_ns
            )
    signature = json.dumps({"cache_version": 2, "packages": packages, "assets": assets}, sort_keys=True)
    return hashlib.sha256(signature.encode()).hexdigest(), packages


@lru_cache(maxsize=32)
def model_file_digest(path, size, modified_at, changed_at):
    # File stats invalidate the in-process memo. Only the digest enters the portable cache key.
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    info = path.stat()
    if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (size, modified_at, changed_at):
        raise OSError(f"Model file changed while reading: {path}")
    return digest.hexdigest()


def read_features(path, key):
    try:
        with np.load(path, allow_pickle=False) as cached:
            if cached["cache_key"].item() != key:
                return None
            vector = cached["image_vector"]
            score = float(cached["aesthetic_score"].item())
            if valid_features(vector, score):
                return vector, score
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        pass
    return None


def valid_features(vector, score):
    return (vector.shape == (EMBEDDING_SIZE,)
            and np.all(np.isfinite(vector))
            and np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)
            and 1 <= score <= 10)


def write_atomic(path, write):
    file = NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".", delete=False)
    temporary = Path(file.name)
    try:
        with file:
            write(file)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Models:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device

    @cached_property
    def siglip(self):
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(SIGLIP_DIR, local_files_only=True)
        model = AutoModel.from_pretrained(SIGLIP_DIR, local_files_only=True)
        return processor, model.to(self.device).eval()

    @cached_property
    def nima(self):
        import pyiqa

        return pyiqa.create_metric(
            NIMA_METRIC, device=self.device, pretrained_model_path=str(NIMA_FILE)
        )

    def image_features(self, image):
        from torchvision.transforms.functional import to_tensor

        processor, siglip = self.siglip
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            embedding = siglip.get_image_features(**inputs).pooler_output
            vector = self.torch.nn.functional.normalize(embedding, dim=-1)[0].cpu().numpy()
            score = float(self.nima(to_tensor(image).unsqueeze(0)).item())
        return vector, score

    def text_vector(self, query):
        processor, siglip = self.siglip
        inputs = processor(
            text=[query], padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            embedding = siglip.get_text_features(**inputs).pooler_output
            return self.torch.nn.functional.normalize(embedding, dim=-1)[0].cpu().numpy()


def configure_model_cache():
    os.environ["HF_HOME"] = str(MODEL_CACHE / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE / "huggingface/hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = str(MODEL_CACHE / "torch")
    os.environ["MPLCONFIGDIR"] = str(PROJECT / "cache/matplotlib")


def build_features(catalog_path, photo_ids, query, requested_device):
    started = perf_counter()
    catalog_path = catalog_path.resolve()
    catalog = read_catalog(catalog_path)
    photos = catalog["photos"]
    if photo_ids:
        unknown = set(photo_ids) - {photo["photo_id"] for photo in photos}
        if unknown:
            raise ValueError(f"Unknown photo_id: {', '.join(sorted(unknown))}")
        photos = [photo for photo in photos if photo["photo_id"] in photo_ids]

    configure_model_cache()
    import torch

    device = requested_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment")
        torch.cuda.reset_peak_memory_stats()
    models = Models(torch, device)
    signature, packages = model_signature()
    cache_dir = PROJECT / "cache" / "features" / catalog["dataset_id"] / catalog["version"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    cache_hits = 0
    for photo in photos:
        photo_id = photo["photo_id"]
        cache_path = cache_dir / f"{photo_id}.npz"
        try:
            image_path = (catalog_path.parent / photo["image_path"]).resolve()
            if Path(photo["image_path"]).is_absolute() or not image_path.is_relative_to(catalog_path.parent):
                raise ValueError("image_path must be relative to the catalog directory")
            image_bytes = image_path.read_bytes()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            key = f"{signature}:{image_hash}"
            cached = read_features(cache_path, key)
            if cached is None:
                with Image.open(io.BytesIO(image_bytes)) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            failures.append({"photo_id": photo_id, "reason": str(error)})
            continue

        if cached is None:
            vector, score = models.image_features(image)
            if not valid_features(vector, score):
                failures.append({"photo_id": photo_id, "reason": "Invalid model output"})
                continue
            write_atomic(
                cache_path,
                lambda file: np.savez_compressed(
                    file, cache_key=key, photo_id=photo_id, image_sha256=image_hash,
                    image_vector=vector, aesthetic_score=score,
                ),
            )
        else:
            vector, score = cached
            cache_hits += 1
        results.append((photo, vector, score))

    summary = {
        "catalog": str(catalog_path),
        "cache_dir": str(cache_dir),
        "models": {"image_text": f"google/{SIGLIP_DIR.name}", "aesthetics": NIMA_METRIC},
        "packages": packages,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "requested": len(photos),
        "encoded": len(results) - cache_hits,
        "cache_hits": cache_hits,
        "failures": failures,
    }
    if query and results:
        query_vector = models.text_vector(query)
        ranking = [
            {
                "photo_id": photo["photo_id"],
                "person_count": photo.get("person_count"),
                "framing": photo.get("framing"),
                "cosine_similarity": float(vector @ query_vector),
                "aesthetic_score": score,
            }
            for photo, vector, score in results
        ]
        summary["query"] = query
        summary["text_ranking"] = sorted(
            ranking, key=lambda row: row["cosine_similarity"], reverse=True
        )
    if device == "cuda":
        summary["peak_allocated_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    summary["elapsed_seconds"] = round(perf_counter() - started, 2)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    write_atomic(cache_dir / "last_run.json", lambda file: file.write((payload + "\n").encode("utf-8")))
    return payload, bool(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PROJECT / "data/demo/catalog.json")
    parser.add_argument("--photo-ids", nargs="+", help="Process a subset of photo IDs.")
    parser.add_argument("--query", help="Optional text query for inspecting the ranking.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    summary, failed = build_features(args.catalog, args.photo_ids, args.query, args.device)
    print(summary)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
