"""Recommend up to three photos from an existing feature cache."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .preprocess import (
    EMBEDDING_SIZE, PROJECT, Models, configure_model_cache, model_signature,
    read_catalog, read_features,
)
from .ranking import filter_photos


FORMAL_CATALOG = PROJECT / "data/formal/catalog.json"
NEAR_DUPLICATE_COSINE = 0.98
UNSPECIFIED_VIEWPOINTS = {"机位待细分", "未定位_暂不细分"}


def select_diverse(ranked, limit):
    selected = []
    deferred = []
    used_viewpoints = set()
    for candidate in ranked:
        if len(selected) == limit:
            break
        row, vector, viewpoint = candidate
        if any(float(vector @ other) >= NEAR_DUPLICATE_COSINE for _, other in selected):
            continue
        if viewpoint is not None and viewpoint in used_viewpoints:
            deferred.append(candidate)
            continue
        selected.append((row, vector))
        used_viewpoints.add(viewpoint)
    for row, vector, _ in deferred:
        if len(selected) == limit:
            break
        if all(float(vector @ other) < NEAR_DUPLICATE_COSINE for _, other in selected):
            selected.append((row, vector))
    return [row for row, _ in selected]


def recommend(
    *, catalog_path=FORMAL_CATALOG, query="", location_id=None, person_count=None,
    framing=None, pose=None, forbidden_poses=(), required_objects=(),
    aesthetic_weight=0.2, limit=3,
):
    """Apply structured filters and rank the remaining photos using cached features."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 3:
        raise ValueError("limit must be an integer between 1 and 3")
    if not math.isfinite(aesthetic_weight) or not 0 <= aesthetic_weight <= 1:
        raise ValueError("aesthetic_weight must be between 0 and 1")
    query = query.strip()
    catalog_path = Path(catalog_path).resolve()
    catalog = read_catalog(catalog_path)
    matches, pending, excluded = filter_photos(
        catalog["photos"], location_id=location_id, person_count=person_count,
        framing=framing, pose=pose, forbidden_poses=forbidden_poses,
        required_objects=required_objects,
    )
    result = {
        "recommendations": [],
        "filter_counts": {
            "matched": len(matches), "pending": len(pending), "excluded": len(excluded),
        },
        "pending": pending,
        "excluded_reason_counts": dict(Counter(
            reason for photo in excluded for reason in photo["reasons"]
        )),
        "missing_features": [],
        "reason": None,
    }
    if not matches:
        result["reason"] = "no_confirmed_matches"
        return result

    candidates, result["missing_features"] = load_candidates(catalog_path, catalog, matches)
    if not candidates:
        result["reason"] = "features_unavailable"
        return result

    similarities = [None] * len(candidates)
    if query:
        configure_model_cache()
        import torch

        models = Models(torch, "cuda" if torch.cuda.is_available() else "cpu")
        similarities = text_similarities(candidates, models.text_vector(query))

    result["recommendations"] = rank_candidates(candidates, similarities, aesthetic_weight, limit)
    return result


def load_candidates(catalog_path, catalog, photos):
    signature, _ = model_signature()
    cache_dir = PROJECT / "cache/features" / catalog["dataset_id"] / catalog["version"]
    candidates = []
    missing = []
    for photo in photos:
        relative_path = Path(photo["image_path"])
        image_path = (catalog_path.parent / relative_path).resolve()
        if relative_path.is_absolute() or not image_path.is_relative_to(catalog_path.parent):
            raise ValueError(f"Invalid image_path for {photo['photo_id']}")
        try:
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        except OSError:
            missing.append(photo["photo_id"])
            continue
        features = read_features(cache_dir / f"{photo['photo_id']}.npz", f"{signature}:{image_hash}")
        if features is None:
            missing.append(photo["photo_id"])
            continue
        vector, score = features
        candidates.append((photo, vector, score))
    return candidates, missing


def text_similarities(candidates, query_vector):
    if query_vector.shape != (EMBEDDING_SIZE,) or not np.isfinite(query_vector).all():
        raise RuntimeError("Invalid SigLIP query embedding")
    return [float(vector @ query_vector) for _, vector, _ in candidates]


def rank_candidates(candidates, similarities, aesthetic_weight, limit):
    ranked = []
    if similarities and similarities[0] is not None:
        minimum, maximum = min(similarities), max(similarities)
    for (photo, vector, aesthetic), similarity in zip(candidates, similarities):
        if similarity is None:
            score = (aesthetic - 1) / 9
        elif maximum == minimum:
            score = (1 - aesthetic_weight) * 0.5 + aesthetic_weight * (aesthetic - 1) / 9
        else:
            relevance = (similarity - minimum) / (maximum - minimum)
            score = (1 - aesthetic_weight) * relevance + aesthetic_weight * (aesthetic - 1) / 9
        group = photo.get("viewpoint_group")
        viewpoint = (photo.get("location_id"), group) if group and group not in UNSPECIFIED_VIEWPOINTS else None
        row = {
            "photo_id": photo["photo_id"],
            "resource_id": photo["resource_id"],
            "image_path": photo["image_path"],
            "cosine_similarity": similarity,
            "aesthetic_score": aesthetic,
            "score": score,
            "location_id": photo.get("location_id"),
            "viewpoint_group": group,
            "viewpoint_status": photo.get("viewpoint_status"),
        }
        ranked.append((row, vector, viewpoint))
    ranked.sort(key=lambda item: (-item[0]["score"], item[0]["photo_id"]))
    return select_diverse(ranked, limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=FORMAL_CATALOG)
    parser.add_argument("--location-id")
    parser.add_argument("--person-count", type=int)
    parser.add_argument("--framing")
    parser.add_argument("--pose")
    parser.add_argument("--forbidden-pose", action="append", default=[])
    parser.add_argument("--required-object", action="append", default=[])
    parser.add_argument("--query", default="")
    parser.add_argument("--aesthetic-weight", type=float, default=0.2)
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3)
    args = vars(parser.parse_args())
    args["catalog_path"] = args.pop("catalog")
    args["required_objects"] = args.pop("required_object")
    args["forbidden_poses"] = args.pop("forbidden_pose")
    print(json.dumps(recommend(**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
