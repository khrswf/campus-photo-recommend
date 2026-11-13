"""Compare text-only and aesthetic-weighted ranking on the same candidates."""

import json
from math import log2
from statistics import mean

from .preprocess import PROJECT, Models, configure_model_cache, read_catalog, write_atomic
from .ranking import filter_photos
from .recommend import FORMAL_CATALOG, load_candidates, rank_candidates, text_similarities


def ndcg_at_three(recommended, ideal):
    def discounted_gain(grades):
        return sum(grade / log2(rank + 2) for rank, grade in enumerate(grades[:3]))

    return discounted_gain(recommended) / discounted_gain(ideal)


def main():
    spec = json.loads((PROJECT / "data/eval/ab_cases.json").read_text(encoding="utf-8"))
    catalog = read_catalog(FORMAL_CATALOG)
    if (spec["dataset_id"], spec["version"]) != (catalog["dataset_id"], catalog["version"]):
        raise ValueError("Evaluation cases must match the catalog version")
    cases = spec["validation"] + spec["test"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case ID")

    features, missing = load_candidates(FORMAL_CATALOG, catalog, catalog["photos"])
    if missing:
        raise RuntimeError(f"Missing features for: {', '.join(missing)}")
    by_id = {row[0]["photo_id"]: row for row in features}
    configure_model_cache()
    import torch

    model = Models(torch, "cuda" if torch.cuda.is_available() else "cpu")

    def prepare(case):
        matches, pending, excluded = filter_photos(catalog["photos"], **case["conditions"])
        ids = {photo["photo_id"] for photo in matches}
        high, partial = set(case["relevance_2"]), set(case["relevance_1"])
        if len(ids) < 4 or not high or high & partial or (high | partial) - ids:
            raise ValueError(f"Check candidates and relevance labels for {case['id']}")
        candidates = [by_id[photo["photo_id"]] for photo in matches]
        grades = dict.fromkeys(high, 2) | dict.fromkeys(partial, 1)
        ideal = sorted((grades.get(photo_id, 0) for photo_id in ids), reverse=True)
        similarities = text_similarities(candidates, model.text_vector(case["query"]))
        return {
            "id": case["id"],
            "query": case["query"],
            "conditions": case["conditions"],
            "counts": {"matched": len(matches), "pending": len(pending), "excluded": len(excluded)},
        }, candidates, similarities, grades, ideal

    def score(prepared, weight):
        _, candidates, similarities, grades, ideal = prepared
        rows = rank_candidates(candidates, similarities, weight, 3)
        for row in rows:
            row["relevance"] = grades.get(row["photo_id"], 0)
        return {
            "recommendations": rows,
            "ndcg_at_3": ndcg_at_three([row["relevance"] for row in rows], ideal),
        }

    validation = [prepare(case) for case in spec["validation"]]
    weights = [0.0, *spec["validation_weights"]]
    validation_scores = [
        {"weight": weight, "mean_ndcg_at_3": mean(score(case, weight)["ndcg_at_3"] for case in validation)}
        for weight in weights
    ]
    # Keep a positive weight in B so the held-out test compares the two methods.
    chosen = max(validation_scores[1:], key=lambda row: (row["mean_ndcg_at_3"], -row["weight"]))
    weight = chosen["weight"]
    tests = []
    for case in spec["test"]:
        prepared = prepare(case)
        tests.append({**prepared[0], "A": score(prepared, 0), "B": score(prepared, weight)})

    result = {
        "dataset_id": catalog["dataset_id"],
        "version": catalog["version"],
        "validation": [case[0] for case in validation],
        "validation_scores": validation_scores,
        "chosen_aesthetic_weight": weight,
        "test_mean_ndcg_at_3": {
            group: mean(case[group]["ndcg_at_3"] for case in tests) for group in ("A", "B")
        },
        "test": tests,
    }
    path = PROJECT / "reports/AB结果.json"
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_atomic(path, lambda file: file.write(payload))
    print(f"B weight: {weight:g}; test nDCG@3: {result['test_mean_ndcg_at_3']}")
    print(path)


if __name__ == "__main__":
    main()
