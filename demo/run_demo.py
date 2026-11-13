"""Run the three reference requests against a locally running service."""

import argparse
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, help="Save the actual request and response pairs as JSON.")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"{url}/health", timeout=10) as response:
        if json.load(response) != {"status": "ok"}:
            raise RuntimeError("Unexpected response from /health")

    cases = json.loads((PROJECT / "demo/requests.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        payload = json.dumps(case["request"], ensure_ascii=False).encode("utf-8")
        request = Request(f"{url}/recommend", data=payload, headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=90) as response:
            result = json.load(response)
        photos = result["recommendations"]
        ids = [photo["photo_id"] for photo in photos]
        if (ids != case["expected_ids"] or result["filter_counts"] != case["expected_counts"]
                or result["reason"] != case["expected_reason"] or result["missing_features"]):
            raise RuntimeError(f"{case['id']}: unexpected recommendations: {ids}; {result['filter_counts']}")
        for photo in photos:
            path = PROJECT / "data/formal" / photo["image_path"]
            if photo["resource_id"] != photo["photo_id"] or not path.is_file():
                raise RuntimeError(f"Missing photo resource: {photo['photo_id']}")
        print(f"{case['id']}: {', '.join(ids) or result['reason']}")
        results.append({"id": case["id"], "request": case["request"], "response": result})

    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
