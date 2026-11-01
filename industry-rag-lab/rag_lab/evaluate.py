"""Reproducible retrieval metrics against a labelled JSONL test set.

Example: python -m rag_lab.evaluate --dataset evals/ground_truth.jsonl
Requires an already-ingested corpus and a running API.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx


def retrieval_metrics(rows: list[dict], predictions: list[list[str]]) -> dict[str, float]:
    """Returns Recall@K (query-level hit rate) and reciprocal-rank average."""
    if not rows or len(rows) != len(predictions):
        raise ValueError("Need equally sized, nonempty labelled queries and predictions")
    hit_count = 0
    reciprocal_sum = 0.0
    for row, actual in zip(rows, predictions):
        expected = set(row["expected_sources"])
        if not expected:
            raise ValueError("Each labelled query needs at least one expected source")
        for rank, source in enumerate(actual, 1):
            if source in expected:
                hit_count += 1
                reciprocal_sum += 1 / rank
                break
    return {"recall_at_k": hit_count / len(rows), "mrr_at_k": reciprocal_sum / len(rows),
            "queries": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/ground_truth.jsonl")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    key = os.getenv("RAG_API_KEY")
    if not key:
        parser.error("Set RAG_API_KEY to the tenant API key")
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    predictions = []
    with httpx.Client(timeout=120) as client:
        for row in rows:
            response = client.post(f"{args.base_url.rstrip('/')}/v1/search",
                                   headers={"X-API-Key": key},
                                   json={"question": row["question"], "top_k": args.top_k})
            response.raise_for_status()
            predictions.append([s["file_name"] for s in response.json()["sources"]])
    print(json.dumps(retrieval_metrics(rows, predictions), indent=2))


if __name__ == "__main__":
    main()
