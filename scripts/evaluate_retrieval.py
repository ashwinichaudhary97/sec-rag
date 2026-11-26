import hashlib
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from sec_rag.evaluation import evidence_recall_at_k, ndcg_at_k
from sec_rag.retrieval import dense_search


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(results):
    answerable = [result for result in results if result["answerable"]]

    summary = {
        "questions": len(results),
        "answerable_questions": len(answerable),
        "p95_retrieval_ms": float(
            np.percentile([result["retrieval_ms"] for result in results], 95)
        ),
    }

    for metric in (
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
    ):
        summary[metric] = (
            float(np.mean([result[metric] for result in answerable]))
            if answerable
            else None
        )

    return summary


def main():
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))

    experiment_id = config["experiment_id"]
    settings = config["embedding"]

    dev_path = Path("data/eval/dev.jsonl")

    dataset_hash = hashlib.sha256(dev_path.read_bytes()).hexdigest()

    if dataset_hash != (
        "09977be282f26ade4595b0444424780093f0afb84cf719b1972c59198ea00238"
    ):
        raise ValueError("Development dataset hash mismatch")

    questions = load_jsonl(dev_path)

    directory = Path("data/indexes") / experiment_id

    chunks = load_jsonl(directory / "chunks.jsonl")
    index = faiss.read_index(str(directory / "index.faiss"))

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk count mismatch")

    model = SentenceTransformer(
        settings["model_id"],
        revision=settings["revision"],
        device=settings["device"],
    )

    candidate_k = config["retrieval"]["candidate_k"]
    final_k = config["retrieval"]["final_k"]

    # Warm up the model before measuring retrieval latency.
    dense_search(
        "What does a company disclose in its annual report?",
        model,
        index,
        chunks,
        k=candidate_k,
    )

    results = []
    by_category = defaultdict(list)

    for number, question in enumerate(questions, start=1):
        started = perf_counter()

        candidates = dense_search(
            question["question"],
            model,
            index,
            chunks,
            k=candidate_k,
        )

        retrieved = candidates[:final_k]

        latency_ms = (perf_counter() - started) * 1000

        result = {
            "question_id": question["question_id"],
            "category": question["category"],
            "answerable": question["answerable"],
            "retrieval_ms": latency_ms,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                }
                for chunk in retrieved
            ],
        }

        if question["answerable"]:
            gold = question["gold_evidence"]

            for k in (5, 10):
                result[f"recall_at_{k}"] = evidence_recall_at_k(retrieved, gold, k)

                result[f"ndcg_at_{k}"] = ndcg_at_k(retrieved, chunks, gold, k)

        results.append(result)
        by_category[question["category"]].append(result)

        print(
            f"[{number}/{len(questions)}] "
            f"{question['question_id']} "
            f"{latency_ms:.1f} ms",
            flush=True,
        )

    summary = {
        "experiment_id": experiment_id,
        "split": "dev",
        "dataset_sha256": dataset_hash,
        "config": config,
        "indexed_chunks": len(chunks),
        "overall": summarize(results),
        "by_category": {
            category: summarize(items)
            for category, items in sorted(by_category.items())
        },
    }

    output = Path("results") / experiment_id
    output.mkdir(parents=True, exist_ok=True)

    (output / "retrieval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output / "retrieval_details.jsonl").open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

    print("\nRetrieval evaluation summary")
    print(json.dumps(summary["overall"], indent=2))

    print("\nResults by category")
    print(json.dumps(summary["by_category"], indent=2))

    print("\nResults saved to:", output)
    print("E00 retrieval evaluation: PASS")


if __name__ == "__main__":
    main()
