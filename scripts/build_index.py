import hashlib
import json
from pathlib import Path
from time import perf_counter

import faiss
import yaml
from sentence_transformers import SentenceTransformer

from sec_rag.chunking import chunk_document, token_count
from sec_rag.retrieval import build_dense_index


def main():
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))

    settings = config["embedding"]
    limit = config["chunking"]["max_tokens"]

    model = SentenceTransformer(
        settings["model_id"],
        revision=settings["revision"],
        device=settings["device"],
    )

    if limit > model.max_seq_length:
        raise ValueError("Chunk limit exceeds model sequence length")

    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))

    chunks = []

    for record in manifest:
        path = Path(record["parsed_path"])
        content = path.read_bytes()

        if hashlib.sha256(content).hexdigest() != record["parsed_sha256"]:
            raise ValueError(f"Corpus hash mismatch: {path}")

        document = json.loads(content.decode("utf-8"))

        chunks.extend(chunk_document(document, model.tokenizer, limit))

    if any(token_count(chunk["text"], model.tokenizer) > limit for chunk in chunks):
        raise ValueError("Chunk exceeds embedding token limit")

    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")

    print("Chunks:", len(chunks), flush=True)
    print("Device:", model.device, flush=True)

    started = perf_counter()

    index = build_dense_index(
        chunks,
        model,
        batch_size=settings["batch_size"],
    )

    elapsed = perf_counter() - started

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunk count mismatch")

    output = Path("data/indexes") / config["experiment_id"]
    output.mkdir(parents=True, exist_ok=True)

    chunks_path = output / "chunks.jsonl"
    index_path = output / "index.faiss"

    with chunks_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    faiss.write_index(index, str(index_path))

    restored = faiss.read_index(str(index_path))

    if restored.ntotal != len(chunks) or restored.d != index.d:
        raise ValueError("Saved index verification failed")

    print("\nIndex build summary")
    print("Indexed chunks:", restored.ntotal)
    print("Dimensions:", restored.d)
    print("Index type:", type(restored).__name__)
    print("Embedding and indexing seconds:", round(elapsed, 2))
    print("Index size MiB:", round(index_path.stat().st_size / 2**20, 2))
    print("Saved index:", index_path)
    print("Saved chunks:", chunks_path)
    print("Index build: PASS")


if __name__ == "__main__":
    main()
