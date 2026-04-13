"""Side-by-side comparison of OpenSearch and LanceDB for vector search.

Runs the same queries against both systems and compares:
- Query latency
- Result quality (overlap in top-k)
- Storage model (references vs inline data)
- Operational complexity
"""

import time
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import torch
import typer
from opensearchpy import OpenSearch
from transformers import AutoModel, AutoProcessor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LANCEDB_DIR = DATA_DIR / "lancedb"

app = typer.Typer()


def get_opensearch():
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )


def get_lancedb():
    db = lancedb.connect(str(LANCEDB_DIR))
    return db.open_table("coco_clip_embeddings")


def search_opensearch(client, vector, k=10):
    start = time.time()
    resp = client.search(
        index="coco-clip-embeddings",
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
        },
    )
    elapsed = time.time() - start
    hits = resp["hits"]["hits"]
    return elapsed, hits


def search_lancedb(table, vector, k=10):
    start = time.time()
    results = table.search(vector).limit(k).to_pandas()
    elapsed = time.time() - start
    return elapsed, results


def text_to_vector(query: str) -> list[float]:
    """Encode a text query using the same SigLIP 2 model."""
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").eval()
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")

    inputs = processor(text=[query], return_tensors="pt", padding="max_length")
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].numpy().tolist()


def compare_query(os_client, lance_table, query_vector: list[float], query_label: str, k: int = 5):
    """Run the same query against both systems and compare."""
    os_time, os_hits = search_opensearch(os_client, query_vector, k)
    lance_time, lance_results = search_lancedb(lance_table, query_vector, k)

    typer.secho(f"\nQuery: {query_label}", bold=True)
    typer.echo(f"{'Rank':<6} {'OpenSearch':<60} {'LanceDB':<60}")
    typer.echo("-" * 126)

    os_ids = []
    lance_ids = []

    for i in range(k):
        # OpenSearch result
        if i < len(os_hits):
            src = os_hits[i]["_source"]
            os_ids.append(src["image_id"])
            os_col = f"id={src['image_id']:<8} score={os_hits[i]['_score']:.4f}  {src['caption'][:30]}..."
        else:
            os_col = "-"

        # LanceDB result
        if i < len(lance_results):
            row = lance_results.iloc[i]
            lance_ids.append(row["image_id"])
            img_size = len(row["image_bytes"]) if row["image_bytes"] else 0
            lance_col = f"id={row['image_id']:<8} dist={row['_distance']:.4f}  {row['caption'][:20]}... ({img_size:,}b inline)"
        else:
            lance_col = "-"

        typer.echo(f"{i + 1:<6} {os_col:<60} {lance_col:<60}")

    # Overlap
    overlap = set(os_ids) & set(lance_ids)
    typer.echo(f"  Top-{k} overlap: {len(overlap)}/{k} matching image IDs")
    typer.echo(f"  Latency: OpenSearch {os_time*1000:.1f}ms | LanceDB {lance_time*1000:.1f}ms")

    return os_time, lance_time


@app.command()
def main():
    """Run the same queries against OpenSearch and LanceDB, comparing results."""
    typer.secho("=" * 60, bold=True)
    typer.secho("OpenSearch vs LanceDB — Side-by-Side Comparison", bold=True)
    typer.secho("Same CLIP embeddings, same queries, different storage models.", dim=True)
    typer.secho("=" * 60, bold=True)

    os_client = get_opensearch()
    lance_table = get_lancedb()

    # Load precomputed embeddings from parquet
    df = pd.read_parquet(EMBEDDINGS_DIR / "image_embeddings.parquet")

    os_times = []
    lance_times = []

    # Query 1: Image similarity (use a sample image embedding)
    typer.secho("\n1. Image -> Image Search", bold=True)
    ot, lt = compare_query(
        os_client, lance_table,
        df.iloc[42]["vector"],
        f"Similar to '{df.iloc[42]['file_name']}'",
    )
    os_times.append(ot)
    lance_times.append(lt)

    # Query 2: Text -> Image search (cross-modal)
    typer.secho("\n2. Text -> Image Search (cross-modal)", bold=True)
    text_queries = [
        "a dog playing in the park",
        "a busy city street at night",
        "food on a kitchen table",
    ]

    for query in text_queries:
        vec = text_to_vector(query)
        ot, lt = compare_query(os_client, lance_table, vec, query)
        os_times.append(ot)
        lance_times.append(lt)

    # Summary
    typer.echo("")
    typer.secho("=" * 60, bold=True)
    typer.secho("Performance Summary", bold=True)
    typer.secho("=" * 60, bold=True)
    typer.echo(f"\nAverage latency:")
    typer.echo(f"  OpenSearch: {np.mean(os_times)*1000:.1f}ms")
    typer.echo(f"  LanceDB:    {np.mean(lance_times)*1000:.1f}ms")
    typer.echo(f"\nStorage Model Comparison:")
    typer.echo(f"  OpenSearch:")
    typer.echo(f"    - Vectors + metadata in kNN index")
    typer.echo(f"    - Images stored externally (S3, CDN, filesystem)")
    typer.echo(f"    - Requires Docker/server infrastructure")
    typer.echo(f"    - Image retrieval = separate HTTP call")
    typer.echo(f"  LanceDB:")
    typer.echo(f"    - Vectors + metadata + images in one Lance table")
    typer.echo(f"    - No external storage needed")
    typer.echo(f"    - No server — embedded, just files on disk")
    typer.echo(f"    - Image bytes returned with search results")


if __name__ == "__main__":
    app()
