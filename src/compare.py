"""Side-by-side comparison of OpenSearch and LanceDB for vector search.

Runs the same queries against both systems and compares:
- Query latency (p50/p95/p99 over --runs repeated measurements)
- Result quality (overlap in top-k)
- Storage model (references vs inline data)
- Operational complexity

DEPLOYMENT NOTE: This benchmark compares two fundamentally different deployment
models. OpenSearch runs as a separate service (Docker container, JVM, HTTP REST
API over localhost). LanceDB runs embedded in-process with data on local disk.
A remote LanceDB instance writing to S3 would show materially different ingestion
and query latency. See the README for context on what these numbers do and don't
tell you.
"""

import time
from pathlib import Path
from typing import Optional

import lancedb
import numpy as np
import pandas as pd
import torch
import typer
from opensearchpy import OpenSearch
from transformers import AutoModel, AutoProcessor

from src.load_lancedb import connect_lancedb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LANCEDB_DIR = DATA_DIR / "lancedb"

DEFAULT_RUNS = 30
DEFAULT_WARMUP = 5

app = typer.Typer()


def get_opensearch():
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )


def get_lancedb_table(
    storage_uri: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: str = "us-east-1",
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
):
    db, _, _ = connect_lancedb(storage_uri, endpoint_url, region, access_key_id, secret_access_key)
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


def _latency_stats(times_s: list[float]) -> dict:
    arr = np.array(times_s) * 1000  # convert to ms
    return {
        "mean": float(np.mean(arr)),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "max":  float(np.max(arr)),
        "n":    len(arr),
    }


def compare_query(
    os_client,
    lance_table,
    query_vector: list[float],
    query_label: str,
    k: int = 5,
    runs: int = DEFAULT_RUNS,
    warmup: int = DEFAULT_WARMUP,
):
    """Run the same query against both systems repeatedly and report latency stats."""

    # Warmup — results discarded
    for _ in range(warmup):
        search_opensearch(os_client, query_vector, k)
        search_lancedb(lance_table, query_vector, k)

    os_all: list[float] = []
    lance_all: list[float] = []

    # Capture result sets from first measurement run for display
    os_time_0, os_hits = search_opensearch(os_client, query_vector, k)
    lance_time_0, lance_results = search_lancedb(lance_table, query_vector, k)
    os_all.append(os_time_0)
    lance_all.append(lance_time_0)

    for _ in range(runs - 1):
        ot, _ = search_opensearch(os_client, query_vector, k)
        lt, _ = search_lancedb(lance_table, query_vector, k)
        os_all.append(ot)
        lance_all.append(lt)

    os_stats = _latency_stats(os_all)
    lance_stats = _latency_stats(lance_all)

    typer.secho(f"\nQuery: {query_label}", bold=True)
    typer.echo(f"{'Rank':<6} {'OpenSearch':<60} {'LanceDB':<60}")
    typer.echo("-" * 126)

    os_ids = []
    lance_ids = []

    for i in range(k):
        if i < len(os_hits):
            src = os_hits[i]["_source"]
            os_ids.append(src["image_id"])
            os_col = f"id={src['image_id']:<8} score={os_hits[i]['_score']:.4f}  {src['caption'][:30]}..."
        else:
            os_col = "-"

        if i < len(lance_results):
            row = lance_results.iloc[i]
            lance_ids.append(row["image_id"])
            img_size = len(row["image_bytes"]) if row["image_bytes"] else 0
            lance_col = f"id={row['image_id']:<8} dist={row['_distance']:.4f}  {row['caption'][:20]}... ({img_size:,}b inline)"
        else:
            lance_col = "-"

        typer.echo(f"{i + 1:<6} {os_col:<60} {lance_col:<60}")

    overlap = set(os_ids) & set(lance_ids)
    typer.echo(f"  Top-{k} overlap: {len(overlap)}/{k} matching image IDs")
    typer.echo(
        f"  Latency ({runs} runs, {warmup} warmup) — "
        f"OpenSearch: mean={os_stats['mean']:.1f}ms  p50={os_stats['p50']:.1f}ms  "
        f"p95={os_stats['p95']:.1f}ms  p99={os_stats['p99']:.1f}ms | "
        f"LanceDB: mean={lance_stats['mean']:.1f}ms  p50={lance_stats['p50']:.1f}ms  "
        f"p95={lance_stats['p95']:.1f}ms  p99={lance_stats['p99']:.1f}ms"
    )

    return os_stats, lance_stats


@app.command()
def main(
    runs: int = typer.Option(DEFAULT_RUNS, help="Number of timed query repetitions per query."),
    warmup: int = typer.Option(DEFAULT_WARMUP, help="Warmup iterations before timing starts."),
    k: int = typer.Option(5, help="Number of results to retrieve per query."),
    lancedb_uri: Optional[str] = typer.Option(
        None,
        "--lancedb-uri",
        help=(
            "LanceDB storage URI. Defaults to local data/lancedb. "
            "Use 's3://bucket/path' to benchmark against S3 or DigitalOcean Spaces."
        ),
    ),
    endpoint_url: Optional[str] = typer.Option(
        None,
        "--endpoint-url",
        envvar="AWS_ENDPOINT_URL",
        help="Custom S3-compatible endpoint URL (e.g. https://nyc3.digitaloceanspaces.com).",
    ),
    region: str = typer.Option(
        "us-east-1",
        "--region",
        envvar="AWS_DEFAULT_REGION",
        help="Storage region.",
    ),
    access_key_id: Optional[str] = typer.Option(
        None, "--access-key-id", envvar="AWS_ACCESS_KEY_ID",
        help="S3 / Spaces access key ID.",
    ),
    secret_access_key: Optional[str] = typer.Option(
        None, "--secret-access-key", envvar="AWS_SECRET_ACCESS_KEY",
        help="S3 / Spaces secret access key.",
    ),
):
    """Run the same queries against OpenSearch and LanceDB, comparing results.

    By default, LanceDB reads from local disk. To benchmark against a remote
    object store (S3, DigitalOcean Spaces) pass --lancedb-uri:

      uv run python -m src.cli compare \\
        --lancedb-uri s3://my-space/coco \\
        --endpoint-url https://nyc3.digitaloceanspaces.com \\
        --region nyc3
    """
    is_remote = bool(lancedb_uri and (
        lancedb_uri.startswith("s3://") or
        lancedb_uri.startswith("gs://") or
        lancedb_uri.startswith("az://")
    ))
    lance_backend_label = f"remote ({lancedb_uri})" if is_remote else "embedded, local disk"

    typer.secho("=" * 70, bold=True)
    typer.secho("OpenSearch vs LanceDB — Side-by-Side Comparison", bold=True)
    typer.secho("Same CLIP embeddings, same queries, different storage models.", dim=True)
    typer.secho("=" * 70, bold=True)

    # Surface the deployment asymmetry prominently so readers aren't misled.
    typer.secho("\n*** DEPLOYMENT CONTEXT ***", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        f"  OpenSearch: client/server  — Docker container, JVM, HTTP REST API (localhost)\n"
        f"  LanceDB:    {lance_backend_label}\n"
        "\n"
        "  These are different deployment models, not just different implementations.\n"
        "  LanceDB backed by a remote object store will show higher latency than an\n"
        "  embedded local-disk deployment. The cost section models S3 separately.\n"
    )
    typer.secho("=" * 70, bold=True)

    os_client = get_opensearch()
    lance_table = get_lancedb_table(lancedb_uri, endpoint_url, region, access_key_id, secret_access_key)

    df = pd.read_parquet(EMBEDDINGS_DIR / "image_embeddings.parquet")

    all_os_stats: list[dict] = []
    all_lance_stats: list[dict] = []

    # Query 1: Image similarity
    typer.secho("\n1. Image -> Image Search", bold=True)
    os_s, lt_s = compare_query(
        os_client, lance_table,
        df.iloc[42]["vector"],
        f"Similar to '{df.iloc[42]['file_name']}'",
        k=k, runs=runs, warmup=warmup,
    )
    all_os_stats.append(os_s)
    all_lance_stats.append(lt_s)

    # Query 2: Text -> Image search (cross-modal)
    typer.secho("\n2. Text -> Image Search (cross-modal)", bold=True)
    text_queries = [
        "a dog playing in the park",
        "a busy city street at night",
        "food on a kitchen table",
    ]

    for query in text_queries:
        vec = text_to_vector(query)
        os_s, lt_s = compare_query(os_client, lance_table, vec, query, k=k, runs=runs, warmup=warmup)
        all_os_stats.append(os_s)
        all_lance_stats.append(lt_s)

    # Aggregate across all queries
    os_p50s = [s["p50"] for s in all_os_stats]
    os_p95s = [s["p95"] for s in all_os_stats]
    lt_p50s = [s["p50"] for s in all_lance_stats]
    lt_p95s = [s["p95"] for s in all_lance_stats]

    typer.echo("")
    typer.secho("=" * 70, bold=True)
    typer.secho("Performance Summary", bold=True)
    typer.secho("=" * 70, bold=True)
    typer.echo(f"\nQuery latency across {len(all_os_stats)} query types ({runs} runs + {warmup} warmup each):")
    typer.echo(f"  {'':25} {'mean p50':>10} {'mean p95':>10}")
    typer.echo(f"  {'OpenSearch':25} {np.mean(os_p50s):>9.1f}ms {np.mean(os_p95s):>9.1f}ms")
    lancedb_row_label = f"LanceDB ({lance_backend_label})"[:25]
    typer.echo(f"  {lancedb_row_label:25} {np.mean(lt_p50s):>9.1f}ms {np.mean(lt_p95s):>9.1f}ms")

    if not is_remote:
        typer.echo(
            "\n  Note: LanceDB numbers are for embedded local-disk. "
            "Re-run with --lancedb-uri s3://... to benchmark the S3 deployment."
        )

    typer.echo(f"\nStorage Model Comparison:")
    typer.echo(f"  OpenSearch  (client/server):")
    typer.echo(f"    - Vectors + metadata in kNN index (JVM heap)")
    typer.echo(f"    - Images stored externally (S3, CDN, filesystem)")
    typer.echo(f"    - Requires Docker/server infrastructure")
    typer.echo(f"    - Image retrieval = separate HTTP call after search")
    typer.echo(f"  LanceDB     ({lance_backend_label}):")
    typer.echo(f"    - Vectors + metadata + images in one Lance table")
    typer.echo(f"    - No server — embedded, files on disk or object store")
    typer.echo(f"    - Image bytes returned with search results")


if __name__ == "__main__":
    app()
