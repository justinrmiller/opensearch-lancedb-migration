"""Load CLIP embeddings into OpenSearch with kNN index.

Demonstrates the OpenSearch approach:
- Vectors are stored in a kNN index
- Images are referenced by URL/path (NOT stored in OpenSearch)
- Requires a running OpenSearch instance (see docker-compose.yml)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from opensearchpy import OpenSearch, helpers

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
INDEX_NAME = "coco-clip-embeddings"

app = typer.Typer()


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )


def create_index(client: OpenSearch, dim: int):
    """Create a kNN index optimized for CLIP embeddings."""
    if client.indices.exists(index=INDEX_NAME):
        typer.secho(f"Index '{INDEX_NAME}' already exists. Deleting...", fg=typer.colors.YELLOW)
        client.indices.delete(index=INDEX_NAME)

    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 256,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 256,
                            "m": 16,
                        },
                    },
                },
                "image_id": {"type": "integer"},
                "file_name": {"type": "keyword"},
                "caption": {"type": "text"},
                # NOTE: OpenSearch stores a *reference* to the image, not the image itself.
                # The actual image must be served from a separate storage system
                # (S3, CDN, local filesystem, etc.)
                "image_path": {"type": "keyword"},
                "coco_url": {"type": "keyword"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
            }
        },
    }

    client.indices.create(index=INDEX_NAME, body=body)
    typer.secho(f"Created index '{INDEX_NAME}' (dim={dim})", fg=typer.colors.GREEN)


def bulk_index(client: OpenSearch, df: pd.DataFrame):
    """Bulk-index image embeddings."""

    def generate_actions():
        for _, row in df.iterrows():
            yield {
                "_index": INDEX_NAME,
                "_id": f"img_{row['image_id']}",
                "_source": {
                    "embedding": row["vector"],
                    "image_id": int(row["image_id"]),
                    "file_name": row["file_name"],
                    "caption": row["caption"],
                    "image_path": f"data/coco_images/{row['file_name']}",
                    "coco_url": row["coco_url"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                },
            }

    typer.echo("Indexing into OpenSearch...")
    start = time.time()

    success, errors = helpers.bulk(client, generate_actions(), chunk_size=500, request_timeout=120)

    elapsed = time.time() - start
    typer.secho(f"Indexed {success} documents in {elapsed:.1f}s", fg=typer.colors.GREEN)
    if errors:
        typer.secho(f"Errors: {len(errors)}", fg=typer.colors.RED)

    # Force refresh so data is searchable
    client.indices.refresh(index=INDEX_NAME)


def search(client: OpenSearch, query_vector: list[float], k: int = 5) -> list[dict]:
    """Run a kNN search against the index."""
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }
    resp = client.search(index=INDEX_NAME, body=body)
    return resp["hits"]["hits"]


def demo_search(client: OpenSearch, df: pd.DataFrame):
    """Run a sample search to verify the index works."""
    typer.secho("\n--- Sample kNN Search (query: first image embedding) ---", bold=True)

    query_vec = df.iloc[0]["vector"]
    results = search(client, query_vec, k=5)

    typer.echo(f"{'Rank':<6} {'Score':<10} {'Image ID':<12} {'Caption':<50} {'Image Reference'}")
    typer.echo("-" * 130)
    for i, hit in enumerate(results, 1):
        src = hit["_source"]
        caption = src["caption"][:48] + "..." if len(src["caption"]) > 48 else src["caption"]
        typer.echo(
            f"{i:<6} {hit['_score']:<10.4f} {src['image_id']:<12} {caption:<50} {src['image_path']}"
        )

    typer.echo(
        "\nNote: OpenSearch stores image paths/URLs as references. "
        "The actual images must be hosted separately (S3, CDN, filesystem)."
    )


@app.command()
def main():
    """Load CLIP embeddings into an OpenSearch kNN index."""
    df = pd.read_parquet(EMBEDDINGS_DIR / "image_embeddings.parquet")
    dim = len(df.iloc[0]["vector"])

    typer.echo(f"Loaded {len(df)} rows from parquet (vector dim={dim})")

    client = get_client()

    create_index(client, dim)
    bulk_index(client, df)
    demo_search(client, df)

    count = client.count(index=INDEX_NAME)["count"]
    typer.secho(f"\nOpenSearch index '{INDEX_NAME}' has {count} documents.", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
