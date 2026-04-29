"""Load CLIP embeddings into LanceDB.

Demonstrates the LanceDB approach:
- Vectors are stored alongside metadata in a single Lance table
- Images are stored DIRECTLY in the table as binary blobs
- Works with local disk (default) or S3-compatible object stores
  (AWS S3, DigitalOcean Spaces, MinIO, etc.)

DigitalOcean Spaces quick-start:
  export AWS_ACCESS_KEY_ID=<spaces-key>
  export AWS_SECRET_ACCESS_KEY=<spaces-secret>
  uv run python -m src.cli lancedb \\
    --storage-uri s3://my-space/coco \\
    --endpoint-url https://sfo3.digitaloceanspaces.com \\
    --region sfo3
"""

import math
import os
import time
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow.parquet as pq
import typer
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LANCEDB_DIR = DATA_DIR / "lancedb"

TABLE_NAME = "coco_clip_embeddings"

app = typer.Typer()

BATCH_SIZE = 4096


def _build_storage_options(
    endpoint_url: Optional[str],
    region: str,
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
) -> dict:
    """Build the storage_options dict passed to lancedb.connect() for S3-compatible stores."""
    opts: dict = {}
    if endpoint_url:
        opts["endpoint"] = endpoint_url
    if access_key_id:
        opts["aws_access_key_id"] = access_key_id
    if secret_access_key:
        opts["aws_secret_access_key"] = secret_access_key
    if region:
        opts["aws_region"] = region
    return opts


def connect_lancedb(
    storage_uri: Optional[str],
    endpoint_url: Optional[str],
    region: str,
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
):
    """Connect to LanceDB at the given URI (local path or s3://)."""
    uri = storage_uri or str(LANCEDB_DIR)
    is_remote = uri.startswith("s3://") or uri.startswith("gs://") or uri.startswith("az://")

    if is_remote:
        opts = _build_storage_options(endpoint_url, region, access_key_id, secret_access_key)
        typer.echo(f"Connecting to LanceDB at {uri}")
        if endpoint_url:
            typer.echo(f"  Endpoint: {endpoint_url}  Region: {region}")
        return lancedb.connect(uri, storage_options=opts if opts else None), uri, True

    Path(uri).mkdir(parents=True, exist_ok=True)
    typer.echo(f"Connecting to LanceDB at {uri}  (local disk)")
    return lancedb.connect(uri), uri, False


def create_table(db, parquet_path: Path, is_remote: bool):
    """Stream parquet into a LanceDB table in fixed-size batches."""
    try:
        db.drop_table(TABLE_NAME)
        typer.secho(f"Dropped existing table '{TABLE_NAME}'", fg=typer.colors.YELLOW)
    except Exception:
        pass

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows

    typer.echo(f"Inserting {total_rows:,} records into LanceDB in batches of {BATCH_SIZE:,}...")
    start = time.time()

    table = None
    with tqdm(total=total_rows, desc="Loading", unit="row") as pbar:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE):
            # Convert to pandas to avoid PyArrow list child field naming
            # mismatch ("item" vs "element") when LanceDB casts to
            # fixed_size_list internally.
            batch_df = batch.to_pandas()
            if table is None:
                table = db.create_table(TABLE_NAME, data=batch_df)
            else:
                table.add(batch_df)
            pbar.update(len(batch_df))

    elapsed = time.time() - start
    rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
    backend = "object store (remote)" if is_remote else "local disk"
    typer.secho(
        f"Inserted {total_rows:,} records in {elapsed:.1f}s  ({rows_per_sec:,.0f} rows/s)  [{backend}]",
        fg=typer.colors.GREEN,
    )
    if is_remote:
        typer.echo(
            "  ^ Remote write rate is bounded by object-store PUT latency.\n"
            "  Local disk ingestion is typically 3-10× faster."
        )
    else:
        typer.echo(
            "  ^ Local disk ingestion. S3-backed LanceDB will be slower for writes\n"
            "    due to object-store PUT latency."
        )

    return table, total_rows


def create_hnsw_index(table, num_rows: int, vector_dim: int):
    """Create an IVF_HNSW_SQ index with parameters tuned to the table stats.

    Heuristics:
    - num_partitions: 1 for tables under 1M rows (single HNSW graph is
      optimal), otherwise sqrt(num_rows) to keep partition sizes manageable.
    - m: 20 (default) for ≤100k rows, 32 for larger tables where extra
      graph connectivity improves recall.
    - ef_construction: 300 (default) is a good balance; bump to 400 for
      tables over 500k rows.
    - metric: cosine, since embeddings are L2-normalized.
    """
    if num_rows < 1_000_000:
        num_partitions = 1
    else:
        num_partitions = max(1, int(math.sqrt(num_rows)))

    m = 32 if num_rows > 100_000 else 20
    ef_construction = 400 if num_rows > 500_000 else 300

    typer.echo(f"\n--- Creating IVF_HNSW_SQ index ---")
    typer.echo(f"  rows={num_rows:,}  dim={vector_dim}  metric=cosine")
    typer.echo(f"  num_partitions={num_partitions}  m={m}  ef_construction={ef_construction}")

    start = time.time()
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_HNSW_SQ",
        num_partitions=num_partitions,
        m=m,
        ef_construction=ef_construction,
        replace=True,
    )
    elapsed = time.time() - start
    typer.secho(f"  Index created in {elapsed:.1f}s", fg=typer.colors.GREEN)


def demo_search(table):
    """Run a sample vector search using the first row as the query."""
    typer.secho("\n--- Sample Vector Search (query: first image embedding) ---", bold=True)

    sample = table.head(1).to_pandas()
    query_vec = sample.iloc[0]["vector"]

    start = time.time()
    results = table.search(query_vec).limit(5).to_pandas()
    elapsed = time.time() - start

    typer.secho(f"LanceDB Results ({elapsed*1000:.1f}ms)", bold=True)
    typer.echo(f"{'Rank':<6} {'Distance':<10} {'Image ID':<12} {'Caption':<50} {'Image Data'}")
    typer.echo("-" * 130)
    for i, (_, row) in enumerate(results.iterrows(), 1):
        img_size = len(row["image_bytes"]) if row["image_bytes"] else 0
        caption = str(row["caption"])
        caption = caption[:48] + "..." if len(caption) > 48 else caption
        typer.echo(
            f"{i:<6} {row['_distance']:<10.4f} {row['image_id']:<12} {caption:<50} {img_size:,} bytes inline"
        )

    typer.secho(
        "\nKey advantage: The image bytes are stored directly in LanceDB. "
        "No S3, no CDN, no external file server — the images live with the vectors and metadata.",
        fg=typer.colors.GREEN,
    )


@app.command()
def main(
    storage_uri: Optional[str] = typer.Option(
        None,
        "--storage-uri",
        help=(
            "LanceDB storage URI. Defaults to local data/lancedb. "
            "Use 's3://bucket/path' for S3 or DigitalOcean Spaces."
        ),
    ),
    endpoint_url: Optional[str] = typer.Option(
        None,
        "--endpoint-url",
        envvar="AWS_ENDPOINT_URL",
        help=(
            "Custom S3-compatible endpoint URL. "
            "For DigitalOcean Spaces use e.g. https://sfo3.digitaloceanspaces.com"
        ),
    ),
    region: str = typer.Option(
        "us-east-1",
        "--region",
        envvar="AWS_DEFAULT_REGION",
        help="Storage region (e.g. sfo3 for DigitalOcean Spaces, us-east-1 for AWS).",
    ),
    access_key_id: Optional[str] = typer.Option(
        None,
        "--access-key-id",
        envvar="AWS_ACCESS_KEY_ID",
        help="S3 / Spaces access key ID. Defaults to AWS_ACCESS_KEY_ID env var.",
    ),
    secret_access_key: Optional[str] = typer.Option(
        None,
        "--secret-access-key",
        envvar="AWS_SECRET_ACCESS_KEY",
        help="S3 / Spaces secret access key. Defaults to AWS_SECRET_ACCESS_KEY env var.",
    ),
):
    """Load CLIP embeddings and images into a LanceDB table.

    Supports local disk (default) and S3-compatible object stores including
    AWS S3 and DigitalOcean Spaces.

    Examples:

      # Local disk (default)
      uv run python -m src.cli lancedb

      # AWS S3
      uv run python -m src.cli lancedb --storage-uri s3://my-bucket/coco

      # DigitalOcean Spaces (sfo3 datacenter)
      export AWS_ACCESS_KEY_ID=<key>
      export AWS_SECRET_ACCESS_KEY=<secret>
      uv run python -m src.cli lancedb \\
        --storage-uri s3://my-space/coco \\
        --endpoint-url https://sfo3.digitaloceanspaces.com \\
        --region sfo3
    """
    overall_start = time.time()

    parquet_path = EMBEDDINGS_DIR / "image_embeddings.parquet"
    if not parquet_path.exists():
        typer.secho(f"Embeddings not found: {parquet_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    pf = pq.ParquetFile(parquet_path)
    typer.echo(f"Found {pf.metadata.num_rows:,} rows in {parquet_path.name}")

    db, resolved_uri, is_remote = connect_lancedb(
        storage_uri, endpoint_url, region, access_key_id, secret_access_key
    )

    table, total_rows = create_table(db, parquet_path, is_remote)

    sample = table.head(1).to_pandas()
    vector_dim = len(sample.iloc[0]["vector"])
    create_hnsw_index(table, num_rows=total_rows, vector_dim=vector_dim)

    demo_search(table)

    row_count = table.count_rows()
    typer.secho(f"\nLanceDB table '{TABLE_NAME}' has {row_count:,} rows.", fg=typer.colors.GREEN, bold=True)

    if not is_remote:
        lance_size = sum(f.stat().st_size for f in Path(resolved_uri).rglob("*") if f.is_file())
        typer.echo(f"LanceDB total size on disk: {lance_size / 1024 / 1024:.1f} MB (includes images + vectors + metadata)")

    overall_elapsed = time.time() - overall_start
    typer.secho(f"Total time: {overall_elapsed:.1f}s", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
