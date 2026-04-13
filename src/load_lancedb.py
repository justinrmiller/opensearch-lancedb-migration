"""Load CLIP embeddings into LanceDB.

Demonstrates the LanceDB approach:
- Vectors are stored alongside metadata in a single Lance table
- Images are stored DIRECTLY in the table as binary blobs
- No external infrastructure required — it's just files on disk
- The data, vectors, and images all live together
"""

import math
import time
from pathlib import Path

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


def create_table(parquet_path: Path):
    """Create a LanceDB table by streaming batches from a parquet file.

    Reads the parquet in fixed-size record batches so memory usage stays
    roughly constant regardless of dataset size.
    """
    db = lancedb.connect(str(LANCEDB_DIR))

    # Drop existing table if present
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
    typer.secho(f"Inserted {total_rows:,} records in {elapsed:.1f}s", fg=typer.colors.GREEN)

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

    # Grab a single vector from the table to use as a query
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
def main():
    """Load CLIP embeddings and images into a LanceDB table."""
    overall_start = time.time()

    parquet_path = EMBEDDINGS_DIR / "image_embeddings.parquet"
    if not parquet_path.exists():
        typer.secho(f"Embeddings not found: {parquet_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Read vector dimension from parquet schema without loading data
    pf = pq.ParquetFile(parquet_path)
    typer.echo(f"Found {pf.metadata.num_rows:,} rows in {parquet_path.name}")

    table, total_rows = create_table(parquet_path)

    # Get vector dim from the first row
    sample = table.head(1).to_pandas()
    vector_dim = len(sample.iloc[0]["vector"])
    create_hnsw_index(table, num_rows=total_rows, vector_dim=vector_dim)

    demo_search(table)

    row_count = table.count_rows()
    typer.secho(f"\nLanceDB table '{TABLE_NAME}' has {row_count:,} rows.", fg=typer.colors.GREEN, bold=True)

    # Show storage comparison
    lance_size = sum(f.stat().st_size for f in Path(LANCEDB_DIR).rglob("*") if f.is_file())
    typer.echo(f"LanceDB total size on disk: {lance_size / 1024 / 1024:.1f} MB (includes images + vectors + metadata)")

    overall_elapsed = time.time() - overall_start
    typer.secho(f"Total time: {overall_elapsed:.1f}s", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
