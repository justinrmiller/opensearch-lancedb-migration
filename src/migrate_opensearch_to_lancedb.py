"""Migrate a live OpenSearch kNN index to LanceDB.

This script connects to a running OpenSearch instance, scrolls through all
documents in the CLIP embeddings index, and writes them into a LanceDB table —
optionally pulling in the actual image files so they live inline with the data.

This demonstrates a realistic migration path:
  OpenSearch (vectors + metadata + image *references*)
      → LanceDB (vectors + metadata + image *bytes*)
"""

import json
import time
from pathlib import Path

import lancedb
import typer
from opensearchpy import OpenSearch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "coco_images"
LANCEDB_DIR = DATA_DIR / "lancedb_migrated"
CHECKPOINT_FILE = DATA_DIR / "migration_checkpoint.json"

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
SOURCE_INDEX = "coco-clip-embeddings"
TARGET_TABLE = "coco_clip_migrated"

PAGE_SIZE = 500  # Documents fetched per OpenSearch request
WRITE_BATCH_SIZE = 500  # Rows written to LanceDB per batch (caps peak RAM usage)

app = typer.Typer()


def get_opensearch_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )


def resolve_image_bytes(doc: dict) -> bytes:
    """Try to load the actual image bytes from the local path reference.

    In a production migration, you might pull from S3 or a CDN instead.
    This shows the key difference: OpenSearch only had a *reference* to the image,
    but LanceDB can store the actual image data.
    """
    image_path = doc.get("image_path", "")
    if image_path:
        # The OpenSearch doc stores a relative path like "data/coco_images/000000001234.jpg"
        full_path = Path(__file__).resolve().parent.parent / image_path
        if full_path.exists():
            return full_path.read_bytes()
    return b""


def doc_to_record(doc: dict) -> dict:
    """Transform an OpenSearch document into a LanceDB record."""
    return {
        "vector": doc["embedding"],
        "image_id": doc["image_id"],
        "file_name": doc["file_name"],
        "caption": doc["caption"],
        "coco_url": doc.get("coco_url", ""),
        "width": doc.get("width", 0),
        "height": doc.get("height", 0),
        # The migration enrichment: actual image bytes now live with the data
        "image_bytes": resolve_image_bytes(doc),
    }


def fetch_page(client: OpenSearch, search_after: str | None) -> tuple[list[dict], str | None]:
    """Fetch one page of documents using search_after for resumable pagination.

    Unlike the scroll API, search_after does not hold server-side cursor state,
    so it survives process restarts and can be checkpointed reliably.
    """
    body: dict = {
        "query": {"match_all": {}},
        "size": PAGE_SIZE,
        "sort": [{"_id": "asc"}],
    }
    if search_after is not None:
        body["search_after"] = [search_after]

    resp = client.search(index=SOURCE_INDEX, body=body)
    hits = resp["hits"]["hits"]
    if not hits:
        return [], None
    last_sort_value = hits[-1]["sort"][0]
    return [h["_source"] for h in hits], last_sort_value


def load_checkpoint() -> dict | None:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return None


def save_checkpoint(docs_written: int, last_sort_value: str | None) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"docs_written": docs_written, "last_sort_value": last_sort_value})
    )


def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def migrate_streaming(
    client: OpenSearch,
    db,
    total: int,
    resume_from: int,
    last_sort_value: str | None,
) -> tuple:
    """Stream documents from OpenSearch and write to LanceDB in batches.

    Only WRITE_BATCH_SIZE records are held in memory at a time, so peak RAM
    is proportional to batch size rather than total dataset size.
    After each batch is committed, the checkpoint is updated — a restart
    resumes from the last committed batch rather than from the beginning.
    """
    docs_written = resume_from
    images_resolved = 0
    table = None

    if resume_from > 0:
        try:
            table = db.open_table(TARGET_TABLE)
            typer.echo(f"Resuming: table already has {table.count_rows()} rows")
        except Exception:
            typer.secho(
                "Warning: checkpoint exists but table not found — restarting from scratch.",
                fg=typer.colors.YELLOW,
            )
            docs_written = 0
            last_sort_value = None

    with typer.progressbar(length=total, label="Migrating documents") as progress:
        if docs_written > 0:
            progress.update(docs_written)

        while True:
            hits, last_sort_value = fetch_page(client, last_sort_value)
            if not hits:
                break

            batch = [doc_to_record(doc) for doc in hits]
            images_resolved += sum(1 for r in batch if r["image_bytes"])

            if table is None:
                table = db.create_table(TARGET_TABLE, data=batch)
            else:
                table.add(batch)

            docs_written += len(batch)
            save_checkpoint(docs_written, last_sort_value)
            progress.update(len(batch))

    return table, docs_written, images_resolved


def verify_migration(table, client: OpenSearch) -> None:
    """Verify the migrated data by running a sample search."""
    typer.secho("\n--- Migration Verification ---", bold=True)

    resp = client.search(index=SOURCE_INDEX, body={"query": {"match_all": {}}, "size": 1})
    hits = resp["hits"]["hits"]
    if not hits:
        typer.echo("No documents available to verify against.")
        return

    query_vec = hits[0]["_source"]["embedding"]
    results = table.search(query_vec).limit(3).to_pandas()

    typer.echo(f"Sample search returned {len(results)} results:")
    for _, row in results.iterrows():
        img_size = len(row["image_bytes"]) if row["image_bytes"] else 0
        status = "image inline" if img_size > 0 else "no image"
        typer.echo(
            f"  id={row['image_id']:>6d} | "
            f"{status} ({img_size:,} bytes) | {row['caption'][:50]}..."
        )

    lance_size = sum(f.stat().st_size for f in Path(LANCEDB_DIR).rglob("*") if f.is_file())
    typer.echo(f"\nLanceDB migrated size: {lance_size / 1024 / 1024:.1f} MB")


@app.command()
def main(
    force: bool = typer.Option(
        False, "--force", help="Drop existing table and checkpoint, restart from scratch."
    ),
) -> None:
    """Migrate a live OpenSearch kNN index to LanceDB with inline image storage."""
    typer.secho("OpenSearch -> LanceDB Migration\n", bold=True)

    client = get_opensearch_client()
    total = client.count(index=SOURCE_INDEX)["count"]
    typer.echo(f"Found {total} documents in OpenSearch index '{SOURCE_INDEX}'")

    if not total:
        typer.secho(
            "No documents found in OpenSearch. Run load_opensearch.py first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Warn upfront if images are missing so the user isn't surprised by empty image_bytes
    image_count = sum(1 for _ in IMAGES_DIR.glob("*.jpg")) if IMAGES_DIR.exists() else 0
    if image_count == 0:
        typer.secho(
            "Warning: No images found in data/coco_images/. Documents will be migrated "
            "without inline image bytes. Run download_coco.py first to include images.",
            fg=typer.colors.YELLOW,
        )

    db = lancedb.connect(str(LANCEDB_DIR))
    checkpoint = load_checkpoint()

    if force:
        try:
            db.drop_table(TARGET_TABLE)
            typer.secho(f"Dropped existing table '{TARGET_TABLE}'", fg=typer.colors.YELLOW)
        except Exception:
            pass
        clear_checkpoint()
        checkpoint = None
    elif checkpoint:
        typer.echo(
            f"Resuming from checkpoint: {checkpoint['docs_written']} docs already written. "
            "Use --force to restart from scratch."
        )

    resume_from = checkpoint["docs_written"] if checkpoint else 0
    last_sort_value = checkpoint["last_sort_value"] if checkpoint else None

    start = time.time()
    table, docs_written, images_resolved = migrate_streaming(
        client, db, total, resume_from, last_sort_value
    )
    elapsed = time.time() - start

    typer.secho(f"\nWrote {docs_written} rows to LanceDB in {elapsed:.1f}s", fg=typer.colors.GREEN)
    typer.echo(f"Resolved {images_resolved}/{docs_written} image files to inline bytes")

    if image_count == 0 and docs_written > 0:
        typer.secho(
            "Reminder: Re-run after downloading images (download_coco.py) to populate inline bytes.",
            fg=typer.colors.YELLOW,
        )

    clear_checkpoint()

    # Step 3: Verify
    verify_migration(table, client)

    typer.secho("\nMigration complete!", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        "\nWhat changed: OpenSearch stored image *paths* — the images had to live elsewhere. "
        "LanceDB now stores the actual image bytes inline, so your data is fully self-contained. "
        "No S3 bucket, no CDN, no file server needed."
    )


if __name__ == "__main__":
    app()
