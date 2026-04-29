"""Unified CLI for the OpenSearch-to-LanceDB migration demo."""

import typer

from src.download_coco import main as download_main
from src.embeddings import main as embeddings_main
from src.load_opensearch import main as opensearch_main
from src.load_lancedb import main as lancedb_main
from src.migrate_opensearch_to_lancedb import main as migrate_main
from src.compare import main as compare_main
from src.cost_estimate import main as cost_main
from src.upload_huggingface import main as upload_main

app = typer.Typer(
    name="clip-search",
    help="Compare OpenSearch and LanceDB for CLIP image embedding search.",
    no_args_is_help=True,
)

app.command(name="download", help="Download COCO 2017 images and captions.")(download_main)
app.command(name="embed", help="Generate SigLIP 2 image embeddings.")(embeddings_main)
app.command(name="opensearch", help="Load embeddings into OpenSearch.")(opensearch_main)
app.command(name="lancedb", help="Load embeddings into LanceDB (local disk or S3/DigitalOcean Spaces).")(lancedb_main)
app.command(name="migrate", help="Migrate from OpenSearch to LanceDB.")(migrate_main)
app.command(name="compare", help="Side-by-side comparison of both systems.")(compare_main)
app.command(name="cost", help="Estimate hourly AWS cost for OpenSearch vs LanceDB.")(cost_main)
app.command(name="upload", help="Upload LanceDB table to Hugging Face.")(upload_main)


if __name__ == "__main__":
    app()
