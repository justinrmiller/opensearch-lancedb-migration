"""Generate SigLIP 2 image embeddings for COCO images.

Outputs a single parquet file containing vectors, metadata, and image bytes.
Supports checkpointing: vectors are saved every CHECKPOINT_INTERVAL batches,
so a crash or interruption resumes from the last checkpoint instead of starting over.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from PIL import Image
from transformers import AutoModel, AutoProcessor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "coco_images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

MODEL_ID = "google/siglip2-so400m-patch14-384"
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 50  # save checkpoint every N batches (~1600 images)

CHECKPOINT_FILE = EMBEDDINGS_DIR / "checkpoint.json"
PARTIAL_FILE = EMBEDDINGS_DIR / "image_embeddings_partial.npy"
FINAL_FILE = EMBEDDINGS_DIR / "image_embeddings.parquet"

app = typer.Typer()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str):
    """Load SigLIP 2 model and processor."""
    typer.echo(f"Loading {MODEL_ID} on {device}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def load_checkpoint() -> tuple[int, np.ndarray | None]:
    """Load checkpoint state. Returns (images_completed, partial_embeddings)."""
    if not CHECKPOINT_FILE.exists():
        return 0, None

    with open(CHECKPOINT_FILE) as f:
        state = json.load(f)

    completed = state.get("images_completed", 0)
    if completed > 0 and PARTIAL_FILE.exists():
        partial = np.load(PARTIAL_FILE)
        return completed, partial

    return 0, None


def save_checkpoint(images_completed: int, embeddings: np.ndarray):
    """Save checkpoint: partial embeddings + progress counter."""
    np.save(PARTIAL_FILE, embeddings)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"images_completed": images_completed}, f)


def clear_checkpoint():
    """Remove checkpoint files after successful completion."""
    for f in (CHECKPOINT_FILE, PARTIAL_FILE):
        if f.exists():
            f.unlink()


def generate_image_embeddings(
    model, processor, metadata: list[dict], device: str,
    resume_from: int = 0, prior_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Generate embeddings for all images in batches with periodic checkpointing."""
    all_embeddings = []
    if prior_embeddings is not None:
        all_embeddings.append(prior_embeddings)

    remaining = metadata[resume_from:]
    batches_since_checkpoint = 0
    images_done = resume_from

    with typer.progressbar(range(0, len(remaining), BATCH_SIZE), label="Embedding images") as progress:
        for start_idx in progress:
            batch_meta = remaining[start_idx : start_idx + BATCH_SIZE]
            images = []

            for item in batch_meta:
                img_path = IMAGES_DIR / item["file_name"]
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    typer.echo(f"  Skipping {item['file_name']}: {e}")
                    # Placeholder: 1x1 black image (processor handles resizing)
                    images.append(Image.new("RGB", (1, 1)))

            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                output = model.get_image_features(**inputs)
                image_features = output.pooler_output if hasattr(output, "pooler_output") else output
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu().numpy())
            images_done += len(batch_meta)
            batches_since_checkpoint += 1

            if batches_since_checkpoint >= CHECKPOINT_INTERVAL:
                combined = np.concatenate(all_embeddings, axis=0)
                save_checkpoint(images_done, combined)
                batches_since_checkpoint = 0

    return np.concatenate(all_embeddings, axis=0)


def read_image_bytes(file_name: str) -> bytes:
    """Read image file as raw bytes."""
    img_path = IMAGES_DIR / file_name
    if img_path.exists():
        return img_path.read_bytes()
    return b""


def build_parquet(metadata: list[dict], embeddings: np.ndarray):
    """Combine metadata, embeddings, and image bytes into a parquet file."""
    typer.echo("Reading image files for parquet output...")

    records = []
    with typer.progressbar(range(len(metadata)), label="Building parquet") as progress:
        for i in progress:
            item = metadata[i]
            records.append({
                "image_id": item["image_id"],
                "file_name": item["file_name"],
                "caption": item.get("caption", ""),
                "coco_url": item.get("coco_url", ""),
                "width": item.get("width", 0),
                "height": item.get("height", 0),
                "split": item.get("split", ""),
                "vector": embeddings[i].tolist(),
                "image_bytes": read_image_bytes(item["file_name"]),
            })

    df = pd.DataFrame(records)
    df.to_parquet(FINAL_FILE, index=False)
    file_size_mb = FINAL_FILE.stat().st_size / (1024 * 1024)
    typer.secho(
        f"Saved {len(df)} rows to {FINAL_FILE.name} ({file_size_mb:.1f} MB)",
        fg=typer.colors.GREEN,
    )


@app.command()
def main(
    force: bool = typer.Option(
        False, "--force", help="Discard any existing checkpoint and start from scratch."
    ),
):
    """Generate SigLIP 2 image embeddings for COCO images (with checkpointing)."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        typer.secho("Run download first to get the dataset.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with open(meta_path) as f:
        metadata = json.load(f)

    typer.echo(f"Loaded metadata for {len(metadata)} images")

    # Handle checkpointing
    resume_from = 0
    prior_embeddings = None

    if force:
        clear_checkpoint()
    else:
        resume_from, prior_embeddings = load_checkpoint()
        if resume_from > 0:
            typer.secho(
                f"Resuming from checkpoint: {resume_from}/{len(metadata)} images already embedded. "
                "Use --force to restart from scratch.",
                fg=typer.colors.YELLOW,
            )

    if resume_from >= len(metadata):
        typer.secho("All images already embedded (checkpoint is complete).", fg=typer.colors.GREEN)
        if prior_embeddings is not None:
            build_parquet(metadata, prior_embeddings)
            clear_checkpoint()
        return

    device = get_device()
    model, processor = load_model(device)

    # Generate image embeddings
    remaining = len(metadata) - resume_from
    typer.echo(f"\n--- Generating image embeddings for {remaining} images ({resume_from} already done) ---")
    image_embeddings = generate_image_embeddings(
        model, processor, metadata, device,
        resume_from=resume_from, prior_embeddings=prior_embeddings,
    )

    typer.secho(f"\nEmbeddings complete: {image_embeddings.shape}", fg=typer.colors.GREEN)
    typer.echo(f"Embedding dimension: {image_embeddings.shape[1]}")

    # Build final parquet with vectors + metadata + image bytes
    build_parquet(metadata, image_embeddings)
    clear_checkpoint()


if __name__ == "__main__":
    app()
