"""Download COCO 2017 images and captions.

Supported splits:
  train     118K images (with captions)
  val         5K images (with captions)
  test       41K images (no captions)
  unlabeled 123K images (no captions)
"""

import json
from pathlib import Path

import requests
import typer
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "coco_images"

IMAGE_ZIPS = {
    "train":     "http://images.cocodataset.org/zips/train2017.zip",
    "val":       "http://images.cocodataset.org/zips/val2017.zip",
    "test":      "http://images.cocodataset.org/zips/test2017.zip",
    "unlabeled": "http://images.cocodataset.org/zips/unlabeled2017.zip",
}

# Each annotation archive and the split files it contains.
ANNOTATION_ARCHIVES = [
    {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "files": [
            {"split": "train", "filename": "captions_train2017.json", "has_captions": True},
            {"split": "val",   "filename": "captions_val2017.json",   "has_captions": True},
        ],
    },
    {
        "url": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
        "files": [
            {"split": "test", "filename": "image_info_test2017.json", "has_captions": False},
        ],
    },
    {
        "url": "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
        "files": [
            {"split": "unlabeled", "filename": "image_info_unlabeled2017.json", "has_captions": False},
        ],
    },
]

ALL_SPLITS = ["train", "val", "test", "unlabeled"]

app = typer.Typer()


def _get_needed_archives(splits: list[str]) -> list[dict]:
    """Return only the archives that contain files matching the requested splits."""
    needed = []
    for archive in ANNOTATION_ARCHIVES:
        matching_files = [f for f in archive["files"] if f["split"] in splits]
        if matching_files:
            needed.append({**archive, "files": matching_files})
    return needed


def download_annotations(splits: list[str]):
    """Download and extract COCO 2017 annotation files for the requested splits."""
    import io
    import zipfile

    archives = _get_needed_archives(splits)

    for archive in archives:
        missing = [
            f for f in archive["files"]
            if not (DATA_DIR / f["filename"]).exists()
        ]
        if not missing:
            for f in archive["files"]:
                typer.echo(f"  {f['filename']} already exists.")
            continue

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = archive["url"]
        typer.echo(f"Downloading {url.split('/')[-1]}...")

        resp = requests.get(url, stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                buf.write(chunk)
                pbar.update(len(chunk))

        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            for f in missing:
                dest = DATA_DIR / f["filename"]
                zip_path = f"annotations/{f['filename']}"
                try:
                    with zf.open(zip_path) as src:
                        dest.write_bytes(src.read())
                    typer.echo(f"  Extracted {f['filename']}")
                except KeyError:
                    typer.secho(f"  {zip_path} not found in archive, skipping.", fg=typer.colors.YELLOW)


def load_image_metadata(splits: list[str], limit: int | None = None) -> list[dict]:
    """Load image metadata from annotation files for the requested splits."""
    seen_ids: set[int] = set()
    results = []

    archives = _get_needed_archives(splits)

    for archive in archives:
        for file_info in archive["files"]:
            fname = file_info["filename"]
            json_path = DATA_DIR / fname
            if not json_path.exists():
                typer.secho(f"  Missing {fname}, skipping.", fg=typer.colors.YELLOW)
                continue

            with open(json_path) as f:
                data = json.load(f)

            caption_map: dict[int, str] = {}
            if file_info["has_captions"]:
                for ann in data["annotations"]:
                    img_id = ann["image_id"]
                    if img_id not in caption_map:
                        caption_map[img_id] = ann["caption"]

            for img in data["images"]:
                img_id = img["id"]
                if img_id in seen_ids:
                    continue
                seen_ids.add(img_id)

                results.append({
                    "image_id": img_id,
                    "file_name": img["file_name"],
                    "coco_url": img.get("coco_url", ""),
                    "caption": caption_map.get(img_id, ""),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "split": file_info["split"],
                })
                if limit and len(results) >= limit:
                    return results

    return results


def download_images(splits: list[str]):
    """Download and extract COCO image zips for the requested splits."""
    import io
    import zipfile

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        url = IMAGE_ZIPS[split]
        zip_name = url.split("/")[-1]

        # Use a sentinel file to track completed extractions per split
        sentinel = IMAGES_DIR / f".extracted_{split}2017"
        if sentinel.exists():
            typer.echo(f"  {zip_name} already extracted.")
            continue

        typer.echo(f"Downloading {zip_name}...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc=zip_name) as pbar:
            for chunk in resp.iter_content(chunk_size=65536):
                buf.write(chunk)
                pbar.update(len(chunk))

        buf.seek(0)
        typer.echo(f"  Extracting {zip_name}...")
        with zipfile.ZipFile(buf) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            for member in tqdm(members, desc="Extracting", unit="file"):
                dest = IMAGES_DIR / Path(member.filename).name
                dest.write_bytes(zf.read(member))
        sentinel.touch()
        typer.echo(f"  Extracted {zip_name}.")


@app.command()
def main(
    split: str = typer.Option(
        "all",
        help="Which split(s): 'train', 'val', 'test', 'unlabeled', or 'all'.",
    ),
):
    """Download COCO 2017 images and captions."""
    if split == "all":
        splits = ALL_SPLITS
    elif split in ALL_SPLITS:
        splits = [split]
    else:
        typer.secho(
            f"Unknown split '{split}'. Use 'train', 'val', 'test', 'unlabeled', or 'all'.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    captionless = [s for s in splits if s in ("test", "unlabeled")]
    if captionless:
        typer.echo(f"Note: {captionless} splits have no captions (image info only).")

    download_annotations(splits)
    download_images(splits)

    metadata = load_image_metadata(splits)
    typer.echo(f"Found {len(metadata)} unique images (splits={splits})")

    meta_path = DATA_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    typer.echo(f"Saved metadata for {len(metadata)} items to {meta_path}")


if __name__ == "__main__":
    app()
