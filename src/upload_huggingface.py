"""Upload LanceDB table to a Hugging Face dataset repo."""

from pathlib import Path

import typer
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
LANCEDB_DIR = DATA_DIR / "lancedb"
DATASET_CARD = PROJECT_DIR / "DATASET_CARD.md"

app = typer.Typer()


def _collect_files(directory: Path) -> list[Path]:
    """Return all files under a directory, sorted by path."""
    return sorted(f for f in directory.rglob("*") if f.is_file())


def _upload_files_with_progress(
    api, files: list[Path], base_dir: Path, repo_id: str, path_prefix: str,
):
    """Upload a list of files one by one with a tqdm progress bar on total bytes."""
    total_bytes = sum(f.stat().st_size for f in files)

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"Uploading {path_prefix}") as pbar:
        for file_path in files:
            rel_path = file_path.relative_to(base_dir)
            path_in_repo = f"{path_prefix}/{rel_path}"
            file_size = file_path.stat().st_size

            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            pbar.update(file_size)


@app.command()
def main(
    repo_id: str = typer.Argument(..., help="Hugging Face repo to upload to (e.g. username/my-dataset)."),
    private: bool = typer.Option(False, "--private", help="Create the repo as private."),
):
    """Upload LanceDB table to a Hugging Face dataset repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        typer.secho("huggingface_hub is not installed. Run: pip install huggingface_hub", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not LANCEDB_DIR.exists():
        typer.secho(f"LanceDB directory not found: {LANCEDB_DIR}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    typer.echo(f"Repo: https://huggingface.co/datasets/{repo_id}")

    # Upload dataset card as README.md
    if DATASET_CARD.exists():
        typer.echo("Uploading dataset card...")
        api.upload_file(
            path_or_fileobj=str(DATASET_CARD),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        typer.secho("  Dataset card uploaded.", fg=typer.colors.GREEN)

    # Upload LanceDB files
    files = _collect_files(LANCEDB_DIR)
    total_size = sum(f.stat().st_size for f in files)
    typer.echo(f"Uploading {len(files)} LanceDB files ({total_size / 1024 / 1024:.1f} MB)...")
    _upload_files_with_progress(api, files, LANCEDB_DIR, repo_id, "lancedb")
    typer.secho("  LanceDB upload complete.", fg=typer.colors.GREEN)

    typer.secho(f"\nDone. View at https://huggingface.co/datasets/{repo_id}", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
