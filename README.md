# OpenSearch → LanceDB Migration Demo

A hands-on project comparing **OpenSearch** and **LanceDB** for vector search using **SigLIP 2** image embeddings on the COCO 2017 dataset.

**Key takeaway:** LanceDB stores images *inline* with vectors and metadata — no S3, no CDN, no external file server. OpenSearch only stores references.

## Architecture

```
COCO 2017 Images
       │
       ▼
 SigLIP 2 (so400m-patch14-384)
       │
       └── image embeddings (N × 1152)
               │
       ┌───────┴────────┐
       ▼                ▼
   OpenSearch         LanceDB
   (Docker)           (local files)
   - vectors          - vectors
   - metadata         - metadata
   - image REFS       - image BYTES ← the difference
       │                    ▲
       └────────────────────┘
         migration script
```

## COCO 2017 Dataset Splits

| Split | Images | Captions |
|-------|--------|----------|
| train | 118K | Yes |
| val | 5K | Yes |
| test | 41K | No |
| unlabeled | 123K | No |

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/) or [Podman](https://podman.io/) (for OpenSearch)

### 1. Install dependencies

```bash
uv sync
```

### 2. Download COCO 2017 dataset

Downloads the official image zip archives from `images.cocodataset.org` and extracts them locally.

```bash
# Just val for a quick test (~1 GB, 5k images)
uv run python -m src.cli download --split val

# Train only (~18 GB, 118k images)
uv run python -m src.cli download --split train

# All splits (~45 GB, 287k images)
uv run python -m src.cli download
```

### 3. Generate SigLIP 2 embeddings

```bash
uv run python -m src.cli embed
```

Generates image embeddings using [Google's SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-384) (SoViT-400M, 384px, 1152-dim). Supports **checkpointing** — if interrupted, re-run the same command and it resumes from where it left off. Use `--force` to start fresh.

### 4. Start OpenSearch

```bash
# Docker
docker compose up -d

# Or Podman
podman compose up -d
```

Wait a few seconds for it to be ready:

```bash
curl -s http://localhost:9200 | python -m json.tool
```

To check index status at any time:

```bash
curl -s http://localhost:9200/_cat/indices?v
```

### 5. Load into OpenSearch

```bash
uv run python -m src.cli opensearch
```

### 6. Load into LanceDB

```bash
uv run python -m src.cli lancedb
```

### 7. Compare them side by side

```bash
uv run python -m src.cli compare
```

### 8. Run the migration

Migrate a live OpenSearch index to LanceDB, pulling images inline:

```bash
uv run python -m src.cli migrate
```

### 9. Upload to Hugging Face

Upload the LanceDB table to a Hugging Face dataset repo. Requires `huggingface-cli login` first.

```bash
# Upload LanceDB files
uv run python -m src.cli upload username/my-dataset

# Private repo
uv run python -m src.cli upload username/my-dataset --private
```

### 10. Launch the Streamlit app

Search for similar images interactively:

```bash
uv run streamlit run src/app.py
```

### CLI help

```bash
uv run python -m src.cli --help
```

## Subcommands

| Command | Purpose |
|---|---|
| `download` | Downloads COCO 2017 images and captions |
| `embed` | Generates SigLIP 2 image embeddings (with checkpointing) |
| `opensearch` | Creates kNN index and bulk-loads into OpenSearch |
| `lancedb` | Creates LanceDB table with inline image storage |
| `compare` | Runs same queries against both, compares latency and results |
| `migrate` | Live migration from OpenSearch → LanceDB |
| `cost` | Estimates hourly AWS cost for OpenSearch vs LanceDB |
| `upload` | Uploads LanceDB table to Hugging Face |

## OpenSearch vs LanceDB — Key Differences

| | OpenSearch | LanceDB |
|---|---|---|
| **Infrastructure** | Docker container, JVM, REST API | Embedded library, just files on disk |
| **Image storage** | External (S3/CDN/filesystem path) | Inline binary blobs in Lance table |
| **Setup** | `docker compose up`, index mapping, bulk API | `pip install lancedb`, `db.create_table()` |
| **Query** | REST API with kNN query DSL | Python method call: `table.search(vec)` |
| **Scaling** | Horizontal (add nodes) | Disk-based, columnar (Lance format) |
| **Self-contained** | No — images live elsewhere | Yes — data, vectors, images all together |

## Cleanup

```bash
docker compose down -v   # Or: podman compose down -v
rm -rf data/             # Remove downloaded images, embeddings, LanceDB files
```
