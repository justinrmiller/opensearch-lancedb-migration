"""Hourly and monthly cost estimator: OpenSearch vs LanceDB for vector search workloads.

Two commands:

  cost        — both systems run on dedicated servers (Droplet/EC2)
  cost-direct — LanceDB reads directly from object storage (S3/Spaces) with no
                dedicated server; compute is absorbed by the calling process

OpenSearch always requires a dedicated server (JVM heap must stay resident).
LanceDB is an embedded library — you can connect("s3://...") from any process
and skip the dedicated server entirely.

Coverage: AWS (us-east-1) and DigitalOcean (sfo3).
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True)


class Provider(str, Enum):
    aws = "aws"
    do = "do"
    all = "all"


# ---------------------------------------------------------------------------
# AWS Pricing (us-east-1 on-demand, approximate early 2025)
# ---------------------------------------------------------------------------

@dataclass
class OSInstance:
    name: str
    vcpu: int
    ram_gb: float
    price_per_hr: float

OPENSEARCH_INSTANCES = [
    OSInstance("r6g.large.search",    2,   16,  0.167),
    OSInstance("r6g.xlarge.search",   4,   32,  0.334),
    OSInstance("r6g.2xlarge.search",  8,   64,  0.668),
    OSInstance("r6g.4xlarge.search", 16,  128,  1.336),
    OSInstance("r6g.8xlarge.search", 32,  256,  2.672),
]

@dataclass
class EC2Instance:
    name: str
    vcpu: int
    ram_gb: float
    price_per_hr: float

EC2_INSTANCES = [
    EC2Instance("c6g.medium",  1,  2,  0.0340),
    EC2Instance("c6g.large",   2,  4,  0.0680),
    EC2Instance("c6g.xlarge",  4,  8,  0.1360),
    EC2Instance("c6g.2xlarge", 8, 16,  0.2720),
]

AWS_EBS_GP3_PER_GB_HR = 0.08   / (30 * 24)
AWS_S3_PER_GB_HR      = 0.023  / (30 * 24)
AWS_S3_GET_PER_1K     = 0.0004


# ---------------------------------------------------------------------------
# DigitalOcean Pricing (sfo3 on-demand, approximate early 2025)
# ---------------------------------------------------------------------------

@dataclass
class DODroplet:
    name: str
    vcpu: int
    ram_gb: float
    disk_gb: int
    price_per_hr: float

DO_DROPLETS = [
    DODroplet("gp-2vcpu-8gb",    2,   8,  25,  0.09375),   # $63/mo
    DODroplet("gp-4vcpu-16gb",   4,  16,  50,  0.17500),   # $126/mo
    DODroplet("gp-8vcpu-32gb",   8,  32, 100,  0.35000),   # $252/mo
    DODroplet("gp-16vcpu-64gb", 16,  64, 200,  0.70000),   # $504/mo
    DODroplet("gp-32vcpu-128gb",32, 128, 400,  1.40000),   # $1008/mo
]

DO_MEM_DROPLETS = [
    DODroplet("m3-2vcpu-16gb",   2,  16,  50,  0.14583),   # $105/mo
    DODroplet("m3-4vcpu-32gb",   4,  32, 100,  0.29167),   # $210/mo
    DODroplet("m3-8vcpu-64gb",   8,  64, 200,  0.58333),   # $420/mo
    DODroplet("m3-16vcpu-128gb",16, 128, 400,  1.16667),   # $840/mo
]

DO_SPACES_BASE_PER_MO    = 5.00    # covers first 250 GB + 1 TB outbound
DO_SPACES_BASE_GB        = 250
DO_SPACES_OVERAGE_PER_GB = 0.02    # per GB/mo beyond 250 GB
DO_SPACES_GET_PER_1K     = 0.0004  # same rate as AWS S3

# GETs per query:
# - server mode: index pages cached in Droplet/EC2 RAM; only cold misses hit storage (~20/query)
# - direct mode: no local cache; every query reads index pages from object storage (~50/query)
GETS_PER_QUERY_SERVER = 20
GETS_PER_QUERY_DIRECT = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_gb(gb: float) -> str:
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    if gb >= 1024:
        return f"{gb / 1024:.2f} TB"
    return f"{gb:.1f} GB"

def _hr_to_mo(price_per_hr: float) -> float:
    return price_per_hr * 24 * 30


# ---------------------------------------------------------------------------
# Shared sizing formulas
# ---------------------------------------------------------------------------

def vector_storage_gb(num_docs: int, dims: int) -> float:
    return (num_docs * dims * 4) / (1024 ** 3)

def opensearch_heap_required_gb(num_docs: int, dims: int) -> float:
    """JVM heap for kNN vectors + HNSW graph: 1.1× raw × 2× JVM overhead."""
    return vector_storage_gb(num_docs, dims) * 1.1 * 2.0

def opensearch_index_storage_gb(num_docs: int, dims: int) -> float:
    return vector_storage_gb(num_docs, dims) * 1.5

def lancedb_total_storage_gb(num_docs: int, dims: int, avg_image_kb: float) -> float:
    vec_gb = vector_storage_gb(num_docs, dims) * 1.05
    img_gb = (num_docs * avg_image_kb * 1024) / (1024 ** 3)
    return vec_gb + img_gb

def lancedb_working_set_gb(num_docs: int, dims: int) -> float:
    """~5% of vector data held in memory-mapped pages during active queries."""
    return vector_storage_gb(num_docs, dims) * 0.05

def image_storage_gb(num_docs: int, avg_image_kb: float) -> float:
    return (num_docs * avg_image_kb * 1024) / (1024 ** 3)

def do_spaces_cost_per_mo(storage_gb: float) -> float:
    if storage_gb <= DO_SPACES_BASE_GB:
        return DO_SPACES_BASE_PER_MO
    return DO_SPACES_BASE_PER_MO + (storage_gb - DO_SPACES_BASE_GB) * DO_SPACES_OVERAGE_PER_GB


# ---------------------------------------------------------------------------
# Instance pickers
# ---------------------------------------------------------------------------

def pick_aws_opensearch_instance(heap_gb: float) -> Optional[OSInstance]:
    for inst in OPENSEARCH_INSTANCES:
        if inst.ram_gb / 2 >= heap_gb:
            return inst
    return None

def aws_opensearch_cluster(heap_gb: float) -> tuple[OSInstance, int]:
    largest = OPENSEARCH_INSTANCES[-1]
    nodes = max(3, math.ceil(heap_gb / (largest.ram_gb / 2)))
    return largest, nodes

def pick_aws_lancedb_instance(num_docs: int, dims: int) -> EC2Instance:
    working_set = lancedb_working_set_gb(num_docs, dims)
    for inst in EC2_INSTANCES:
        if inst.ram_gb >= max(working_set * 2, 2.0):
            return inst
    return EC2_INSTANCES[-1]

def pick_do_opensearch_droplet(heap_gb: float) -> Optional[DODroplet]:
    """Usable heap = (RAM − 2 GB OS/Docker overhead) ÷ 2."""
    for d in DO_DROPLETS + DO_MEM_DROPLETS:
        if (d.ram_gb - 2) * 0.5 >= heap_gb:
            return d
    return None

def pick_do_lancedb_droplet(num_docs: int, dims: int) -> DODroplet:
    working_set = lancedb_working_set_gb(num_docs, dims)
    for d in DO_DROPLETS:
        if d.ram_gb >= max(working_set * 2, 2.0):
            return d
    return DO_DROPLETS[-1]


# ---------------------------------------------------------------------------
# Print blocks (one per system × provider)
# ---------------------------------------------------------------------------

def _block_os_aws(num_docs, dims, queries_per_hr, avg_image_kb):
    heap        = opensearch_heap_required_gb(num_docs, dims)
    idx_gb      = opensearch_index_storage_gb(num_docs, dims)
    img_gb      = image_storage_gb(num_docs, avg_image_kb)
    img_hr      = img_gb * AWS_S3_PER_GB_HR
    get_hr      = (queries_per_hr * 10 * 0.5 / 1000) * AWS_S3_GET_PER_1K
    inst        = pick_aws_opensearch_instance(heap)

    typer.secho("  OpenSearch  (AWS Managed Service, always-on)", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"    Heap required:  {_fmt_gb(heap)}")
    if inst:
        compute_hr = inst.price_per_hr
        ebs_hr     = idx_gb * AWS_EBS_GP3_PER_GB_HR
        typer.echo(f"    Instance:       {inst.name}  ({inst.ram_gb:.0f} GB RAM, {inst.vcpu} vCPU)")
    else:
        ci, nodes  = aws_opensearch_cluster(heap)
        compute_hr = ci.price_per_hr * nodes
        ebs_hr     = idx_gb * 2 * AWS_EBS_GP3_PER_GB_HR
        typer.echo(f"    Cluster:        {nodes}× {ci.name}  ({ci.ram_gb:.0f} GB RAM each)")
    total_hr = compute_hr + ebs_hr + img_hr + get_hr
    typer.echo(f"    Compute/hr:     ${compute_hr:.4f}  (cannot scale to zero)")
    typer.echo(f"    EBS index/hr:   ${ebs_hr:.5f}  ({_fmt_gb(idx_gb)})")
    typer.echo(f"    S3 images/hr:   ${img_hr:.5f}  ({_fmt_gb(img_gb)})")
    typer.echo(f"    S3 GETs/hr:     ${get_hr:.5f}")
    typer.secho(f"    Total:          ${total_hr:.4f}/hr   (${_hr_to_mo(total_hr):,.0f}/mo)", fg=typer.colors.BLUE)
    typer.echo(f"    Idle (no queries): ${_hr_to_mo(total_hr):,.0f}/mo  (compute still runs)")
    return total_hr


def _block_os_do(num_docs, dims, queries_per_hr, avg_image_kb):
    heap        = opensearch_heap_required_gb(num_docs, dims)
    img_gb      = image_storage_gb(num_docs, avg_image_kb)
    img_mo      = do_spaces_cost_per_mo(img_gb)
    get_hr      = (queries_per_hr * 10 * 0.5 / 1000) * DO_SPACES_GET_PER_1K
    droplet     = pick_do_opensearch_droplet(heap)

    typer.secho("  OpenSearch  (Docker on Droplet, always-on)", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"    Heap required:  {_fmt_gb(heap)}")
    if droplet:
        usable = (droplet.ram_gb - 2) * 0.5
        compute_hr = droplet.price_per_hr
        typer.echo(f"    Droplet:        {droplet.name}  ({droplet.ram_gb:.0f} GB RAM, {droplet.vcpu} vCPU, {droplet.disk_gb} GB NVMe)")
        typer.echo(f"    Usable heap:    {_fmt_gb(usable)}  (RAM − 2 GB overhead, ÷ 2)")
    else:
        typer.secho("    !! Exceeds single Droplet capacity.", fg=typer.colors.RED)
        droplet    = DO_DROPLETS[-1]
        compute_hr = droplet.price_per_hr
    img_hr   = img_mo / (30 * 24)
    total_hr = compute_hr + img_hr + get_hr
    typer.echo(f"    Compute/hr:     ${compute_hr:.5f}  (cannot scale to zero)")
    typer.echo(f"    Spaces images:  ${img_mo:.2f}/mo  ({_fmt_gb(img_gb)})")
    typer.echo(f"    Spaces GETs/hr: ${get_hr:.5f}")
    typer.secho(f"    Total:          ${total_hr:.4f}/hr   (${_hr_to_mo(total_hr):,.0f}/mo)", fg=typer.colors.BLUE)
    typer.echo(f"    Idle (no queries): ${_hr_to_mo(total_hr):,.0f}/mo  (Droplet still runs)")
    return total_hr


def _block_lance_aws_server(num_docs, dims, queries_per_hr, avg_image_kb, utilization):
    store_gb    = lancedb_total_storage_gb(num_docs, dims, avg_image_kb)
    store_hr    = store_gb * AWS_S3_PER_GB_HR
    get_hr      = (queries_per_hr * GETS_PER_QUERY_SERVER / 1000) * AWS_S3_GET_PER_1K
    inst        = pick_aws_lancedb_instance(num_docs, dims)
    compute_hr  = inst.price_per_hr
    total_hr    = compute_hr + store_hr + get_hr
    idle_hr     = compute_hr * utilization + store_hr

    typer.secho("  LanceDB     (EC2 server + S3)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"    Working set:    {_fmt_gb(lancedb_working_set_gb(num_docs, dims))}  (memory-mapped)")
    typer.echo(f"    Instance:       {inst.name}  ({inst.ram_gb:.0f} GB RAM, {inst.vcpu} vCPU)")
    typer.echo(f"    Compute/hr:     ${compute_hr:.4f}")
    typer.echo(f"    S3 storage/hr:  ${store_hr:.5f}  ({_fmt_gb(store_gb)})")
    typer.echo(f"    S3 GETs/hr:     ${get_hr:.5f}  ({GETS_PER_QUERY_SERVER} GETs/query, cached)")
    typer.secho(f"    Total:          ${total_hr:.4f}/hr   (${_hr_to_mo(total_hr):,.0f}/mo)", fg=typer.colors.GREEN)
    if utilization < 1.0:
        typer.secho(f"    At {utilization*100:.0f}% util:      ${total_hr * utilization:.4f}/hr   (${_hr_to_mo(total_hr * utilization):,.0f}/mo)", fg=typer.colors.GREEN, bold=True)
    idle_mo = store_hr * 24 * 30
    typer.echo(f"    Idle (no queries): ${idle_mo:.2f}/mo  (EC2 stopped, only S3 storage)")
    return total_hr


def _block_lance_do_server(num_docs, dims, queries_per_hr, avg_image_kb, utilization):
    store_gb    = lancedb_total_storage_gb(num_docs, dims, avg_image_kb)
    spaces_mo   = do_spaces_cost_per_mo(store_gb)
    spaces_hr   = spaces_mo / (30 * 24)
    get_hr      = (queries_per_hr * GETS_PER_QUERY_SERVER / 1000) * DO_SPACES_GET_PER_1K
    droplet     = pick_do_lancedb_droplet(num_docs, dims)
    compute_hr  = droplet.price_per_hr
    total_hr    = compute_hr + spaces_hr + get_hr

    typer.secho("  LanceDB     (Droplet server + Spaces)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"    Working set:    {_fmt_gb(lancedb_working_set_gb(num_docs, dims))}  (memory-mapped)")
    typer.echo(f"    Droplet:        {droplet.name}  ({droplet.ram_gb:.0f} GB RAM, {droplet.vcpu} vCPU, {droplet.disk_gb} GB NVMe)")
    typer.echo(f"    Compute/hr:     ${compute_hr:.5f}")
    typer.echo(f"    Spaces/mo:      ${spaces_mo:.2f}  ({_fmt_gb(store_gb)})")
    typer.echo(f"    Spaces GETs/hr: ${get_hr:.5f}  ({GETS_PER_QUERY_SERVER} GETs/query, cached)")
    typer.secho(f"    Total:          ${total_hr:.4f}/hr   (${_hr_to_mo(total_hr):,.0f}/mo)", fg=typer.colors.GREEN)
    if utilization < 1.0:
        typer.secho(f"    At {utilization*100:.0f}% util:      ${total_hr * utilization:.4f}/hr   (${_hr_to_mo(total_hr * utilization):,.0f}/mo)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"    Idle (no queries): ${spaces_mo:.2f}/mo  (Droplet off, only Spaces storage)")
    return total_hr


def _block_lance_direct(num_docs, dims, queries_per_hr, avg_image_kb, get_cost_per_1k):
    """LanceDB queried directly from object storage — no dedicated server.

    Compute cost is $0 as dedicated LanceDB infrastructure: queries run inside
    the calling process (web server, notebook, cloud function, etc.).
    GETs per query is higher than server mode because there is no resident
    in-memory page cache between requests.
    """
    store_gb  = lancedb_total_storage_gb(num_docs, dims, avg_image_kb)
    get_hr    = (queries_per_hr * GETS_PER_QUERY_DIRECT / 1000) * get_cost_per_1k

    typer.secho("  LanceDB     (direct from object storage — no dedicated server)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"    Dedicated compute: $0.00/hr  (runs inside your existing process)")
    typer.echo(f"    Storage:           {_fmt_gb(store_gb)}")
    typer.echo(f"    GETs/hr:           ${get_hr:.5f}  ({GETS_PER_QUERY_DIRECT} GETs/query, no server-side cache)")
    return store_gb, get_hr


def _block_lance_direct_aws(num_docs, dims, queries_per_hr, avg_image_kb):
    store_gb, get_hr = _block_lance_direct(num_docs, dims, queries_per_hr, avg_image_kb, AWS_S3_GET_PER_1K)
    store_hr  = store_gb * AWS_S3_PER_GB_HR
    total_hr  = store_hr + get_hr
    typer.echo(f"    S3 storage/hr:     ${store_hr:.5f}")
    typer.secho(f"    Total:             ${total_hr:.5f}/hr   (${_hr_to_mo(total_hr):.2f}/mo)", fg=typer.colors.GREEN)
    idle_mo = store_hr * 24 * 30
    typer.echo(f"    Idle (no queries): ${idle_mo:.2f}/mo  (only S3 storage, zero compute)")
    return total_hr


def _block_lance_direct_do(num_docs, dims, queries_per_hr, avg_image_kb):
    store_gb, get_hr = _block_lance_direct(num_docs, dims, queries_per_hr, avg_image_kb, DO_SPACES_GET_PER_1K)
    spaces_mo = do_spaces_cost_per_mo(store_gb)
    spaces_hr = spaces_mo / (30 * 24)
    total_hr  = spaces_hr + get_hr
    typer.echo(f"    Spaces/mo:         ${spaces_mo:.2f}  (base plan covers first {DO_SPACES_BASE_GB} GB)")
    typer.secho(f"    Total:             ${total_hr:.5f}/hr   (${_hr_to_mo(total_hr):.2f}/mo)", fg=typer.colors.GREEN)
    typer.echo(f"    Idle (no queries): ${spaces_mo:.2f}/mo  (only Spaces storage, zero compute)")
    return total_hr


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

_FOOTER = (
    "\n* Idle = data uploaded, no active query traffic."
    "\n  Pricing: AWS us-east-1 on-demand / DO sfo3, ~early 2025."
    "\n  Actual costs vary by reserved pricing, data transfer, and multi-AZ."
)

_PARAMS = dict(
    dims=typer.Option(1152, help="Vector dimension (default: 1152 for SigLIP 2 so400m)."),
    queries_per_hr=typer.Option(100, help="Expected query rate (queries per hour)."),
    avg_image_kb=typer.Option(200.0, help="Average image size in KB."),
    utilization=typer.Option(1.0, help="LanceDB server utilization 0–1 (only used in server mode)."),
    provider=typer.Option(Provider.all, help="Pricing to show: aws, do, or all."),
)


@app.command(name="cost")
def cost_server(
    num_docs: int = typer.Argument(..., help="Number of documents (vectors)."),
    dims: int = _PARAMS["dims"],
    queries_per_hr: int = _PARAMS["queries_per_hr"],
    avg_image_kb: float = _PARAMS["avg_image_kb"],
    utilization: float = _PARAMS["utilization"],
    provider: Provider = _PARAMS["provider"],
) -> None:
    """OpenSearch (server) vs LanceDB (dedicated Droplet/EC2 + object storage).

    Use this when you want a persistent LanceDB server that keeps index pages
    warm in RAM between requests — lower per-query latency, higher fixed cost.
    """
    typer.secho("Cost Estimate — Server Mode", bold=True)
    typer.echo(
        f"  Both systems run on dedicated servers.\n"
        f"  {num_docs:,} docs | {dims} dims | {queries_per_hr} q/hr | avg image {avg_image_kb:.0f} KB\n"
    )

    if provider in (Provider.aws, Provider.all):
        typer.secho("═══ AWS ════════════════════════════════════════════", fg=typer.colors.CYAN, bold=True)
        os_hr  = _block_os_aws(num_docs, dims, queries_per_hr, avg_image_kb)
        typer.echo("")
        lb_hr  = _block_lance_aws_server(num_docs, dims, queries_per_hr, avg_image_kb, utilization)
        typer.echo(f"\n  OpenSearch is ~{os_hr/lb_hr:.1f}× more expensive than LanceDB (always-on)")

    if provider in (Provider.do, Provider.all):
        typer.secho("\n═══ DigitalOcean ════════════════════════════════════", fg=typer.colors.MAGENTA, bold=True)
        os_hr  = _block_os_do(num_docs, dims, queries_per_hr, avg_image_kb)
        typer.echo("")
        lb_hr  = _block_lance_do_server(num_docs, dims, queries_per_hr, avg_image_kb, utilization)
        typer.echo(f"\n  OpenSearch is ~{os_hr/lb_hr:.1f}× more expensive than LanceDB (always-on)")

    typer.echo(_FOOTER)


@app.command(name="cost-direct")
def cost_direct(
    num_docs: int = typer.Argument(..., help="Number of documents (vectors)."),
    dims: int = _PARAMS["dims"],
    queries_per_hr: int = _PARAMS["queries_per_hr"],
    avg_image_kb: float = _PARAMS["avg_image_kb"],
    provider: Provider = _PARAMS["provider"],
) -> None:
    """OpenSearch (server) vs LanceDB (queried directly from S3/Spaces, no dedicated server).

    LanceDB is an embedded library. You can lancedb.connect('s3://...') from any
    Python process — your web app, a notebook, a cloud function — and skip the
    dedicated server entirely. The query compute is absorbed into whatever you are
    already running.

    Trade-off vs server mode:
      - Cost:    lower (no dedicated compute)
      - Latency: higher per query — index pages are read from object storage on
                 each request without a warm in-memory cache between calls.
                 Best suited for low-to-medium query rates or batch workloads.
    """
    typer.secho("Cost Estimate — Direct from Object Storage Mode", bold=True)
    typer.echo(
        f"  OpenSearch needs a dedicated server; LanceDB has no dedicated server.\n"
        f"  {num_docs:,} docs | {dims} dims | {queries_per_hr} q/hr | avg image {avg_image_kb:.0f} KB\n"
    )

    if provider in (Provider.aws, Provider.all):
        typer.secho("═══ AWS ════════════════════════════════════════════", fg=typer.colors.CYAN, bold=True)
        os_hr = _block_os_aws(num_docs, dims, queries_per_hr, avg_image_kb)
        typer.echo("")
        lb_hr = _block_lance_direct_aws(num_docs, dims, queries_per_hr, avg_image_kb)
        typer.echo(
            f"\n  OpenSearch dedicated server: ${_hr_to_mo(os_hr):,.0f}/mo\n"
            f"  LanceDB direct from S3:      ${_hr_to_mo(lb_hr):.2f}/mo\n"
            f"  Difference:                  ~{os_hr/lb_hr:.0f}× (LanceDB cheaper)"
        )

    if provider in (Provider.do, Provider.all):
        typer.secho("\n═══ DigitalOcean ════════════════════════════════════", fg=typer.colors.MAGENTA, bold=True)
        os_hr = _block_os_do(num_docs, dims, queries_per_hr, avg_image_kb)
        typer.echo("")
        lb_hr = _block_lance_direct_do(num_docs, dims, queries_per_hr, avg_image_kb)
        typer.echo(
            f"\n  OpenSearch Droplet: ${_hr_to_mo(os_hr):,.0f}/mo\n"
            f"  LanceDB direct:     ${_hr_to_mo(lb_hr):.2f}/mo\n"
            f"  Difference:         ~{os_hr/lb_hr:.0f}× (LanceDB cheaper)"
        )

    typer.echo(_FOOTER)


if __name__ == "__main__":
    app()
