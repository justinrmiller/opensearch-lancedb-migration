"""Hourly and monthly cost estimator: OpenSearch vs LanceDB for vector search workloads.

Models deployments on AWS (us-east-1) and DigitalOcean (sfo3).

OpenSearch: AWS OpenSearch Service (managed) or Docker on a Droplet.
            Vectors must fit in JVM heap for kNN — memory is the binding constraint.
LanceDB:    Self-hosted on EC2/Droplet + S3/Spaces. Compute only needed during queries;
            for a static dataset the Droplet can be powered off — only storage costs remain.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import typer

app = typer.Typer()


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

# General Purpose droplets (dedicated vCPU)
DO_DROPLETS = [
    DODroplet("gp-2vcpu-8gb",    2,   8,  25,  0.09375),   # $63/mo
    DODroplet("gp-4vcpu-16gb",   4,  16,  50,  0.17500),   # $126/mo
    DODroplet("gp-8vcpu-32gb",   8,  32, 100,  0.35000),   # $252/mo
    DODroplet("gp-16vcpu-64gb", 16,  64, 200,  0.70000),   # $504/mo
    DODroplet("gp-32vcpu-128gb",32, 128, 400,  1.40000),   # $1008/mo
]

# Memory-Optimized droplets (if we need more RAM per vCPU)
DO_MEM_DROPLETS = [
    DODroplet("m3-2vcpu-16gb",   2,  16,  50,  0.14583),   # $105/mo
    DODroplet("m3-4vcpu-32gb",   4,  32, 100,  0.29167),   # $210/mo
    DODroplet("m3-8vcpu-64gb",   8,  64, 200,  0.58333),   # $420/mo
    DODroplet("m3-16vcpu-128gb",16, 128, 400,  1.16667),   # $840/mo
]

# Spaces: $5/mo base covers 250 GB + 1 TB outbound. Overage: $0.02/GB/mo.
DO_SPACES_BASE_PER_MO     = 5.00        # covers first 250 GB
DO_SPACES_BASE_GB         = 250
DO_SPACES_OVERAGE_PER_GB  = 0.02        # per GB/mo beyond 250 GB
DO_SPACES_GET_PER_1K      = 0.0004      # same as S3 (DO matches AWS GET pricing)


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
# Shared sizing logic (provider-agnostic)
# ---------------------------------------------------------------------------

def vector_storage_gb(num_docs: int, dims: int) -> float:
    return (num_docs * dims * 4) / (1024 ** 3)


def opensearch_heap_required_gb(num_docs: int, dims: int) -> float:
    """Estimated JVM heap for kNN vectors + HNSW graph.

    Rule of thumb: 1.1× raw vector size for HNSW edges (m=16), then ×2
    for JVM object overhead and GC headroom.
    """
    return vector_storage_gb(num_docs, dims) * 1.1 * 2.0


def opensearch_index_storage_gb(num_docs: int, dims: int) -> float:
    return vector_storage_gb(num_docs, dims) * 1.5


def lancedb_vector_storage_gb(num_docs: int, dims: int) -> float:
    return vector_storage_gb(num_docs, dims) * 1.05


def image_storage_gb(num_docs: int, avg_image_kb: float) -> float:
    return (num_docs * avg_image_kb * 1024) / (1024 ** 3)


def lancedb_working_set_gb(num_docs: int, dims: int) -> float:
    """LanceDB only memory-maps the pages it needs: ~5% of vector data."""
    return vector_storage_gb(num_docs, dims) * 0.05


def s3_get_cost_per_hr(queries_per_hr: int, gets_per_query: int = 20) -> float:
    return (queries_per_hr * gets_per_query / 1000) * AWS_S3_GET_PER_1K


# ---------------------------------------------------------------------------
# AWS sizing
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


# ---------------------------------------------------------------------------
# DigitalOcean sizing
# ---------------------------------------------------------------------------

def pick_do_opensearch_droplet(heap_gb: float) -> Optional[DODroplet]:
    """
    OpenSearch runs in Docker on the Droplet. Usable heap = (RAM - 2GB_os) * 0.5.
    We reserve 2 GB for the OS/Docker daemon, then split the rest 50/50 for
    the JVM heap and the OS page cache / other processes.
    """
    for d in DO_DROPLETS + DO_MEM_DROPLETS:
        usable_heap = (d.ram_gb - 2) * 0.5
        if usable_heap >= heap_gb:
            return d
    return None


def pick_do_lancedb_droplet(num_docs: int, dims: int) -> DODroplet:
    """LanceDB working set is tiny; pick the smallest Droplet that fits it."""
    working_set = lancedb_working_set_gb(num_docs, dims)
    for d in DO_DROPLETS:
        if d.ram_gb >= max(working_set * 2, 2.0):
            return d
    return DO_DROPLETS[-1]


def do_spaces_cost_per_mo(storage_gb: float) -> float:
    if storage_gb <= DO_SPACES_BASE_GB:
        return DO_SPACES_BASE_PER_MO
    return DO_SPACES_BASE_PER_MO + (storage_gb - DO_SPACES_BASE_GB) * DO_SPACES_OVERAGE_PER_GB


# ---------------------------------------------------------------------------
# Output sections
# ---------------------------------------------------------------------------

def _print_aws_section(
    num_docs: int,
    dims: int,
    queries_per_hr: int,
    avg_image_kb: float,
    utilization: float,
):
    heap_needed   = opensearch_heap_required_gb(num_docs, dims)
    os_instance   = pick_aws_opensearch_instance(heap_needed)
    os_storage_gb = opensearch_index_storage_gb(num_docs, dims)
    os_s3_img_gb  = image_storage_gb(num_docs, avg_image_kb)
    os_s3_img_hr  = os_s3_img_gb * AWS_S3_PER_GB_HR
    os_s3_get_hr  = s3_get_cost_per_hr(queries_per_hr * 10 * 0.5)

    typer.secho("\n═══ AWS ═══════════════════════════════════════════", fg=typer.colors.CYAN, bold=True)

    typer.secho("  OpenSearch (AWS Managed Service)", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"    Heap required:    {_fmt_gb(heap_needed)}")
    if os_instance:
        os_nodes        = 1
        os_compute_hr   = os_instance.price_per_hr
        os_ebs_hr       = os_storage_gb * AWS_EBS_GP3_PER_GB_HR
        typer.echo(f"    Instance:         {os_instance.name}  ({os_instance.ram_gb:.0f} GB RAM, {os_instance.vcpu} vCPU)")
    else:
        inst, os_nodes  = aws_opensearch_cluster(heap_needed)
        os_compute_hr   = inst.price_per_hr * os_nodes
        os_ebs_hr       = os_storage_gb * 2 * AWS_EBS_GP3_PER_GB_HR
        typer.echo(f"    Cluster:          {os_nodes}× {inst.name}  ({inst.ram_gb:.0f} GB RAM each)")
    os_total_hr = os_compute_hr + os_ebs_hr + os_s3_img_hr + os_s3_get_hr
    typer.echo(f"    Compute/hr:       ${os_compute_hr:.4f}  (always-on — cannot scale to zero)")
    typer.echo(f"    EBS storage/hr:   ${os_ebs_hr:.5f}  ({_fmt_gb(os_storage_gb)})")
    typer.echo(f"    S3 images/hr:     ${os_s3_img_hr:.5f}  ({_fmt_gb(os_s3_img_gb)})")
    typer.echo(f"    S3 GETs/hr:       ${os_s3_get_hr:.5f}")
    typer.secho(f"    Total:            ${os_total_hr:.4f}/hr   (${_hr_to_mo(os_total_hr):,.0f}/mo)", fg=typer.colors.BLUE)
    typer.echo(f"    Static-dataset*:  ${_hr_to_mo(os_total_hr):,.0f}/mo  (compute still runs 24/7)")

    lance_vec_gb    = lancedb_vector_storage_gb(num_docs, dims)
    lance_img_gb    = image_storage_gb(num_docs, avg_image_kb)
    lance_total_gb  = lance_vec_gb + lance_img_gb
    lance_store_hr  = lance_total_gb * AWS_S3_PER_GB_HR
    lance_get_hr    = s3_get_cost_per_hr(queries_per_hr)
    lance_inst      = pick_aws_lancedb_instance(num_docs, dims)
    lance_comp_hr   = lance_inst.price_per_hr
    lance_total_hr  = lance_comp_hr + lance_store_hr + lance_get_hr
    lance_idle_hr   = lance_comp_hr * utilization + lance_store_hr + lance_get_hr

    typer.echo("")
    typer.secho("  LanceDB (EC2 + S3)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"    Working set:      {_fmt_gb(lancedb_working_set_gb(num_docs, dims))}")
    typer.echo(f"    Instance:         {lance_inst.name}  ({lance_inst.ram_gb:.0f} GB RAM, {lance_inst.vcpu} vCPU)")
    typer.echo(f"    Compute/hr:       ${lance_comp_hr:.4f}")
    typer.echo(f"    S3 storage/hr:    ${lance_store_hr:.5f}  ({_fmt_gb(lance_total_gb)})")
    typer.echo(f"    S3 GETs/hr:       ${lance_get_hr:.5f}")
    typer.secho(f"    Total (always-on): ${lance_total_hr:.4f}/hr  (${_hr_to_mo(lance_total_hr):,.0f}/mo)", fg=typer.colors.GREEN)
    if utilization < 1.0:
        typer.secho(f"    Total ({utilization*100:.0f}% util):   ${lance_idle_hr:.4f}/hr  (${_hr_to_mo(lance_idle_hr):,.0f}/mo)", fg=typer.colors.GREEN, bold=True)
    static_mo = lance_store_hr * 24 * 30
    typer.echo(f"    Static-dataset*:  ${static_mo:.2f}/mo  (EC2 stopped, only S3 storage)")

    ratio = os_total_hr / lance_total_hr
    typer.echo(f"\n    OpenSearch is ~{ratio:.1f}× more expensive than LanceDB (always-on)")

    return os_total_hr, lance_total_hr, _hr_to_mo(os_total_hr), _hr_to_mo(lance_total_hr), lance_store_hr * 24 * 30


def _print_do_section(
    num_docs: int,
    dims: int,
    queries_per_hr: int,
    avg_image_kb: float,
    utilization: float,
):
    heap_needed      = opensearch_heap_required_gb(num_docs, dims)
    os_storage_gb    = opensearch_index_storage_gb(num_docs, dims)
    os_droplet       = pick_do_opensearch_droplet(heap_needed)
    os_img_gb        = image_storage_gb(num_docs, avg_image_kb)
    os_img_spaces_mo = do_spaces_cost_per_mo(os_img_gb)
    os_get_hr        = (queries_per_hr * 10 * 0.5 / 1000) * DO_SPACES_GET_PER_1K

    typer.secho("\n═══ DigitalOcean ═══════════════════════════════════", fg=typer.colors.MAGENTA, bold=True)

    typer.secho("  OpenSearch (Docker on Droplet)", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"    Heap required:    {_fmt_gb(heap_needed)}")
    if os_droplet:
        usable_heap = (os_droplet.ram_gb - 2) * 0.5
        os_compute_hr = os_droplet.price_per_hr
        typer.echo(f"    Droplet:          {os_droplet.name}  ({os_droplet.ram_gb:.0f} GB RAM, {os_droplet.vcpu} vCPU, {os_droplet.disk_gb} GB NVMe)")
        typer.echo(f"    Usable heap:      {_fmt_gb(usable_heap)}  (RAM − 2 GB OS overhead, ÷ 2)")
    else:
        typer.secho("    !! Dataset too large for a single Droplet — consider sharding.", fg=typer.colors.RED)
        os_droplet  = DO_DROPLETS[-1]
        os_compute_hr = os_droplet.price_per_hr
    typer.echo(f"    Compute/hr:       ${os_compute_hr:.5f}  (always-on)")
    typer.echo(f"    Spaces images/mo: ${os_img_spaces_mo:.2f}  ({_fmt_gb(os_img_gb)} — stored in Spaces)")
    typer.echo(f"    Spaces GETs/hr:   ${os_get_hr:.5f}")
    os_img_hr = os_img_spaces_mo / (30 * 24)
    os_total_hr = os_compute_hr + os_img_hr + os_get_hr
    typer.secho(f"    Total:            ${os_total_hr:.4f}/hr   (${_hr_to_mo(os_total_hr):,.0f}/mo)", fg=typer.colors.BLUE)
    typer.echo(f"    Static-dataset*:  ${_hr_to_mo(os_total_hr):,.0f}/mo  (Droplet still runs 24/7)")

    lance_vec_gb      = lancedb_vector_storage_gb(num_docs, dims)
    lance_img_gb      = image_storage_gb(num_docs, avg_image_kb)
    lance_total_gb    = lance_vec_gb + lance_img_gb
    lance_spaces_mo   = do_spaces_cost_per_mo(lance_total_gb)
    lance_get_hr      = (queries_per_hr / 1000) * DO_SPACES_GET_PER_1K
    lance_droplet     = pick_do_lancedb_droplet(num_docs, dims)
    lance_compute_hr  = lance_droplet.price_per_hr
    lance_spaces_hr   = lance_spaces_mo / (30 * 24)
    lance_total_hr    = lance_compute_hr + lance_spaces_hr + lance_get_hr
    lance_idle_hr     = lance_compute_hr * utilization + lance_spaces_hr + lance_get_hr

    typer.echo("")
    typer.secho("  LanceDB (Droplet + Spaces)", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"    Working set:      {_fmt_gb(lancedb_working_set_gb(num_docs, dims))}")
    typer.echo(f"    Droplet:          {lance_droplet.name}  ({lance_droplet.ram_gb:.0f} GB RAM, {lance_droplet.vcpu} vCPU, {lance_droplet.disk_gb} GB NVMe)")
    typer.echo(f"    Compute/hr:       ${lance_compute_hr:.5f}")
    typer.echo(f"    Spaces/mo:        ${lance_spaces_mo:.2f}  ({_fmt_gb(lance_total_gb)})")
    typer.echo(f"    Spaces GETs/hr:   ${lance_get_hr:.5f}")
    typer.secho(f"    Total (always-on): ${lance_total_hr:.4f}/hr  (${_hr_to_mo(lance_total_hr):,.0f}/mo)", fg=typer.colors.GREEN)
    if utilization < 1.0:
        typer.secho(f"    Total ({utilization*100:.0f}% util):   ${lance_idle_hr:.4f}/hr  (${_hr_to_mo(lance_idle_hr):,.0f}/mo)", fg=typer.colors.GREEN, bold=True)
    static_mo = lance_spaces_mo
    typer.echo(f"    Static-dataset*:  ${static_mo:.2f}/mo  (Droplet off, only Spaces storage)")

    ratio = os_total_hr / lance_total_hr
    typer.echo(f"\n    OpenSearch is ~{ratio:.1f}× more expensive than LanceDB (always-on)")

    return os_total_hr, lance_total_hr, _hr_to_mo(os_total_hr), _hr_to_mo(lance_total_hr), static_mo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    num_docs: int = typer.Argument(..., help="Number of documents (vectors) to store."),
    dims: int = typer.Option(1152, help="Vector dimension (default: 1152 for SigLIP 2 so400m)."),
    queries_per_hr: int = typer.Option(100, help="Expected query rate (queries per hour)."),
    avg_image_kb: float = typer.Option(200.0, help="Average image size in KB (for LanceDB inline storage)."),
    utilization: float = typer.Option(1.0, help="LanceDB compute utilization 0–1. Use <1 if the instance is idle most of the time."),
    provider: Provider = typer.Option(Provider.all, help="Cloud provider pricing to show: aws, do (DigitalOcean), or all."),
) -> None:
    """Estimate hourly and monthly cost for OpenSearch vs LanceDB at a given scale.

    Covers both AWS (managed OpenSearch + EC2/S3) and DigitalOcean
    (Docker Droplet + Spaces).

    The 'static-dataset' line shows cost when data is loaded and sitting idle
    (no active query workload):
      - OpenSearch: compute still runs 24/7 (cannot be stopped)
      - LanceDB:    Droplet/EC2 can be powered off; only storage cost remains
    """
    typer.secho("Cost Estimate: OpenSearch vs LanceDB", bold=True)
    typer.echo(
        f"  {num_docs:,} docs | {dims} dims | {queries_per_hr} q/hr | "
        f"avg image {avg_image_kb:.0f} KB\n"
    )

    if provider in (Provider.aws, Provider.all):
        _print_aws_section(num_docs, dims, queries_per_hr, avg_image_kb, utilization)

    if provider in (Provider.do, Provider.all):
        _print_do_section(num_docs, dims, queries_per_hr, avg_image_kb, utilization)

    typer.echo(
        "\n* Static-dataset = data uploaded, no query traffic."
        "\n  Pricing: AWS us-east-1 on-demand / DO sfo3, ~early 2025."
        "\n  Actual costs vary by reserved pricing, data transfer, and multi-AZ."
    )


if __name__ == "__main__":
    app()
