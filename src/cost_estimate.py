"""Hourly cost estimator: OpenSearch vs LanceDB for vector search workloads.

Models AWS-based deployments using approximate us-east-1 on-demand pricing.

OpenSearch: AWS OpenSearch Service (managed), vectors must fit in JVM heap for kNN.
LanceDB:    Self-hosted on EC2 + S3 (or local disk), compute only needed during queries.
"""

from dataclasses import dataclass

import typer

app = typer.Typer()

# ---------------------------------------------------------------------------
# Pricing data (us-east-1 on-demand, approximate as of early 2025)
# ---------------------------------------------------------------------------

# AWS OpenSearch Service — r6g (memory-optimized, Graviton2) instances
# kNN requires vectors to fit in JVM heap, so memory-optimized is the right family.
@dataclass
class OSInstance:
    name: str
    vcpu: int
    ram_gb: float
    price_per_hr: float  # USD

OPENSEARCH_INSTANCES = [
    OSInstance("r6g.large.search",    2,   16,  0.167),
    OSInstance("r6g.xlarge.search",   4,   32,  0.334),
    OSInstance("r6g.2xlarge.search",  8,   64,  0.668),
    OSInstance("r6g.4xlarge.search", 16,  128,  1.336),
    OSInstance("r6g.8xlarge.search", 32,  256,  2.672),
]

# EC2 c6g (compute-optimized, Graviton2) — for LanceDB query serving
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

# Storage
EBS_GP3_PER_GB_HR  = 0.08   / (30 * 24)   # $0.08/GB/month
S3_PER_GB_HR       = 0.023  / (30 * 24)   # $0.023/GB/month
S3_GET_PER_1K      = 0.0004                # per 1,000 GET requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_gb(gb: float) -> str:
    """Human-readable size: show MB when < 1 GB, TB when > 1024 GB."""
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    if gb >= 1024:
        return f"{gb / 1024:.2f} TB"
    return f"{gb:.1f} GB"


# ---------------------------------------------------------------------------
# Core sizing logic
# ---------------------------------------------------------------------------

def vector_storage_gb(num_docs: int, dims: int) -> float:
    """Raw vector data in GB (float32)."""
    return (num_docs * dims * 4) / (1024 ** 3)


def opensearch_heap_required_gb(num_docs: int, dims: int) -> float:
    """Estimated JVM heap needed for kNN vectors + HNSW graph.

    OpenSearch loads the entire HNSW graph into the JVM heap for kNN search.
    Rule of thumb: 1.1× raw vector size for graph edges (m=16), then double
    for JVM object overhead and GC headroom.
    """
    raw_gb = vector_storage_gb(num_docs, dims)
    hnsw_overhead = 1.1
    jvm_overhead  = 2.0
    return raw_gb * hnsw_overhead * jvm_overhead


def pick_opensearch_instance(heap_gb: float) -> OSInstance | None:
    """Return the smallest single instance whose RAM fits the required heap."""
    # OpenSearch recommends heap = RAM/2, so usable heap ≈ ram_gb / 2
    for inst in OPENSEARCH_INSTANCES:
        usable_heap = inst.ram_gb / 2
        if usable_heap >= heap_gb:
            return inst
    return None  # exceeds largest listed instance — caller should use multi-node


def opensearch_cluster_nodes(heap_gb: float) -> tuple[OSInstance, int]:
    """Return the (instance, node_count) for a multi-node cluster.

    Uses the largest available instance (r6g.8xlarge, 128 GB usable heap)
    and adds nodes until the total usable heap covers the requirement.
    A minimum of 3 nodes is enforced for production HA (primary + 2 replicas).
    """
    largest = OPENSEARCH_INSTANCES[-1]
    usable_per_node = largest.ram_gb / 2
    import math
    nodes = max(3, math.ceil(heap_gb / usable_per_node))
    return largest, nodes


def opensearch_index_storage_gb(num_docs: int, dims: int) -> float:
    """EBS storage for the OpenSearch index (vectors + metadata + HNSW graph)."""
    raw_gb = vector_storage_gb(num_docs, dims)
    return raw_gb * 1.5  # ~50% overhead for Lucene segment metadata


def opensearch_s3_image_storage_gb(num_docs: int, avg_image_kb: float) -> float:
    """S3 storage for images that OpenSearch cannot store inline.

    OpenSearch stores only a path reference; images must live in S3 (or a CDN).
    """
    unique_images = num_docs
    return (unique_images * avg_image_kb * 1024) / (1024 ** 3)


def lancedb_storage_gb(num_docs: int, dims: int, avg_image_kb: float) -> float:
    """S3/disk storage for LanceDB (columnar vectors + optional inline images)."""
    vector_gb  = vector_storage_gb(num_docs, dims) * 1.05  # Lance columnar overhead
    image_gb   = (num_docs * avg_image_kb * 1024) / (1024 ** 3)
    return vector_gb + image_gb


def pick_lancedb_instance(num_docs: int, dims: int) -> EC2Instance:
    """Return a suitable EC2 instance for LanceDB query serving.

    LanceDB does NOT load all vectors into RAM — it memory-maps columnar
    files and reads only the pages needed per query. Working set ≈ 5% of
    total vector data for typical query patterns.
    """
    working_set_gb = vector_storage_gb(num_docs, dims) * 0.05
    for inst in EC2_INSTANCES:
        if inst.ram_gb >= max(working_set_gb * 2, 2.0):  # 2× working set, min 2 GB
            return inst
    return EC2_INSTANCES[-1]


def s3_query_cost_per_hr(queries_per_hr: int) -> float:
    """S3 GET request cost for LanceDB reading index pages during queries.

    Assumes ~20 S3 GETs per query (ANN index page reads).
    """
    gets_per_hr = queries_per_hr * 20
    return (gets_per_hr / 1000) * S3_GET_PER_1K


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    num_docs: int   = typer.Argument(..., help="Number of documents (vectors) to store."),
    dims: int       = typer.Option(1152, help="Vector dimension (default: 1152 for SigLIP 2 so400m)."),
    queries_per_hr: int = typer.Option(100, help="Expected query rate (queries per hour)."),
    avg_image_kb: float = typer.Option(200.0, help="Average image size in KB (only relevant for LanceDB inline storage)."),
    utilization: float  = typer.Option(1.0,   help="LanceDB compute utilization 0–1 (e.g. 0.1 = instance idle 90%% of the time). Use 1.0 for always-on."),
) -> None:
    """Estimate hourly AWS cost for OpenSearch vs LanceDB at a given document count."""

    typer.secho("Cost Estimate: OpenSearch vs LanceDB", bold=True)
    typer.echo(
        f"  {num_docs:,} documents | {dims} dims | {queries_per_hr} queries/hr | "
        f"region: us-east-1 (on-demand)\n"
    )

    # -----------------------------------------------------------------------
    # OpenSearch
    # -----------------------------------------------------------------------
    heap_needed     = opensearch_heap_required_gb(num_docs, dims)
    os_instance     = pick_opensearch_instance(heap_needed)
    os_storage_gb   = opensearch_index_storage_gb(num_docs, dims)

    typer.secho("--- OpenSearch (AWS Managed Service) ---", fg=typer.colors.BLUE, bold=True)
    typer.echo(f"  Heap required (vectors + HNSW):  {_fmt_gb(heap_needed)}")

    os_s3_image_gb   = opensearch_s3_image_storage_gb(num_docs, avg_image_kb)
    os_s3_image_cost = os_s3_image_gb * S3_PER_GB_HR
    # S3 GETs to serve images from query results: k=10 results/query, ~50% are image-modality
    os_s3_get_cost   = s3_query_cost_per_hr(queries_per_hr * 10 * 0.5)

    if os_instance:
        os_nodes        = 1
        os_compute_cost = os_instance.price_per_hr
        os_ebs_cost     = os_storage_gb * EBS_GP3_PER_GB_HR
        typer.echo(f"  Deployment:                      single node")
        typer.echo(f"  Instance:                        {os_instance.name}  ({os_instance.ram_gb:.0f} GB RAM, {os_instance.vcpu} vCPU)")
        typer.echo(f"  Compute (always-on):             ${os_compute_cost:.4f}/hr")
    else:
        cluster_inst, os_nodes = opensearch_cluster_nodes(heap_needed)
        os_compute_cost = cluster_inst.price_per_hr * os_nodes
        # EBS is per-node; replicas double total storage — assume 1 replica shard set.
        os_ebs_cost     = os_storage_gb * 2 * EBS_GP3_PER_GB_HR
        typer.echo(f"  Deployment:                      {os_nodes}-node cluster (exceeds single-instance limit)")
        typer.echo(f"  Instance per node:               {cluster_inst.name}  ({cluster_inst.ram_gb:.0f} GB RAM, {cluster_inst.vcpu} vCPU)")
        typer.echo(f"  Compute ({os_nodes} nodes, always-on):    ${os_compute_cost:.2f}/hr")

    os_total = os_compute_cost + os_ebs_cost + os_s3_image_cost + os_s3_get_cost
    typer.echo(f"  EBS gp3 index ({_fmt_gb(os_storage_gb)}{', 2× replicated' if os_nodes > 1 else ''}):  ${os_ebs_cost:.4f}/hr")
    typer.echo(f"  S3 images ({_fmt_gb(os_s3_image_gb)}, {num_docs:,} files):    ${os_s3_image_cost:.4f}/hr")
    typer.echo(f"  S3 GET (image fetches, {queries_per_hr} q/hr):  ${os_s3_get_cost:.5f}/hr")
    typer.secho(f"  Total:                           ${os_total:.4f}/hr", fg=typer.colors.BLUE, bold=True)
    typer.echo(
        "  Note: OpenSearch cannot scale to zero — the domain runs 24/7 "
        "regardless of query volume."
    )

    # -----------------------------------------------------------------------
    # LanceDB
    # -----------------------------------------------------------------------
    lance_storage   = lancedb_storage_gb(num_docs, dims, avg_image_kb)
    lance_storage_cost = lance_storage * S3_PER_GB_HR
    lance_s3_req_cost  = s3_query_cost_per_hr(queries_per_hr)
    lance_instance     = pick_lancedb_instance(num_docs, dims)
    lance_compute_cost = lance_instance.price_per_hr * utilization
    lance_total_alwayson = lance_instance.price_per_hr + lance_storage_cost + lance_s3_req_cost
    lance_total_util     = lance_compute_cost + lance_storage_cost + lance_s3_req_cost

    typer.echo("")
    typer.secho("--- LanceDB (EC2 + S3) ---", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Working set (memory-mapped):     {_fmt_gb(vector_storage_gb(num_docs, dims) * 0.05)}")
    typer.echo(f"  Recommended instance:            {lance_instance.name}  ({lance_instance.ram_gb:.0f} GB RAM, {lance_instance.vcpu} vCPU)")
    typer.echo(f"  Compute (always-on):             ${lance_instance.price_per_hr:.4f}/hr")
    if utilization < 1.0:
        typer.echo(f"  Compute ({utilization*100:.0f}% utilization):      ${lance_compute_cost:.4f}/hr")
    typer.echo(f"  S3 storage ({_fmt_gb(lance_storage)}):            ${lance_storage_cost:.4f}/hr")
    typer.echo(f"  S3 GET requests ({queries_per_hr} q/hr):      ${lance_s3_req_cost:.5f}/hr")
    if utilization < 1.0:
        typer.secho(f"  Total (always-on):               ${lance_total_alwayson:.4f}/hr", fg=typer.colors.GREEN)
        typer.secho(f"  Total ({utilization*100:.0f}% utilization):         ${lance_total_util:.4f}/hr", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho(f"  Total:                           ${lance_total_alwayson:.4f}/hr", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        "  Note: LanceDB reads only needed pages via memory-mapping — it does not "
        "load all vectors into RAM."
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    typer.echo("")
    typer.secho("--- Summary ---", bold=True)
    ratio = os_total / lance_total_alwayson if lance_total_alwayson > 0 else float("inf")
    typer.echo(f"  OpenSearch:            ${os_total:.2f}/hr   (${os_total * 24 * 30:,.0f}/month)")
    typer.echo(f"  LanceDB (always-on):   ${lance_total_alwayson:.2f}/hr   (${lance_total_alwayson * 24 * 30:,.0f}/month)")
    if utilization < 1.0:
        typer.echo(f"  LanceDB ({utilization*100:.0f}% util):     ${lance_total_util:.2f}/hr   (${lance_total_util * 24 * 30:,.0f}/month)")
    typer.secho(f"  OpenSearch is ~{ratio:.1f}× more expensive than LanceDB (always-on) at this scale.", bold=True)
    typer.echo("")
    typer.echo(
        "Pricing source: AWS us-east-1 on-demand, ~early 2025. "
        "Actual costs vary by reserved/savings-plan discounts, data transfer, and multi-AZ."
    )


if __name__ == "__main__":
    app()
