"""Tests for attributing job resources to the nodes they run on."""

from sltop.slurm.usage import calculate_node_usage


def test_usage_per_node_gres(make_job, make_node):
    """A job's per-node GRES is charged to each node it holds."""
    nodes = [make_node(name="b0"), make_node(name="b1")]
    job = make_job(
        nodelist="b[0-1]",
        tres_per_node="gres/gpu:4",
        tres_alloc="cpu=64,mem=256G,gres/gpu=8",
    )

    usage = calculate_node_usage(nodes, [job])
    assert usage["b0"]["gpus"] == {"batch": 4}
    assert usage["b1"]["gpus"] == {"batch": 4}


def test_usage_spreads_allocation_over_nodes(make_job, make_node):
    """Without per-node GRES, the total allocation is split evenly."""
    nodes = [make_node(name="b0"), make_node(name="b1")]
    job = make_job(nodelist="b[0-1]", tres_alloc="cpu=64,mem=256G,gres/gpu=8")

    usage = calculate_node_usage(nodes, [job])
    assert usage["b0"] == {
        "cpus": {"batch": 32},
        "gpus": {"batch": 4},
        "memory": {"batch": 131072},
    }


def test_usage_accumulates_by_partition(make_job, make_node):
    """Jobs on one node add up, and are kept apart by partition."""
    nodes = [make_node(name="b0")]
    jobs = [
        make_job(job_id=1, partition="batch", cpus=8, tres_alloc="cpu=8"),
        make_job(job_id=2, partition="batch", cpus=16, tres_alloc="cpu=16"),
        make_job(job_id=3, partition="dev", cpus=4, tres_alloc="cpu=4"),
    ]

    usage = calculate_node_usage(nodes, jobs)
    assert usage["b0"]["cpus"] == {"batch": 24, "dev": 4}


def test_usage_ignores_jobs_which_are_not_running(make_job, make_node):
    """A pending job holds nothing, so it is not charged to any node."""
    nodes = [make_node(name="b0")]
    job = make_job(
        job_state="PENDING", nodelist="", tres_alloc="", tres_req="cpu=8"
    )

    usage = calculate_node_usage(nodes, [job])
    assert usage["b0"] == {"cpus": {}, "gpus": {}, "memory": {}}


def test_usage_ignores_unknown_nodes(make_job, make_node):
    """Jobs on nodes which are not shown, e.g. in another partition, are cut."""
    nodes = [make_node(name="b0")]
    job = make_job(nodelist="c9", tres_alloc="cpu=8")

    usage = calculate_node_usage(nodes, [job])
    assert set(usage) == {"b0"}
    assert usage["b0"]["cpus"] == {}


def test_usage_falls_back_to_cpu_count(make_job, make_node):
    """A job with no CPU TRES is charged the CPU count squeue reports."""
    nodes = [make_node(name="b0")]
    job = make_job(nodelist="b0", cpus=12, tres_alloc="mem=16G")

    usage = calculate_node_usage(nodes, [job])
    assert usage["b0"]["cpus"] == {"batch": 12}
