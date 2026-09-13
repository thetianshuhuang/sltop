"""Tests for reading nodes and expanding nodelists."""

import pytest

from sltop.slurm import nodes
from sltop.slurm.nodes import Node, expand_nodelist, get_nodes


@pytest.mark.parametrize(
    "gres,expected",
    [
        ("", 0),
        ("(null)", 0),
        ("gpu:8", 8),
        ("gpu:h100:4", 4),
        ("gpu:a100:2(IDX:0-1)", 2),
        ("gpu:a100:2,gpu:h100:1", 3),
        ("shard:16,gpu:4", 4),
        ("shard:16", 0),
        ("gpu:no_count", 0),
    ],
)
def test_parse_gpu_count(gres, expected):
    """GPU counts are summed over the GRES entries which are GPUs."""
    assert Node._parse_gpu_count(gres) == expected


def test_node_from_record():
    """A node record maps onto the fields sltop displays."""
    node = Node.from_record(
        {
            "NodeName": "b0",
            "CPUTot": "64",
            "RealMemory": "512000",
            "Gres": "gpu:h100:8",
            "Arch": "x86_64",
            "State": "MIXED",
            "Partitions": "dev,batch",
        }
    )
    assert node == Node(
        name="b0",
        cpus=64,
        memory=512000,
        gpus=8,
        architecture="x86_64",
        state="MIXED",
        partitions=["dev", "batch"],
    )


def test_node_from_record_defaults():
    """An empty record still produces a usable node."""
    node = Node.from_record({})
    assert node.name == "unknown"
    assert node.state == "UNKNOWN"
    assert (node.cpus, node.memory, node.gpus) == (0, 0, 0)
    assert node.partitions == []


def test_get_nodes(monkeypatch):
    """Nodes come back sorted by name, and can be filtered by partition."""
    monkeypatch.setattr(
        nodes.scontrol,
        "show",
        lambda *a, **k: [
            {"NodeName": "b1", "Partitions": "batch"},
            {"NodeName": "b0", "Partitions": "dev,batch"},
        ],
    )

    assert [n.name for n in get_nodes()] == ["b0", "b1"]
    assert [n.name for n in get_nodes(partition="dev")] == ["b0"]
    assert get_nodes(partition="gpu") == []


@pytest.mark.parametrize(
    "nodelist,expected",
    [
        ("", []),
        ("b0", ["b0"]),
        ("b0,c1", ["b0", "c1"]),
        ("b[0-2]", ["b0", "b1", "b2"]),
        ("b[0-2],c1", ["b0", "b1", "b2", "c1"]),
        ("b[0,3-4]", ["b0", "b3", "b4"]),
        ("robo-gh00[1-2]", ["robo-gh001", "robo-gh002"]),
        ("b[08-10]", ["b08", "b09", "b10"]),
        ("b[1-2]x[3-4]", ["b1x3", "b1x4", "b2x3", "b2x4"]),
    ],
)
def test_expand_nodelist(nodelist, expected):
    """Ranges expand the way `scontrol show hostnames` expands them."""
    assert expand_nodelist(nodelist) == expected


def test_expand_nodelist_falls_back_to_scontrol(monkeypatch):
    """Syntax which is not understood is handed to scontrol."""
    monkeypatch.setattr(
        nodes.subprocess, "check_output", lambda *a, **k: "b0\nb1\n"
    )
    assert expand_nodelist("b[x-y]") == ["b0", "b1"]


def test_expand_nodelist_fallback_unavailable(monkeypatch):
    """If scontrol cannot be run either, the nodelist expands to nothing."""

    def fail(*args, **kwargs):
        raise FileNotFoundError("scontrol")

    monkeypatch.setattr(nodes.subprocess, "check_output", fail)
    assert expand_nodelist("b[x-y]") == []


def test_get_slurm_version(monkeypatch):
    """The version is the second word of `squeue --version`."""
    monkeypatch.setattr(
        nodes.subprocess, "check_output", lambda *a, **k: "slurm-wlm 23.11.4\n"
    )
    assert nodes.get_slurm_version() == "23.11.4"


def test_get_slurm_version_unavailable(monkeypatch):
    """A missing squeue reports an unknown version rather than raising."""

    def fail(*args, **kwargs):
        raise FileNotFoundError("squeue")

    monkeypatch.setattr(nodes.subprocess, "check_output", fail)
    assert nodes.get_slurm_version() == "unknown"
