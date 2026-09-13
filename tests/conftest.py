"""Shared fixtures for the slurm module tests."""

import pytest

from sltop.slurm.jobs import Job
from sltop.slurm.nodes import Node

_JOB_DEFAULTS = {
    "job_id": 1,
    "partition": "batch",
    "name": "job",
    "user_name": "alice",
    "job_state": "RUNNING",
    "start_time": 0,
    "nice": 0,
    "node_count": 1,
    "nodelist": "b0",
    "tres_per_node": "",
    "state_reason": "None",
    "cpus": 8,
    "tres_alloc": "",
    "tres_req": "",
    "submit_time": 0,
    "req_nodes": "",
    "command": "/bin/true",
}

_NODE_DEFAULTS = {
    "name": "b0",
    "cpus": 64,
    "memory": 512000,
    "gpus": 8,
    "architecture": "x86_64",
    "state": "MIXED",
    "partitions": ["batch"],
}


@pytest.fixture
def make_job():
    """Returns a factory for Jobs, with everything unset defaulted."""

    def factory(**kwargs) -> Job:
        return Job(**{**_JOB_DEFAULTS, **kwargs})

    return factory


@pytest.fixture
def make_node():
    """Returns a factory for Nodes, with everything unset defaulted."""

    def factory(**kwargs) -> Node:
        return Node(**{**_NODE_DEFAULTS, **kwargs})

    return factory
