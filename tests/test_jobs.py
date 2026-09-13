"""Tests for reading jobs, and for the ordering they are displayed in."""

import pytest

from sltop.slurm import jobs as jobs_module
from sltop.slurm.jobs import _SQUEUE_SEPARATOR, Job, _sort_jobs, get_jobs


@pytest.mark.parametrize(
    "value,expected",
    [
        ("375", 375),
        ("375M", 375),
        ("375G", 384000),
        ("1T", 1024 * 1024),
        ("512K", 0),  # Rounded down to a whole MB.
        ("1.5G", 1536),
        ("", 0),
        ("lots", 0),
        ("10X", 0),
    ],
)
def test_parse_memory(value, expected):
    """Memory sizes are converted to MB, and junk reads as zero."""
    assert Job._parse_memory(value) == expected


def test_parse_tres():
    """TRES parses to counts, with memory in MB and the `gres/` prefix cut."""
    assert Job._parse_tres("cpu=64,mem=375G,node=1,gres/gpu=4") == {
        "cpu": 64,
        "mem": 384000,
        "node": 1,
        "gpu": 4,
    }


def test_parse_tres_empty():
    """An unset allocation parses to nothing rather than raising."""
    assert Job._parse_tres("") == {}


@pytest.mark.parametrize(
    "job_id,expected", [("12345", 12345), ("12345_7", 12345), ("nan", 0)]
)
def test_parse_job_id(job_id, expected):
    """Array tasks report the ID of the array they belong to."""
    assert Job._parse_job_id(job_id) == expected


def test_gres_strips_prefix(make_job):
    """Both spellings of the GRES prefix are removed."""
    assert make_job(tres_per_node="gres/gpu:4").gres == "gpu:4"
    assert make_job(tres_per_node="gres:gpu:4").gres == "gpu:4"


@pytest.mark.parametrize(
    "tres_per_node,expected",
    [
        ("gres/gpu:4", {"gpu": 4}),
        ("gres:gpu:4", {"gpu": 4}),
        ("gres/gpu:h100:2", {"gpu": 2}),
        ("cpu:8,gpu:1", {"cpu": 8, "gpu": 1}),
        ("", {}),
        ("gpu", {}),
    ],
)
def test_get_resources_per_node(make_job, tres_per_node, expected):
    """Per-node GRES parses to counts keyed by resource."""
    assert make_job(tres_per_node=tres_per_node).get_resources_per_node() == (
        expected
    )


def test_time_used(make_job, monkeypatch):
    """Running jobs report elapsed time; anything else reports nothing."""
    monkeypatch.setattr(jobs_module.time, "time", lambda: 3661.0)
    assert make_job(job_state="RUNNING", start_time=0).time_used == "1:01:01"
    assert make_job(job_state="PENDING", start_time=0).time_used == "-"


def test_time_queued(make_job, monkeypatch):
    """Queued time is measured from submission, if it is known."""
    monkeypatch.setattr(jobs_module.time, "time", lambda: 3661.0)
    assert make_job(submit_time=1).time_queued == "1:01:00"
    assert make_job(submit_time=0).time_queued == "-"


def test_partitions(make_job):
    """A job may be eligible for more than one partition."""
    assert make_job(partition="dev,batch").partitions == ["dev", "batch"]


def test_job_from_record_running():
    """A running job keeps its allocation, and drops the request."""
    job = Job.from_record(
        {
            "JobId": "123",
            "Partition": "batch",
            "JobName": "train",
            "UserId": "alice(1000)",
            "JobState": "RUNNING",
            "Nice": "0",
            "NumNodes": "2",
            "NodeList": "b[0-1]",
            "TresPerNode": "gres/gpu:4",
            "NumCPUs": "64",
            "AllocTRES": "cpu=64,mem=375G,gres/gpu=8",
            "ReqTRES": "cpu=64,mem=375G,gres/gpu=8",
            "Command": "/bin/train.sh",
            "Reason": "None",
        }
    )
    assert job.job_id == 123
    assert job.user_name == "alice"
    assert job.node_count == 2
    assert job.get_resources_total() == {"cpu": 64, "mem": 384000, "gpu": 8}
    assert job.tres_req == ""


def test_job_from_record_pending():
    """A pending job holds nothing, so its request is kept instead."""
    job = Job.from_record(
        {
            "JobId": "124",
            "JobState": "PENDING",
            "ReqTRES": "cpu=8,mem=16G,gres/gpu=1",
            "Reason": "Resources",
        }
    )
    assert job.tres_alloc == ""
    assert job.get_resources_requested() == {"cpu": 8, "mem": 16384, "gpu": 1}


def test_job_from_record_pending_reports_request_as_alloc():
    """A pending job's request is reported in the allocation field."""
    job = Job.from_record(
        {"JobId": "124", "JobState": "PENDING", "AllocTRES": "cpu=8,mem=16G"}
    )
    assert job.tres_alloc == ""
    assert job.get_resources_requested() == {"cpu": 8, "mem": 16384}


def test_job_from_record_legacy_tres():
    """Older Slurm versions report the allocation as `TRES`."""
    job = Job.from_record(
        {"JobId": "125", "JobState": "RUNNING", "TRES": "cpu=8,gres/gpu=1"}
    )
    assert job.get_resources_total() == {"cpu": 8, "gpu": 1}


def test_job_from_record_defaults():
    """An empty record still produces a usable job."""
    job = Job.from_record({})
    assert job.job_id == 0
    assert job.job_state == "UNKNOWN"
    assert job.state_reason == "None"


def test_sort_jobs(make_job):
    """Jobs are grouped by state and reason, then ordered within the group."""
    running_old = make_job(job_id=1, job_state="RUNNING", start_time=100)
    running_new = make_job(job_id=2, job_state="RUNNING", start_time=200)
    resources = make_job(
        job_id=3, job_state="PENDING", state_reason="Resources"
    )
    priority_nice = make_job(
        job_id=4, job_state="PENDING", state_reason="Priority", nice=50
    )
    priority = make_job(
        job_id=5, job_state="PENDING", state_reason="Priority", nice=0
    )
    qos = make_job(
        job_id=6, job_state="PENDING", state_reason="QOSMaxGRESPerUser"
    )
    dependency = make_job(
        job_id=7, job_state="PENDING", state_reason="Dependency"
    )

    order = _sort_jobs(
        [
            dependency,
            priority_nice,
            qos,
            running_new,
            resources,
            priority,
            running_old,
        ]
    )
    assert [j.job_id for j in order] == [1, 2, 3, 5, 4, 6, 7]


def test_sort_jobs_is_a_total_order(make_job):
    """Every job survives the sort, whatever its reason is."""
    unknown = make_job(job_id=8, job_state="PENDING", state_reason="BeginTime")
    running = make_job(job_id=9, job_state="RUNNING")
    assert len(_sort_jobs([unknown, running])) == 2


def _squeue_line(**fields) -> str:
    """Builds a line of the squeue output sltop asks for."""
    return _SQUEUE_SEPARATOR.join(
        fields.get(key, "").ljust(width)
        for key, _, width in jobs_module._SQUEUE_FIELDS
    )


def test_get_jobs(monkeypatch):
    """Jobs are read from squeue, which is padded to a fixed field width."""
    lines = [
        _squeue_line(
            JobId="1",
            Partition="batch",
            UserId="alice",
            JobState="RUNNING",
            NodeList="b0",
            JobName="train",
        ),
        _squeue_line(
            JobId="2",
            Partition="batch",
            UserId="bob",
            JobState="PENDING",
            Reason="Resources",
            JobName="eval",
        ),
    ]
    monkeypatch.setattr(
        jobs_module.subprocess, "check_output", lambda *a, **k: "\n".join(lines)
    )

    jobs = get_jobs()
    assert [(j.job_id, j.name, j.user_name) for j in jobs] == [
        (1, "train", "alice"),
        (2, "eval", "bob"),
    ]


def test_get_jobs_filters(monkeypatch):
    """Finished, invalid and out-of-partition jobs are dropped."""
    lines = [
        _squeue_line(JobId="1", Partition="batch", JobState="RUNNING"),
        _squeue_line(JobId="2", Partition="batch", JobState="COMPLETED"),
        _squeue_line(JobId="3", Partition="dev", JobState="RUNNING"),
        _squeue_line(
            JobId="4",
            Partition="batch",
            JobState="PENDING",
            Reason="DependencyNeverSatisfied",
        ),
    ]
    monkeypatch.setattr(
        jobs_module.subprocess, "check_output", lambda *a, **k: "\n".join(lines)
    )

    assert [j.job_id for j in get_jobs()] == [1, 3]
    assert [j.job_id for j in get_jobs(include_invalid=True)] == [1, 3, 4]
    assert [j.job_id for j in get_jobs(partition="batch")] == [1]


def test_get_jobs_falls_back_to_scontrol(monkeypatch):
    """If squeue cannot supply the TRES fields, scontrol is used instead."""

    def fail(*args, **kwargs):
        raise FileNotFoundError("squeue")

    monkeypatch.setattr(jobs_module.subprocess, "check_output", fail)
    monkeypatch.setattr(
        jobs_module.scontrol,
        "show",
        lambda *a, **k: [{"JobId": "7", "JobState": "RUNNING"}],
    )
    assert [j.job_id for j in get_jobs()] == [7]
