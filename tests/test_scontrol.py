"""Tests for parsing the `key=value` output of scontrol."""

import subprocess

import pytest

from sltop.slurm import scontrol


def test_parse_record():
    """Fields are split on the ` Key=` which follows them."""
    record = scontrol.parse_record(
        "JobId=123 JobName=my job Partition=batch NumCPUs=8"
    )
    assert record == {
        "JobId": "123",
        "JobName": "my job",
        "Partition": "batch",
        "NumCPUs": "8",
    }


def test_parse_record_punctuated_keys():
    """Keys may contain `/`, `:` and `.`, and values may contain `=`."""
    record = scontrol.parse_record(
        "CPUs/Task=4 ReqB:S:C:T=0:0:*:* Comment=a=b MinMemoryNode=0.5"
    )
    assert record == {
        "CPUs/Task": "4",
        "ReqB:S:C:T": "0:0:*:*",
        "Comment": "a=b",
        "MinMemoryNode": "0.5",
    }


def test_parse_record_duplicate_key_keeps_first():
    """A key which appears twice keeps the value it was first given."""
    assert scontrol.parse_record("State=UP State=DOWN") == {"State": "UP"}


def test_parse_record_empty():
    """A line with no pairs parses as an empty record."""
    assert scontrol.parse_record("No jobs in the system") == {}


@pytest.mark.parametrize("value", ["", "(null)", "N/A"])
def test_get_unset(value):
    """The placeholders Slurm prints for an unset field become the default."""
    assert scontrol.get({"Command": value}, "Command", "-") == "-"


def test_get_missing_key():
    """A key which is absent entirely also becomes the default."""
    assert scontrol.get({}, "Command", "-") == "-"
    assert scontrol.get({}, "Command") == ""


def test_get_list():
    """A comma separated field splits, and an unset one is empty."""
    assert scontrol.get_list({"Partitions": "dev,batch"}, "Partitions") == [
        "dev",
        "batch",
    ]
    assert scontrol.get_list({"Partitions": "(null)"}, "Partitions") == []


def test_get_int():
    """Integers parse, ranges take their lower bound, junk takes the default."""
    assert scontrol.get_int({"NumCPUs": "64"}, "NumCPUs") == 64
    assert scontrol.get_int({"NumNodes": "1-4"}, "NumNodes") == 1
    assert scontrol.get_int({"Nice": "-100"}, "Nice") == -100
    assert scontrol.get_int({"NumCPUs": "Unknown"}, "NumCPUs", 7) == 7
    assert scontrol.get_int({}, "NumCPUs", 7) == 7


def test_get_time():
    """Timestamps parse, and placeholders take the default."""
    record = {"StartTime": "2026-09-12T04:45:31", "EndTime": "Unknown"}
    assert scontrol.get_time(record, "StartTime") > 0
    assert scontrol.get_time(record, "EndTime") == 0
    assert scontrol.get_time({}, "StartTime") == 0


def test_show(monkeypatch):
    """Each line becomes a record, and blank lines are dropped."""
    output = "NodeName=b0 CPUTot=64\nNodeName=b1 CPUTot=32\n\n"
    monkeypatch.setattr(
        scontrol.subprocess, "check_output", lambda *a, **k: output
    )

    records = scontrol.show("node")
    assert [r["NodeName"] for r in records] == ["b0", "b1"]


def test_show_contains_filter(monkeypatch):
    """`contains` drops lines before they are parsed."""
    output = "JobId=1 Partition=dev\nJobId=2 Partition=batch\n"
    monkeypatch.setattr(
        scontrol.subprocess, "check_output", lambda *a, **k: output
    )

    records = scontrol.show("job", contains="dev")
    assert [r["JobId"] for r in records] == ["1"]


def test_show_unavailable(monkeypatch):
    """A missing or failing scontrol yields no records rather than raising."""

    def fail(*args, **kwargs):
        raise FileNotFoundError("scontrol")

    monkeypatch.setattr(scontrol.subprocess, "check_output", fail)
    assert scontrol.show("node") == []

    def nonzero(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "scontrol")

    monkeypatch.setattr(scontrol.subprocess, "check_output", nonzero)
    assert scontrol.show("node") == []
