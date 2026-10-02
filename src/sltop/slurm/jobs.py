"""Module interacting with Slurm via scontrol to get job info."""

import dataclasses
import datetime
import subprocess
import time

from . import scontrol

# Fields are separated by the unit separator, which job names and commands
# cannot contain, unlike any printable character.
_SQUEUE_SEPARATOR = "\x1f"

# Fields to read from squeue, as (scontrol key, squeue field, width), so that
# squeue output can be turned into the records `scontrol show job` returns.
# squeue pads and truncates each field to its width, so these are generous.
_SQUEUE_FIELDS = (
    ("JobId", "jobid", 24),
    ("ArrayJobId", "arrayjobid", 24),
    ("ArrayTaskId", "arraytaskid", 64),
    ("Partition", "partition", 24),
    ("UserId", "username", 24),
    ("JobState", "state", 16),
    ("StartTime", "starttime", 24),
    ("SubmitTime", "submittime", 24),
    ("Nice", "nice", 12),
    ("NumNodes", "numnodes", 10),
    ("NodeList", "nodelist", 64),
    ("ReqNodeList", "reqnodes", 64),
    ("TresPerNode", "tres-per-node", 48),
    ("Reason", "reason", 32),
    ("NumCPUs", "numcpus", 10),
    ("AllocTRES", "tres-alloc", 128),
    ("CpusPerTres", "cpus-per-tres", 48),
    ("MemPerTres", "mem-per-tres", 48),
    # squeue does not say whether this is per node or per CPU, unlike scontrol,
    # which reports either MinMemoryNode or MinMemoryCPU.
    ("MinMemory", "minmemory", 16),
    ("Command", "command", 256),
    ("JobName", "name", 256),
)

# Base states which Slurm considers finished; scontrol keeps reporting these
# jobs for a few minutes after they complete, but squeue hides them by default.
_FINISHED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}

# Whether the memory of a pending job is per node or per CPU, which squeue does
# not report, as {job ID: scontrol memory fields}. A job's request is fixed once
# it is submitted, so each job only needs to be looked up once.
_MEMORY_FIELDS = ("MinMemoryNode", "MinMemoryCPU")
_memory_cache: dict[str, dict[str, str]] = {}


@dataclasses.dataclass
class Job:
    """Represents a single Slurm job."""

    job_id: int
    partition: str
    name: str
    user_name: str
    job_state: str
    start_time: int
    nice: int
    node_count: int
    nodelist: str
    tres_per_node: str
    state_reason: str
    cpus: int
    tres_alloc: str  # Total allocated TRES, e.g. "cpu=64,mem=375G,gres/gpu=4"
    # TRES the job asked for, in the same format, with defaults filled in
    tres_req: str
    submit_time: int
    req_nodes: str  # Nodes the job explicitly asked for, if any
    command: str
    array_job_id: int  # ID of the array the job belongs to, if any
    # Task index, or for tasks still pending together, their indices along
    # with any throttle, e.g. "0-30%4"; empty if the job is not an array.
    array_task_id: str

    @property
    def display_id(self) -> str:
        """Returns the job ID as squeue shows it, e.g. "25190_1".

        Array tasks which are still pending together are shown as one job,
        with their indices in brackets, e.g. "25578_[0-30%4]".
        """
        if not self.array_task_id:
            return str(self.job_id)
        if self.array_task_id.isdigit():
            return f"{self.array_job_id}_{self.array_task_id}"
        return f"{self.array_job_id}_[{self.array_task_id}]"

    @property
    def partitions(self) -> list[str]:
        """Returns the partitions the job may run in."""
        return self.partition.split(",")

    @property
    def gres(self) -> str:
        """Returns the generic resources per node, e.g. "gpu:h100:2".

        Slurm prefixes these with "gres/" or "gres:" depending on the version.
        """
        return self.tres_per_node.removeprefix("gres/").removeprefix("gres:")

    @property
    def time_used(self) -> str:
        """Returns the time used by the job formatted as H:MM:SS."""
        if self.job_state != "RUNNING":
            return "-"

        now = int(time.time())
        diff = now - self.start_time
        return str(datetime.timedelta(seconds=diff))

    @property
    def time_queued(self) -> str:
        """Returns how long the job has been queued, formatted as H:MM:SS."""
        if not self.submit_time:
            return "-"

        return str(
            datetime.timedelta(seconds=int(time.time()) - self.submit_time)
        )

    @staticmethod
    def _parse_memory(mem_str: str) -> int:
        """Parses a memory size such as "375G" into MB.

        A value with no unit suffix is already in MB.
        """
        units = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
        try:
            unit = mem_str[-1].upper()
            if unit.isdigit():
                return int(mem_str)  # Default MB

            return int(float(mem_str[:-1]) * units[unit])
        except (IndexError, KeyError, ValueError):
            return 0

    def get_resources_per_node(self) -> dict:
        """Parses tres_per_node into a dictionary {type: count}.

        Examples:
            "gpu:4" -> {'gpu': 4}
            "cpu:8,gpu:1" -> {'cpu': 8, 'gpu': 1}
        """
        res = {}
        for part in self.gres.split(","):
            if ":" in part:
                # key:val or key:type:val
                sub = part.split(":")
                key = sub[0]
                try:
                    val = int(sub[-1])
                    res[key] = val
                except ValueError:
                    pass
        return res

    def get_resources_total(self) -> dict:
        """Returns the resources allocated to the job, as {type: count}."""
        return Job._parse_tres(self.tres_alloc)

    def get_resources_requested(self) -> dict:
        """Returns the resources the job asked for, as {type: count}."""
        return Job._parse_tres(self.tres_req)

    @staticmethod
    def _parse_tres(tres: str) -> dict:
        """Parses a TRES string into a dictionary {type: count}.

        Memory is converted to MB.

        Examples:
            "cpu=64,mem=375G,node=1,gres/gpu=4"
                -> {'cpu': 64, 'mem': 384000, 'node': 1, 'gpu': 4}
        """
        res = {}
        for part in tres.split(","):
            key, _, val = part.partition("=")
            key = key.rsplit("/", 1)[-1]  # "gres/gpu" -> "gpu"
            if key == "mem":
                res[key] = Job._parse_memory(val)
            else:
                try:
                    res[key] = int(val)
                except ValueError:
                    pass
        return res

    @staticmethod
    def _parse_per_gres(value: str) -> dict:
        """Parses a per-GRES amount into a dictionary {gres: amount}.

        Examples:
            "gres/gpu:30" -> {'gpu': '30'}
            "gpu:96000" -> {'gpu': '96000'}
        """
        res = {}
        for part in value.split(","):
            fields = part.removeprefix("gres/").removeprefix("gres:").split(":")
            if len(fields) >= 2:
                res[fields[0]] = fields[-1]
        return res

    @staticmethod
    def _resolve_request(tres_req: str, data: dict) -> str:
        """Fills the defaults Slurm leaves out of a pending job's request.

        Slurm reports the request as submitted: CPUs per GPU are never applied
        to it, memory per CPU only counts the CPUs asked for directly, and a
        job which relies on its partition's memory per GPU asks for whole nodes
        of memory instead.

        Args:
            tres_req: The request, e.g. "cpu=1,mem=1500000M,gres/gpu=1".
            data: The job's record, for the fields which qualify the request.

        Returns:
            The request with these filled in, in the same format.
        """
        req = Job._parse_tres(tres_req)
        gpus = req.get("gpu", 0)
        if not gpus:
            # Without GPUs, the request already counts every CPU.
            return tres_req

        cpus_per_gpu = Job._parse_per_gres(scontrol.get(data, "CpusPerTres"))
        mem_per_gpu = Job._parse_per_gres(scontrol.get(data, "MemPerTres"))
        if "gpu" in cpus_per_gpu:
            req["cpu"] = max(
                req.get("cpu", 0), scontrol.get_int(cpus_per_gpu, "gpu") * gpus
            )

        # MinMemory comes from squeue, which cannot say which of these it is.
        mem_per_cpu = scontrol.get(data, "MinMemoryCPU")
        mem_per_node = scontrol.get(data, "MinMemoryNode") or scontrol.get(
            data, "MinMemory"
        )
        if mem_per_cpu:
            req["mem"] = Job._parse_memory(mem_per_cpu) * req.get("cpu", 0)
        elif mem_per_node == "0" and "gpu" in mem_per_gpu:
            req["mem"] = Job._parse_memory(mem_per_gpu["gpu"]) * gpus

        return ",".join(f"{key}={value}" for key, value in req.items())

    @staticmethod
    def _parse_job_id(job_id: str) -> int:
        """Parses a job ID, e.g. "12345", or "12345_7" for job array tasks."""
        try:
            return int(job_id.split("_")[0])
        except ValueError:
            return 0

    @classmethod
    def from_record(cls, data: dict) -> "Job":
        """Creates a Job instance from a `scontrol show job` record.

        Args:
            data: Dictionary containing job information from scontrol.

        Returns:
            A Job instance with parsed and validated data.
        """
        state = scontrol.get(data, "JobState", "UNKNOWN")

        # Older Slurm versions report the allocation as "TRES" instead.
        tres_alloc = scontrol.get(data, "AllocTRES") or scontrol.get(
            data, "TRES"
        )

        # Nothing is allocated until a job runs, and squeue reports what a
        # pending job asked for in place of its (empty) allocation.
        tres_req = scontrol.get(data, "ReqTRES")
        if state == "PENDING":
            tres_req = Job._resolve_request(tres_req or tres_alloc, data)
            tres_alloc = ""
        else:
            # squeue cannot report the request once a job has started, so it is
            # dropped here too, to keep the two sources equivalent.
            tres_req = ""

        return cls(
            job_id=Job._parse_job_id(scontrol.get(data, "JobId")),
            partition=scontrol.get(data, "Partition"),
            name=scontrol.get(data, "JobName"),
            # "UserId=alice(1000)" -> "alice"
            user_name=scontrol.get(data, "UserId").split("(")[0],
            job_state=state,
            start_time=scontrol.get_time(data, "StartTime"),
            nice=scontrol.get_int(data, "Nice"),
            node_count=scontrol.get_int(data, "NumNodes"),
            nodelist=scontrol.get(data, "NodeList"),
            tres_per_node=scontrol.get(data, "TresPerNode"),
            state_reason=scontrol.get(data, "Reason", "None"),
            cpus=scontrol.get_int(data, "NumCPUs"),
            tres_alloc=tres_alloc,
            tres_req=tres_req,
            submit_time=scontrol.get_time(data, "SubmitTime"),
            req_nodes=scontrol.get(data, "ReqNodeList"),
            command=scontrol.get(data, "Command"),
            array_job_id=scontrol.get_int(data, "ArrayJobId"),
            array_task_id=scontrol.get(data, "ArrayTaskId"),
        )


def _squeue_records(partition: str | None) -> list[dict[str, str]] | None:
    """Fetches job records from squeue, in the shape scontrol.show returns.

    Unlike `scontrol show job`, squeue can restrict itself to one partition,
    and only formats the fields which are actually used. On a large cluster
    that is the difference between seconds and milliseconds. It does however
    need the TRES fields, which were added in Slurm 19.05.

    Args:
        partition: If set, only fetch jobs in this partition.

    Returns:
        One record per job, or None if squeue could not supply these fields.
    """
    # Every field but the last is suffixed with the separator.
    fields = [
        f"{field}:{width}{_SQUEUE_SEPARATOR}"
        for _, field, width in _SQUEUE_FIELDS
    ]
    fields[-1] = fields[-1].rstrip(_SQUEUE_SEPARATOR)

    command = ["squeue", "--noheader", "--Format=" + ",".join(fields)]
    if partition is not None:
        command.append(f"--partition={partition}")

    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None

    keys = [key for key, _, _ in _SQUEUE_FIELDS]
    records = []
    for line in output.splitlines():
        values = line.split(_SQUEUE_SEPARATOR, len(keys) - 1)
        if len(values) == len(keys):
            records.append(dict(zip(keys, (v.strip() for v in values))))
    return records


def _add_memory_fields(records: list[dict[str, str]]) -> None:
    """Adds whether memory is per node or per CPU to squeue records.

    This only matters for pending jobs whose CPUs are set per GPU, and which
    ask for a particular amount of memory, so only those are looked up.

    Args:
        records: Records from squeue, which are updated in place.
    """
    ambiguous = {
        r["JobId"]: r
        for r in records
        if r.get("JobState") == "PENDING"
        and scontrol.get(r, "CpusPerTres")
        and scontrol.get(r, "MinMemory", "0") != "0"
    }

    for job_id in ambiguous.keys() - _memory_cache.keys():
        fields = {}
        for record in scontrol.show("job", name=job_id):
            # Showing an array shows all of its tasks.
            if record.get("JobId") == job_id:
                fields = {k: record[k] for k in _MEMORY_FIELDS if k in record}
        _memory_cache[job_id] = fields

    # Jobs which are no longer pending will not be looked up again.
    for job_id in _memory_cache.keys() - ambiguous.keys():
        del _memory_cache[job_id]

    for job_id, record in ambiguous.items():
        record.update(_memory_cache[job_id])


def _sort_jobs(jobs: list[Job]) -> list[Job]:
    """Sorts jobs by state, then by reason and nice value.

    The ordering is:

    1. Running jobs, in decreasing order of execution time (longest first).
    2. Pending jobs waiting for resources (Reason: Resources).
    3. Pending jobs with reason Priority, sorted by nice.
    4. Pending jobs with reason Dependency, sorted by nice.
    5. Failed/Cancelled/Other.
    """
    # Tasks of the same array are kept together.
    jobs.sort(key=lambda j: (j.array_job_id or j.job_id, j.job_id))

    job_categories = {}
    for j in jobs:
        if j.job_state == "RUNNING":
            category = "RUNNING"
        else:
            category = j.state_reason

        if category not in job_categories:
            job_categories[category] = []
        job_categories[category].append(j)

    for category, category_jobs in job_categories.items():
        if category != "RUNNING":
            category_jobs.sort(key=lambda j: j.nice)

    sorted_jobs = (
        sorted(job_categories.pop("RUNNING", []), key=lambda j: j.start_time)
        + job_categories.pop("Resources", [])
        + job_categories.pop("Priority", [])
    )

    for k in job_categories:
        if "qos" in k.lower():
            sorted_jobs += job_categories[k]
    job_categories = {
        k: v for k, v in job_categories.items() if "qos" not in k.lower()
    }

    sorted_jobs += job_categories.pop("Dependency", [])

    for k, v in job_categories.items():
        sorted_jobs += v

    return sorted_jobs


def get_jobs(
    include_invalid: bool = False, partition: str | None = None
) -> list[Job]:
    """Fetches the jobs which are currently queued or running, in display order.

    Args:
        include_invalid: Whether to include jobs which can never run, i.e. jobs
            whose dependencies can never be satisfied.
        partition: If set, only include jobs in this partition.
    """
    records = _squeue_records(partition)
    if records is None:
        records = scontrol.show("job", contains=partition)
    else:
        _add_memory_fields(records)

    jobs = [Job.from_record(r) for r in records]
    jobs = [j for j in jobs if j.job_state not in _FINISHED_STATES]
    if partition is not None:
        jobs = [j for j in jobs if partition in j.partitions]
    if not include_invalid:
        jobs = [j for j in jobs if j.state_reason != "DependencyNeverSatisfied"]
    return _sort_jobs(jobs)
