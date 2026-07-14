"""Estimate node-hours for an AiiDA process tree.

Usage:
    uv run python scripts/count_node_hours.py PK
    uv run python scripts/count_node_hours.py --profile PROFILE_NAME PK

The public helper is ``count_node_hours(profile_name, pk)``. It loads the
requested AiiDA profile, walks the process tree below ``pk``, and estimates
node-hours for all descendant CalcJobs from scheduler accounting metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import aiida
from aiida import orm

DEFAULT_PROFILE_NAME = "presto-pg"


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None

    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.lower() in {"unknown", "n/a", "none", "null"}:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_number(value)
    if number is None:
        return None
    return int(number)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, Mapping) and "date" in value:
        value = value["date"]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _parse_slurm_elapsed(value: Any) -> int | None:
    """Parse Slurm elapsed time strings such as ``DD-HH:MM:SS`` or raw seconds."""
    seconds = _to_int(value)
    if seconds is not None:
        return seconds
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text.lower() in {"unknown", "n/a"}:
        return None

    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = _to_int(day_text) or 0

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(part) for part in parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (int(part) for part in parts)
        else:
            return None
    except ValueError:
        return None

    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _split_sacct_fields(line: str) -> list[str]:
    fields = line.split("|")
    if fields and fields[-1] == "":
        fields = fields[:-1]
    return fields


def _parse_sacct_stdout(stdout: str, scheduler_job_id: str | None) -> dict[str, str]:
    """Return the best matching Slurm ``sacct --parsable`` row as a dictionary."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return {}

    header = _split_sacct_fields(lines[0])
    rows = []
    for line in lines[1:]:
        fields = _split_sacct_fields(line)
        if len(fields) < len(header):
            fields.extend([""] * (len(header) - len(fields)))
        rows.append(dict(zip(header, fields, strict=False)))

    if scheduler_job_id:
        for row in rows:
            if (
                row.get("JobID") == scheduler_job_id
                or row.get("JobIDRaw") == scheduler_job_id
            ):
                return row

    for row in rows:
        job_id = row.get("JobID", "")
        if job_id and "." not in job_id:
            return row

    return rows[0] if rows else {}


def _walltime_from_detailed_info(
    node: orm.CalcJobNode,
) -> tuple[int | None, str | None, list[str]]:
    detailed = node.get_detailed_job_info() or {}
    warnings = []

    if detailed.get("retval") not in (None, 0):
        warnings.append(f"detailed_job_info retval={detailed.get('retval')}")
    if detailed.get("stderr"):
        warnings.append("detailed_job_info has stderr")

    stdout = detailed.get("stdout")
    if not isinstance(stdout, str):
        return None, None, warnings

    row = _parse_sacct_stdout(stdout, node.get_job_id())
    for key in ("ElapsedRaw", "ElapsedRawSeconds", "Elapsed"):
        seconds = _parse_slurm_elapsed(row.get(key))
        if seconds is not None:
            return seconds, f"detailed_job_info.stdout:{key}", warnings

    return None, None, warnings


def _nodes_from_detailed_info(node: orm.CalcJobNode) -> tuple[int | None, str | None]:
    detailed = node.get_detailed_job_info() or {}
    stdout = detailed.get("stdout")
    if not isinstance(stdout, str):
        return None, None

    row = _parse_sacct_stdout(stdout, node.get_job_id())
    for key in ("AllocNodes", "NNodes", "NumNodes"):
        nodes = _to_int(row.get(key))
        if nodes is not None and nodes > 0:
            return nodes, f"detailed_job_info.stdout:{key}"

    return None, None


def _walltime_from_last_job_info(
    node: orm.CalcJobNode,
) -> tuple[int | None, str | None]:
    last_info = node.get_last_job_info()
    if not isinstance(last_info, Mapping):
        return None, None

    for key in ("wallclock_time_seconds", "requested_wallclock_time_seconds"):
        seconds = _to_int(last_info.get(key))
        if seconds is not None:
            source = f"last_job_info:{key}"
            if key.startswith("requested_"):
                source += ":requested_limit"
            return seconds, source

    dispatch_time = _parse_datetime(last_info.get("dispatch_time"))
    finish_time = _parse_datetime(last_info.get("finish_time"))
    if dispatch_time and finish_time:
        return (
            int((finish_time - dispatch_time).total_seconds()),
            "last_job_info:finish-dispatch",
        )

    return None, None


def _nodes_from_last_job_info(node: orm.CalcJobNode) -> tuple[int | None, str | None]:
    last_info = node.get_last_job_info()
    if not isinstance(last_info, Mapping):
        return None, None

    nodes = _to_int(last_info.get("num_machines"))
    if nodes is not None and nodes > 0:
        return nodes, "last_job_info:num_machines"

    allocated_machines = last_info.get("allocated_machines")
    if allocated_machines:
        try:
            return len(allocated_machines), "last_job_info:allocated_machines"
        except TypeError:
            pass

    return None, None


def _walltime_from_options(node: orm.CalcJobNode) -> tuple[int | None, str | None]:
    seconds = _to_int(node.get_option("max_wallclock_seconds"))
    if seconds is not None:
        return seconds, "metadata.options:max_wallclock_seconds:requested_limit"
    return None, None


def _nodes_from_resources(node: orm.CalcJobNode) -> tuple[int | None, str | None]:
    resources = node.get_option("resources") or {}
    if not isinstance(resources, Mapping):
        return None, None

    nodes = _to_int(resources.get("num_machines"))
    if nodes is not None and nodes > 0:
        return nodes, "metadata.options.resources:num_machines"

    total_mpiprocs = _to_int(resources.get("tot_num_mpiprocs"))
    mpiprocs_per_machine = _to_int(resources.get("num_mpiprocs_per_machine"))
    if total_mpiprocs and mpiprocs_per_machine:
        return math.ceil(total_mpiprocs / mpiprocs_per_machine), (
            "metadata.options.resources:ceil(tot_num_mpiprocs/num_mpiprocs_per_machine)"
        )

    return None, None


def _process_state(node: orm.ProcessNode) -> str | None:
    state = getattr(node, "process_state", None)
    if state is None:
        return None
    return getattr(state, "value", str(state))


def _estimate_calcjob_node_hours(node: orm.CalcJobNode) -> dict[str, Any]:
    warnings = []

    walltime_seconds, walltime_source, detailed_warnings = _walltime_from_detailed_info(
        node
    )
    warnings.extend(detailed_warnings)
    if walltime_seconds is None:
        walltime_seconds, walltime_source = _walltime_from_last_job_info(node)
    if walltime_seconds is None:
        walltime_seconds, walltime_source = _walltime_from_options(node)

    nodes, nodes_source = _nodes_from_detailed_info(node)
    if nodes is None:
        nodes, nodes_source = _nodes_from_last_job_info(node)
    if nodes is None:
        nodes, nodes_source = _nodes_from_resources(node)
    if nodes is None:
        nodes = 1
        nodes_source = "fallback:assumed_one_node"
        warnings.append("node count missing; assumed one node")

    node_hours = None
    if walltime_seconds is not None:
        node_hours = walltime_seconds * nodes / 3600.0
    else:
        warnings.append("walltime missing; excluded from total")

    return {
        "pk": node.pk,
        "uuid": node.uuid,
        "label": node.process_label,
        "process_type": node.process_type,
        "process_state": _process_state(node),
        "scheduler_job_id": node.get_job_id(),
        "computer": node.computer.label if node.computer else None,
        "walltime_seconds": walltime_seconds,
        "walltime_source": walltime_source,
        "nodes": nodes,
        "nodes_source": nodes_source,
        "node_hours": node_hours,
        "warnings": warnings,
    }


def _collect_process_tree(
    root: orm.ProcessNode,
) -> tuple[list[orm.ProcessNode], dict[int, list[orm.ProcessNode]]]:
    processes = []
    children_by_pk = {}
    stack = [root]

    while stack:
        node = stack.pop()
        processes.append(node)
        children = list(node.called)
        children_by_pk[node.pk] = children
        stack.extend(reversed(children))

    return processes, children_by_pk


def _estimate_calcjobs_sequential(
    calcjobs: list[orm.CalcJobNode],
) -> list[dict[str, Any]]:
    records_by_pk: dict[int, dict[str, Any]] = {}
    records = []

    for node in calcjobs:
        if node.pk not in records_by_pk:
            records_by_pk[node.pk] = _estimate_calcjob_node_hours(node)
        records.append(records_by_pk[node.pk])

    return records


def _summarize_tree(
    node: orm.ProcessNode,
    children_by_pk: Mapping[int, list[orm.ProcessNode]],
    records_by_pk: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    record = records_by_pk.get(node.pk)
    is_calcjob = isinstance(node, orm.CalcJobNode)

    if is_calcjob:
        node_hours = record["node_hours"] if record else None
        return {
            "pk": node.pk,
            "uuid": node.uuid,
            "label": node.process_label,
            "process_type": node.process_type,
            "process_state": _process_state(node),
            "is_calcjob": True,
            "scheduler_job_id": record.get("scheduler_job_id") if record else None,
            "walltime_seconds": record.get("walltime_seconds") if record else None,
            "nodes": record.get("nodes") if record else None,
            "node_hours": node_hours or 0.0,
            "calcjobs_total": 1,
            "calcjobs_counted": 1 if node_hours is not None else 0,
            "calcjobs_skipped": 0 if node_hours is not None else 1,
            "warnings": record.get("warnings", []) if record else [],
            "children": [],
        }

    children = [
        _summarize_tree(child, children_by_pk, records_by_pk)
        for child in children_by_pk.get(node.pk, [])
    ]
    return {
        "pk": node.pk,
        "uuid": node.uuid,
        "label": node.process_label,
        "process_type": node.process_type,
        "process_state": _process_state(node),
        "is_calcjob": False,
        "node_hours": sum(child["node_hours"] for child in children),
        "calcjobs_total": sum(child["calcjobs_total"] for child in children),
        "calcjobs_counted": sum(child["calcjobs_counted"] for child in children),
        "calcjobs_skipped": sum(child["calcjobs_skipped"] for child in children),
        "children": children,
    }


def count_node_hours(profile_name: str, pk: int) -> dict[str, Any]:
    """Estimate node-hours for a WorkChain/Process and all descendant CalcJobs.

    The estimate is intentionally best effort. It prefers actual scheduler
    accounting data, falls back to AiiDA last scheduler poll, then falls back
    to requested wallclock limits when actual runtime is unavailable.
    """
    aiida.load_profile(profile_name, allow_switch=True)
    root = orm.load_node(int(pk))

    if not isinstance(root, orm.ProcessNode):
        raise TypeError(
            f"PK {pk} is a {root.__class__.__name__}, not an AiiDA ProcessNode"
        )

    processes, children_by_pk = _collect_process_tree(root)
    calcjobs = [node for node in processes if isinstance(node, orm.CalcJobNode)]
    records = _estimate_calcjobs_sequential(calcjobs)
    records_by_pk = {record["pk"]: record for record in records}
    tree = _summarize_tree(root, children_by_pk, records_by_pk)

    return {
        "profile": profile_name,
        "root_pk": root.pk,
        "root_uuid": root.uuid,
        "root_label": root.process_label,
        "root_process_type": root.process_type,
        "root_process_state": _process_state(root),
        "calcjobs_total": tree["calcjobs_total"],
        "calcjobs_counted": tree["calcjobs_counted"],
        "calcjobs_skipped": tree["calcjobs_skipped"],
        "total_node_hours": tree["node_hours"],
        "tree": tree,
        "records": records,
    }


def _format_hours(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _format_node_label(node: Mapping[str, Any]) -> str:
    label = "{}<{}>".format(node["label"], node["pk"])
    state = node.get("process_state")

    if node["is_calcjob"]:
        parts = [label]
        if state:
            parts.append(str(state))
        if node.get("scheduler_job_id"):
            parts.append("job={}".format(node["scheduler_job_id"]))
        parts.append("nodes={}".format(node.get("nodes", "n/a")))
        parts.append("wall={}s".format(node.get("walltime_seconds") or "n/a"))
        if node.get("warnings"):
            parts.append("warnings={}".format(len(node["warnings"])))
        return " ".join(parts)

    parts = [label]
    if state:
        parts.append(str(state))
    parts.append(
        "calcjobs={}/{}".format(node["calcjobs_counted"], node["calcjobs_total"])
    )
    if node["calcjobs_skipped"]:
        parts.append("skipped={}".format(node["calcjobs_skipped"]))
    return " ".join(parts)


def _collect_display_rows(
    node: Mapping[str, Any],
    max_depth: int,
    prefix: str = "",
    depth: int = 0,
    is_last: bool = True,
    is_root: bool = True,
) -> list[tuple[float, str]]:
    line_prefix = "" if is_root else f"{prefix}+- "
    rows = [(node["node_hours"], f"{line_prefix}{_format_node_label(node)}")]
    children = node.get("children", [])
    child_prefix = "" if is_root else prefix + ("   " if is_last else "|  ")

    if depth >= max_depth:
        if children:
            hidden_hours = sum(child["node_hours"] for child in children)
            hidden_calcjobs = sum(child["calcjobs_total"] for child in children)
            rows.append(
                (
                    hidden_hours,
                    f"{child_prefix}+- ... {len(children)} children hidden ({hidden_calcjobs} CalcJobs)",
                )
            )
        return rows

    for index, child in enumerate(children):
        rows.extend(
            _collect_display_rows(
                child,
                max_depth=max_depth,
                prefix=child_prefix,
                depth=depth + 1,
                is_last=index == len(children) - 1,
                is_root=False,
            )
        )

    return rows


def _truncate_middle(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _print_summary(result: Mapping[str, Any], max_depth: int = 2) -> None:
    total = result["total_node_hours"]
    rows = _collect_display_rows(result["tree"], max_depth=max_depth)
    hour_strings = [_format_hours(hours) for hours, _ in rows]
    hours_width = max(len("NODE_HOURS"), *(len(value) for value in hour_strings))
    percent_width = len("100.0%")
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    tree_width = max(30, terminal_width - hours_width - percent_width - 6)

    print(
        "profile: {}  calcjobs: {}/{} counted  depth: {}".format(
            result["profile"],
            result["calcjobs_counted"],
            result["calcjobs_total"],
            max_depth,
        )
    )
    if result["calcjobs_skipped"]:
        print("skipped_calcjobs: {}".format(result["calcjobs_skipped"]))
    print(
        "{hours:>{hours_width}}  {tree:<{tree_width}}  {pct:>{percent_width}}".format(
            hours="NODE_HOURS",
            tree="TREE",
            pct="PCT",
            hours_width=hours_width,
            tree_width=tree_width,
            percent_width=percent_width,
        )
    )

    for hours, tree_text in rows:
        percent = 0.0 if total == 0 else hours / total * 100.0
        print(
            f"{_format_hours(hours):>{hours_width}}  "
            f"{_truncate_middle(tree_text, tree_width):<{tree_width}}  "
            f"{percent:>{percent_width - 1}.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pk", type=int, help="PK of the root WorkChain/Process node")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        help=f"AiiDA profile name to load (default: {DEFAULT_PROFILE_NAME})",
    )
    parser.add_argument(
        "--depth", type=int, default=2, help="tree display depth (default: 2)"
    )
    parser.add_argument(
        "--json", action="store_true", help="print JSON instead of a tree summary"
    )
    args = parser.parse_args()

    result = count_node_hours(args.profile, args.pk)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_summary(result, max_depth=args.depth)


if __name__ == "__main__":
    main()
