from scripts.count_node_hours import _parse_sacct_stdout, _parse_slurm_elapsed


def test_parse_sacct_stdout_prefers_exact_scheduler_job_id():
    stdout = (
        "JobID|JobIDRaw|ElapsedRaw|AllocNodes|\n"
        "123.batch|123.batch|3600|1|\n"
        "123|123|7200|2|\n"
    )

    row = _parse_sacct_stdout(stdout, "123")

    assert row["JobID"] == "123"
    assert row["ElapsedRaw"] == "7200"
    assert row["AllocNodes"] == "2"


def test_parse_sacct_stdout_preserves_empty_columns():
    stdout = (
        "Account|AdminComment|AllocCPUS|AllocNodes|JobID|ElapsedRaw|\n"
        "mr33||256|1|7232060|1119|\n"
    )

    row = _parse_sacct_stdout(stdout, "7232060")

    assert row["AdminComment"] == ""
    assert row["AllocNodes"] == "1"
    assert row["ElapsedRaw"] == "1119"


def test_parse_slurm_elapsed_accepts_day_hour_format():
    assert _parse_slurm_elapsed("1-02:03:04") == 93784


def test_print_tree_hides_children_beyond_depth(capsys):
    from scripts.count_node_hours import _print_summary

    result = {
        "profile": "presto-pg",
        "total_node_hours": 3.0,
        "calcjobs_counted": 2,
        "calcjobs_total": 2,
        "calcjobs_skipped": 0,
        "tree": {
            "pk": 1,
            "label": "RootWorkChain",
            "process_state": "finished",
            "is_calcjob": False,
            "node_hours": 3.0,
            "calcjobs_counted": 2,
            "calcjobs_total": 2,
            "calcjobs_skipped": 0,
            "children": [
                {
                    "pk": 2,
                    "label": "ChildWorkChain",
                    "process_state": "finished",
                    "is_calcjob": False,
                    "node_hours": 3.0,
                    "calcjobs_counted": 2,
                    "calcjobs_total": 2,
                    "calcjobs_skipped": 0,
                    "children": [
                        {
                            "pk": 3,
                            "label": "PwCalculation",
                            "process_state": "finished",
                            "is_calcjob": True,
                            "scheduler_job_id": "123",
                            "nodes": 2,
                            "walltime_seconds": 3600,
                            "node_hours": 2.0,
                            "calcjobs_counted": 1,
                            "calcjobs_total": 1,
                            "calcjobs_skipped": 0,
                            "warnings": [],
                            "children": [],
                        }
                    ],
                }
            ],
        },
    }

    _print_summary(result, max_depth=1)

    output = capsys.readouterr().out
    assert "NODE_HOURS" in output
    assert "TREE" in output
    assert "PCT" in output
    assert "RootWorkChain<1>" in output
    assert "ChildWorkChain<2>" in output
    assert "children hidden" in output
    assert "PwCalculation<3>" not in output


def test_estimate_calcjobs_sequential_preserves_order_and_caches(monkeypatch):
    from scripts import count_node_hours as module

    class FakeNode:
        def __init__(self, pk):
            self.pk = pk

    calls = []

    def fake_estimate(node):
        calls.append(node.pk)
        return {"pk": node.pk, "node_hours": node.pk}

    monkeypatch.setattr(module, "_estimate_calcjob_node_hours", fake_estimate)

    records = module._estimate_calcjobs_sequential(
        [FakeNode(2), FakeNode(1), FakeNode(2), FakeNode(3), FakeNode(1)]
    )

    assert [record["pk"] for record in records] == [2, 1, 2, 3, 1]
    assert calls == [2, 1, 3]
