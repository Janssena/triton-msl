"""Two mechanism pins: disjoint scheduling and failure/coverage accounting."""
from types import SimpleNamespace

from scripts import run_project_tests as gate
from tests import conftest


def test_lane_partition_is_lossless_and_explicit(monkeypatch):
    monkeypatch.setattr(conftest.platform, "system", lambda: "Darwin")

    class Item:
        def __init__(self, name, performance):
            self.nodeid, self.performance = name, performance

        def get_closest_marker(self, name):
            assert name == "performance_sentinel"
            return object() if self.performance else None

    original = [Item("correctness", False), Item("sentinel", True)]
    partitions = {}
    for lane in ("all", "correctness", "performance"):
        deselected = []
        config = SimpleNamespace(getoption=lambda name: lane,
                                 hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
        selected = list(original)
        conftest.pytest_collection_modifyitems(config, selected)
        assert set(selected).isdisjoint(deselected)
        assert set(selected) | set(deselected) == set(original)
        partitions[lane] = selected
    assert partitions["all"] == original
    assert partitions["correctness"] == [original[0]]
    assert partitions["performance"] == [original[1]]


def test_job_receipt_requires_every_floor_to_execute_and_never_qualifies_claims():
    observer = gate.Outcomes()
    observer.collected = sorted(gate.PERFORMANCE_IDS)
    observer.statuses = dict.fromkeys(observer.collected, "passed")
    result = observer.result("performance", 0)
    assert result["classification"] == "UNQUALIFIED_FLOOR_PASS"
    assert result["release_qualified"] is False
    result.update(phase="performance", source_unchanged=True)
    receipt = dict(job="performance", process_rc=0, source_unchanged=True, results=[result])
    assert gate.job_succeeded(receipt, False)
    assert not gate.job_succeeded(receipt, True)  # Timing is not collection evidence.
    for field, bad in (("classification", "FAILED_OR_INCOMPLETE"), ("source_unchanged", False),
                       ("phase", "another-job"), ("rc", 1)):
        altered = dict(result, **{field: bad})
        assert not gate.job_succeeded(dict(receipt, results=[altered]), False)
    assert not gate.job_succeeded(dict(receipt, results=[]), False)
    node = observer.collected[0]
    for bad_status in ("failed", "skipped", "xfailed"):
        observer.statuses[node] = bad_status
        assert observer.result("performance", 0)["classification"] == "FAILED_OR_INCOMPLETE"
    del observer.statuses[node]
    assert observer.result("performance", 0)["classification"] == "FAILED_OR_INCOMPLETE"
    observer.statuses[node] = "passed"
    assert observer.result("performance", 1)["classification"] == "FAILED_OR_INCOMPLETE"
    # Setup/call success cannot conceal teardown failure, nor can a later success erase it.
    observer.pytest_runtest_logreport(SimpleNamespace(nodeid=node, when="teardown", failed=True))
    observer.pytest_runtest_logreport(SimpleNamespace(nodeid=node, when="call", failed=False, skipped=False, outcome="passed"))
    assert observer.result("performance", 0)["classification"] == "FAILED_OR_INCOMPLETE"
    observer.collected.append(node)
    assert not observer.result("performance", 0)["accounting_valid"]
