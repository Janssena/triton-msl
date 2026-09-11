"""Local scheduler: separate required correctness and throughput-floor jobs.

Floor outcomes are NOT qualified claims. 'context' is an explicit before/core/after
same-process diagnostic. Raw failures survive; no automatic retry or threshold edits.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_OPTIONS = ["-o", "junit_logging=all", "-o", "junit_log_passing_tests=true"]
PERFORMANCE_IDS = frozenset({
    "tests/test_fast_matmul_perf.py::test_fast_matmul_throughput[dtype0]",
    "tests/test_fast_matmul_perf.py::test_fast_matmul_throughput[dtype1]",
    "tests/test_fast_matmul_perf.py::test_fast_matmul_fp16out_throughput",
    "tests/test_compile_shader_perf.py::test_vector_add_fast_path_throughput",
})


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def fingerprint():
    paths = [ROOT / "pyproject.toml"]
    for directory in ("triton_msl", "tests", "scripts"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def job_succeeded(receipt, collect_only):
    results = receipt.get("results")
    phases = ["before", "core", "after"] if receipt["job"] == "context" else [receipt["job"]]
    allowed = {"COLLECTION_ONLY"} if collect_only else {"CORRECTNESS_PASS", "UNQUALIFIED_FLOOR_PASS"}
    return (receipt.get("process_rc") == 0 and receipt.get("source_unchanged") is True
            and isinstance(results, list) and [r.get("phase") for r in results] == phases
            and all(r.get("rc") == 0 and r.get("accounting_valid") is True and r.get("source_unchanged") is True
                    and r.get("classification") in allowed and r.get("release_qualified") is False for r in results))


class Outcomes:
    def __init__(self):
        self.collected, self.deselected, self.statuses = [], [], {}

    def pytest_deselected(self, items):
        self.deselected.extend(i.nodeid for i in items)

    def pytest_collection_finish(self, session):
        import triton_msl
        if Path(triton_msl.__file__).resolve() != ROOT / "triton_msl/__init__.py":
            raise RuntimeError("Wrong implementation in collecting/running interpreter")
        self.collected = [i.nodeid for i in session.items]

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.statuses[report.nodeid] = "failed"
        elif self.statuses.get(report.nodeid) != "failed" and (report.when == "call" or report.skipped):
            self.statuses[report.nodeid] = report.outcome

    def result(self, lane, rc, collect_only=False):
        selected, excluded = set(self.collected), set(self.deselected)
        accounting = (len(selected) == len(self.collected) and bool(selected) and not selected & excluded)
        accounting &= selected == PERFORMANCE_IDS if lane == "performance" else excluded == PERFORMANCE_IDS
        complete = accounting and set(self.statuses) == selected
        passing = rc == 0 and complete and "failed" not in self.statuses.values() and "passed" in self.statuses.values()
        if lane == "performance":
            passing &= all(s == "passed" for s in self.statuses.values())
        if collect_only:
            classification = "COLLECTION_ONLY" if rc == 0 and accounting else "INVALID_COLLECTION"
        elif not passing:
            classification = "FAILED_OR_INCOMPLETE"
        else:
            classification = "CORRECTNESS_PASS" if lane == "correctness" else "UNQUALIFIED_FLOOR_PASS"
        return dict(lane=lane, rc=rc, classification=classification, accounting_valid=bool(accounting),
                    counts=dict(collections.Counter(self.statuses.values())), collected=self.collected,
                    deselected=self.deselected, node_statuses=self.statuses, release_qualified=False)


def worker(args):
    # Retain import and teardown diagnostics too, not just pytest's captured output.
    raw = (args.output / "raw.txt").open("x", buffering=1)
    os.dup2(raw.fileno(), 1)
    os.dup2(raw.fileno(), 2)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from triton_msl.profiling.cache_session import CACHE_KEYS, fresh_cache_environment
    os.environ.update(fresh_cache_environment(dict(os.environ, **{k: str(args.output / k) for k in CACHE_KEYS})))
    import pytest
    import triton_msl
    import Metal
    device = Metal.MTLCreateSystemDefaultDevice()
    assert Path(triton_msl.__file__).resolve() == ROOT / "triton_msl/__init__.py"
    before = fingerprint()
    write_json(args.output / "source-freeze.json", before)
    write_json(args.output / "environment.json", dict(python=sys.executable, platform=platform.platform(),
               device_name=str(device.name()) if device is not None else None,
               root=str(ROOT), policy={k: v for k, v in os.environ.items() if k.startswith("TRITON_")}))
    report = ROOT / "reports/perf_baseline.json"
    original = report.read_bytes()
    phases = [(args.worker, args.worker)] if args.worker != "context" else [
        ("before", "performance"), ("core", "correctness"), ("after", "performance")]
    results = []
    try:
        for label, lane in phases:
            # Snapshot, not attached telemetry or per-sample utilization evidence.
            with (args.output / (label + "-clients-before.txt")).open("x") as clients:
                subprocess.run(["ps", "-axo", "pid,ppid,comm"], stdout=clients, stderr=clients, check=True)
            observer = Outcomes()
            command = ["tests", "--project-lane=" + lane, "--color=no", "--tb=short", "-q", "-ra",
                       "--junitxml=" + str(args.output / (label + ".xml"))]
            command += CAPTURE_OPTIONS
            command += ["--randomly-seed=" + str(args.seed)] if lane == "correctness" else ["-p", "no:randomly"]
            if args.collect_only:
                command.append("--collect-only")
            print("PHASE", label, command, flush=True)
            rc = int(pytest.main(command, plugins=[observer]))
            result = observer.result(lane, rc, args.collect_only)
            result["phase"] = label
            result["source_unchanged"] = fingerprint() == before
            write_json(args.output / (label + ".json"), result)
            write_json(args.output / (label + "-perf-report.json"), json.loads(report.read_text()))
            with (args.output / (label + "-clients-after.txt")).open("x") as clients:
                subprocess.run(["ps", "-axo", "pid,ppid,comm"], stdout=clients, stderr=clients, check=True)
            results.append(result)
            # Context mode retains all outcomes; a failing prefix does not select a rerun.
    finally:
        report.write_bytes(original)
    write_json(args.output / "results.json", results)
    return 0 if all(r["source_unchanged"] and r["classification"] in
                    ("CORRECTNESS_PASS", "UNQUALIFIED_FLOOR_PASS", "COLLECTION_ONLY") for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", "correctness", "performance", "context"), default="both")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; never overwritten")
    parser.add_argument("--seed", type=int, default=423001)
    parser.add_argument("--timeout", type=int, default=2400, help="Per-worker wall-clock cap in seconds")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--worker", choices=("correctness", "performance", "context"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    if os.environ.get("PYTEST_ADDOPTS"):
        parser.error("PYTEST_ADDOPTS would alter the frozen selection; clear it explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = fingerprint()
    write_json(args.output / "source-freeze.json", frozen)
    report = ROOT / "reports/perf_baseline.json"
    original_report = report.read_bytes()
    # Cooperating callers only: this does NOT exclude OS compositing/unrelated GPU clients.
    with (Path(tempfile.gettempdir()) / "triton-msl-project-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = ("correctness", "performance") if args.mode == "both" else (args.mode,)
        receipts = []
        for job in jobs:
            out = args.output / job
            out.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", job, "--output", str(out),
                       "--seed", str(args.seed)] + (["--collect-only"] if args.collect_only else [])
            with (out / "process-stdout.txt").open("x") as stdout, (out / "process-stderr.txt").open("x") as stderr:
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
                    try:
                        receipt = dict(job=job, process_rc=process.wait(timeout=args.timeout))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)  # Only this newly created worker's process group.
                        process.wait()
                        receipt = dict(job=job, process_error="TIMEOUT")
                except OSError as error:
                    receipt = dict(job=job, process_error=repr(error))
                finally:
                    # The worker's finally cannot run after a timeout/forced termination.
                    if report.read_bytes() != original_report:
                        (out / "interrupted-perf-report.json").write_bytes(report.read_bytes())
                        report.write_bytes(original_report)
            try:
                receipt["results"] = json.loads((out / "results.json").read_text())
                receipt["source_unchanged"] = (fingerprint() == frozen and
                    json.loads((out / "source-freeze.json").read_text()) == frozen)
            except (OSError, ValueError) as error:
                receipt["results"] = []
                receipt["result_error"] = repr(error)
            receipts.append(receipt)
        summary = dict(mode=args.mode, jobs=receipts, release_qualified=False,
                       performance_obligation="OPEN: controlled qualification and claims matrix required")
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"evidence": str(args.output), "processes": [r.get("process_rc") for r in receipts],
                          "release_qualified": False}))
        return 0 if all(job_succeeded(r, args.collect_only) for r in receipts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
