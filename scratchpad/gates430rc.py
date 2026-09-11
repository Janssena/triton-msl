"""Exact-tree gates, no retries; complete status maps and retained uncapped output."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/Users/bledden/Documents/triton-validation-evidence/2026-09-10-rc030rc1-gates')
MODE = sys.argv[1]
assert MODE in ('focused', 'project', 'ratchet')
OUT = BASE / (sys.argv[2] if len(sys.argv) > 2 else MODE)
OUT.mkdir(exist_ok=False)
raw = (OUT / 'raw.txt').open('x', buffering=1)
os.dup2(raw.fileno(), 1)
os.dup2(raw.fileno(), 2)
sys.path[:0] = [str(ROOT), str(ROOT/'tests'), str(ROOT/'scripts')]
os.chdir(ROOT)
from triton_msl.profiling.cache_session import CACHE_KEYS, fresh_cache_environment
os.environ.update(fresh_cache_environment(dict(os.environ, **{k: str(OUT/k) for k in CACHE_KEYS})))
os.environ.update(TRITON_DEFAULT_BACKEND='metal', TRITON_MSL_COMPILE_SHADER='1')
import pytest
import triton_msl
import torch
import Metal
assert torch.backends.mps.is_available() and Metal.MTLCreateSystemDefaultDevice() is not None, "GPU unavailable: no gate credit"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
freeze = json.loads(Path('/Users/bledden/Documents/triton-validation-evidence/2026-09-10-rc030rc1/RC-SOURCE-FREEZE-rc3.json').read_text())
assert {p: sha(ROOT/p) for p in freeze} == freeze

class Outcomes:
    def __init__(self):
        self.statuses, self.collected = {}, []
    def pytest_collection_finish(self, session):
        assert Path(triton_msl.__file__).resolve() == ROOT/'triton_msl/__init__.py'
        self.collected = [i.nodeid for i in session.items]
        assert len(self.collected) == len(set(self.collected))
    def pytest_runtest_setup(self, item):
        assert Path(triton_msl.__file__).resolve() == ROOT/'triton_msl/__init__.py'
    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.statuses[report.nodeid] = 'failed'
        elif self.statuses.get(report.nodeid) != 'failed' and (report.when == 'call' or report.skipped):
            self.statuses[report.nodeid] = report.outcome

args = ['--color=no', '--tb=short', '--junitxml='+str(OUT/'junit.xml')]
if MODE == 'focused':
    args += ['tests/test_emitter_outcomes.py', 'tests/test_emitter.py',
             'tests/test_unsupported_op_refusal.py', 'tests/test_mept_m5_default_gpu.py',
             'tests/test_mept_m3c_gt1024_gpu.py', 'tests/test_reduce_loop_carried.py',
             'tests/test_loop_pointer_contract.py', '-p', 'no:randomly', '-q', '-ra']
elif MODE == 'project':
    args += ['tests', '--randomly-seed=437001', '-q', '-ra']
else:
    args += ['/Users/bledden/Documents/triton/python/test/unit/language/test_core.py',
             '-p', 'conftest_metal', '--device', 'cpu', '-p', 'no:randomly', '-v', '-rf', '--no-header']
(OUT/'run.json').write_text(json.dumps(dict(root=str(ROOT), args=args, python=sys.executable), indent=2)+'\n')
report = ROOT/'reports/perf_baseline.json'
before = report.read_bytes()
observer = Outcomes()
start = time.monotonic()
try:
    rc = int(pytest.main(args, plugins=[observer]))
finally:
    (OUT/'generated-perf-baseline.json').write_bytes(report.read_bytes())
    report.write_bytes(before)
source_ok = {p: sha(ROOT/p) for p in freeze} == freeze
complete = set(observer.statuses) == set(observer.collected)
for name, value in [('node-statuses', observer.statuses), ('collected', observer.collected)]:
    (OUT/(name+'.json')).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
baseline_equal = None
if MODE == 'ratchet':
    baseline = json.loads((BASE.parent/'2026-09-09-p415/ratchet/node-statuses.json').read_text())
    baseline_equal = len(observer.statuses) == 9342 and observer.statuses == baseline
result = dict(rc=rc, seconds=time.monotonic()-start, counts=dict(collections.Counter(observer.statuses.values())),
              collected=len(observer.collected), complete=complete, source_freeze_verified=source_ok,
              report_restored=report.read_bytes()==before, baseline_equal=baseline_equal)
(OUT/'exit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if rc==0 and complete and source_ok and baseline_equal is not False else 1)
