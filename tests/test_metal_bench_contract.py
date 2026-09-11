"""CPU-only benchmark boundaries: one invocation, completed work, one timing domain."""
import pytest
from triton_msl.profiling import metal_bench as bench


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_wall_samples_execute_once_and_synchronize(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(bench.time, "perf_counter", clock)
    calls = []

    def work():
        calls.append("work")
        clock.now += .001

    def sync():
        calls.append("sync")
        clock.now += .002

    result = bench.metal_do_bench(work, warmup=2, rep=3, synchronize=sync)
    assert calls.count("work") == 5  # two warmups + three samples, never eight
    assert calls[-2:] == ["work", "sync"]
    assert result == pytest.approx(3.0)  # queued work's completion is inside timing
    # The driver calls without an explicit callback; pin that default too.
    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", sync)
    calls.clear()
    assert bench.metal_do_bench(work, warmup=2, rep=3) == pytest.approx(3.0)
    assert calls.count("work") == 5 and calls[-2:] == ["work", "sync"]
    row = bench.metal_do_bench(work, warmup=0, rep=3, return_metadata=True)
    assert row["value"] == pytest.approx(3.0)
    assert row["clock"] == row["requested_clock"] == "wall" and row["unit"] == "ms"
    assert row["samples"] == 3 and row["valid_gpu_timestamp_samples"] is None
    assert "host overhead" in row["boundary"] and row["fallback_reason"] is None


def test_command_buffers_wait_before_timestamps(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(bench.time, "perf_counter", clock)
    commands = []

    class Command:
        done = False

        def waitUntilCompleted(self):
            self.done = True
            clock.now += .002

        def GPUStartTime(self):
            assert self.done
            return 1.0

        def GPUEndTime(self):
            assert self.done
            return 1.00025

        def error(self):
            return None

    def work():
        clock.now += .001
        cmd = Command()
        commands.append(cmd)
        return cmd

    def wrong_sync():
        pytest.fail("command completion must not drain an unrelated MPS queue")

    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", wrong_sync)
    # A returned command must not silently change the default scalar's clock.
    assert bench.metal_do_bench(work, warmup=1, rep=3) == pytest.approx(3.0)
    assert len(commands) == 4 and all(c.done for c in commands)
    for mode in ("gpu", "auto"):
        commands.clear()
        row = bench.metal_do_bench(work, warmup=1, rep=3, clock=mode, return_metadata=True)
        assert row["clock"] == "gpu" and row["requested_clock"] == mode
        assert row["value"] == pytest.approx(.25) and row["unit"] == "ms"
        assert row["valid_gpu_timestamp_samples"] == 3 and row["fallback_reason"] is None
        assert "entire-workload command buffer" in row["boundary"]
        assert len(commands) == 4 and all(c.done for c in commands)
    # Explicit completion is honored even when fn returns a waitable buffer:
    # it may have other work on a different queue which that buffer cannot cover.
    extra_completions = []

    def complete_other_work():
        assert commands[-1].done
        extra_completions.append(1)
        clock.now += .004

    assert bench.metal_do_bench(work, warmup=1, rep=3, synchronize=complete_other_work) == pytest.approx(7.0)
    assert len(extra_completions) == 4


def test_invalid_timestamp_falls_back_for_whole_series_without_relaunch(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(bench.time, "perf_counter", clock)
    calls = []

    class Command:
        def __init__(self, duration):
            self.duration = duration

        def waitUntilCompleted(self):
            clock.now += .002

        def GPUStartTime(self):
            return 1.0

        def GPUEndTime(self):
            return 1.0 + self.duration

    def work():
        duration = [.00025, 0, float("nan")][len(calls)]
        calls.append(duration)
        clock.now += .001
        return Command(duration)

    row = bench.metal_do_bench(work, warmup=0, rep=3, quantiles=[0, .5, 1],
                              clock="auto", return_metadata=True, synchronize=lambda: None)
    assert row["value"] == pytest.approx([3, 3, 3])
    assert row["clock"] == "wall" and row["requested_clock"] == "auto"
    assert row["valid_gpu_timestamp_samples"] == 1
    assert row["fallback_reason"] == "missing_or_invalid_gpu_timestamps"
    assert len(calls) == 3
    calls.clear()
    with pytest.raises(ValueError, match="timestamps"):
        bench.metal_do_bench(work, warmup=0, rep=3, clock="gpu", synchronize=lambda: None)
    assert len(calls) == 3  # explicit GPU request fails, never returns a wall value


def test_invalid_request_rejected_before_work():
    def work():
        pytest.fail("invalid benchmark request must not execute")

    for options in ({"rep": 0}, {"warmup": -1}, {"quantiles": [float("nan")]},
                    {"quantiles": [-.1]}, {"quantiles": [1.1]}, {"quantiles": []},
                    {"clock": "unspecified"}, {"clock": "auto"}):
        with pytest.raises(ValueError):
            bench.metal_do_bench(work, synchronize=lambda: None, **options)


def test_failed_work_is_not_retried():
    calls = []

    def work():
        calls.append(1)
        raise RuntimeError("submission failed")

    with pytest.raises(RuntimeError, match="submission failed"):
        bench.metal_do_bench(work, warmup=0, rep=3, synchronize=lambda: None)
    assert calls == [1]
    # A command-buffer failure is also a failed measurement, not a fallback time.
    from types import SimpleNamespace
    command = SimpleNamespace(waitUntilCompleted=lambda: None, error=lambda: "GPU failure")
    calls.clear()

    def failed_command():
        calls.append(1)
        return command

    with pytest.raises(RuntimeError, match="GPU failure"):
        bench.metal_do_bench(failed_command, warmup=0, rep=3, synchronize=lambda: None)
    assert calls == [1]
