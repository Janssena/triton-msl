"""Real mmap/ctypes lifetimes, fake Metal allocation outcomes; no GPU required."""
import types
import sys

import pytest

from triton_msl.buffer_pool import MetalBufferPool, PAGE_SIZE


@pytest.fixture(autouse=True)
def metal_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, 'Metal', types.SimpleNamespace(MTLResourceStorageModeShared=0))


class Device:
    def __init__(self, *, no_copy=None, allocated=None, failure=None):
        self.no_copy = no_copy
        self.allocated = allocated
        self.failure = failure
        self.backing = None
        self.requests = []

    def newBufferWithBytesNoCopy_length_options_deallocator_(self, view, size, options, deallocator):
        # Retain the mmap itself, not its exported ctypes/memoryview owner.
        self.backing = next(obj.obj for obj in view._objects.values() if isinstance(obj, memoryview))
        self.requests.append(('no_copy', size, options))
        assert deallocator is None
        return self.no_copy

    def newBufferWithLength_options_(self, size, options):
        self.requests.append(('allocated', size, options))
        if self.failure is not None:
            raise self.failure
        return self.allocated


def test_no_copy_failure_closes_mapping_and_reaches_fallback():
    allocation = object()
    device = Device(allocated=allocation)
    pool = MetalBufferPool(device)
    buffer, mapping, size = pool.acquire(64)
    assert (buffer, mapping, size) == (allocation, None, PAGE_SIZE)
    assert device.backing.closed
    assert device.requests == [('no_copy', PAGE_SIZE, 0), ('allocated', PAGE_SIZE, 0)]
    pool.release(buffer, mapping, size)
    assert not pool._free[size]


def test_both_allocations_fail_loudly_after_mapping_cleanup():
    device = Device()
    pool = MetalBufferPool(device)
    with pytest.raises(MemoryError, match=f'Metal buffer allocation failed for {PAGE_SIZE} bytes'):
        pool.acquire(1)
    assert device.backing.closed
    assert len(device.requests) == 2
    assert not pool._free[PAGE_SIZE]


def test_fallback_exception_propagates_without_replacing_it_with_cleanup_error():
    failure = RuntimeError('allocation API failure')
    device = Device(failure=failure)
    with pytest.raises(RuntimeError) as raised:
        MetalBufferPool(device).acquire(256)
    assert raised.value is failure
    assert device.backing.closed
    assert len(device.requests) == 2


def test_successful_zero_copy_reuses_live_mapping_until_pool_disposes_it():
    buffer = object()
    device = Device(no_copy=buffer)
    pool = MetalBufferPool(device)
    first = pool.acquire(64)
    assert first == (buffer, device.backing, PAGE_SIZE)
    assert not device.backing.closed
    first[1][0:4] = b'live'
    pool.release(*first)
    second = pool.acquire(64)
    assert second == first and second[1][0:4] == b'live'
    assert device.requests == [('no_copy', PAGE_SIZE, 0)]
    pool._max_per_class = 0
    pool.release(*second)
    assert device.backing.closed


@pytest.mark.parametrize('nbytes', [1, 4, 8])
def test_scalar_allocation_failure_is_loud(nbytes):
    device = Device()
    with pytest.raises(MemoryError, match='Metal scalar buffer allocation failed'):
        MetalBufferPool(device).acquire_scalar(nbytes)
    assert device.requests == [('allocated', max(nbytes, 4), 0)]


def test_scalar_success_and_reuse_are_unchanged():
    allocation = object()
    device = Device(allocated=allocation)
    pool = MetalBufferPool(device)
    assert pool.acquire_scalar(4) is allocation
    pool.release_scalar(allocation, 4)
    assert pool.acquire_scalar(4) is allocation
    assert device.requests == [('allocated', 4, 0)]
