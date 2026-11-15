# SPDX-License-Identifier: Apache-2.0
"""``close()`` of a native connector must not hold the GIL while it waits.

``close()`` joins the worker threads. A worker blocked in backend I/O that
nothing interrupts keeps it waiting for as long as the backend stays silent.
Held with the GIL, that wait freezes every Python thread of the process; with
the GIL released only the caller waits.

The vehicle is the file-system connector: a worker opening a FIFO blocks until
someone opens its other end, which is what a hung mount looks like to it. The
check runs in a subprocess so that a regression fails the test instead of
freezing the test run.
"""

# Standard
import subprocess
import sys
import textwrap

# Third Party
import pytest

pytest.importorskip("lmcache.lmcache_fs")

_SCRIPT = textwrap.dedent(
    """
    import os, select, tempfile, threading, time
    import torch  # noqa: F401
    from lmcache.lmcache_fs import LMCacheFSClient
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.distributed.key_codec import encode_storage_key
    from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
        FSNativeL2AdapterConfig,
    )

    base = tempfile.mkdtemp(prefix="close-gil-")
    config = FSNativeL2AdapterConfig.from_dict({"base_path": base, "num_workers": 2})
    client = LMCacheFSClient(
        config.base_path, config.num_workers, config.relative_tmp_dir,
        config.use_odirect, config.read_ahead_size or 0, config.num_read_workers,
        config.read_max_bytes_in_flight,
    )

    def wait_completion(future_id):
        poller = select.poll()
        poller.register(client.event_fd(), select.POLLIN)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if poller.poll(100):
                for done in client.drain_completions():
                    if done[0] == future_id:
                        return done
        raise AssertionError("no completion")

    keys = [
        encode_storage_key(ObjectKey(
            chunk_hash=bytes([i]) * 32, model_name="m", kv_rank=0,
            object_group_id=0, cache_salt=""))
        for i in range(8)
    ]
    buffers = [bytearray(4096) for _ in keys]
    views = [memoryview(b) for b in buffers]
    done = wait_completion(client.submit_batch_set(keys, views))
    assert done[1], done

    fifos = []
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if os.path.isfile(path) and name.endswith(".data"):
            os.remove(path)
            os.mkfifo(path)
            fifos.append(path)
    for key, buffer in zip(keys, buffers):
        client.submit_batch_get([key], [memoryview(buffer)])  # blocks in open()
    time.sleep(0.5)

    closer = threading.Thread(target=client.close, daemon=True)
    closer.start()
    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(1)
            time.sleep(0.02)

    threading.Thread(target=tick, daemon=True).start()
    time.sleep(1.5)  # the main thread itself needs the GIL to wake up
    stop.set()
    assert closer.is_alive(), "close() returned although its workers are blocked"
    assert len(ticks) > 20, f"only {len(ticks)} ticks in 1.5 s: close() holds the GIL"

    # Let the blocked open() calls return so close() can finish. A worker that
    # is freed goes on to the next queued request and blocks on another FIFO,
    # so keep opening them all until close() is done.
    deadline = time.monotonic() + 15
    while closer.is_alive() and time.monotonic() < deadline:
        for path in fifos:
            try:
                os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                pass  # nobody is blocked on this FIFO right now
        time.sleep(0.05)
    assert not closer.is_alive(), "close() did not finish once its workers were free"
    """
)


def test_close_does_not_freeze_the_process_while_workers_are_blocked():
    try:
        done = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the process froze: close() holds the GIL while it waits")
    assert done.returncode == 0, done.stdout + done.stderr
