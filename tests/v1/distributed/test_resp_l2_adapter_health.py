# SPDX-License-Identifier: Apache-2.0
"""Health probing and reconnection of the RESP (Redis) L2 adapter.

These tests use the real C++ Redis connector against a minimal in-process RESP
server that answers ``PING`` and ``EXISTS``. Stopping and restarting that
server on the same port reproduces what the connector does when its backend is
missing at startup, killed while running, and brought back.
"""

# Standard
import select
import socket
import subprocess
import sys
import textwrap
import threading
import time

# Third Party
import pytest

pytest.importorskip("lmcache.lmcache_redis")

# First Party
from lmcache.lmcache_redis import LMCacheRedisClient
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L2HealthConfig
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import RESPL2AdapterConfig
from lmcache.v1.distributed.l2_health_monitor import L2HealthMonitor
from lmcache.v1.platform import consume_fd


class _MiniRespServer:
    """Answers PING with +PONG and EXISTS with :0, over any number of
    connections, until ``stop`` closes them all."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._listener: socket.socket | None = None
        self._connections: list[socket.socket] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.port))
        listener.listen(16)
        self._listener = listener
        threading.Thread(
            target=self._accept_loop, args=(listener,), daemon=True
        ).start()

    def stop(self) -> None:
        """Close the listener and every open connection, as a killed server does."""
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            listener.close()
        with self._lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()

    def _accept_loop(self, listener: socket.socket) -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with self._lock:
                self._connections.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    @staticmethod
    def _serve(conn: socket.socket) -> None:
        buffer = b""
        while True:
            try:
                data = conn.recv(65536)
            except OSError:
                return
            if not data:
                return
            buffer += data
            # Commands arrive as RESP arrays of bulk strings.
            while True:
                parsed = _parse_command(buffer)
                if parsed is None:
                    break
                command, buffer = parsed
                try:
                    conn.sendall(b"+PONG\r\n" if command == "PING" else b":0\r\n")
                except OSError:
                    return


def _parse_command(buffer: bytes) -> tuple[str, bytes] | None:
    """Split one RESP command off ``buffer``; ``None`` if it is incomplete."""
    if not buffer.startswith(b"*"):
        return None
    parts = buffer.split(b"\r\n")
    count = int(parts[0][1:])
    needed = 1 + 2 * count
    if len(parts) < needed + 1:
        return None
    command = parts[2].decode().upper()
    consumed = sum(len(p) + 2 for p in parts[:needed])
    return command, buffer[consumed:]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def _lookup_completes(adapter, timeout_s: float = 5.0) -> bool:
    """Whether a lookup through the adapter completes (with any result)."""
    task_id = adapter.submit_lookup_and_lock_task([_key(1)], {})
    poller = select.poll()
    poller.register(adapter.get_lookup_and_lock_event_fd(), select.POLLIN)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if poller.poll(100):
            try:
                consume_fd(adapter.get_lookup_and_lock_event_fd())
            except BlockingIOError:
                pass
        if adapter.query_lookup_and_lock_result(task_id) is not None:
            return True
    return False


def _rounds_until(monitor: L2HealthMonitor, state: str, max_rounds: int = 20) -> None:
    for _ in range(max_rounds):
        monitor.check_once()
        if monitor.report_status()["adapters"][0]["state"] == state:
            return
        time.sleep(0.2)
    raise AssertionError(
        f"adapter never became {state}: {monitor.report_status()['adapters'][0]}"
    )


@pytest.fixture
def make_stack():
    """Build a RESP adapter, a monitor over it, and the server it talks to.

    ``make_stack(server_up=True)`` starts the server before the adapter, so the
    adapter's connector connects at creation; with ``False`` nothing listens
    when the adapter is created.
    """
    created: list[tuple[_MiniRespServer, object]] = []

    def make(server_up: bool):
        port = _free_port()
        server = _MiniRespServer(port)
        if server_up:
            server.start()
        adapter = create_l2_adapter(
            RESPL2AdapterConfig.from_dict({"host": "127.0.0.1", "port": port})
        )
        monitor = L2HealthMonitor(
            L2HealthConfig(check_interval_s=0, probe_timeout_s=1.0, failure_threshold=2)
        )
        monitor.register(0, adapter, "resp")
        created.append((server, adapter))
        return server, adapter, monitor

    yield make
    for server, adapter in created:
        server.stop()
        adapter.close()


def test_the_adapter_declares_probe_and_reconnect_support(make_stack):
    _, adapter, _ = make_stack(server_up=True)

    assert adapter.supports_health_probe
    assert adapter.supports_reconnect


def test_backend_missing_at_startup_is_detected_and_recovered_without_restart(
    make_stack,
):
    server, adapter, monitor = make_stack(server_up=False)
    # Nothing is listening: the connector's worker threads failed to connect.
    _rounds_until(monitor, "unavailable")
    assert not monitor.is_available(0)

    server.start()
    _rounds_until(monitor, "healthy")

    assert monitor.is_available(0)
    assert _lookup_completes(adapter)


def test_backend_killed_while_running_is_detected_and_recovered_without_restart(
    make_stack,
):
    server, adapter, monitor = make_stack(server_up=True)
    _rounds_until(monitor, "healthy")
    assert _lookup_completes(adapter)

    server.stop()
    _rounds_until(monitor, "unavailable")
    assert not monitor.is_available(0)

    server.start()
    _rounds_until(monitor, "healthy")

    assert _lookup_completes(adapter)


def test_requests_to_a_dead_backend_fail_at_once(make_stack):
    _, adapter, _ = make_stack(server_up=False)

    # No connection exists, so the lookup must not wait for the backend.
    assert _lookup_completes(adapter, timeout_s=2.0)


# ---------------------------------------------------------------------------
# The connector's own health interface (probe_health, reconnect, close)
# ---------------------------------------------------------------------------


class _SilentServer:
    """Accepts connections and never answers, like a server that is hung."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", port))
        self._listener.listen(16)
        self._held: list[socket.socket] = []
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            self._held.append(conn)

    def stop(self) -> None:
        self._listener.close()
        for conn in self._held:
            conn.close()


def _drain_until(client, count: int, timeout_s: float = 5.0) -> list:
    done: list = []
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    deadline = time.monotonic() + timeout_s
    while len(done) < count:
        assert time.monotonic() < deadline, "completion not delivered in time"
        if poller.poll(100):
            done.extend(client.drain_completions())
    return done


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.02)


@pytest.fixture
def connector_env():
    """Servers and connectors for one test, all stopped at the end."""
    servers: list[_MiniRespServer | _SilentServer] = []
    clients: list[LMCacheRedisClient] = []

    class Env:
        @staticmethod
        def resp_server(started: bool = True) -> _MiniRespServer:
            server = _MiniRespServer(_free_port())
            if started:
                server.start()
            servers.append(server)
            return server

        @staticmethod
        def silent_server() -> _SilentServer:
            server = _SilentServer(_free_port())
            servers.append(server)
            return server

        @staticmethod
        def client(port: int, num_workers: int = 4) -> LMCacheRedisClient:
            created = LMCacheRedisClient("127.0.0.1", port, num_workers)
            clients.append(created)
            return created

    yield Env
    for client in clients:
        client.close()
    for server in servers:
        server.stop()


def test_connector_declares_probe_and_reconnect(connector_env):
    client = connector_env.client(connector_env.resp_server().port)

    assert client.supports_health_probe()
    assert client.supports_reconnect()


def test_probe_is_ok_when_backend_and_connections_are_fine(connector_env):
    client = connector_env.client(connector_env.resp_server().port)

    _wait_until(lambda: client.probe_health(1000)[0] == "ok")


def test_probe_reports_a_missing_backend_as_unreachable(connector_env):
    client = connector_env.client(connector_env.resp_server(started=False).port)

    status, reason = client.probe_health(1000)

    assert status == "unreachable"
    assert reason


def test_backend_down_at_startup_needs_a_reconnect(connector_env):
    server = connector_env.resp_server(started=False)
    client = connector_env.client(server.port)
    time.sleep(0.5)  # every worker has tried to connect and failed

    server.start()

    # the backend is back, but no worker holds a connection
    assert client.probe_health(1000)[0] == "reconnect_needed"
    client.reconnect(3000)
    assert client.probe_health(1000)[0] == "ok"


def test_probe_finds_connections_a_restarted_server_closed(connector_env):
    server = connector_env.resp_server()
    client = connector_env.client(server.port)
    _wait_until(lambda: client.probe_health(1000)[0] == "ok")

    server.stop()
    server.start()

    status, reason = client.probe_health(2000)
    assert status == "reconnect_needed"
    assert "closed" in reason
    client.reconnect(3000)
    assert client.probe_health(1000)[0] == "ok"


def test_probe_gives_up_on_a_server_that_never_answers(connector_env):
    client = connector_env.client(connector_env.silent_server().port)

    start = time.monotonic()
    status, _ = client.probe_health(500)

    assert status == "unreachable"
    assert time.monotonic() - start < 2.0


def test_reconnect_fails_the_request_a_worker_was_blocked_in(connector_env):
    server = connector_env.silent_server()
    client = connector_env.client(server.port)
    time.sleep(0.5)  # every worker is connected
    future_id = client.submit_batch_exists(["k"])
    time.sleep(0.3)  # a worker is now waiting for the reply that never comes

    client.reconnect(3000)

    # The worker ended its I/O and reported the request as failed.
    completions = _drain_until(client, count=1)
    assert completions[0][0] == future_id
    assert completions[0][1] is False


def test_reconnect_raises_while_the_backend_is_down(connector_env):
    server = connector_env.resp_server(started=False)
    client = connector_env.client(server.port)

    with pytest.raises(RuntimeError):
        client.reconnect(3000)


_CLOSE_SCRIPT = textwrap.dedent(
    """
    import select, socket, sys, threading, time
    import torch  # noqa: F401
    from lmcache.lmcache_redis import LMCacheRedisClient

    scenario = sys.argv[1]
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    held = []
    password = ""
    if scenario == "auth":
        # the server accepts and never answers AUTH
        listener.listen(16)
        threading.Thread(
            target=lambda: [held.append(listener.accept()[0]) for _ in range(8)],
            daemon=True,
        ).start()
        password = "secret"
    else:
        # the accept queue holds one connection; later handshakes are dropped
        # and stay pending
        listener.listen(0)
        for _ in range(3):
            filler = socket.socket()
            filler.setblocking(False)
            filler.connect_ex(("127.0.0.1", port))
            held.append(filler)
        time.sleep(0.5)
        probe = socket.socket()
        probe.setblocking(False)
        probe.connect_ex(("127.0.0.1", port))
        time.sleep(0.5)
        if select.select([], [probe], [], 0)[1]:
            sys.exit(77)  # this kernel completes or refuses the handshake

    client = LMCacheRedisClient("127.0.0.1", port, 4, "", password)
    time.sleep(1.0)  # every worker is blocked by now
    start = time.monotonic()
    client.close()
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"close() took {elapsed:.1f}s"
    """
)


@pytest.mark.parametrize("blocked_in", ["auth", "connect"])
def test_closing_a_connector_whose_workers_are_blocked_returns(blocked_in):
    """Workers blocked in the handshake or in AUTH must not keep close() from
    returning, or a reconnect would hang exactly when the backend is silent."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", _CLOSE_SCRIPT, blocked_in],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"close() did not return with workers blocked in {blocked_in}")
    if done.returncode == 77:
        pytest.skip("this kernel does not leave a TCP handshake pending")
    assert done.returncode == 0, done.stdout + done.stderr
