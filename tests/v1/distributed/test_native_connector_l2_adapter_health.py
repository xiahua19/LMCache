# SPDX-License-Identifier: Apache-2.0
"""NativeConnectorL2Adapter forwards the health interface to its native client.

The probing and reconnecting logic lives in the C++ connector and is tested
against the real Redis connector in ``test_resp_l2_adapter_health.py``; here
only the bridge's contract is checked.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryResidency, MemorySegment, MemoryTags
from lmcache.v1.distributed.l2_adapters.health import L2ReconnectError, ProbeStatus
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.platform import create_event_notifier

_TAGS = MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE)


class _ClientWithoutHealth:
    """A native client built before the health interface existed."""

    def __init__(self) -> None:
        self._efd = create_event_notifier()

    def event_fd(self) -> int:
        return self._efd.fileno()

    def drain_completions(self) -> list:
        return []

    def close(self) -> None:
        self._efd.close()


class _ClientWithHealth(_ClientWithoutHealth):
    def __init__(self) -> None:
        super().__init__()
        self.probe_answer: tuple[str, str] = ("ok", "")
        self.probe_timeout_ms: int | None = None
        self.reconnect_error: Exception | None = None
        self.reconnect_timeout_ms: int | None = None

    def supports_health_probe(self) -> bool:
        return True

    def probe_health(self, timeout_ms: int) -> tuple[str, str]:
        self.probe_timeout_ms = timeout_ms
        return self.probe_answer

    def supports_reconnect(self) -> bool:
        return True

    def reconnect(self, timeout_ms: int) -> None:
        self.reconnect_timeout_ms = timeout_ms
        if self.reconnect_error is not None:
            raise self.reconnect_error


@pytest.fixture
def make_adapter():
    adapters: list[NativeConnectorL2Adapter] = []

    def make(client) -> NativeConnectorL2Adapter:
        adapter = NativeConnectorL2Adapter(client, _TAGS)
        adapters.append(adapter)
        return adapter

    yield make
    for adapter in adapters:
        adapter.close()


def test_a_client_without_health_methods_is_not_managed(make_adapter):
    adapter = make_adapter(_ClientWithoutHealth())

    assert not adapter.supports_health_probe
    assert not adapter.supports_reconnect


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ok", ProbeStatus.OK),
        ("unreachable", ProbeStatus.UNREACHABLE),
        ("reconnect_needed", ProbeStatus.RECONNECT_NEEDED),
    ],
)
def test_probe_health_passes_the_clients_answer_through(make_adapter, status, expected):
    client = _ClientWithHealth()
    client.probe_answer = (status, "why")
    adapter = make_adapter(client)

    assert adapter.supports_health_probe
    probe = adapter.probe_health(timeout_s=1.5)

    assert probe.status is expected
    assert probe.reason == "why"
    assert client.probe_timeout_ms == 1500


def test_reconnect_turns_a_client_failure_into_l2_reconnect_error(make_adapter):
    client = _ClientWithHealth()
    adapter = make_adapter(client)

    assert adapter.supports_reconnect
    adapter.reconnect()
    assert client.reconnect_timeout_ms is not None

    client.reconnect_error = RuntimeError("2 of 8 workers could not connect")
    with pytest.raises(L2ReconnectError, match="could not connect"):
        adapter.reconnect()
