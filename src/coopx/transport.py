"""Seeded reference broadcast with explicit sim-time delay and loss.

No MAC/PHY realism is claimed. Each source may send once per frame. Count every
transmission once, regardless of receiver count or eventual loss.

PacketTransport is the CX2 binary path (delay, jitter, loss, disconnect, budget).
ReferenceTransport is unchanged and still carries ASCII CX1 payloads.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import random
from typing import Callable

from coopx.events import EVENT_TYPES, FIELDS, SCHEMA as EVENT_SCHEMA
from coopx.radio_codec import validate_payload

EVENT_KEYS = FIELDS
MAX_PACKET_BYTES = 65536
BUDGET_WINDOW_US = 1_000_000


@dataclass(frozen=True, slots=True)
class Delivery:
    source: str
    receiver: str
    sent_sim_us: int
    available_sim_us: int
    payload: str


class ReferenceTransport:
    def __init__(self, nodes: tuple[str, ...], *, delay_us: int = 100_000,
                 loss_probability: float = 0, seed: int = 0):
        if len(set(nodes)) != len(nodes) or len(nodes) < 2 or any(not n for n in nodes):
            raise ValueError("provide at least two unique named nodes")
        if type(delay_us) is not int or delay_us < 0 or not 0 <= loss_probability <= 1:
            raise ValueError("invalid delay/loss")
        self.nodes = nodes
        self.delay_us = delay_us
        self.loss_probability = loss_probability
        self.sent_bytes = 0
        self.received_bytes = 0
        self.dropped_deliveries = 0
        self._rng = random.Random(seed)
        self._last_frame: dict[str, int] = {}
        self._last_send_us: dict[str, int] = {}
        self._now_us = 0
        self._order = 0
        self._pending = []

    def send(self, source: str, payload: str, *, frame: int, sim_us: int) -> None:
        size = validate_payload(payload)
        if source not in self.nodes:
            raise ValueError("unknown source")
        if type(frame) is not int or frame < 0 or frame <= self._last_frame.get(source, -1):
            raise ValueError("source may send only once per increasing frame")
        if type(sim_us) is not int or sim_us < max(self._now_us, self._last_send_us.get(source, 0)):
            raise ValueError("send time must not go backwards")
        self._last_frame[source] = frame
        self._last_send_us[source] = sim_us
        self.sent_bytes += size
        for receiver in self.nodes:
            if receiver == source:
                continue
            if self._rng.random() < self.loss_probability:
                self.dropped_deliveries += 1
                continue
            delivery = Delivery(source, receiver, sim_us, sim_us + self.delay_us, payload)
            self._order += 1
            heapq.heappush(self._pending, (delivery.available_sim_us, self._order, delivery))

    def receive_until(self, sim_us: int) -> list[Delivery]:
        if type(sim_us) is not int or sim_us < self._now_us:
            raise ValueError("receive clock must not go backwards")
        self._now_us = sim_us
        ready = []
        while self._pending and self._pending[0][0] <= sim_us:
            _, _, delivery = heapq.heappop(self._pending)
            self.received_bytes += len(delivery.payload.encode("ascii"))
            ready.append(delivery)
        return ready


def packet_hash(packet: bytes) -> str:
    return hashlib.sha256(packet).hexdigest()


def make_event(*, t_replay_us: int, type: str, source_id: str,
               dest_id: str | None = None, message_id: str | None = None,
               message_type: str | None = None, evidence_ref: str | None = None,
               bytes: int | None = None, check_codes: list[str] | None = None,
               state: str | None = None, detail: dict | None = None,
               extra_types: frozenset[str] | tuple[str, ...] = ()) -> dict:
    allowed = EVENT_TYPES if not extra_types else (set(EVENT_TYPES) | set(extra_types))
    if type not in allowed:
        raise ValueError(f"unknown event type {type}")
    return {
        "schema": EVENT_SCHEMA,
        "t_replay_us": t_replay_us,
        "type": type,
        "source_id": source_id,
        "dest_id": dest_id,
        "message_id": message_id,
        "message_type": message_type,
        "evidence_ref": evidence_ref,
        "bytes": bytes,
        "check_codes": [] if check_codes is None else list(check_codes),
        "state": state,
        "detail": {} if detail is None else dict(detail),
    }


@dataclass(frozen=True, slots=True)
class PacketDelivery:
    source: str
    receiver: str
    sent_sim_us: int
    available_sim_us: int
    packet: bytes
    message_id: str | None = None
    message_type: str | None = None
    evidence_ref: str | None = None
    tx_id: str | None = None


class PacketTransport:
    """Binary CX2 broadcast: delay + jitter, loss, disconnect, optional byte budget."""

    def __init__(self, nodes: tuple[str, ...], *, delay_us: int = 100_000,
                 jitter_us: int = 0, loss_probability: float = 0, seed: int = 0,
                 budget_bytes_per_s: int | None = None,
                 event_callback: Callable[[dict], None] | None = None,
                 delivery_callback: Callable[["PacketDelivery"], None] | None = None):
        if len(set(nodes)) != len(nodes) or len(nodes) < 2 or any(not n for n in nodes):
            raise ValueError("provide at least two unique named nodes")
        if type(delay_us) is not int or delay_us < 0:
            raise ValueError("invalid delay")
        if type(jitter_us) is not int or jitter_us < 0:
            raise ValueError("invalid jitter")
        if not 0 <= loss_probability <= 1:
            raise ValueError("invalid loss")
        if budget_bytes_per_s is not None and (type(budget_bytes_per_s) is not int or budget_bytes_per_s < 0):
            raise ValueError("invalid budget")
        self.nodes = nodes
        self.delay_us = delay_us
        self.jitter_us = jitter_us
        self.loss_probability = loss_probability
        self.budget_bytes_per_s = budget_bytes_per_s
        self.event_callback = event_callback
        self.delivery_callback = delivery_callback
        self.sent_bytes = 0
        self.received_bytes = 0
        self.drops = {"loss": 0, "disconnected": 0, "budget": 0}
        self.sent_bytes_by_message_type: dict[str, int] = defaultdict(int)
        self.received_bytes_by_message_type: dict[str, int] = defaultdict(int)
        self.drops_by_message_type: dict[str, dict[str, int]] = defaultdict(
            lambda: {"loss": 0, "disconnected": 0, "budget": 0})
        self.events: list[dict] = []
        self._rng = random.Random(seed)
        self._last_send_us: dict[str, int] = {}
        self._now_us = 0
        self._order = 0
        self._tx_seq = 0
        self._pending: list = []
        self._disconnected: set[str] = set()
        self._budget_used: dict[tuple[str, str, int], int] = {}

    def send(self, source: str, packet: bytes, *, sim_us: int,
             message_id: str | None = None, message_type: str | None = None) -> list[dict]:
        if source not in self.nodes:
            raise ValueError("unknown source")
        if not isinstance(packet, (bytes, bytearray)) or not 1 <= len(packet) <= MAX_PACKET_BYTES:
            raise ValueError("packet must be bytes of length 1..65536")
        if type(sim_us) is not int or sim_us < max(self._now_us, self._last_send_us.get(source, 0)):
            raise ValueError("send time must not go backwards")
        packet = bytes(packet)
        size = len(packet)
        ref = packet_hash(packet)
        self._tx_seq += 1
        tx_id = f"{source}:{sim_us}:{self._tx_seq}"
        produced: list[dict] = []
        self._last_send_us[source] = sim_us
        meta = dict(message_id=message_id, message_type=message_type, evidence_ref=ref,
                    bytes=size, tx_id=tx_id)

        if source in self._disconnected:
            for receiver in self.nodes:
                if receiver == source:
                    continue
                produced.append(self._drop(sim_us, source, receiver, "disconnected", **meta))
            return produced

        self.sent_bytes += size
        if message_type is not None:
            self.sent_bytes_by_message_type[message_type] += size
        produced.append(self._emit(make_event(
            t_replay_us=sim_us, type="packet_sent", source_id=source, dest_id=None,
            message_id=message_id, message_type=message_type, evidence_ref=ref,
            bytes=size, detail={"tx_id": tx_id})))

        for receiver in self.nodes:
            if receiver == source:
                continue
            if receiver in self._disconnected:
                produced.append(self._drop(sim_us, source, receiver, "disconnected", **meta))
                continue
            if self.budget_bytes_per_s is not None:
                window = sim_us // BUDGET_WINDOW_US
                key = (source, receiver, window)
                used = self._budget_used.get(key, 0)
                if used + size > self.budget_bytes_per_s:
                    produced.append(self._drop(sim_us, source, receiver, "budget", **meta))
                    continue
                self._budget_used[key] = used + size
            if self._rng.random() < self.loss_probability:
                produced.append(self._drop(sim_us, source, receiver, "loss", **meta))
                continue
            jitter = self._rng.randint(0, self.jitter_us) if self.jitter_us else 0
            delivery = PacketDelivery(
                source, receiver, sim_us, sim_us + self.delay_us + jitter, packet,
                message_id, message_type, ref, tx_id)
            self._order += 1
            heapq.heappush(self._pending, (delivery.available_sim_us, self._order, delivery))
        return produced

    def receive_until(self, sim_us: int) -> list[PacketDelivery]:
        if type(sim_us) is not int or sim_us < self._now_us:
            raise ValueError("receive clock must not go backwards")
        self._now_us = sim_us
        ready = []
        while self._pending and self._pending[0][0] <= sim_us:
            _, _, delivery = heapq.heappop(self._pending)
            self.received_bytes += len(delivery.packet)
            if delivery.message_type is not None:
                self.received_bytes_by_message_type[delivery.message_type] += len(delivery.packet)
            self._emit(make_event(
                t_replay_us=delivery.available_sim_us, type="packet_delivered",
                source_id=delivery.source, dest_id=delivery.receiver,
                message_id=delivery.message_id, message_type=delivery.message_type,
                evidence_ref=delivery.evidence_ref, bytes=len(delivery.packet),
                detail={"tx_id": delivery.tx_id}))
            if self.delivery_callback is not None:
                self.delivery_callback(delivery)
            ready.append(delivery)
        return ready

    def disconnect(self, node: str, *, sim_us: int) -> list[dict]:
        self._require_node_time(node, sim_us)
        if node in self._disconnected:
            return []
        self._disconnected.add(node)
        produced = [self._emit(make_event(
            t_replay_us=sim_us, type="observer_disconnected", source_id=node))]
        kept = []
        for item in self._pending:
            delivery = item[2]
            # Due packets (available_sim_us <= cutoff) stay deliverable.
            if delivery.receiver == node and delivery.available_sim_us > sim_us:
                produced.append(self._drop(
                    sim_us, delivery.source, delivery.receiver, "disconnected",
                    message_id=delivery.message_id, message_type=delivery.message_type,
                    evidence_ref=delivery.evidence_ref, bytes=len(delivery.packet),
                    tx_id=delivery.tx_id, in_flight=True))
            else:
                kept.append(item)
        heapq.heapify(kept)
        self._pending = kept
        return produced

    def reconnect(self, node: str, *, sim_us: int) -> list[dict]:
        self._require_node_time(node, sim_us)
        if node not in self._disconnected:
            return []
        self._disconnected.remove(node)
        return [self._emit(make_event(
            t_replay_us=sim_us, type="observer_reconnected", source_id=node))]

    def _require_node_time(self, node: str, sim_us: int) -> None:
        if node not in self.nodes:
            raise ValueError("unknown node")
        if type(sim_us) is not int or sim_us < self._now_us:
            raise ValueError("control time must not go backwards")
        self._now_us = sim_us

    def _drop(self, sim_us: int, source: str, dest: str, reason: str, *,
              message_id, message_type, evidence_ref, bytes, tx_id=None,
              in_flight: bool = False) -> dict:
        self.drops[reason] += 1
        if message_type is not None:
            self.drops_by_message_type[message_type][reason] += 1
        detail = {"reason": reason, "in_flight": in_flight}
        if tx_id is not None:
            detail["tx_id"] = tx_id
        return self._emit(make_event(
            t_replay_us=sim_us, type="packet_dropped", source_id=source, dest_id=dest,
            message_id=message_id, message_type=message_type, evidence_ref=evidence_ref,
            bytes=bytes, detail=detail))

    def _emit(self, event: dict) -> dict:
        self.events.append(event)
        if self.event_callback is not None:
            self.event_callback(event)
        return event
