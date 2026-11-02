"""Seeded reference broadcast with explicit sim-time delay and loss.

No MAC/PHY realism is claimed. Each source may send once per frame. Count every
transmission once, regardless of receiver count or eventual loss.
"""

from dataclasses import dataclass
import heapq
import random

from coopx.radio_codec import validate_payload


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
