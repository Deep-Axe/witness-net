"""The coordinator's barrier; this class itself never ticks CARLA."""

from dataclasses import dataclass, field


@dataclass
class FrameBarrier:
    """Require all control workers to acknowledge frame N before issuing N+1.

    Perception/API workers deliberately do not participate in this barrier.
    Missing or failed control workers stop progression instead of skipping time.
    """

    workers: frozenset[str]
    frame: int = -1
    _pending: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        self.workers = frozenset(self.workers)
        if not self.workers or any(not w for w in self.workers):
            raise ValueError("at least one named control worker is required")

    @property
    def ready(self) -> bool:
        return not self._pending

    def begin(self, frame: int) -> None:
        if not self.ready:
            raise RuntimeError(f"frame {self.frame} awaits {sorted(self._pending)}")
        if type(frame) is not int or frame <= self.frame:
            raise ValueError("frames must increase")
        self.frame = frame
        self._pending = set(self.workers)

    def acknowledge(self, worker: str, frame: int) -> None:
        if frame != self.frame or worker not in self._pending:
            raise ValueError("unknown, duplicate, or out-of-order acknowledgement")
        self._pending.remove(worker)
