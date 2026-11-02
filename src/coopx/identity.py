"""Ed25519 demo identities; receiver-configured authorization and common causes."""
from dataclasses import dataclass
from enum import Enum
import math
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

class TrustClass(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

@dataclass(frozen=True, slots=True)
class NodeKey:
    _private: Ed25519PrivateKey

    @classmethod
    def generate(cls):
        return cls(Ed25519PrivateKey.generate())

    @property
    def public_key(self) -> bytes:
        return self._private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def sign(self, envelope_bytes: bytes) -> bytes:
        return self._private.sign(envelope_bytes)

    @staticmethod
    def verify(public_key: bytes, envelope_bytes: bytes, signature: bytes) -> bool:
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, envelope_bytes)
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False

@dataclass(frozen=True, slots=True)
class TrustAnchor:
    public_key: bytes
    key_ref: str
    trust_class: TrustClass
    physical_observer: str | None = None
    # xmin,ymin,zmin,xmax,ymax,zmax in the configured spatial frame.
    permitted_region: tuple[float, float, float, float, float, float] | None = None
    common_causes: tuple[str, ...] = ()
    # Receiver-established physical binding, never taken from a sender's claim.
    # A configured body name alone does not anchor a class-B pseudonym.
    anchored: bool = False

    def __post_init__(self):
        Ed25519PublicKey.from_public_bytes(self.public_key)
        if not self.key_ref or not isinstance(self.trust_class, TrustClass):
            raise ValueError("key reference and TrustClass required")
        if self.physical_observer is not None and not self.physical_observer:
            raise ValueError("physical_observer must be nonempty when set")
        if self.permitted_region is not None:
            r = self.permitted_region
            if len(r) != 6 or not all(math.isfinite(v) for v in r) or any(r[i] > r[i+3] for i in range(3)):
                raise ValueError("invalid permitted region")
        if any(not c for c in self.common_causes):
            raise ValueError("common cause identifiers must be nonempty")
        if type(self.anchored) is not bool:
            raise ValueError("anchored must be a boolean")

    def permits(self, xyz: tuple[float, float, float]) -> bool:
        return len(xyz) == 3 and all(math.isfinite(v) for v in xyz) and (
            self.permitted_region is None or all(
                self.permitted_region[i] <= xyz[i] <= self.permitted_region[i+3] for i in range(3)))

TrustAnchors = dict[str, TrustAnchor]
