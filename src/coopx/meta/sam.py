"""SAM 3.1 request builder and box-token parser (contract 7).

Grammar (dev.meta.ai/docs/sam/reading-segmentation): one line per frame,
`<Nf>` then records `id<|box;x1;y1;x2;y2;w;h[;c]|><|mask;...|>`. Box corners
are source pixels; w/h repeat the source size (rescaled defensively). `c` is the
confidence when requested. Masks (base85 rasters) are not decoded: None.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import re

from coopx.meta.client import MetaClient, ModelResult

BOX_TOKEN = re.compile(
    r"<\|box;x1=([^;|]+);y1=([^;|]+);x2=([^;|]+);y2=([^;|]+);w=([^;|]+);h=([^;|]+)"
    r"(?:;c=([^;|]+))?\|>"
)
# SAM takes no instructions; the concept phrase is the whole prompt.
SAM_INSTRUCTIONS = ""


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float | None
    mask: object | None


def _finite(value: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("nonfinite box token")
    return number


def parse_sam_text(text: str, image_size: tuple[int, int]) -> list[Detection]:
    """Parse SAM box tokens and rescale from the token w/h frame to image_size."""
    if not isinstance(text, str) or not text:
        return []
    if (not isinstance(image_size, tuple) or len(image_size) != 2
            or any(type(v) not in (int, float) or v <= 0 for v in image_size)):
        raise ValueError("image_size must be a positive (width, height)")
    width, height = float(image_size[0]), float(image_size[1])
    detections = []
    for match in BOX_TOKEN.finditer(text):
        try:
            x1, y1, x2, y2, token_w, token_h = (_finite(g) for g in match.groups()[:6])
            score = None if match.group(7) is None else _finite(match.group(7))
        except ValueError:
            continue
        if token_w <= 0 or token_h <= 0:
            continue
        sx, sy = width / token_w, height / token_h
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        detections.append(Detection(box, score, None))
    return detections


def _png_data_url(image_png: bytes) -> str:
    if not isinstance(image_png, (bytes, bytearray)) or not image_png:
        raise ValueError("image_png must be nonempty bytes")
    media = "image/jpeg" if bytes(image_png[:2]) == b"\xff\xd8" else "image/png"
    return f"data:{media};base64," + base64.b64encode(bytes(image_png)).decode("ascii")


def segment(client: MetaClient, image_png: bytes, concept: str, *,
            image_size: tuple[int, int], query_time_us: int,
            stage: str = "segment") -> tuple[list[Detection], ModelResult]:
    if not isinstance(concept, str) or not concept.strip() or len(concept) > 64:
        raise ValueError("concept must be a nonempty noun phrase ≤ 64 characters")
    phrase = concept.strip()
    result = client.respond(
        model="sam-3.1",
        instructions=SAM_INSTRUCTIONS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": _png_data_url(image_png)},
            ],
        }],
        query_time_us=query_time_us,
        stage=stage,
    )
    return parse_sam_text(result.text, image_size), result
