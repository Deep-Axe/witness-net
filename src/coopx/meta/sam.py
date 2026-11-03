"""SAM 3.1 request builder, box tokens, and mask decode (contracts 7 and 12).

Grammar (dev.meta.ai/docs/sam/reading-segmentation): one line per frame,
`<Nf>` then records `id<|box;…|><|mask;…|>`. Box corners are source pixels;
w/h repeat the source size (rescaled defensively). Mask rasters are H×W,
anchored at the box, then nearest-neighbour resized to the original image.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import re

import numpy as np

from coopx.meta.client import MetaClient, ModelResult
from coopx.meta.mask import MaskError, decode_mask_raster

TOKEN = re.compile(r"<\|(box|mask);([^|]+)\|>")
SAM_INSTRUCTIONS = ""


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float | None
    mask: np.ndarray | None  # bool (image_height, image_width) or None


def _finite(value: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("nonfinite box token")
    return number


def _fields(body: str) -> dict[str, str]:
    fields = {}
    for part in body.split(";"):
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _nearest_bool(canvas: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = canvas.shape
    if src_h == height and src_w == width:
        return canvas
    ys = np.minimum((np.arange(height) * src_h) // height, src_h - 1)
    xs = np.minimum((np.arange(width) * src_w) // width, src_w - 1)
    return canvas[ys[:, None], xs]


def _decode_mask(fields: dict[str, str], box, token_size, image_size) -> np.ndarray | None:
    """HxW raster pasted at the box in token space, then scaled to image_size."""
    data = fields.get("data")
    if not data:
        return None
    try:
        height_s, width_s, rest = data.split(",", 2)
        raster_h, raster_w = int(height_s), int(width_s)
        if raster_h <= 0 or raster_w <= 0 or not rest:
            return None
        marker, payload = rest[0], rest
        encoding = {"~": "lossless", "!": "one_bit"}.get(marker)
        if encoding is None:
            return None
        bits = decode_mask_raster(encoding, payload, raster_w, raster_h)
    except (ValueError, MaskError, OverflowError):
        return None
    raster = np.frombuffer(bytes(bits), dtype=np.uint8).reshape(raster_h, raster_w).astype(bool)
    token_w, token_h = int(token_size[0]), int(token_size[1])
    if token_w <= 0 or token_h <= 0:
        return None
    canvas = np.zeros((token_h, token_w), dtype=bool)
    x1, y1, x2, y2 = box
    y0 = max(0, min(token_h, int(round(y1))))
    x0 = max(0, min(token_w, int(round(x1))))
    y1c = min(token_h, y0 + raster_h)
    x1c = min(token_w, x0 + raster_w)
    canvas[y0:y1c, x0:x1c] = raster[: y1c - y0, : x1c - x0]
    image_w, image_h = int(image_size[0]), int(image_size[1])
    return _nearest_bool(canvas, image_h, image_w)


def parse_sam_text(text: str, image_size: tuple[int, int]) -> list[Detection]:
    """Parse SAM box+mask tokens and rescale from the token w/h frame to image_size."""
    if not isinstance(text, str) or not text:
        return []
    if (not isinstance(image_size, tuple) or len(image_size) != 2
            or any(type(v) not in (int, float) or v <= 0 for v in image_size)):
        raise ValueError("image_size must be a positive (width, height)")
    width, height = float(image_size[0]), float(image_size[1])
    detections: list[Detection] = []
    pending = None
    for kind, body in TOKEN.findall(text):
        fields = _fields(body)
        if kind == "box":
            if pending is not None:
                parsed = _box_only(pending, width, height)
                if parsed is not None:
                    detections.append(parsed)
            pending = fields
        elif kind == "mask" and pending is not None:
            parsed = _with_mask(pending, fields, width, height, image_size)
            if parsed is not None:
                detections.append(parsed)
            pending = None
    if pending is not None:
        parsed = _box_only(pending, width, height)
        if parsed is not None:
            detections.append(parsed)
    return detections


def _box_geometry(fields: dict[str, str], width: float, height: float):
    try:
        x1, y1, x2, y2 = (_finite(fields[k]) for k in ("x1", "y1", "x2", "y2"))
        token_w, token_h = _finite(fields["w"]), _finite(fields["h"])
        score = None if "c" not in fields else _finite(fields["c"])
    except (KeyError, ValueError):
        return None
    if token_w <= 0 or token_h <= 0:
        return None
    sx, sy = width / token_w, height / token_h
    box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    return box, score, (token_w, token_h), (x1, y1, x2, y2)


def _box_only(fields: dict[str, str], width: float, height: float) -> Detection | None:
    parsed = _box_geometry(fields, width, height)
    if parsed is None:
        return None
    box, score, _token, _src = parsed
    return Detection(box, score, None)


def _with_mask(box_fields, mask_fields, width, height, image_size) -> Detection | None:
    parsed = _box_geometry(box_fields, width, height)
    if parsed is None:
        return None
    box, score, token_size, src_box = parsed
    mask = _decode_mask(mask_fields, src_box, token_size, image_size)
    return Detection(box, score, mask)


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
