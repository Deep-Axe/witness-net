"""SAM 3.1 mask codec (contract 12). Port of @meta-sam/parser decodeMaskToRaster.

Wire: data=H,W,<enc><payload> with enc ~ lossless / ! one_bit. Payload is a
custom base85 (printable ASCII minus wire delimiters), not Adobe ASCII85.
"""

from __future__ import annotations

MASK32 = 0xFFFFFFFF
TOP = 1 << 24
BOTTOM = 1 << 16
RADIX = 85
PREFIX_LENGTH = 5
INCREMENT = 14
COUNT_LIMIT = 4096
CONTEXT_COUNT = 1 << 12
PADDING = 2
SYMBOLS = 256
TOTAL_INDEX = SYMBOLS
SPATIAL_MODES = 5
ZERO_INCREMENT = 32
ZERO_LIMIT = 16384
ORDER_ONE_INCREMENT = 56
ORDER_ONE_LIMIT = 8192
ORDER_TWO_STEP = 14 * 44
ORDER_TWO_LIMIT = 14 * 3584
ORDER_ZERO_STEP = 2 * 16
ORDER_ZERO_LIMIT = 2 * 2048
ORDER_ZERO_INITIAL = 2
ORDER_ONE_INITIAL = 1

_EXCLUDED = frozenset('"\\,;<>|')
_ALPHABET = "".join(
    chr(code) for code in range(0x21, 0x7F) if chr(code) not in _EXCLUDED)
_CHAR_VALUES = [-1] * 128
for _index, _ch in enumerate(_ALPHABET):
    _CHAR_VALUES[ord(_ch)] = _index


def _u32(value: int) -> int:
    return value & MASK32


class MaskError(ValueError):
    """Malformed mask payload; callers turn this into mask=None."""


def _digit(character: str) -> int:
    code = ord(character)
    value = _CHAR_VALUES[code] if code < 128 else -1
    if value < 0 or value >= RADIX:
        raise MaskError("Mask payload contains an invalid character.")
    return value


def _pack(data: bytes | bytearray) -> str:
    length = len(data)
    prefix = [""] * PREFIX_LENGTH
    remaining = length
    for index in range(PREFIX_LENGTH - 1, -1, -1):
        prefix[index] = _ALPHABET[remaining % RADIX]
        remaining //= RADIX
    if remaining != 0:
        raise MaskError("Mask payload is too large.")
    out = prefix
    offset = 0
    full = length - (length % 4)
    while offset < full:
        value = ((data[offset] << 24) | (data[offset + 1] << 16)
                 | (data[offset + 2] << 8) | data[offset + 3]) & MASK32
        digits = [0] * 5
        for index in range(4, -1, -1):
            digits[index] = value % RADIX
            value = (value - digits[index]) // RADIX
        out.extend(_ALPHABET[d] for d in digits)
        offset += 4
    remainder = length - offset
    if remainder:
        value = 0
        for index in range(4):
            value = ((value << 8) | (data[offset + index] if index < remainder else 0)) & MASK32
        digits = [0] * 5
        for index in range(4, -1, -1):
            digits[index] = value % RADIX
            value = (value - digits[index]) // RADIX
        out.extend(_ALPHABET[digits[index]] for index in range(remainder + 1))
    return "".join(out)


def _unpack(payload: str) -> bytes:
    if len(payload) < PREFIX_LENGTH:
        raise MaskError("Mask payload is missing its length prefix.")
    length = 0
    for index in range(PREFIX_LENGTH):
        length = length * RADIX + _digit(payload[index])
    remainder = length % 4
    expected = PREFIX_LENGTH + (length // 4) * 5 + (0 if remainder == 0 else remainder + 1)
    if len(payload) != expected:
        raise MaskError("Mask payload length does not match its prefix.")
    output = bytearray(length)
    source = PREFIX_LENGTH
    destination = 0
    full = length - remainder
    while destination < full:
        value = 0
        for index in range(5):
            value = value * RADIX + _digit(payload[source + index])
        if value > MASK32:
            raise MaskError("Mask payload contains an out-of-range group.")
        source += 5
        output[destination] = (value >> 24) & 0xFF
        output[destination + 1] = (value >> 16) & 0xFF
        output[destination + 2] = (value >> 8) & 0xFF
        output[destination + 3] = value & 0xFF
        destination += 4
    if remainder:
        value = 0
        for index in range(5):
            value = value * RADIX + (
                _digit(payload[source + index]) if index < remainder + 1 else RADIX - 1)
        if value > MASK32:
            raise MaskError("Mask payload contains an out-of-range tail.")
        for index in range(remainder):
            output[destination + index] = (value >> (24 - index * 8)) & 0xFF
    if _pack(output) != payload:
        raise MaskError("Mask payload is not in canonical form.")
    return bytes(output)


class _RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = MASK32
        self.output = bytearray()

    def encode(self, cumulative: int, frequency: int, total: int) -> None:
        scaled = self.range // total
        self.low = _u32(self.low + scaled * cumulative)
        self.range = scaled * frequency
        while True:
            if _u32(self.low ^ _u32(self.low + self.range)) < TOP:
                pass
            elif self.range < BOTTOM:
                self.range = _u32(-self.low) & (BOTTOM - 1)
            else:
                break
            self.output.append((self.low >> 24) & 0xFF)
            self.low = _u32(self.low << 8)
            self.range = _u32(self.range << 8)

    def finish(self) -> bytes:
        for _ in range(4):
            self.output.append((self.low >> 24) & 0xFF)
            self.low = _u32(self.low << 8)
        return bytes(self.output)


class _RangeDecoder:
    def __init__(self, data: bytes):
        if len(data) < 4:
            raise MaskError("Mask payload is missing its finalization.")
        self.input = data
        self.position = 0
        self.low = 0
        self.range = MASK32
        self.code = 0
        self.scaled = 0
        for _ in range(4):
            self.code = _u32((self.code << 8) | self._read())

    def frequency(self, total: int) -> int:
        self.scaled = self.range // total
        if self.scaled == 0:
            raise MaskError("Mask payload has invalid coding state.")
        value = _u32(self.code - self.low) // self.scaled
        return total - 1 if value >= total else value

    def decode(self, cumulative: int, frequency: int) -> None:
        self.low = _u32(self.low + self.scaled * cumulative)
        self.range = self.scaled * frequency
        while True:
            if _u32(self.low ^ _u32(self.low + self.range)) < TOP:
                pass
            elif self.range < BOTTOM:
                self.range = _u32(-self.low) & (BOTTOM - 1)
            else:
                break
            self.code = _u32((self.code << 8) | self._read())
            self.low = _u32(self.low << 8)
            self.range = _u32(self.range << 8)

    def _read(self) -> int:
        if self.position >= len(self.input):
            raise MaskError("Mask payload ended before decoding completed.")
        value = self.input[self.position]
        self.position += 1
        return value


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    left_d, up_d, ul_d = abs(estimate - left), abs(estimate - up), abs(estimate - upper_left)
    if left_d <= up_d and left_d <= ul_d:
        return left
    return up if up_d <= ul_d else upper_left


def _predict_byte(output, offset: int, x: int, y: int, mode: int, width: int) -> int:
    if mode < 0 or mode >= SPATIAL_MODES:
        raise MaskError("Lossless mask payload uses an invalid predictor.")
    left = output[offset - 1] if x > 0 else 0
    up = output[offset - width] if y > 0 else 0
    if mode == 1:
        return up
    if mode == 2:
        return left
    if mode == 3:
        return (left + up) >> 1
    upper_left = output[offset - width - 1] if x > 0 and y > 0 else 0
    if mode == 0:
        return _paeth(left, up, upper_left)
    if upper_left >= max(left, up):
        return min(left, up)
    if upper_left <= min(left, up):
        return max(left, up)
    return (left + up - upper_left) & 0xFF


def _unzig(value: int, prediction: int) -> int:
    delta = -((value + 1) >> 1) if value & 1 else value >> 1
    return (prediction + delta) & 0xFF


def _zero_context(residuals, offset: int, x: int, y: int, width: int) -> int:
    left = 1 if x == 0 or residuals[offset - 1] == 0 else 0
    up = 1 if y == 0 or residuals[offset - width] == 0 else 0
    upper_left = 1 if x == 0 or y == 0 or residuals[offset - width - 1] == 0 else 0
    upper_right = 1 if y == 0 or x >= width - 1 or residuals[offset - width + 1] == 0 else 0
    return left | (up << 1) | (upper_left << 2) | (upper_right << 3)


def _decode_lossless(payload: str, width: int, height: int) -> bytearray:
    if not payload.startswith("~"):
        raise MaskError("lossless mask payloads must start with ~.")
    packed = _unpack(payload[1:])
    if len(packed) < 5:
        raise MaskError("Lossless mask payload is truncated.")
    selector = packed[0]
    decoder = _RangeDecoder(packed[1:])
    length = width * height
    output = bytearray(length)
    residuals = bytearray(length)
    zero_counts = [1] * 32
    order_one = [ORDER_ONE_INITIAL] * (SYMBOLS * SYMBOLS)
    order_one_totals = [SYMBOLS * ORDER_ONE_INITIAL] * SYMBOLS
    order_zero = [ORDER_ZERO_INITIAL] * SYMBOLS
    order_zero_total = SYMBOLS * ORDER_ZERO_INITIAL
    order_two: dict[int, list[int]] = {}
    previous = 0
    for offset in range(length):
        x, y = offset % width, offset // width
        zero_offset = _zero_context(residuals, offset, x, y, width) << 1
        zero, nonzero = zero_counts[zero_offset], zero_counts[zero_offset + 1]
        bit = 0 if decoder.frequency(zero + nonzero) < zero else 1
        decoder.decode(0 if bit == 0 else zero, zero if bit == 0 else nonzero)
        zero_counts[zero_offset + bit] += ZERO_INCREMENT
        if zero_counts[zero_offset] + zero_counts[zero_offset + 1] >= ZERO_LIMIT:
            zero_counts[zero_offset] = zero_counts[zero_offset] >> 1 or 1
            zero_counts[zero_offset + 1] = zero_counts[zero_offset + 1] >> 1 or 1
        symbol = 0
        if bit == 1:
            above = residuals[offset - width] if offset >= width else 0
            key = (previous << 8) | above
            second = order_two.get(key)
            if second is None:
                second = [0] * (SYMBOLS + 1)
                order_two[key] = second
            first_offset = previous << 8
            excluded_zero = second[0] + order_one[first_offset] + order_zero[0]
            total = second[TOTAL_INDEX] + order_one_totals[previous] + order_zero_total - excluded_zero
            target = decoder.frequency(total)
            cumulative = 0
            symbol = 1
            frequency = second[1] + order_one[first_offset + 1] + order_zero[1]
            while cumulative + frequency <= target:
                cumulative += frequency
                symbol += 1
                if symbol >= SYMBOLS:
                    raise MaskError("Lossless mask payload is malformed.")
                frequency = second[symbol] + order_one[first_offset + symbol] + order_zero[symbol]
            decoder.decode(cumulative, frequency)
            second[symbol] += ORDER_TWO_STEP
            second[TOTAL_INDEX] += ORDER_TWO_STEP
            if second[TOTAL_INDEX] >= ORDER_TWO_LIMIT:
                total_after = 0
                for value in range(SYMBOLS):
                    second[value] >>= 1
                    total_after += second[value]
                second[TOTAL_INDEX] = total_after
            order_one[first_offset + symbol] += ORDER_ONE_INCREMENT
            order_one_totals[previous] += ORDER_ONE_INCREMENT
            if order_one_totals[previous] >= ORDER_ONE_LIMIT:
                total_after = 0
                for value in range(SYMBOLS):
                    scaled = order_one[first_offset + value] >> 1 or 1
                    order_one[first_offset + value] = scaled
                    total_after += scaled
                order_one_totals[previous] = total_after
            order_zero[symbol] += ORDER_ZERO_STEP
            order_zero_total += ORDER_ZERO_STEP
            if order_zero_total >= ORDER_ZERO_LIMIT:
                total_after = 0
                for value in range(SYMBOLS):
                    order_zero[value] >>= 1
                    total_after += order_zero[value]
                order_zero_total = total_after
        residuals[offset] = symbol
        output[offset] = _unzig(symbol, _predict_byte(output, offset, x, y, selector, width))
        previous = symbol
    return output


def _context_at(raster, position: int, prior: int, earlier: int) -> int:
    return (
        raster[position - 1]
        | (raster[position - 2] << 1)
        | (raster[prior - 2] << 2)
        | (raster[prior - 1] << 3)
        | (raster[prior] << 4)
        | (raster[prior + 1] << 5)
        | (raster[prior + 2] << 6)
        | (raster[earlier - 2] << 7)
        | (raster[earlier - 1] << 8)
        | (raster[earlier] << 9)
        | (raster[earlier + 1] << 10)
        | (raster[earlier + 2] << 11)
    )


def _update_counts(counts, offset: int, bit: int) -> None:
    counts[offset + bit] += INCREMENT
    if counts[offset] + counts[offset + 1] >= COUNT_LIMIT:
        counts[offset] = counts[offset] >> 1 or 1
        counts[offset + 1] = counts[offset + 1] >> 1 or 1


def encode_one_bit(raster, width: int, height: int) -> str:
    """Canonical one_bit payload including the leading ! marker. Tests only."""
    if width <= 0 or height <= 0 or width * height != len(raster):
        raise MaskError("Mask raster dimensions do not match its complete payload.")
    encoder = _RangeEncoder()
    counts = [1] * (CONTEXT_COUNT * 2)
    padded_width = width + 2 * PADDING
    padded = [0] * (padded_width * (height + PADDING))
    for y in range(height):
        row = (y + PADDING) * padded_width + PADDING
        for x in range(width):
            position = row + x
            context = _context_at(padded, position, position - padded_width,
                                  position - 2 * padded_width)
            offset = context << 1
            zero, one = counts[offset], counts[offset + 1]
            bit = int(raster[y * width + x])
            if bit not in (0, 1):
                raise MaskError("Decoded mask is not binary.")
            if bit == 0:
                encoder.encode(0, zero, zero + one)
            else:
                encoder.encode(zero, one, zero + one)
            padded[position] = bit
            _update_counts(counts, offset, bit)
    return "!" + _pack(encoder.finish())


def _decode_one_bit(payload: str, width: int, height: int) -> bytearray:
    decoder = _RangeDecoder(_unpack(payload[1:]))
    counts = [1] * (CONTEXT_COUNT * 2)
    padded_width = width + 2 * PADDING
    padded = [0] * (padded_width * (height + PADDING))
    output = bytearray(width * height)
    for y in range(height):
        row = (y + PADDING) * padded_width + PADDING
        for x in range(width):
            position = row + x
            context = _context_at(padded, position, position - padded_width,
                                  position - 2 * padded_width)
            offset = context << 1
            zero, one = counts[offset], counts[offset + 1]
            bit = 0 if decoder.frequency(zero + one) < zero else 1
            if bit == 0:
                decoder.decode(0, zero)
            else:
                decoder.decode(zero, one)
            padded[position] = bit
            output[y * width + x] = bit
            _update_counts(counts, offset, bit)
    return output


def decode_mask_raster(encoding: str, payload: str, width: int, height: int) -> bytearray:
    """Row-major 0/1 bytes, length width*height. Payload includes the ~ or ! marker."""
    if encoding not in ("one_bit", "lossless"):
        raise MaskError(f"Unsupported complete mask encoding: {encoding}.")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise MaskError("Mask dimensions must be positive integers.")
    if type(payload) is not str:
        raise MaskError("Mask payload must be a string.")
    if encoding == "one_bit" and not payload.startswith("!"):
        raise MaskError("one_bit mask payloads must start with !.")
    if encoding == "lossless" and not payload.startswith("~"):
        raise MaskError("lossless mask payloads must start with ~.")
    if encoding == "lossless":
        coverage = _decode_lossless(payload, width, height)
        return bytearray(1 if value >= 129 else 0 for value in coverage)
    raster = _decode_one_bit(payload, width, height)
    if encode_one_bit(raster, width, height) != payload:
        raise MaskError("Mask payload is not canonical or has invalid finalization.")
    return raster
