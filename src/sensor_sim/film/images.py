"""EXR helpers and a gray PNG whose samples are the literal uint8 values."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_gray_png(path: Path, image: np.ndarray) -> None:
    """Write an 8-bit gray PNG. Values are stored as given, including 0 and 1."""
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"gray PNG needs HxW, got {array.shape}")
    height, width = array.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def read_gray_png(path: Path) -> np.ndarray:
    """Read an 8-bit gray PNG back to uint8. This does not scale 1 to 255."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    offset = 8
    width = height = None
    compressed = b""
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
            if bit_depth != 8 or color_type != 0:
                raise ValueError(f"{path} is not 8-bit gray (depth={bit_depth} color={color_type})")
        elif tag == b"IDAT":
            compressed += chunk_data
        elif tag == b"IEND":
            break
    if width is None or height is None:
        raise ValueError(f"{path} has no IHDR")
    raw = zlib.decompress(compressed)
    rows = []
    stride = width + 1
    for row in range(height):
        start = row * stride
        if raw[start] != 0:
            raise ValueError(f"{path} uses a PNG filter other than None")
        rows.append(np.frombuffer(raw[start + 1 : start + 1 + width], dtype=np.uint8))
    return np.vstack(rows)


def write_exr(path: Path, image: np.ndarray, channel_names: list[str]) -> None:
    """Write a float EXR. A HxW array becomes one channel."""
    import mitsuba as mi

    array = np.ascontiguousarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[2] != len(channel_names):
        raise ValueError(f"{path} has {array.shape[2]} channels, names={channel_names}")
    mi.Bitmap(array, channel_names=list(channel_names)).write(str(path))


def read_exr(path: Path) -> dict[str, np.ndarray]:
    """Load an EXR. The Mitsuba variant must already be set."""
    import mitsuba as mi

    bitmap = mi.Bitmap(str(path))
    return {name: np.array(image, dtype=np.float64) for name, image in bitmap.split()}
