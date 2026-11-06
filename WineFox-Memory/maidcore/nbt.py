"""Read-only, standard-library NBT reader for Minecraft .maid archive exports.
This intentionally does NOT write game save data.
"""
from __future__ import annotations

import gzip
import io
import struct
from pathlib import Path

MAX_UNCOMPRESSED = 64 * 1024 * 1024
MAX_COLLECTION_ITEMS = 1_000_000
MAX_NESTING = 100


class NBTError(ValueError):
    """Archive was not valid or was too large to parse safely."""


class _NBTReader:
    def __init__(self, data: bytes):
        self.stream = io.BytesIO(data)

    def _read(self, n: int) -> bytes:
        b = self.stream.read(n)
        if len(b) != n:
            raise NBTError("NBT file is truncated")
        return b

    def _unpack(self, fmt: str):
        return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]

    def _count(self) -> int:
        n = self._unpack(">i")
        if not 0 <= n <= MAX_COLLECTION_ITEMS:
            raise NBTError("Unexpectedly large or negative NBT collection")
        return n

    def _string(self) -> str:
        n = self._unpack(">H")
        return self._read(n).decode("utf-8", errors="replace")

    def _value(self, tag: int, depth: int = 0):
        if depth > MAX_NESTING:
            raise NBTError("NBT nesting limit exceeded")
        if tag in (1, 2, 3, 4, 5, 6):
            return self._unpack({1: ">b", 2: ">h", 3: ">i", 4: ">q", 5: ">f", 6: ">d"}[tag])
        if tag == 8:
            return self._string()
        if tag in (7, 11, 12):
            n = self._count()
            if tag == 7:
                return list(self._read(n))
            fmt = {11: ">i", 12: ">q"}[tag]
            size = struct.calcsize(fmt)
            return [struct.unpack(fmt, self._read(size))[0] for _ in range(n)]
        if tag == 9:
            element_tag = self._unpack(">B")
            n = self._count()
            return [self._value(element_tag, depth + 1) for _ in range(n)]
        if tag == 10:
            result = {}
            while True:
                item_tag = self._unpack(">B")
                if item_tag == 0:
                    break
                key = self._string()
                result[key] = self._value(item_tag, depth + 1)
                if len(result) > MAX_COLLECTION_ITEMS:
                    raise NBTError("Too many compound keys")
            return result
        raise NBTError(f"Unsupported NBT type: {tag}")

    def root(self) -> dict:
        if self._unpack(">B") != 10:
            raise NBTError("Expected root compound (NBT type 10)")
        self._string()  # Root compound's name, often empty.
        result = self._value(10)
        if self.stream.read(1):
            raise NBTError("Unexpected trailing NBT data")
        return result


def read_maid_file(filename: str | Path) -> dict:
    """Load the contents of a gzip-compressed .maid file without modifying it."""
    path = Path(filename)
    # GzipFile streams so we can put a bound on decompression.
    try:
        with gzip.open(path, "rb") as source:
            raw = source.read(MAX_UNCOMPRESSED + 1)
    except (OSError, EOFError) as exc:
        raise NBTError(f"Cannot decompress .maid file: {exc}") from exc
    if len(raw) > MAX_UNCOMPRESSED:
        raise NBTError("Decompressed .maid archive exceeded size limit")
    data = _NBTReader(raw).root()
    if not isinstance(data.get("data"), dict):
        raise NBTError("Expected 'data' compound in .maid archive")
    return data
