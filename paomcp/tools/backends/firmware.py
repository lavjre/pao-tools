"""Firmware extraction + carving (P2): unblob/binwalk when present, pure carve always.

``extract_firmware`` never fails just because external tools are missing:

1. try ``unblob`` (best fidelity) when on PATH,
2. else try ``binwalk -e`` when on PATH,
3. always run the pure-Python carver (magic scan for ELF/SquashFS/JFFS2/UBI/
   gzip/lzma/zlib/uImage/TRX + ASCII strings) so Windows/CI without firmware
   tooling still gets a typed inventory.

Output caps (2 GiB expanded, 600 s) blunt zip-bombs. Every finding carries
offsets so ``decompile_binary`` / ``analyze_elf`` can follow up per carved
file. Hostile firmware must run in a disposable container; the carver itself
only reads.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from paomcp.tools.backends.foundation import build_backend_response
from paomcp.tools.backends.heap_schemas import PARSER_VERSION

logger = logging.getLogger(__name__)

FIRMWARE_EXTRACT_SCHEMA = "pao.firmware-extract.v1"
MAX_FIRMWARE_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_CARVED_FILES = 256
MAX_STRINGS = 512

_MAGIC: Tuple[Tuple[str, bytes, str], ...] = (
    # ELF matched on \x7fELF + class only: OSABI varies (0x00 SysV, 0x03
    # Linux) and must not zero the elf_count. Class/endianness refined below.
    ("elf", b"\x7fELF", "ELF"),
    ("squashfs", b"hsqs", "SquashFS LE"),
    ("squashfs-be", b"sqsh", "SquashFS BE"),
    ("jffs2", b"\x85\x19", "JFFS2"),
    ("ubi", b"UBI#", "UBI"),
    ("gzip", b"\x1f\x8b\x08", "gzip"),
    ("zlib", b"\x78\x9c", "zlib"),
    ("zip", b"PK\x03\x04", "ZIP"),
    ("lzma", b"\x5d\x00\x00", "lzma"),
    ("uimage", b"\x27\x05\x19\x56", "uImage"),
    ("trx", b"HDR0", "TRX"),
    ("cramfs", b"Compressed ROMFS", "CramFS"),
    ("ubifs", b"UBI#", "UBI/UBIFS"),
    ("cpio", b"070701", "cpio newc"),
    ("dtb", b"\xd0\x0d\xfe\xed", "device-tree"),
    ("android-boot", b"ANDROID!", "Android bootimg"),
    ("fit", b"\xd0\x0d\xfe\xed", "FIT/device-tree"),
)


def _refine_elf(data: bytes, offset: int) -> Dict[str, Any]:
    """Refine a generic ELF hit with class/endianness when bytes exist."""
    if offset + 6 <= len(data):
        ei_class = data[offset + 4]
        ei_data = data[offset + 5]
        if ei_class in (1, 2) and ei_data in (1, 2):
            bits = "32-bit" if ei_class == 1 else "64-bit"
            endian = "LE" if ei_data == 1 else "BE"
            return {
                "kind": f"elf{32 if ei_class == 1 else 64}-{endian.lower()}",
                "description": f"ELF {bits} {endian}",
            }
    return {"kind": "elf", "description": "ELF (header truncated)"}


_STRING_RE = re.compile(rb"[\x20-\x7e]{6,}")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def carve_buffer(
    data: bytes, *, max_files: int = MAX_CARVED_FILES
) -> List[Dict[str, Any]]:
    """Scan bytes for known magics; returns offset/type/description hits.

    Same-offset duplicates (one magic is a prefix family of another) keep
    the first table entry only. ``size_hint`` is the byte distance to the
    next hit -- a carve bound, not a parsed size.
    """
    hits: List[Dict[str, Any]] = []
    for kind, magic, description in _MAGIC:
        start = 0
        while len(hits) < max_files * 2:
            idx = data.find(magic, start)
            if idx < 0:
                break
            if kind == "elf":
                refined = _refine_elf(data, idx)
                kind, description = refined["kind"], refined["description"]
            hits.append(
                {
                    "kind": kind,
                    "offset": idx,
                    "description": description,
                    "size_hint": None,
                }
            )
            start = idx + max(1, len(magic))
    hits.sort(key=lambda h: (int(h["offset"]), str(h["kind"])))
    deduped: List[Dict[str, Any]] = []
    seen_offsets = set()
    for hit in hits:
        if hit["offset"] in seen_offsets:
            continue
        seen_offsets.add(hit["offset"])
        deduped.append(hit)
        if len(deduped) >= max_files:
            break
    for position, hit in enumerate(deduped):
        if position + 1 < len(deduped):
            hit["size_hint"] = int(deduped[position + 1]["offset"]) - int(hit["offset"])
    return deduped


def extract_strings(data: bytes, *, max_strings: int = MAX_STRINGS) -> List[str]:
    """Printable ASCII strings >= 6 chars, capped (pure)."""
    out: List[str] = []
    for match in _STRING_RE.finditer(data):
        try:
            out.append(match.group(0).decode("ascii"))
        except UnicodeDecodeError:
            continue
        if len(out) >= max_strings:
            break
    return out


def _try_unblob(
    image_path: str, out_dir: str, timeout: int
) -> Optional[Dict[str, Any]]:
    exe = shutil.which("unblob")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-e", out_dir, image_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)}
    count = 0
    for _root, _dirs, files in os.walk(out_dir):
        count += len(files)
    return {"ok": proc.returncode == 0, "exit_code": proc.returncode, "files": count}


def _try_binwalk(
    image_path: str, out_dir: str, timeout: int
) -> Optional[Dict[str, Any]]:
    exe = shutil.which("binwalk")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-e", "-C", out_dir, image_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)}
    count = 0
    for _root, _dirs, files in os.walk(out_dir):
        count += len(files)
    return {"ok": proc.returncode == 0, "exit_code": proc.returncode, "files": count}


def _enforce_expanded_budget(output_dir: str, cap_bytes: int) -> None:
    """Fail closed when an extractor expanded past the zip-bomb budget."""
    total = 0
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
            if total > cap_bytes:
                raise ValueError(
                    f"extracted output exceeds the {cap_bytes}-byte budget; "
                    "refusing to inventory a suspected zip-bomb"
                )


def extract_firmware(
    *,
    image_path: str,
    output_dir: Optional[str] = None,
    timeout_seconds: int = 600,
    max_bytes: int = MAX_FIRMWARE_BYTES,
    cancellation_check: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Extract + carve one firmware image; external tools optional."""
    extraction_id = uuid.uuid4().hex[:12]
    if not os.path.isfile(image_path):
        raise ValueError(f"firmware image not found: {image_path}")
    size = os.path.getsize(image_path)
    if size <= 0 or size > max_bytes:
        raise ValueError(f"firmware size {size} outside 1-{max_bytes} bytes")
    if timeout_seconds <= 0 or timeout_seconds > 3600:
        raise ValueError("timeout_seconds must be 1-3600")
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(image_path), f".extract-{extraction_id}"
        )
    os.makedirs(output_dir, exist_ok=True)
    started = time.monotonic()
    deadline = started + timeout_seconds

    def _budget_left() -> float:
        return max(1.0, deadline - time.monotonic())

    def _poll_cancel() -> None:
        if cancellation_check is not None:
            cancellation_check()

    method = "carve-only"
    external: Optional[Dict[str, Any]] = None
    # Fallback chain: unblob -> binwalk -> carve-only. A *failed* run falls
    # through to the next stage (previously `elif` skipped binwalk whenever
    # unblob merely existed but failed).
    for tool_name, runner, subdir in (
        ("unblob", _try_unblob, "unblob"),
        ("binwalk", _try_binwalk, "binwalk"),
    ):
        _poll_cancel()
        try:
            result = runner(
                image_path, os.path.join(output_dir, subdir), int(_budget_left())
            )
        except (OSError, subprocess.SubprocessError) as exc:
            result = {"ok": False, "error": str(exc)}
        if result is None:
            continue  # tool not installed; try the next stage
        external = {"tool": tool_name, **result}
        if result.get("ok"):
            method = tool_name
            break
        method = f"{tool_name}-failed-carve-only"
    _poll_cancel()
    _enforce_expanded_budget(output_dir, MAX_EXPANDED_BYTES)
    with open(image_path, "rb") as handle:
        head = handle.read(min(size, 64 * 1024 * 1024))
        tail = b""
        tail_base = 0
        if size > len(head):
            tail_window = min(size - len(head), 16 * 1024 * 1024)
            handle.seek(size - tail_window)
            tail = handle.read(tail_window)
            tail_base = size - tail_window
    _poll_cancel()
    carved = carve_buffer(head)
    if tail:
        for hit in carve_buffer(tail):
            hit["offset"] = int(hit["offset"]) + tail_base
            if len(carved) < MAX_CARVED_FILES and all(
                h["offset"] != hit["offset"] for h in carved
            ):
                carved.append(hit)
        carved.sort(key=lambda h: (int(h["offset"]), str(h["kind"])))
    strings = extract_strings(head[: 4 * 1024 * 1024])
    elfs = [h for h in carved if str(h["kind"]).startswith("elf")]
    return build_backend_response(
        "extract_firmware",
        parsed={
            "schema": FIRMWARE_EXTRACT_SCHEMA,
            "supported": True,
            "verdict": "extracted",
            "extraction_id": extraction_id,
            "image": {"path": image_path, "size": size, "sha256": _sha256(image_path)},
            "method": method,
            "external": external,
            "output_dir": output_dir,
            "carved_count": len(carved),
            "carved": carved,
            "elf_count": len(elfs),
            "strings_sample": strings[:64],
            "string_count": len(strings),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "next": "run analyze_elf/decompile_binary on carved ELF offsets",
            "safety": "treat output as untrusted; analyze in a disposable container",
        },
        raw_responses=[],
        state="idle",
        parser_version=PARSER_VERSION,
    )


__all__ = [
    "extract_firmware",
    "carve_buffer",
    "extract_strings",
    "FIRMWARE_EXTRACT_SCHEMA",
]
