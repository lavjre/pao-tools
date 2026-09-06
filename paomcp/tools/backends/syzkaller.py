"""Syzkaller sidecar bridge (P2): config generation + crash ingest.

Syzkaller never runs inside the pao-tools container (it needs its own VM +
kernel build). This module is the handoff:

* :func:`generate_syz_config` renders a ``syz-manager.cfg`` from the existing
  QEMU target spec (image, ssh key/port, workdir, kernel src/obj) with strict
  validation. The operator boots ``syz-manager`` elsewhere.
* :func:`ingest_syz_crash` takes a syzkaller crash log (file or text), reuses
  the typed ``analyze_kernel_oops`` pipeline for fault kind / RIP / trace /
  fingerprint, and returns linked-triage output ready for
  ``link_evidence_batch``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from paomcp.tools.backends.foundation import build_backend_response
from paomcp.tools.backends.heap_schemas import PARSER_VERSION
from paomcp.tools.backends.kernel_oops import analyze_kernel_oops

SYZ_CONFIG_SCHEMA = "pao.syz-config.v1"
SYZ_CRASH_SCHEMA = "pao.syz-crash.v1"
MAX_LOG_BYTES = 1024 * 1024


def generate_syz_config(
    *,
    kernel_src: str,
    kernel_obj: Optional[str] = None,
    image_path: str,
    ssh_key_path: str,
    workdir: str,
    syzkaller_path: str = "/syzkaller",
    target_arch: str = "amd64",
    qemu_args: Optional[str] = None,
    repro_timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Render a syz-manager.cfg dict from a QEMU target spec (pure)."""
    if not kernel_src:
        raise ValueError("kernel_src is required (kernel source checkout)")
    if not image_path:
        raise ValueError("image_path is required (bootable QEMU image)")
    if not ssh_key_path:
        raise ValueError("ssh_key_path is required (syzkaller SSH key)")
    if not workdir:
        raise ValueError("workdir is required (syzkaller workdir)")
    forbidden = ("-virtfs", "-net ", "daemonize", "-snapshot=off")
    extra = qemu_args or ""
    for token in forbidden:
        if token in extra:
            raise ValueError(f"qemu_args forbids host-escape flag {token!r}")
    cfg = {
        "target": f"linux/{target_arch}",
        "http": "127.0.0.1:56741",
        "workdir": workdir,
        "kernel_src": kernel_src,
        "kernel_obj": kernel_obj or kernel_src,
        "image": image_path,
        "sshkey": ssh_key_path,
        "syzkaller": syzkaller_path,
        "type": "qemu",
        "reproduce": True,
        "repro_timeout": repro_timeout_seconds,
        "vm": {
            "count": 4,
            "kernel": kernel_obj or kernel_src,
            "cpu": 2,
            "mem": 2048,
            "qemu_args": extra,
        },
    }
    return build_backend_response(
        "generate_syz_config",
        parsed={
            "schema": SYZ_CONFIG_SCHEMA,
            "supported": True,
            "verdict": "rendered",
            "config": cfg,
            "config_json": json.dumps(cfg, indent=2, sort_keys=True),
            "next": "run syz-manager -config syz-manager.cfg on the sidecar VM host",
        },
        raw_responses=[],
        state="idle",
        parser_version=PARSER_VERSION,
    )


def ingest_syz_crash(
    *,
    log_text: Optional[str] = None,
    log_path: Optional[str] = None,
    max_bytes: int = MAX_LOG_BYTES,
) -> Dict[str, Any]:
    """Triage a syzkaller crash log through the kernel-oops pipeline."""
    if (log_text is None) == (log_path is None):
        raise ValueError("pass exactly one of log_text / log_path")
    if log_path is not None:
        if not os.path.isfile(log_path):
            raise ValueError(f"crash log not found: {log_path}")
        size = os.path.getsize(log_path)
        if size <= 0 or size > max_bytes:
            raise ValueError(f"crash log size {size} outside 1-{max_bytes}")
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            log_text = handle.read(max_bytes)
    assert log_text is not None
    triage = analyze_kernel_oops(log_text=log_text)
    parsed = dict(triage.get("parsed") or {})
    return build_backend_response(
        "ingest_syz_crash",
        parsed={
            "schema": SYZ_CRASH_SCHEMA,
            "supported": True,
            "verdict": parsed.get("verdict", "triaged"),
            "fault_kind": parsed.get("fault_kind"),
            "fingerprint": parsed.get("fingerprint"),
            "rip": parsed.get("rip"),
            "frames": parsed.get("frames") or parsed.get("trace"),
            "oops": parsed,
            "next": "link as crash evidence via link_evidence_batch",
        },
        raw_responses=[triage],
        state="idle",
        parser_version=PARSER_VERSION,
    )


__all__ = [
    "generate_syz_config",
    "ingest_syz_crash",
    "SYZ_CONFIG_SCHEMA",
    "SYZ_CRASH_SCHEMA",
]
