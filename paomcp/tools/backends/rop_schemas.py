"""Shared ROP schemas and calling-convention constants (vong 3 split).

Single source of truth for every S7 stage; gadget/libc/chain modules
import from here instead of forking their own copies.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

PARSER_VERSION = "1"
GADGET_LIST_SCHEMA = "pao.gadget-list.v1"
GADGET_SEQUENCE_SCHEMA = "pao.gadget-sequence.v1"
GADGET_INSPECTION_SCHEMA = "pao.gadget-inspection.v1"
LIBC_IDENTIFICATION_SCHEMA = "pao.libc-identification.v1"
ONE_GADGET_SCHEMA = "pao.one-gadgets.v1"
ROP_CHAIN_SCHEMA = "pao.rop-chain.v1"
SIGRETURN_FRAME_SCHEMA = "pao.sigreturn-frame.v1"
CHAIN_CONSTRAINTS_SCHEMA = "pao.chain-constraints.v1"
CHAIN_VALIDATION_SCHEMA = "pao.chain-validation.v1"
CHAIN_EXPORT_SCHEMA = "pao.chain-export.v1"
CHAIN_EXPORT_FORMATS: Tuple[str, ...] = ("raw", "python", "pwntools")
DEFAULT_ONE_GADGET_TIMEOUT_SEC = 30.0
_SYSV_AMD64_ARG_REGS: Tuple[str, ...] = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
_LINUX_AMD64_SYSCALL_ARG_REGS: Tuple[str, ...] = (
    "rdi",
    "rsi",
    "rdx",
    "r10",
    "r8",
    "r9",
)

__all__ = [
    "PARSER_VERSION",
    "GADGET_LIST_SCHEMA",
    "GADGET_SEQUENCE_SCHEMA",
    "GADGET_INSPECTION_SCHEMA",
    "LIBC_IDENTIFICATION_SCHEMA",
    "ONE_GADGET_SCHEMA",
    "ROP_CHAIN_SCHEMA",
    "SIGRETURN_FRAME_SCHEMA",
    "CHAIN_CONSTRAINTS_SCHEMA",
    "CHAIN_VALIDATION_SCHEMA",
    "CHAIN_EXPORT_SCHEMA",
    "CHAIN_EXPORT_FORMATS",
    "DEFAULT_ONE_GADGET_TIMEOUT_SEC",
    "_SYSV_AMD64_ARG_REGS",
    "_LINUX_AMD64_SYSCALL_ARG_REGS",
]
