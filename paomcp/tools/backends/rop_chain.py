"""ROP chain construction, validation and export (split from rop.py)."""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
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
from paomcp.tools.backends.foundation import build_backend_response, parse_int
from paomcp.tools.backends.rop_schemas import (
    CHAIN_CONSTRAINTS_SCHEMA,
    CHAIN_EXPORT_FORMATS,
    CHAIN_EXPORT_SCHEMA,
    CHAIN_VALIDATION_SCHEMA,
    PARSER_VERSION,
    ROP_CHAIN_SCHEMA,
    SIGRETURN_FRAME_SCHEMA,
    _LINUX_AMD64_SYSCALL_ARG_REGS,
    _SYSV_AMD64_ARG_REGS,
)
from paomcp.tools.backends.rop_gadgets import GadgetIndex
from paomcp.tools.backends.rop_libc import LibcIdentifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# typed ROP chain construction and sigreturn frame builder
# ---------------------------------------------------------------------------
class ChainBuilderError(RuntimeError):
    """Raised when a chain build request is malformed or a gadget is missing."""


class ChainBuilder:
    """Compose typed ROP chains from a live gadget index and libc metadata.

    Supported ``kind`` values today:

    * ``ret2libc``    -- x86_64 SysV: load rdi/rsi/rdx via ``pop <reg>``
      gadgets then transfer control to a target libc symbol (typically
      ``system``, ``execve`` or ``one_gadget``).  Stack alignment is
      computed, not assumed: the chain is entered via ``ret`` with rsp at
      8 (mod 16) on the first slot, every argument pops exactly two slots,
      and the target is entered by the final pop -- so rsp lands on the
      16-byte phase required by ``movaps``-using functions exactly when the
      total pop count is even.  A pure pop-args chain pops ``2*n + 1``
      slots (always odd), so ``align_stack=True`` appends a single bare
      ``ret`` to make the count even; the parity check is explicit so a
      chain shape that already leaves an even pop count skips the ``ret``.
    * ``ret2syscall`` -- x86_64 Linux: load rax with a syscall number,
      then set rdi/rsi/rdx/r10/r8/r9 (SYSCALL replaces rcx with r10)
      via pop gadgets, then execute a ``syscall ; ret`` gadget.

    ``ret2csu``, ``ret2dlresolve`` and stack-pivot flavors are documented
    as deliberate follow-ups: they need multi-gadget solver plumbing
    that is scoped to a later pass.  What ships here is what a real CTF
    ret2libc / ret2syscall exploit uses first.

    The sibling ``inspect_sigreturn_frame`` builds a typed x86_64
    sigreturn frame -- the payload a ``mov rax, 0xf ; syscall`` gadget
    hands to the kernel to force arbitrary register state before a
    single ``ret``.  We wrap :class:`pwnlib.rop.srop.SigreturnFrame` so
    the layout stays authoritative but expose it under the typed backend
    envelope every other heap/ROP tool uses.
    """

    def __init__(self, gadgets: Optional[GadgetIndex] = None) -> None:
        self.gadgets = gadgets if gadgets is not None else GadgetIndex()

    # ------------------------------------------------------------------
    # Chain layout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_arch(arch: Optional[str]) -> str:
        return GadgetIndex._canonical_arch(arch)

    @staticmethod
    def _pointer_bytes(arch: Optional[str]) -> int:
        bytes_ = GadgetIndex._pointer_bytes_for_arch(arch)
        if bytes_ is None:
            raise ChainBuilderError(
                f"cannot build a ROP chain without a known pointer width for arch={arch!r}"
            )
        return bytes_

    def _find_pop_reg(
        self,
        *,
        binary_path: str,
        register: str,
        arch: str,
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
    ) -> Optional[Dict[str, Any]]:
        """Locate a ``pop <register> ; ret`` gadget (shortest wins)."""
        # We require the *simplest* possible gadget so an accidental side
        # effect (extra pop / mov) cannot corrupt an unrelated register
        # the caller was about to load themselves.
        pattern = rf"^\s*pop\s+{re.escape(register)}\s*;\s*ret\s*$"
        response = self.gadgets.find_gadget_sequence(
            binary_path=binary_path,
            steps=[pattern],
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            prefer_shortest=True,
        )
        step = (response.get("parsed", {}).get("steps") or [None])[0]
        return step["gadget"] if step and step.get("matched") else None

    def _find_bare_ret(
        self,
        *,
        binary_path: str,
        arch: str,
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
    ) -> Optional[Dict[str, Any]]:
        response = self.gadgets.find_gadget_sequence(
            binary_path=binary_path,
            steps=[r"^\s*ret\s*$"],
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            prefer_shortest=True,
        )
        step = (response.get("parsed", {}).get("steps") or [None])[0]
        return step["gadget"] if step and step.get("matched") else None

    def _find_syscall_ret(
        self,
        *,
        binary_path: str,
        arch: str,
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
    ) -> Optional[Dict[str, Any]]:
        response = self.gadgets.find_gadget_sequence(
            binary_path=binary_path,
            steps=[r"^\s*syscall\s*;\s*ret\s*$"],
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            prefer_shortest=True,
        )
        step = (response.get("parsed", {}).get("steps") or [None])[0]
        return step["gadget"] if step and step.get("matched") else None

    @staticmethod
    def _encode_entries(
        entries: Sequence[Mapping[str, Any]], pointer_bytes: int
    ) -> bytes:
        buf = bytearray()
        for entry in entries:
            kind = entry.get("kind")
            if kind == "gadget":
                value = int(entry["address"])
            elif kind == "target":
                value = int(entry["address"])
            elif kind == "immediate":
                value = int(entry["value"])
            elif kind == "raw_bytes":
                raw = entry.get("bytes") or b""
                if isinstance(raw, str):
                    raw = bytes.fromhex(raw)
                buf.extend(raw)
                continue
            else:
                raise ChainBuilderError(f"unsupported chain entry kind: {kind!r}")
            try:
                buf.extend(int(value).to_bytes(pointer_bytes, "little", signed=False))
            except OverflowError as exc:
                raise ChainBuilderError(
                    f"chain entry value {value:#x} does not fit "
                    f"{pointer_bytes * 8}-bit pointer"
                ) from exc
        return bytes(buf)

    # ------------------------------------------------------------------
    # ret2libc
    # ------------------------------------------------------------------
    def _ret2libc_amd64(
        self,
        *,
        binary_path: str,
        target_address: int,
        args: Sequence[int],
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
        align_stack: bool,
        target_symbol: Optional[str],
    ) -> Dict[str, Any]:
        if len(args) > len(_SYSV_AMD64_ARG_REGS):
            raise ChainBuilderError(
                f"ret2libc supports up to {len(_SYSV_AMD64_ARG_REGS)} args, "
                f"got {len(args)}"
            )
        entries: List[Dict[str, Any]] = []
        assumptions: List[str] = [
            "x86_64 SysV: first args land in rdi, rsi, rdx, rcx, r8, r9",
            "The target function must reach a 16-byte-aligned rsp on entry",
        ]
        missing: List[str] = []
        arg_details: List[Dict[str, Any]] = []
        for index, value in enumerate(args):
            register = _SYSV_AMD64_ARG_REGS[index]
            gadget = self._find_pop_reg(
                binary_path=binary_path,
                register=register,
                arch="x86_64",
                image_base=image_base,
                bad_bytes=bad_bytes,
            )
            arg_details.append(
                {
                    "index": index,
                    "register": register,
                    "value": int(value),
                    "gadget": gadget,
                }
            )
            if gadget is None:
                missing.append(f"pop {register} ; ret")
                continue
            entries.append(
                {
                    "kind": "gadget",
                    "address": gadget["address"],
                    "instructions": gadget["instruction_text"],
                    "purpose": f"load {register}",
                }
            )
            entries.append(
                {
                    "kind": "immediate",
                    "value": int(value),
                    "purpose": f"{register} = {value:#x}",
                }
            )
        if align_stack:
            # SysV alignment model (documented so a caller can verify the
            # emitted chain, not just trust that a ``ret`` exists):
            #
            #   * The chain is entered via ``ret`` from the overflowed frame.
            #     When that pivot ``ret`` executes, rsp points at the first
            #     chain slot and is 8 (mod 16) -- the standard epilogue state
            #     (the ABI leaves the saved return-address slot at 8 mod 16).
            #   * Each argument contributes exactly two 8-byte pops: the
            #     ``pop <reg>`` consumes the immediate and the gadget's own
            #     ``ret`` consumes the next slot.  A bare ``ret`` contributes
            #     one pop.
            #   * The target is entered by the final pop, so rsp must be 8
            #     (mod 16) there -- i.e. the total pop count from the pivot
            #     slot through the target pop must be EVEN.
            #   * A pure pop-args chain pops ``2*len(args) + 1`` slots, which
            #     is always odd, so a bare ``ret`` is appended to reach an
            #     even count and land rsp on the correct phase.  The parity
            #     check is written explicitly rather than as an unconditional
            #     insert, so a future chain-shape change (a leading pivot or
            #     extra-grooming gadget that adds slots) re-derives the
            #     alignment instead of blindly appending a ``ret`` that would
            #     flip an already-correct phase.
            pop_count = 2 * len(args) + 1
            if pop_count % 2 == 1:
                ret_gadget = self._find_bare_ret(
                    binary_path=binary_path,
                    arch="x86_64",
                    image_base=image_base,
                    bad_bytes=bad_bytes,
                )
                if ret_gadget is None:
                    assumptions.append(
                        "no bare 'ret' gadget available for stack alignment; "
                        "the target may crash on movaps if rsp is not 16-byte aligned"
                    )
                else:
                    entries.append(
                        {
                            "kind": "gadget",
                            "address": ret_gadget["address"],
                            "instructions": ret_gadget["instruction_text"],
                            "purpose": "align rsp to 16 bytes before the target call",
                        }
                    )
            else:
                assumptions.append(
                    "chain pop count is already even; rsp is on the correct "
                    "16-byte phase at target entry without an alignment ret"
                )
        entries.append(
            {
                "kind": "target",
                "address": int(target_address),
                "symbol": target_symbol,
                "purpose": (
                    f"call {target_symbol}" if target_symbol else "call target"
                ),
            }
        )
        pointer_bytes = 8
        try:
            raw_bytes = self._encode_entries(entries, pointer_bytes)
        except ChainBuilderError:
            # Missing gadgets already recorded; skip encoding.
            raw_bytes = b""
        verdict = "built" if not missing else "missing_gadgets"
        return {
            "schema": ROP_CHAIN_SCHEMA,
            "kind": "ret2libc",
            "arch": "x86_64",
            "pointer_bytes": pointer_bytes,
            "target": {
                "address": int(target_address),
                "symbol": target_symbol,
            },
            "args": arg_details,
            "align_stack": align_stack,
            "entries": entries,
            "raw_bytes": raw_bytes.hex(),
            "raw_bytes_length": len(raw_bytes),
            "missing_gadgets": missing,
            "assumptions": assumptions,
            "verdict": verdict,
            "reason": (
                None
                if not missing
                else "one or more required pop <reg> gadgets were not found"
            ),
        }

    # ------------------------------------------------------------------
    # ret2syscall
    # ------------------------------------------------------------------
    def _ret2syscall_amd64(
        self,
        *,
        binary_path: str,
        syscall_number: int,
        args: Sequence[int],
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        if len(args) > len(_LINUX_AMD64_SYSCALL_ARG_REGS):
            raise ChainBuilderError(
                f"ret2syscall supports up to {len(_LINUX_AMD64_SYSCALL_ARG_REGS)} args, "
                f"got {len(args)}"
            )
        entries: List[Dict[str, Any]] = []
        assumptions: List[str] = [
            "Linux x86_64 syscall ABI: rax = number; args in rdi, rsi, rdx, r10, r8, r9",
            "SYSCALL clobbers rcx and r11 -- caller must not rely on them post-syscall",
        ]
        missing: List[str] = []
        arg_details: List[Dict[str, Any]] = []
        pop_rax = self._find_pop_reg(
            binary_path=binary_path,
            register="rax",
            arch="x86_64",
            image_base=image_base,
            bad_bytes=bad_bytes,
        )
        if pop_rax is None:
            missing.append("pop rax ; ret")
        else:
            entries.append(
                {
                    "kind": "gadget",
                    "address": pop_rax["address"],
                    "instructions": pop_rax["instruction_text"],
                    "purpose": "load rax (syscall number)",
                }
            )
            entries.append(
                {
                    "kind": "immediate",
                    "value": int(syscall_number),
                    "purpose": f"rax = {syscall_number}",
                }
            )
        for index, value in enumerate(args):
            register = _LINUX_AMD64_SYSCALL_ARG_REGS[index]
            gadget = self._find_pop_reg(
                binary_path=binary_path,
                register=register,
                arch="x86_64",
                image_base=image_base,
                bad_bytes=bad_bytes,
            )
            arg_details.append(
                {
                    "index": index,
                    "register": register,
                    "value": int(value),
                    "gadget": gadget,
                }
            )
            if gadget is None:
                missing.append(f"pop {register} ; ret")
                continue
            entries.append(
                {
                    "kind": "gadget",
                    "address": gadget["address"],
                    "instructions": gadget["instruction_text"],
                    "purpose": f"load {register}",
                }
            )
            entries.append(
                {
                    "kind": "immediate",
                    "value": int(value),
                    "purpose": f"{register} = {value:#x}",
                }
            )
        syscall_gadget = self._find_syscall_ret(
            binary_path=binary_path,
            arch="x86_64",
            image_base=image_base,
            bad_bytes=bad_bytes,
        )
        if syscall_gadget is None:
            missing.append("syscall ; ret")
        else:
            entries.append(
                {
                    "kind": "gadget",
                    "address": syscall_gadget["address"],
                    "instructions": syscall_gadget["instruction_text"],
                    "purpose": "trigger syscall",
                }
            )
        pointer_bytes = 8
        try:
            raw_bytes = self._encode_entries(entries, pointer_bytes)
        except ChainBuilderError:
            raw_bytes = b""
        verdict = "built" if not missing else "missing_gadgets"
        return {
            "schema": ROP_CHAIN_SCHEMA,
            "kind": "ret2syscall",
            "arch": "x86_64",
            "pointer_bytes": pointer_bytes,
            "target": {
                "syscall_number": int(syscall_number),
            },
            "args": arg_details,
            "entries": entries,
            "raw_bytes": raw_bytes.hex(),
            "raw_bytes_length": len(raw_bytes),
            "missing_gadgets": missing,
            "assumptions": assumptions,
            "verdict": verdict,
            "reason": (
                None if not missing else "one or more required gadgets were not found"
            ),
        }

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def build_rop_chain(
        self,
        *,
        binary_path: str,
        kind: str,
        arch: str = "x86_64",
        image_base: Optional[int] = None,
        bad_bytes: Optional[Sequence[Any]] = None,
        target_address: Optional[int] = None,
        target_symbol: Optional[str] = None,
        args: Optional[Sequence[int]] = None,
        align_stack: bool = True,
        syscall_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Assemble a typed ROP chain for the requested exploit shape."""
        canonical_arch = self._canonical_arch(arch)
        if canonical_arch != "x86_64":
            parsed = {
                "schema": ROP_CHAIN_SCHEMA,
                "kind": kind,
                "arch": arch,
                "supported": False,
                "verdict": "unsupported_arch",
                "reason": (
                    f"chain construction currently supports x86_64 only; "
                    f"got arch={arch!r}"
                ),
                "entries": [],
                "raw_bytes": "",
                "raw_bytes_length": 0,
                "assumptions": [],
            }
            return build_backend_response(
                "build_rop_chain",
                parsed=parsed,
                raw_responses=[],
                state="STATIC",
                parser_version=PARSER_VERSION,
                warnings=None,
            )

        args_seq = list(args or [])
        if kind == "ret2libc":
            if target_address is None:
                raise ChainBuilderError(
                    "ret2libc requires a resolved target_address "
                    "(compute libc_base + symbol_offset via identify_libc first)"
                )
            parsed = self._ret2libc_amd64(
                binary_path=binary_path,
                target_address=int(target_address),
                args=args_seq,
                image_base=image_base,
                bad_bytes=bad_bytes,
                align_stack=align_stack,
                target_symbol=target_symbol,
            )
        elif kind == "ret2syscall":
            if syscall_number is None:
                raise ChainBuilderError("ret2syscall requires a syscall_number")
            parsed = self._ret2syscall_amd64(
                binary_path=binary_path,
                syscall_number=int(syscall_number),
                args=args_seq,
                image_base=image_base,
                bad_bytes=bad_bytes,
            )
        else:
            raise ChainBuilderError(
                f"unsupported chain kind {kind!r}; expected 'ret2libc' or 'ret2syscall'"
            )
        return build_backend_response(
            "build_rop_chain",
            parsed=parsed,
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    # ------------------------------------------------------------------
    # SROP frame builder
    # ------------------------------------------------------------------
    _SIGRETURN_SUPPORTED_ARCHES: Tuple[str, ...] = ("x86_64", "i386")

    def inspect_sigreturn_frame(
        self,
        *,
        arch: str = "x86_64",
        registers: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """Build a typed sigreturn frame for the requested target arch.

        Delegates the byte-level layout to pwntools' SigreturnFrame so
        the offsets stay authoritative, but presents the result under
        the typed backend envelope every other stage uses: the raw
        bytes, the frame size, and every register slot's byte offset
        so a caller can hand-patch fields after the initial build.
        """
        canonical_arch = self._canonical_arch(arch)
        if canonical_arch not in self._SIGRETURN_SUPPORTED_ARCHES:
            return build_backend_response(
                "inspect_sigreturn_frame",
                parsed={
                    "schema": SIGRETURN_FRAME_SCHEMA,
                    "arch": arch,
                    "supported": False,
                    "verdict": "unsupported_arch",
                    "reason": (
                        f"sigreturn frame builder currently supports "
                        f"{self._SIGRETURN_SUPPORTED_ARCHES}; got arch={arch!r}"
                    ),
                    "frame_size": 0,
                    "frame_bytes": "",
                    "register_offsets": {},
                    "registers": {},
                    "syscall_register": None,
                    "syscall_number": None,
                },
                raw_responses=[],
                state="STATIC",
                parser_version=PARSER_VERSION,
                warnings=None,
            )
        from pwnlib.context import context as _pwn_context  # type: ignore[import-untyped]
        from pwnlib.rop.srop import SigreturnFrame  # type: ignore[import-untyped]  # local: pwntools is optional

        pwn_arch = "amd64" if canonical_arch == "x86_64" else "i386"
        # pwntools' SigreturnFrame is context-sensitive (arch + bits
        # come from the process-wide pwnlib.context) and blows up on
        # a stale bits value from a previous i386 call.  Scope the
        # context per call so this method is thread-safe and idempotent.
        pwn_bits = 64 if pwn_arch == "amd64" else 32
        # pwntools' ``context.kernel`` refers to the *kernel CPU arch*,
        # not the OS.  For a 32-bit i386 process running on a modern
        # 64-bit Linux kernel the sigcontext layout is the amd64 one;
        # amd64 processes ignore the field.  The context is scoped so
        # this method stays thread-safe.
        with _pwn_context.local(
            arch=pwn_arch,
            bits=pwn_bits,
            endian="little",
            kernel="amd64",
        ):
            frame = SigreturnFrame(arch=pwn_arch)
        assigned: Dict[str, int] = {}
        if registers:
            for name, value in registers.items():
                if name not in frame.registers.values():
                    raise ChainBuilderError(
                        f"register '{name}' is not part of a {canonical_arch} "
                        "sigreturn frame"
                    )
                # SigreturnFrame supports attribute assignment for every
                # register in ``registers.values()``.
                setattr(frame, name, int(value))
                assigned[name] = int(value)
        raw = bytes(frame)
        offsets: Dict[str, int] = {
            name: offset for offset, name in frame.registers.items()
        }
        # rt_sigreturn syscall number differs per arch: 15 on x86_64,
        # 173 on i386.  Surfaced so a caller building the trigger chain
        # does not have to look it up.
        syscall_number = 15 if canonical_arch == "x86_64" else 173
        return build_backend_response(
            "inspect_sigreturn_frame",
            parsed={
                "schema": SIGRETURN_FRAME_SCHEMA,
                "arch": canonical_arch,
                "supported": True,
                "verdict": "built",
                "frame_size": len(raw),
                "frame_bytes": raw.hex(),
                "register_offsets": offsets,
                "registers": assigned,
                "syscall_register": "rax" if canonical_arch == "x86_64" else "eax",
                "syscall_number": syscall_number,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    # ------------------------------------------------------------------
    # Chain constraints explanation
    # ------------------------------------------------------------------
    @staticmethod
    def explain_chain_constraints(
        *,
        chain: Sequence[Mapping[str, Any]],
        arch: str = "x86_64",
    ) -> Dict[str, Any]:
        """Summarize per-slot side effects and cross-slot register aliasing.

        Walks ``chain`` (as returned by :meth:`build_rop_chain`'s
        ``entries`` list) and reports which registers each gadget writes
        or clobbers, which slots hold immediates the chain treats as
        register values, and any register a later slot depends on that
        an earlier slot clobbers.  The output is intentionally advisory
        -- the goal is to help a caller notice a chain layout that is
        obviously wrong before they hand it to the target process.
        """
        canonical_arch = ChainBuilder._canonical_arch(arch)
        slots: List[Dict[str, Any]] = []
        register_state: Dict[str, str] = {}
        risky: List[Dict[str, Any]] = []
        for index, entry in enumerate(chain):
            kind = entry.get("kind")
            slot: Dict[str, Any] = {
                "index": index,
                "kind": kind,
                "purpose": entry.get("purpose"),
            }
            if kind == "gadget":
                # analyze the gadget text via GadgetIndex helpers
                instructions = str(entry.get("instructions") or "")
                lines = [
                    line.strip() for line in instructions.split(";") if line.strip()
                ]
                reads: List[str] = []
                writes: List[str] = []
                clobbers: List[str] = []
                for text in lines:
                    parsed = GadgetIndex._parse_instruction(text)
                    if parsed is None:
                        continue
                    mnemonic, operands = parsed
                    effects = GadgetIndex._instruction_side_effects(
                        mnemonic, operands, arch
                    )
                    reads.extend(effects["reads"])
                    writes.extend(effects["writes"])
                    clobbers.extend(effects["clobbers"])
                slot["reads"] = list(dict.fromkeys(reads))
                slot["writes"] = list(dict.fromkeys(writes))
                slot["clobbers"] = list(dict.fromkeys(clobbers))
                # Track: any previously-loaded arg register that this
                # gadget writes will get overwritten -- that is only a
                # problem if the *next* target read expects it to still
                # hold the earlier value.  We surface the write for the
                # caller to reason about; we do not decide correctness.
                for reg in slot["writes"] + slot["clobbers"]:
                    if reg in register_state and register_state[reg] != slot["purpose"]:
                        risky.append(
                            {
                                "slot_index": index,
                                "register": reg,
                                "previously_loaded_by": register_state[reg],
                                "now_overwritten_by": slot["purpose"]
                                or f"slot {index}",
                            }
                        )
                for reg in slot["writes"]:
                    register_state[reg] = slot["purpose"] or f"slot {index}"
            elif kind == "immediate":
                slot["value"] = entry.get("value")
            elif kind == "target":
                slot["address"] = entry.get("address")
                slot["symbol"] = entry.get("symbol")
            slots.append(slot)
        return build_backend_response(
            "explain_chain_constraints",
            parsed={
                "schema": CHAIN_CONSTRAINTS_SCHEMA,
                "arch": canonical_arch,
                "slot_count": len(slots),
                "slots": slots,
                "register_state_after_chain": register_state,
                "risky_overwrites": risky,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


class ChainValidatorError(RuntimeError):
    """Raised when a chain validation input is malformed."""


class ChainValidator:
    """Static validation and export for typed ROP chains.

    Validation is deliberately static: we check every gadget address
    lands in an executable mapping, verify no encoded pointer contains
    a forbidden byte, and surface a register-clobber summary drawn from
    the chain-constraint :meth:`ChainBuilder.explain_chain_constraints` analyzer.
    Dynamic sandbox validation (execute the chain against a real
    debugger inferior) requires a live target and is documented as a
    future chain-validation extension rather than faked here.

    Export supports three formats:

    * ``raw``     -- hex-encoded little-endian pointer blob
    * ``python``  -- ``struct.pack``-flavored snippet suitable for
                     a Python 3 script (no pwntools dependency)
    * ``pwntools`` -- a ``p64(...)`` payload string a caller can
                      paste into a pwntools exploit template
    """

    def __init__(self, builder: Optional[ChainBuilder] = None) -> None:
        self._builder = builder if builder is not None else ChainBuilder()

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mapping_contains(mapping: Mapping[str, Any], address: int) -> bool:
        start = parse_int(mapping.get("start"))
        end = parse_int(mapping.get("end"))
        if start is None or end is None:
            return False
        return start <= address < end

    @staticmethod
    def _mapping_is_executable(mapping: Mapping[str, Any]) -> bool:
        # Accept either a boolean field or a POSIX-style "r-xp" string.
        if mapping.get("executable") is True:
            return True
        perms = str(mapping.get("permissions") or mapping.get("perms") or "")
        return "x" in perms.lower()

    @classmethod
    def _classify_address(
        cls,
        address: int,
        executable_mappings: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        matches: List[Mapping[str, Any]] = [
            m for m in executable_mappings if cls._mapping_contains(m, address)
        ]
        if not matches:
            return {"in_mapping": False, "executable": False, "mapping": None}
        # If multiple overlap, prefer the executable one.
        executable_hits = [m for m in matches if cls._mapping_is_executable(m)]
        chosen = executable_hits[0] if executable_hits else matches[0]
        return {
            "in_mapping": True,
            "executable": cls._mapping_is_executable(chosen),
            "mapping": {
                "start": parse_int(chosen.get("start")),
                "end": parse_int(chosen.get("end")),
                "file": chosen.get("file"),
                "permissions": chosen.get("permissions") or chosen.get("perms"),
            },
        }

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def validate_rop_chain(
        self,
        *,
        chain: Sequence[Mapping[str, Any]],
        executable_mappings: Optional[Sequence[Mapping[str, Any]]] = None,
        bad_bytes: Optional[Sequence[Any]] = None,
        arch: str = "x86_64",
    ) -> Dict[str, Any]:
        """Return a static pass / fail / unknown verdict for ``chain``.

        Checks (each surfaces per-slot evidence):

        * ``exec_mapping``  -- every gadget/target address lies in an
          executable mapping (skipped with ``unknown`` when the caller
          omits ``executable_mappings``)
        * ``bad_bytes``     -- every gadget/target/immediate address,
          when packed little-endian for the arch's pointer width, is
          free of every forbidden byte
        * ``register_clobber`` -- borrowed from
          :meth:`ChainBuilder.explain_chain_constraints`; surfaces
          risky overwrites but does not fail the chain by itself
          because a plausible chain may deliberately reload a register.
        """
        canonical_arch = self._builder._canonical_arch(arch)
        try:
            pointer_bytes = self._builder._pointer_bytes(canonical_arch)
        except ChainBuilderError as exc:
            raise ChainValidatorError(str(exc)) from exc
        normalized_bad_bytes = GadgetIndex._normalize_bad_bytes(bad_bytes)
        executable_mappings_list = list(executable_mappings or [])
        exec_check: Dict[str, Any] = {
            "name": "exec_mapping",
            "verdict": "unknown" if not executable_mappings_list else "pass",
            "issues": [],
            "checked_slots": [],
        }
        bad_byte_check: Dict[str, Any] = {
            "name": "bad_bytes",
            "verdict": "pass",
            "issues": [],
            "bad_bytes": normalized_bad_bytes,
        }
        for index, entry in enumerate(chain):
            kind = entry.get("kind")
            address: Optional[int] = None
            if kind in ("gadget", "target"):
                address = parse_int(entry.get("address"))
            elif kind == "immediate":
                address = parse_int(entry.get("value"))
            if address is None:
                continue
            if kind in ("gadget", "target") and executable_mappings_list:
                classification = self._classify_address(
                    address, executable_mappings_list
                )
                classification["slot_index"] = index
                classification["address"] = address
                exec_check["checked_slots"].append(classification)
                if not classification["in_mapping"]:
                    exec_check["issues"].append(
                        {
                            "slot_index": index,
                            "address": address,
                            "reason": "address is outside every provided mapping",
                        }
                    )
                elif not classification["executable"]:
                    exec_check["issues"].append(
                        {
                            "slot_index": index,
                            "address": address,
                            "reason": "address is in a non-executable mapping",
                        }
                    )
            if normalized_bad_bytes:
                if GadgetIndex._address_contains_bad_byte(
                    address, pointer_bytes, normalized_bad_bytes
                ):
                    bad_byte_check["issues"].append(
                        {
                            "slot_index": index,
                            "address": address,
                            "kind": kind,
                            "reason": "encoded pointer contains a forbidden byte",
                        }
                    )
        if exec_check["issues"]:
            exec_check["verdict"] = "fail"
        if bad_byte_check["issues"]:
            bad_byte_check["verdict"] = "fail"

        clobber_response = self._builder.explain_chain_constraints(
            chain=chain, arch=canonical_arch
        )
        clobber_parsed = clobber_response["parsed"]
        clobber_check = {
            "name": "register_clobber",
            "verdict": ("warn" if clobber_parsed.get("risky_overwrites") else "pass"),
            "risky_overwrites": clobber_parsed.get("risky_overwrites") or [],
        }

        checks = [exec_check, bad_byte_check, clobber_check]
        verdicts = {c["verdict"] for c in checks}
        if "fail" in verdicts:
            overall = "fail"
        elif "unknown" in verdicts:
            overall = "unknown"
        else:
            overall = "pass"
        return build_backend_response(
            "validate_rop_chain",
            parsed={
                "schema": CHAIN_VALIDATION_SCHEMA,
                "arch": canonical_arch,
                "pointer_bytes": pointer_bytes,
                "slot_count": len(chain),
                "verdict": overall,
                "checks": checks,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def export_chain(
        self,
        *,
        chain: Sequence[Mapping[str, Any]],
        format: str = "raw",
        arch: str = "x86_64",
        variable_name: str = "chain",
    ) -> Dict[str, Any]:
        """Serialize a chain into a raw / python / pwntools payload."""
        if format not in CHAIN_EXPORT_FORMATS:
            raise ChainValidatorError(
                f"unsupported export format {format!r}; expected one of "
                f"{CHAIN_EXPORT_FORMATS}"
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable_name):
            raise ChainValidatorError(
                f"variable_name must be a plain identifier, got {variable_name!r}"
            )
        canonical_arch = self._builder._canonical_arch(arch)
        try:
            pointer_bytes = self._builder._pointer_bytes(canonical_arch)
        except ChainBuilderError as exc:
            raise ChainValidatorError(str(exc)) from exc
        try:
            raw_bytes = self._builder._encode_entries(chain, pointer_bytes)
        except ChainBuilderError as exc:
            raise ChainValidatorError(str(exc)) from exc
        text: str
        if format == "raw":
            text = raw_bytes.hex()
        elif format == "python":
            struct_fmt = "<Q" if pointer_bytes == 8 else "<I"
            lines = ["import struct", "", f'{variable_name} = b""']
            for entry in chain:
                kind = entry.get("kind")
                if kind == "raw_bytes":
                    payload = entry.get("bytes") or b""
                    if isinstance(payload, str):
                        payload = bytes.fromhex(payload)
                    lines.append(f"{variable_name} += bytes.fromhex({payload.hex()!r})")
                    continue
                raw_value = (
                    entry.get("address")
                    if kind in ("gadget", "target")
                    else entry.get("value")
                )
                value = parse_int(raw_value)
                if value is None:
                    raise ChainValidatorError(
                        f"chain entry #{len(lines)} is missing a numeric "
                        f"address/value for kind={kind!r}"
                    )
                purpose = entry.get("purpose") or kind
                lines.append(
                    f"{variable_name} += struct.pack({struct_fmt!r}, {value:#x})"
                    f"  # {purpose}"
                )
            text = "\n".join(lines)
        else:  # pwntools
            pack_fn = "p64" if pointer_bytes == 8 else "p32"
            lines = [
                "from pwn import *",
                "",
                f"context.arch = {'amd64' if canonical_arch == 'x86_64' else canonical_arch!r}",
                "",
                f'{variable_name} = b""',
            ]
            for entry in chain:
                kind = entry.get("kind")
                if kind == "raw_bytes":
                    payload = entry.get("bytes") or b""
                    if isinstance(payload, str):
                        payload = bytes.fromhex(payload)
                    lines.append(f"{variable_name} += bytes.fromhex({payload.hex()!r})")
                    continue
                raw_value = (
                    entry.get("address")
                    if kind in ("gadget", "target")
                    else entry.get("value")
                )
                value = parse_int(raw_value)
                if value is None:
                    raise ChainValidatorError(
                        f"chain entry #{len(lines)} is missing a numeric "
                        f"address/value for kind={kind!r}"
                    )
                purpose = entry.get("purpose") or kind
                lines.append(f"{variable_name} += {pack_fn}({value:#x})  # {purpose}")
            text = "\n".join(lines)
        return build_backend_response(
            "export_chain",
            parsed={
                "schema": CHAIN_EXPORT_SCHEMA,
                "arch": canonical_arch,
                "pointer_bytes": pointer_bytes,
                "format": format,
                "slot_count": len(chain),
                "raw_bytes_length": len(raw_bytes),
                "raw_bytes": raw_bytes.hex(),
                "text": text,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


class RopTools:
    """Facade grouping every ROP typed backend under one session attribute."""

    def __init__(
        self,
        gadgets: Optional[GadgetIndex] = None,
        libc: Optional[LibcIdentifier] = None,
        chains: Optional[ChainBuilder] = None,
        validator: Optional[ChainValidator] = None,
    ) -> None:
        self.gadgets = gadgets if gadgets is not None else GadgetIndex()
        self.libc = libc if libc is not None else LibcIdentifier()
        self.chains = chains if chains is not None else ChainBuilder(self.gadgets)
        self.validator = (
            validator if validator is not None else ChainValidator(self.chains)
        )

    # ---- gadget indexing ----------------------------------------
    def find_gadgets(self, **kwargs: Any) -> Dict[str, Any]:
        return self.gadgets.find_gadgets(**kwargs)

    def find_gadget_sequence(self, **kwargs: Any) -> Dict[str, Any]:
        return self.gadgets.find_gadget_sequence(**kwargs)

    def inspect_gadget(self, **kwargs: Any) -> Dict[str, Any]:
        return self.gadgets.inspect_gadget(**kwargs)

    # ---- libc identification ------------------------------------
    def identify_libc(self, **kwargs: Any) -> Dict[str, Any]:
        return self.libc.identify_libc(**kwargs)

    def find_one_gadgets(self, **kwargs: Any) -> Dict[str, Any]:
        return self.libc.find_one_gadgets(**kwargs)

    # ---- chain construction -------------------------------------
    def build_rop_chain(self, **kwargs: Any) -> Dict[str, Any]:
        return self.chains.build_rop_chain(**kwargs)

    def inspect_sigreturn_frame(self, **kwargs: Any) -> Dict[str, Any]:
        return self.chains.inspect_sigreturn_frame(**kwargs)

    def explain_chain_constraints(self, **kwargs: Any) -> Dict[str, Any]:
        return self.chains.explain_chain_constraints(**kwargs)

    # ---- chain validation + export ------------------------------
    def validate_rop_chain(self, **kwargs: Any) -> Dict[str, Any]:
        return self.validator.validate_rop_chain(**kwargs)

    def export_chain(self, **kwargs: Any) -> Dict[str, Any]:
        return self.validator.export_chain(**kwargs)


__all__ = [
    "ChainBuilder",
    "ChainBuilderError",
    "ChainValidator",
    "ChainValidatorError",
    "RopTools",
]
