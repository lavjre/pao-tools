"""Attack-helper framework -- execute exploitation primitives against a live target.

The heap snapshot surface and later *detects* vulnerabilities (House-of-* flags,
overlap/UAF scans, corruption verdicts).  Attack helpers go one step further:
given a confirmed primitive and a target address, they compute the exact
bytes that materialize it and -- only on explicit opt-in -- write them into
the target, with the discipline:

1. live-guard + family classification (a non-glibc target gets a typed
   ``unsupported`` verdict, never a partial write),
2. locate the primitive's anchor in the live heap (tcache bin / chain),
3. compute the encoded payload (PROTECT_PTR safe-linking handled here),
4. dry-run by default -- report the full plan, write nothing,
5. on ``dry_run=false``: snapshot-before (named heap snapshot for the
   structural ledger + ``original_hex`` captured by the guarded writer for
   byte-exact revert), write through ``DebuggerOperations.write_memory``
   (mapping-validated, verify-after built in), then verify semantics,
6. natural idempotency: if the field already holds the exact encoded value,
   re-apply is a no-op.

The first primitive is ``poison_tcache_link`` (glibc tcache-poisoning: the
fd of a freed tcache chunk is re-pointed so a subsequent ``malloc`` returns
an arbitrary writable address).  Later primitives (fastbin-dup re-point,
top-chunk overwrite, ...) join as sibling methods on this component -- the
harness (guarded writer + named snapshot + encode helper) is shared.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.heap import (
    PARSER_VERSION,
    AllocatorDiscovery,
    DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
    encode_safe_linked,
)
from paomcp.tools.backends.inspection import PwndbgInspection

ATTACK_TCACHE_POISON_SCHEMA = "pao.attack.tcache-poison.v1"


class GlibcAttackHelper:
    """Execute glibc heap primitives against a live target (write-capable).

    The first write-capable glibc component: unlike the chunk-to-technique parsers it
    can mutate the target, but only through the guarded writer
    (``DebuggerOperations.write_memory``), only on explicit ``dry_run=false``,
    and only after a named snapshot is captured.
    """

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        discovery: AllocatorDiscovery,
        glibc: Any,
        bins: Any,
        history: Any,
        patch_recorder: Optional[Any] = None,
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self.glibc = glibc
        self.bins = bins
        self.history = history
        self.patch_recorder = patch_recorder

    # ------------------------------------------------------------------
    # Live-target helpers (mirror GlibcTechniqueAnalyzer)
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Attack helpers require a live, stopped inferior; run the "
                "target to a stop before applying a primitive"
            ),
        )

    def _prepare_context(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        self._require_live()
        mappings, libcinfo_parsed, component_status = self.discovery.collect_evidence(
            cancellation_check
        )
        classification = AllocatorDiscovery.classify(mappings, libcinfo_parsed)
        profile = self.discovery.target_profile(mappings)
        return classification, mappings, profile, component_status

    @staticmethod
    def _allocator_summary(classification: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "family": classification.get("family"),
            "variant": classification.get("variant"),
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
            "safe_linking": classification.get("safe_linking"),
        }

    def _unsupported_response(
        self,
        request: str,
        schema: str,
        classification: Mapping[str, Any],
        profile: Mapping[str, Any],
        component_status: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return build_backend_response(
            request,
            parsed={
                "schema": schema,
                "supported": False,
                "verdict": "unsupported",
                "reason": (
                    "attack helpers support glibc only; detected "
                    f"family={classification.get('family')}"
                ),
                "allocator": self._allocator_summary(classification),
                "target_profile": profile,
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    # ------------------------------------------------------------------
    # tcache-poisoning
    # ------------------------------------------------------------------
    def poison_tcache_link(
        self,
        *,
        victim_address: str,
        target_address: str,
        address_kind: str = "chunk",
        dry_run: bool = True,
        snapshot_before: bool = True,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Re-point a freed tcache chunk's ``fd`` so a later malloc returns
        ``target_address``.

        ``victim_address`` is the freed chunk sitting in a tcache bin (find it
        via ``inspect_tcache`` -- ``bins[i].chain[j].chunk_address``).
        ``target_address`` is the *user* pointer the caller wants the following
        ``malloc`` to return.  The chunk header stays untouched; the write
        targets the ``fd`` slot at the chunk's user address.

        ``dry_run`` defaults to ``True``: the full plan (encoded bytes, write
        address, malloc follow-up) is computed and returned but nothing is
        written.  Set ``dry_run=false`` to apply -- which snapshots first
        (named heap snapshot + ``original_hex`` for revert), writes through the
        mapping-validated guarded writer, and verifies by re-read.
        """
        request = "poison_tcache_link"
        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        allocator = self._allocator_summary(classification)

        if classification.get("family") != "glibc":
            return self._unsupported_response(
                request,
                ATTACK_TCACHE_POISON_SCHEMA,
                classification,
                profile,
                component_status,
            )

        pointer_bits = int(profile.get("pointer_bits") or 64)
        byteorder = cast(Literal["little", "big"], profile.get("byteorder") or "little")
        size_sz = pointer_bits // 8

        victim = parse_int(victim_address)
        target = parse_int(target_address)
        if victim is None:
            raise ValueError(f"invalid victim_address: {victim_address!r}")
        if target is None:
            raise ValueError(f"invalid target_address: {target_address!r}")
        victim = self.glibc._chunk_from_user(victim, address_kind, size_sz)
        target_chunk = target - 2 * size_sz

        safe_linking = classification.get("safe_linking")
        safe_linking_enabled = safe_linking == "enabled"
        if safe_linking not in ("enabled", "disabled"):
            # glibc >= 2.32 encodes tcache fds with PROTECT_PTR; older glibc
            # stores them raw.  With the version unresolved we cannot know
            # which encoding to write, so writing either form is a guess that
            # the next malloc will decode into the wrong pointer.  Refuse
            # instead of silently treating "unknown" as "disabled".
            return build_backend_response(
                request,
                parsed={
                    "schema": ATTACK_TCACHE_POISON_SCHEMA,
                    "allocator": allocator,
                    "target_profile": profile,
                    "supported": False,
                    "verdict": "unresolvable",
                    "reason": (
                        f"safe-linking state is {safe_linking!r}; refusing to "
                        "write an fd without knowing whether PROTECT_PTR "
                        "encoding applies (glibc >= 2.32 encodes tcache fds, "
                        "older glibc does not)"
                    ),
                    "component_status": component_status,
                    "target_address": target,
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
            )

        # -- locate the victim in a tcache bin -------------------------
        tcache = self.bins.inspect_tcache(
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )
        tc_parsed = tcache.get("parsed") or {}
        tc_verdict = tc_parsed.get("verdict")
        bins = tc_parsed.get("bins") or []
        victim_bin: Optional[Dict[str, Any]] = None
        victim_entry: Optional[Dict[str, Any]] = None
        for bin_entry in bins:
            for entry in bin_entry.get("chain") or []:
                if entry.get("chunk_address") == victim:
                    victim_bin = bin_entry
                    victim_entry = entry
                    break
            if victim_entry is not None:
                break

        if tc_verdict != "parsed":
            return build_backend_response(
                request,
                parsed={
                    "schema": ATTACK_TCACHE_POISON_SCHEMA,
                    "allocator": allocator,
                    "target_profile": profile,
                    "supported": False,
                    "verdict": "unresolvable",
                    "reason": f"tcache inspection returned verdict={tc_verdict!r}",
                    "component_status": component_status,
                    "target_address": target,
                },
                raw_responses=[tcache],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
            )

        if victim_entry is None:
            return build_backend_response(
                request,
                parsed={
                    "schema": ATTACK_TCACHE_POISON_SCHEMA,
                    "allocator": allocator,
                    "target_profile": profile,
                    "supported": False,
                    "verdict": "victim_not_in_tcache",
                    "reason": (
                        f"chunk {hex(victim)} is not linked into any tcache "
                        "bin; free it once more or use a chunk from "
                        "inspect_tcache"
                    ),
                    "scanned_bins": len(bins),
                    "component_status": component_status,
                    "victim_chunk_address": victim,
                },
                raw_responses=[tcache],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
            )

        assert victim_bin is not None  # set iff victim_entry was found
        victim_user = int(victim_entry["user_address"])
        bin_index = int(victim_bin["index"])
        chain_position = int(victim_entry["index"])
        size_class = victim_bin.get("size_class")
        fd_raw = victim_entry.get("next_raw")
        if fd_raw is None:
            return build_backend_response(
                request,
                parsed={
                    "schema": ATTACK_TCACHE_POISON_SCHEMA,
                    "allocator": allocator,
                    "target_profile": profile,
                    "supported": False,
                    "verdict": "unresolvable",
                    "reason": "victim's fd slot could not be read",
                    "component_status": component_status,
                    "victim_chunk_address": victim,
                },
                raw_responses=[tcache],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
            )
        fd_raw = int(fd_raw)

        original_fd_decoded = (
            fd_raw ^ (victim_user >> 12) if safe_linking_enabled else fd_raw
        )
        encoded_fd = (
            encode_safe_linked(target_chunk, victim_user)
            if safe_linking_enabled
            else target_chunk
        )
        encoded_hex = encoded_fd.to_bytes(size_sz, byteorder).hex()

        # -- idempotency: already poisoned to this exact target ---------
        if fd_raw == encoded_fd:
            return build_backend_response(
                request,
                parsed={
                    "schema": ATTACK_TCACHE_POISON_SCHEMA,
                    "allocator": allocator,
                    "target_profile": profile,
                    "component_status": component_status,
                    "supported": True,
                    "verdict": "already_applied",
                    "reason": (
                        f"victim fd already encodes {hex(target)}; re-apply "
                        "is a no-op"
                    ),
                    "victim_chunk_address": victim,
                    "victim_user_address": victim_user,
                    "victim_bin_index": bin_index,
                    "victim_chain_position": chain_position,
                    "victim_size_class": size_class,
                    "target_address": target,
                    "target_chunk_address": target_chunk,
                    "safe_linking_enabled": safe_linking_enabled,
                    "encoded_fd_hex": encoded_hex,
                    "original_fd_raw_hex": hex(fd_raw),
                    "original_fd_decoded": original_fd_decoded,
                    "snapshot_id": None,
                    "verify": None,
                    "follow_up": self._follow_up(chain_position, size_class, target),
                    "revert": None,
                },
                raw_responses=[tcache],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
            )

        # -- advisory target check + warnings ---------------------------
        warnings: List[str] = []
        target_mapping = mapping_for(target, mappings)
        if target_mapping is None:
            warnings.append(
                f"target {hex(target)} is not inside any mapping; the "
                "returned malloc would fault on dereference"
            )
        elif not target_mapping.get("writable"):
            warnings.append(
                f"target mapping {target_mapping.get('file')} "
                f"[{target_mapping.get('permissions')}] is not writable"
            )
        alignment = 2 * size_sz
        if target_chunk % alignment != 0:
            warnings.append(
                f"target chunk {hex(target_chunk)} is not {alignment}-byte "
                "aligned; malloc may misbehave"
            )

        # -- plan ------------------------------------------------------
        # The fd slot lives at chunk + 2*SIZE_SZ (offset 0 of the user area).
        # The plan anchors on the chunk header so ``address + chunk_offset``
        # resolves exactly to the fd slot; ``write_target_address`` is that
        # resolved location, matching where the write actually lands.
        fd_slot_offset = 2 * size_sz
        write_plan = {
            "address": hex(victim),
            "chunk_offset": fd_slot_offset,
            "byte_count": size_sz,
            "data_hex": encoded_hex,
            "write_target_address": hex(victim_user),
        }
        base_parsed: Dict[str, Any] = {
            "schema": ATTACK_TCACHE_POISON_SCHEMA,
            "allocator": allocator,
            "target_profile": profile,
            "component_status": component_status,
            "supported": True,
            "victim_chunk_address": victim,
            "victim_user_address": victim_user,
            "victim_bin_index": bin_index,
            "victim_chain_position": chain_position,
            "victim_size_class": size_class,
            "target_address": target,
            "target_chunk_address": target_chunk,
            "safe_linking_enabled": safe_linking_enabled,
            "encoded_fd_hex": encoded_hex,
            "original_fd_raw_hex": hex(fd_raw),
            "original_fd_decoded": original_fd_decoded,
            "target_mapping": target_mapping,
            "follow_up": self._follow_up(chain_position, size_class, target),
        }

        if dry_run:
            return build_backend_response(
                request,
                parsed={
                    **base_parsed,
                    "verdict": "dry_run",
                    "snapshot_id": None,
                    "verify": None,
                    "revert": None,
                    "write_plan": write_plan,
                    "note": (
                        "dry run -- nothing written; pass dry_run=false to "
                        "apply (snapshots first, writes through the guarded "
                        "writer, verifies after)"
                    ),
                },
                raw_responses=[tcache],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        # -- apply -----------------------------------------------------
        raw_responses: List[Dict[str, Any]] = [tcache]
        snapshot_id: Optional[str] = None
        if snapshot_before:
            snapshot = self.history.capture_heap_snapshot(
                name=f"pre-poison-{hex(victim)}",
                max_chunks_per_bin=max_chunks_per_bin,
                cancellation_check=cancellation_check,
            )
            raw_responses.append(snapshot)
            snap_parsed = snapshot.get("parsed") or {}
            snapshot_id = snap_parsed.get("snapshot_id")
            if snapshot_id is None:
                warnings.append(
                    "named pre-poison snapshot could not be captured "
                    f"(verdict={snap_parsed.get('verdict')!r})"
                )

        # Route through the patch-recording wrapper (``PwndbgTools.write_``
        # ``memory``) when one is wired, so the poison lands in the same
        # session patch ledger that ``list_memory_patches`` / ``revert_memory_``
        # ``patch`` operate on.  Falls back to the raw guarded writer only in
        # direct-construction contexts (tests) where no ledger exists.
        writer = (
            self.patch_recorder if self.patch_recorder is not None else self.debugger
        )
        write_result = writer.write_memory(
            address=hex(victim_user),
            data_hex=encoded_hex,
            verify=True,
            allow_unsafe=False,
        )
        raw_responses.append(write_result)

        if not write_result.get("success"):
            return build_backend_response(
                request,
                parsed={
                    **base_parsed,
                    "verdict": "write_failed",
                    "reason": write_result.get("error")
                    or (write_result.get("parsed") or {}).get("error"),
                    "snapshot_id": snapshot_id,
                    "verify": None,
                    "revert": None,
                },
                raw_responses=raw_responses,
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        write_parsed = write_result.get("parsed") or {}
        actual_hex = write_parsed.get("actual_hex")
        if actual_hex is not None:
            # actual_hex is the LE byte string as written; recover the stored
            # integer before applying the PROTECT_PTR mask.
            raw_after = int.from_bytes(bytes.fromhex(actual_hex), byteorder)
            decoded_after = (
                raw_after ^ (victim_user >> 12) if safe_linking_enabled else raw_after
            )
        else:
            decoded_after = None
        semantic_match = decoded_after == target_chunk
        verify = {
            "writer_verified": bool(write_parsed.get("verified")),
            "actual_hex": actual_hex,
            "decoded_after": decoded_after,
            "semantic_match": semantic_match,
            "note": (
                "decoded_after is the PROTECT_PTR-decoded fd re-read after "
                "the write; semantic_match confirms it decodes to the "
                "target chunk"
            ),
        }
        revert = {
            "original_hex": write_parsed.get("original_hex"),
            "address": hex(victim_user),
            "patch_id": write_parsed.get("patch_id"),
            "how": (
                "restore by writing original_hex back via the write_memory "
                "tool at this address"
            ),
            "revertible": bool(write_parsed.get("revertible")),
        }

        return build_backend_response(
            request,
            parsed={
                **base_parsed,
                "verdict": "applied",
                "snapshot_id": snapshot_id,
                "verify": verify,
                "revert": revert,
                "write_plan": write_plan,
            },
            raw_responses=raw_responses,
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    @staticmethod
    def _follow_up(
        chain_position: int, size_class: Optional[int], target: int
    ) -> Dict[str, Any]:
        """How many same-size mallocs before ``target`` is returned."""
        return {
            "size_class": size_class,
            "victim_returned_on_malloc": chain_position + 1,
            "target_returned_on_malloc": chain_position + 2,
            "note": (
                f"malloc #{chain_position + 1} pops the victim; malloc "
                f"#{chain_position + 2} of the same size returns {hex(target)}"
            ),
        }


__all__ = [
    "GlibcAttackHelper",
    "ATTACK_TCACHE_POISON_SCHEMA",
]
