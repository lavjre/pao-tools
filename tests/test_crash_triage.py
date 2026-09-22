import copy
import json
import os
import threading
import time

import pytest

import paomcp.tools.backends.crash as crash_backend
import paomcp.tools.backends.crash_core as crash_core_module
from paomcp.artifacts import ArtifactMetadata, WriterNotQuiescentError
from paomcp.tools.backends.crash import (
    MAX_CONFIGURED_CORE_DUMP_BYTES,
    MAX_CYCLIC_ALPHABET_CHARS,
    MAX_CYCLIC_VALUE_CHARS,
    CrashRecordError,
    CrashTriage,
)
from paomcp.tools.crash import (
    _core_reference_for_crash,
    _snapshot_report_provenance,
)


def minimal_elf64_core(segment_bytes: int = 64) -> bytes:
    """Build a minimal but structurally valid little-endian ET_CORE ELF64.

    Carries one program header so the stricter validator (which now rejects
    header-only stubs) accepts it.  ``segment_bytes`` pads the file past the
    program-header table.
    """
    ehsize = 64
    phentsize = 56
    phnum = 2
    phoff = ehsize
    note_offset = phoff + phentsize * phnum
    note_name = b"CORE\x00"
    note_description = b"\x00" * 8
    note_bytes = (
        len(note_name).to_bytes(4, "little")
        + len(note_description).to_bytes(4, "little")
        + (1).to_bytes(4, "little")  # NT_PRSTATUS
        + note_name
        + b"\x00" * 3
        + note_description
    )
    load_offset = note_offset + len(note_bytes)
    header = bytearray(ehsize)
    header[:4] = b"\x7fELF"
    header[4] = 2  # ELFCLASS64
    header[5] = 1  # little-endian
    header[6] = 1  # EV_CURRENT
    header[16:18] = (4).to_bytes(2, "little")  # ET_CORE
    header[18:20] = (62).to_bytes(2, "little")  # EM_X86_64
    header[20:24] = (1).to_bytes(4, "little")  # EV_CURRENT
    header[32:40] = (phoff).to_bytes(8, "little")  # e_phoff
    header[52:54] = (ehsize).to_bytes(2, "little")  # e_ehsize
    header[54:56] = (phentsize).to_bytes(2, "little")  # e_phentsize
    header[56:58] = (phnum).to_bytes(2, "little")  # e_phnum
    note_ph = bytearray(phentsize)
    note_ph[0:4] = (4).to_bytes(4, "little")  # PT_NOTE
    note_ph[8:16] = note_offset.to_bytes(8, "little")
    note_ph[32:40] = len(note_bytes).to_bytes(8, "little")
    note_ph[40:48] = len(note_bytes).to_bytes(8, "little")
    load_ph = bytearray(phentsize)
    load_ph[0:4] = (1).to_bytes(4, "little")  # PT_LOAD
    load_ph[4:8] = (4).to_bytes(4, "little")  # PF_R
    load_ph[8:16] = load_offset.to_bytes(8, "little")
    load_ph[16:24] = (0x400000).to_bytes(8, "little")
    load_ph[32:40] = segment_bytes.to_bytes(8, "little")
    load_ph[40:48] = segment_bytes.to_bytes(8, "little")
    load_ph[48:56] = (0x1000).to_bytes(8, "little")
    return (
        bytes(header)
        + bytes(note_ph)
        + bytes(load_ph)
        + note_bytes
        + b"\x00" * max(0, segment_bytes)
    )


def stop_record(
    *,
    reason="signal-received",
    signal="SIGSEGV",
    identity="1:1",
    live=True,
):
    payload = {
        "reason": reason,
        "thread-id": "1",
        "stopped-threads": "all",
        "frame": {"addr": "0x401000", "func": "main"},
    }
    if signal is not None:
        payload.update({"signal-name": signal, "signal-meaning": "Segmentation fault"})
    return {
        "stop_sequence": 1,
        "captured_at": "2026-07-22T00:00:00+00:00",
        "inferior_epoch": 1,
        "inferior_pid": 123,
        "payload": payload,
        "identity": identity,
        "controller_state": "stopped",
        "is_live": live,
    }


class FakeGdb:
    def __init__(self, record=None):
        self.state = "stopped" if record is not None else "idle"
        self.pid = 123 if record is not None else None
        self.record = record
        self.calls = []
        self.pending_record = stop_record()
        self.core_payload = None
        self._destroyed = False

    def get_state(self):
        return self.state

    def get_inferior_pid(self):
        return self.pid

    def get_last_stop_record(self):
        return dict(self.record) if self.record is not None else None

    @staticmethod
    def quote_mi_argument(value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def execute_mi_command(self, command, timeout_sec=7.0):
        self.calls.append(("mi", command, timeout_sec))
        return {
            "success": True,
            "state": self.state,
            "responses": [{"type": "result", "message": "done", "payload": {}}],
        }

    def run(self):
        self.calls.append(("run",))
        self.record = None
        self.state = "running"
        self.pid = 123
        return {
            "success": True,
            "state": "running",
            "responses": [{"type": "notify", "message": "running", "payload": {}}],
        }

    def continue_execution(self):
        self.calls.append(("continue",))
        self.record = None
        self.state = "running"
        return {"success": True, "state": "running", "responses": []}

    def drain_async(self, timeout_sec=0.0, max_rounds=10):
        self.calls.append(("drain", timeout_sec, max_rounds))
        self.record = self.pending_record
        self.state = "stopped"
        return {
            "success": True,
            "state": "stopped",
            "responses": [
                {
                    "type": "notify",
                    "message": "stopped",
                    "payload": self.pending_record["payload"],
                }
            ],
        }

    def interrupt(self, timeout_sec=1.0):
        self.calls.append(("interrupt", timeout_sec))
        self.state = "stopped"
        return {"success": True, "state": "stopped", "responses": []}

    def apply_fsize_limit(self, max_bytes):
        self.calls.append(("apply_fsize_limit", max_bytes))
        return {
            "applied": True,
            "pid": 999,
            "previous_soft": -1,
            "previous_hard": -1,
            "applied_soft": max_bytes,
        }

    def restore_fsize_limit(self, previous):
        self.calls.append(("restore_fsize_limit", previous["previous_soft"]))
        return True

    def mark_destroyed(self, reason):
        self._destroyed = True
        self.destroyed_reason = reason

    def read_memory_bytes(self, address, num_bytes):
        self.calls.append(("read", address, num_bytes))
        return {
            "success": True,
            "state": "stopped",
            "responses": [
                {
                    "type": "result",
                    "message": "done",
                    "payload": {
                        "memory": [
                            {"begin": "0x7fffffffe000", "contents": "aa" * num_bytes}
                        ]
                    },
                }
            ],
        }

    def execute_command(self, command, timeout_sec=7.0):
        self.calls.append(("cli", command, timeout_sec))
        output_path = command.removeprefix("generate-core-file ")
        payload = self.core_payload
        if payload is None:
            payload = minimal_elf64_core()
        with open(output_path, "wb") as destination:
            destination.write(payload)
        return {
            "success": True,
            "state": "stopped",
            "responses": [{"type": "result", "message": "done", "payload": {}}],
        }


class FakeDebugger:
    timeout_seconds = 3

    @staticmethod
    def snapshot_deadline():
        return time.monotonic() + 3

    def capture_snapshot(self, **kwargs):
        snapshot = {
            "snapshot_version": "1",
            "captured_at": "2026-07-22T00:00:00+00:00",
            "state": "stopped",
            "registers": [
                {"number": 0, "name": "rip", "value": "0x401000"},
                {"number": 1, "name": "rsp", "value": "0x7fffffffe000"},
                {"number": 2, "name": "rbp", "value": "0x7fffffffe100"},
            ],
            "frames": [{"level": 0, "address": "0x401000", "function": "main"}],
            "instructions": [{"address": "0x401000", "instruction": "mov [rax], 1"}],
            "threads": [{"id": "1", "state": "stopped"}],
            "current_thread_id": "1",
            "inferiors": [{"id": "i1", "pid": "123"}],
            "mappings": (
                kwargs["mapping_result"]["parsed"]["mappings"]
                if kwargs["mapping_result"] is not None
                else []
            ),
        }
        return {
            "success": True,
            "usable": True,
            "complete": True,
            "parsed": {"snapshot": snapshot},
            "raw_responses": {"registers": []},
        }

    @staticmethod
    def evaluate_expression(expression):
        return {
            "success": True,
            "parsed": {"value": "(void *) 0x0"},
            "raw_responses": [],
        }

    @staticmethod
    def _memory_contents(result):
        return result["responses"][0]["payload"]["memory"][0]["contents"]

    @staticmethod
    def _quote_cli_argument(value):
        return f'"{value}"'

    @staticmethod
    def _raw_output(result):
        return ""

    @staticmethod
    def _error_message(result, command):
        return result.get("error", f"failed: {command}")


class FakeInspection:
    @staticmethod
    def memory_maps():
        return {
            "success": True,
            "parsed": {
                "mappings": [
                    {
                        "start": "0x400000",
                        "end": "0x402000",
                        "readable": True,
                        "writable": False,
                        "executable": True,
                        "permissions": "r-xp",
                        "file": "/workspace/target",
                    },
                    {
                        "start": "0x7fffffffd000",
                        "end": "0x800000000000",
                        "readable": True,
                        "writable": True,
                        "executable": False,
                        "permissions": "rw-p",
                        "file": "[stack]",
                    },
                ]
            },
            "raw_responses": [{"type": "console", "payload": "vmmap"}],
        }


def adapter(record=None):
    gdb = FakeGdb(record)
    return (
        CrashTriage(
            gdb,
            FakeDebugger(),
            FakeInspection(),
            "alpha",
            session_generation="testgen",
        ),
        gdb,
    )


def test_run_until_crash_returns_stable_id_and_initial_snapshot():
    triage, gdb = adapter()

    result = triage.run_until_crash(arguments=["hello world"], mode="run")

    assert result["success"] is True
    assert result["parsed"]["crashed"] is True
    assert result["parsed"]["crash_id"] == "alpha:testgen:1:1"
    assert (
        result["parsed"]["initial_snapshot"]["analysis"]["category"]
        == "null_page_access"
    )
    assert result["parsed"]["snapshot_complete"] is True
    assert ("mi", '-exec-arguments "hello world"', 7.0) in gdb.calls


def test_run_until_crash_waits_for_new_stop_when_mi_state_is_stale():
    triage, gdb = adapter()

    def stale_state_run():
        gdb.calls.append(("run",))
        gdb.record = None
        gdb.pid = 123
        # Real GDB/MI may not include *running in the first response batch, so
        # the controller can temporarily retain its prior stopped state.
        gdb.state = "stopped"
        return {"success": True, "state": "stopped", "responses": []}

    gdb.run = stale_state_run

    result = triage.run_until_crash(mode="run")

    assert result["success"] is True
    assert result["parsed"]["crashed"] is True
    assert result["parsed"]["crash_id"] == "alpha:testgen:1:1"
    assert any(call[0] == "drain" for call in gdb.calls)


def test_non_crash_stop_is_truthful_and_never_auto_continues():
    triage, gdb = adapter()
    gdb.pending_record = stop_record(reason="breakpoint-hit", signal=None)

    result = triage.run_until_crash(mode="run")

    assert result["success"] is True
    assert result["parsed"]["crashed"] is False
    assert result["parsed"]["stop"]["kind"] == "non_crash_stop"
    assert any("did not auto-resume" in warning for warning in result["warnings"])
    assert not any(call[0] == "continue" for call in gdb.calls)


def test_snapshot_rejects_stale_or_non_live_crash_identity():
    triage, _ = adapter(stop_record())

    with pytest.raises(CrashRecordError, match="stale"):
        triage.capture_crash_snapshot("alpha:testgen:1:99")

    triage, _ = adapter(stop_record(live=False))
    with pytest.raises(CrashRecordError, match="no longer attached"):
        triage.capture_crash_snapshot("alpha:testgen:1:1")


def test_rebuilt_session_generation_rejects_colliding_mi_identity():
    record = stop_record()
    first_gdb = FakeGdb(record)
    second_gdb = FakeGdb(record)
    first = CrashTriage(
        first_gdb,
        FakeDebugger(),
        FakeInspection(),
        "alpha",
        session_generation="generation-a",
    )
    rebuilt = CrashTriage(
        second_gdb,
        FakeDebugger(),
        FakeInspection(),
        "alpha",
        session_generation="generation-b",
    )
    stale_id = first.get_last_stop()["parsed"]["crash_id"]
    current_id = rebuilt.get_last_stop()["parsed"]["crash_id"]

    assert stale_id != current_id
    with pytest.raises(CrashRecordError, match="stale"):
        rebuilt.capture_crash_snapshot(stale_id)


def test_cyclic_offset_reports_exact_bytes_without_hidden_truncation():
    result = CrashTriage.find_cyclic_offset("0x61616162", n=4, endian="little")

    assert result["parsed"]["found"] is True
    assert result["parsed"]["offset"] == 4
    assert result["parsed"]["match_kind"] == "exact"
    assert result["parsed"]["input"]["query_hex"] == "62616161"
    assert result["parsed"]["source"] == "local"


def test_short_cyclic_query_is_typed_ambiguous_with_bounded_unique_candidates():
    result = CrashTriage.find_cyclic_offset("aa", n=4, max_length=4096)
    parsed = result["parsed"]

    assert parsed["found"] is True
    assert parsed["offset"] is None
    assert parsed["match_kind"] == "ambiguous"
    assert parsed["resolution"] == "ambiguous"
    assert parsed["ambiguous"] is True
    assert 1 < parsed["candidate_count"] <= 256
    assert parsed["candidates_truncated"] is True
    assert all(match["kind"] == "partial_candidate" for match in parsed["matches"])
    identities = {(match["offset"], match["query_hex"]) for match in parsed["matches"]}
    assert len(identities) == len(parsed["matches"])


def test_cyclic_inputs_are_bounded_before_strip_or_ascii_encode():
    class OversizedValue(str):
        def __len__(self):
            return MAX_CYCLIC_VALUE_CHARS + 1

        def strip(self, *args, **kwargs):
            raise AssertionError("strip must not run for oversized input")

    class OversizedAlphabet(str):
        def __len__(self):
            return MAX_CYCLIC_ALPHABET_CHARS + 1

        def encode(self, *args, **kwargs):
            raise AssertionError("encode must not run for oversized input")

    with pytest.raises(ValueError, match="value must contain"):
        CrashTriage._decode_cyclic_value(
            OversizedValue("x"),
            endian="little",
            width=None,
        )
    with pytest.raises(ValueError, match="alphabet must contain"):
        CrashTriage._cyclic_parameters(
            n=4,
            endian="little",
            max_length=1024,
            alphabet=OversizedAlphabet("ab"),
        )


def test_snapshot_has_typed_faulting_instruction_classification_and_signature():
    triage, _ = adapter(stop_record())

    result = triage.capture_crash_snapshot("alpha:testgen:1:1")
    snapshot = result["parsed"]["crash_snapshot"]

    faulting = snapshot["analysis"]["faulting_instruction"]
    assert faulting["status"] == "observed"
    assert faulting["address"] == "0x401000"
    assert faulting["address_integer"] == 0x401000
    assert faulting["instruction"] == "mov [rax], 1"
    assert faulting["source"] == "gdb-mi"
    assert faulting["location"]["kind"] == "module_offset"
    assert faulting["location"]["module"] == "target"
    assert faulting["location"]["offset"] == 0x1000
    module_identity = faulting["location"]["module_identity"]
    assert module_identity["kind"] == "file"
    assert module_identity["path"].replace("\\", "/").endswith("/workspace/target")
    assert module_identity["stable"] is True
    classification = snapshot["classification"]
    assert classification["schema"] == "pao.crash-classification.v1"
    assert classification["primary"]["kind"] == "bad_pointer_dereference"
    assert classification["primary"]["assessment"] == "possible"
    assert classification["exploitability"] == "unknown"
    signature = snapshot["signature"]
    assert signature["schema"] == "pao.crash-fingerprint.v1"
    assert len(signature["digest"]) == 64
    assert "0x401000" not in json.dumps(signature["canonical"], sort_keys=True)


def test_memory_direction_is_mnemonic_aware_for_compare_instructions():
    for instruction in (
        "cmp [rax], rdx",
        "cmp qword ptr [rbp-0x8], 0x1",
        "cmp rax, [rbx]",
        "test byte ptr [rcx], 0x1",
        "test [rdx], rax",
        "cmn [x0], 1",
        "tst [x1], 1",
    ):
        assert CrashTriage._memory_direction(instruction) == "read", instruction
    # Register-only compares touch no memory: direction stays unknown.
    assert CrashTriage._memory_direction("cmp rax, rbx") is None
    assert CrashTriage._memory_direction("test eax, eax") is None


def test_fault_access_reports_compare_faults_as_reads():
    read = CrashTriage._fault_access(0x0, 0x401000, "cmp [rax], rdx", 1)
    assert read["access"] == "read"
    assert read["mnemonic"] == "cmp"
    assert read["si_code"] == 1
    write = CrashTriage._fault_access(0x0, 0x401000, "mov [rax], rdx", 1)
    assert write["access"] == "write"


def test_classify_crash_is_conservative_and_reuses_exact_cached_snapshot():
    triage, _ = adapter(stop_record())
    triage.capture_crash_snapshot(
        "alpha:testgen:1:1",
        stack_bytes=512,
        disassembly_bytes=128,
        include_mappings=False,
    )

    classified = triage.classify_crash(
        "alpha:testgen:1:1",
        stack_bytes=None,
        disassembly_bytes=None,
        include_mappings=None,
    )

    assert classified["success"] is True
    assert classified["parsed"]["cached"] is True
    assert classified["parsed"]["capture_options"] == {
        "stack_bytes": 512,
        "disassembly_bytes": 128,
        "include_mappings": False,
    }
    assert classified["parsed"]["classification"]["exploitability"] == "unknown"
    assert classified["parsed"]["signature"]["confidence"] == "low"


def test_report_provenance_reflects_mapping_capture_options_and_status():
    not_requested = _snapshot_report_provenance(
        {
            "component_status": {"mappings": True},
            "sources": {"mappings": "pwndbg-cli", "debugger": "gdb-mi"},
        },
        {"include_mappings": False},
    )
    unavailable = _snapshot_report_provenance(
        {
            "component_status": {"mappings": False},
            "sources": {"mappings": None, "debugger": "gdb-mi"},
        },
        {"include_mappings": True},
    )
    captured = _snapshot_report_provenance(
        {
            "component_status": {"mappings": True},
            "sources": {"mappings": "pwndbg-cli", "debugger": "gdb-mi"},
        },
        {"include_mappings": True},
    )

    assert not_requested["mappings"] == {
        "requested": False,
        "available": None,
        "source": None,
    }
    assert unavailable["mappings"] == {
        "requested": True,
        "available": False,
        "source": "unavailable",
    }
    assert captured["mappings"] == {
        "requested": True,
        "available": True,
        "source": "pwndbg-cli",
    }


def _shifted_snapshot(snapshot, crash_id, delta=0, pc_delta=0):
    shifted = copy.deepcopy(snapshot)
    shifted["crash_id"] = crash_id
    shifted["captured_at"] = f"2026-07-22T00:00:0{crash_id[-1]}+00:00"
    target_mapping = shifted["debugger"]["mappings"][0]
    target_mapping["start"] = hex(0x400000 + delta)
    target_mapping["end"] = hex(0x402000 + delta)
    pc_value = 0x401000 + delta + pc_delta
    for register in shifted["debugger"]["registers"]:
        if register["name"] == "rip":
            register["value"] = hex(pc_value)
    shifted["debugger"]["frames"][0]["address"] = hex(pc_value)
    shifted["debugger"]["instructions"][0]["address"] = hex(pc_value)
    shifted["analysis"]["program_counter"]["value"] = hex(pc_value)
    shifted["analysis"]["program_counter"]["integer"] = pc_value
    shifted["analysis"]["program_counter_mapping"] = copy.deepcopy(target_mapping)
    shifted["analysis"]["faulting_instruction"]["address"] = hex(pc_value)
    shifted["analysis"]["faulting_instruction"]["address_integer"] = pc_value
    shifted["analysis"]["faulting_instruction"]["location"] = (
        CrashTriage._normalized_address(pc_value, target_mapping)
    )
    shifted.pop("classification", None)
    shifted.pop("signature", None)
    classification = CrashTriage._classification_for_snapshot(shifted)
    shifted["classification"] = classification
    shifted["signature"] = CrashTriage._fingerprint_for_snapshot(
        shifted, classification
    )
    return shifted


def test_compare_crashes_is_aslr_normalized_and_evidence_driven():
    triage, _ = adapter(stop_record())
    captured = triage.capture_crash_snapshot("alpha:testgen:1:1")
    base = captured["parsed"]["crash_snapshot"]
    same_under_aslr = _shifted_snapshot(base, "alpha:testgen:1:2", delta=0x100000)
    different_pc = _shifted_snapshot(
        base, "alpha:testgen:1:3", delta=0x200000, pc_delta=4
    )
    triage._remember_snapshot(same_under_aslr)
    triage._remember_snapshot(different_pc)

    same = triage.compare_crashes("alpha:testgen:1:1", "alpha:testgen:1:2")
    different = triage.compare_crashes("alpha:testgen:1:1", "alpha:testgen:1:3")

    assert same["parsed"]["schema"] == "pao.crash-comparison.v1"
    assert same["parsed"]["relationship"] == "same"
    assert same["parsed"]["same_crash_family"] is True
    assert different["parsed"]["relationship"] == "different"
    assert different["parsed"]["same_crash_family"] is False
    assert any(
        item["field"] == "program_counter" and item["status"] == "different"
        for item in different["parsed"]["field_evidence"]
    )


def test_compare_crashes_binds_signature_to_same_basename_target_identity(tmp_path):
    triage, _ = adapter(stop_record())
    captured = triage.capture_crash_snapshot("alpha:testgen:1:1")
    base = captured["parsed"]["crash_snapshot"]
    snapshots = []
    for index, directory_name in enumerate(("first", "second"), start=2):
        binary = tmp_path / directory_name / "target"
        binary.parent.mkdir()
        binary.write_bytes(b"not-an-elf-" + bytes([index]))
        snapshot = copy.deepcopy(base)
        snapshot["crash_id"] = f"alpha:testgen:1:{index}"
        snapshot["debugger"]["mappings"][0]["file"] = str(binary)
        snapshot["analysis"]["program_counter_mapping"]["file"] = str(binary)
        snapshot.pop("target_identity", None)
        snapshot.pop("classification", None)
        snapshot.pop("signature", None)
        classification = CrashTriage._classification_for_snapshot(snapshot)
        snapshot["classification"] = classification
        snapshot["target_identity"] = CrashTriage._target_identity(snapshot)
        snapshot["signature"] = CrashTriage._fingerprint_for_snapshot(
            snapshot, classification
        )
        triage._remember_snapshot(snapshot)
        snapshots.append(snapshot)

    compared = triage.compare_crashes(
        "alpha:testgen:1:2",
        "alpha:testgen:1:3",
    )

    assert snapshots[0]["target_identity"]["name"] == "target"
    assert snapshots[1]["target_identity"]["name"] == "target"
    assert (
        snapshots[0]["target_identity"]["path"]
        != snapshots[1]["target_identity"]["path"]
    )
    assert compared["parsed"]["relationship"] == "different"
    assert compared["parsed"]["confidence"] == "high"
    assert "target identity" in compared["parsed"]["reasons"][0]
    assert any(
        item["field"] == "target_identity" and item["status"] == "different"
        for item in compared["parsed"]["field_evidence"]
    )


def test_anonymous_and_special_mappings_are_never_stable_module_offsets():
    special = CrashTriage._normalized_address(
        0x7FFFFFFFE100,
        {
            "start": "0x7fffffffd000",
            "end": "0x800000000000",
            "file": "[stack]",
        },
    )
    anonymous = CrashTriage._normalized_address(
        0x7F001000,
        {
            "start": "0x7f000000",
            "end": "0x7f100000",
            "file": "anonymous",
        },
    )

    assert special == {
        "kind": "special_mapping",
        "mapping": "[stack]",
        "stable": False,
    }
    assert anonymous == {
        "kind": "anonymous_mapping",
        "mapping": None,
        "stable": False,
    }


def test_compare_crashes_returns_unknown_when_stable_evidence_is_insufficient():
    triage, _ = adapter(stop_record())
    for suffix in ("1", "2"):
        crash_id = f"alpha:testgen:unknown:{suffix}"
        snapshot = {
            "crash_id": crash_id,
            "captured_at": f"2026-07-22T00:00:0{suffix}+00:00",
            "capture_options": {},
            "stop": {"signal": {"name": None}},
            "analysis": {
                "category": "unknown",
                "fault_access": "unknown",
                "program_counter": {"integer": None},
                "fault_address": {"integer": None},
                "faulting_instruction": {"instruction": None},
            },
            "debugger": {"frames": [], "mappings": []},
        }
        classification = CrashTriage._classification_for_snapshot(snapshot)
        snapshot["classification"] = classification
        snapshot["signature"] = CrashTriage._fingerprint_for_snapshot(
            snapshot, classification
        )
        triage._remember_snapshot(snapshot)

    compared = triage.compare_crashes(
        "alpha:testgen:unknown:1", "alpha:testgen:unknown:2"
    )

    assert compared["parsed"]["relationship"] == "unknown"
    assert compared["parsed"]["same_crash_family"] is None
    assert compared["complete"] is False


def test_crash_history_is_bounded_and_returns_immutable_copies(monkeypatch):
    monkeypatch.setattr(crash_backend, "MAX_CRASH_HISTORY_ENTRIES", 2)
    triage, _ = adapter(stop_record())
    captured = triage.capture_crash_snapshot("alpha:testgen:1:1")
    base = captured["parsed"]["crash_snapshot"]
    second = _shifted_snapshot(base, "alpha:testgen:1:2", delta=0x100000)
    third = _shifted_snapshot(base, "alpha:testgen:1:3", delta=0x200000)
    triage._remember_snapshot(second)
    triage._remember_snapshot(third)

    assert triage._history_descriptor("alpha:testgen:1:1") is None
    returned = triage._history_descriptor("alpha:testgen:1:2")
    assert returned is not None
    returned["signature"]["digest"] = "mutated"
    assert (
        triage._history_descriptor("alpha:testgen:1:2")["signature"]["digest"]
        != "mutated"
    )


def test_crash_aware_cyclic_scan_checks_registers_and_bounded_stack():
    triage, _ = adapter(stop_record())
    captured = triage.capture_crash_snapshot("alpha:testgen:1:1")
    snapshot = copy.deepcopy(captured["parsed"]["crash_snapshot"])
    snapshot["debugger"]["registers"].append(
        {"number": 3, "name": "rax", "value": "0x61616162"}
    )
    snapshot["stack"]["data_hex"] = (b"\x00" * 8 + b"baaa" + b"\x00" * 8).hex()
    snapshot.pop("classification", None)
    snapshot.pop("signature", None)
    triage._remember_snapshot(snapshot)

    result = triage.find_cyclic_offsets_for_crash(
        "alpha:testgen:1:1",
        n=4,
        endian="little",
        max_stack_scan_bytes=20,
    )

    assert result["parsed"]["found"] is True
    assert result["parsed"]["crash_scan"]["stack_scanned_bytes"] == 20
    assert any(
        match["offset"] == 4
        and match["evidence"]["kind"] == "register"
        and match["evidence"]["name"] == "rax"
        for match in result["parsed"]["matches"]
    )
    assert any(
        match["offset"] == 4
        and match["evidence"]["kind"] == "stack"
        and match["evidence"]["stack_offset"] == 8
        for match in result["parsed"]["matches"]
    )


def test_cyclic_generation_checks_cancellation_during_large_work():
    calls = 0

    def cancellation_check():
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        CrashTriage.find_cyclic_offset(
            "baaa",
            max_length=256 * 1024,
            cancellation_check=cancellation_check,
        )


def test_core_generation_uses_raw_cli_filename_and_validates_et_core(tmp_path):
    triage, gdb = adapter(stop_record())
    output = tmp_path / "core file"

    result = triage.generate_core_file(
        "alpha:testgen:1:1", str(output), max_size_bytes=4096, min_free_bytes=0
    )

    assert result["success"] is True
    assert result["parsed"]["validation"]["elf_type"] == "ET_CORE"
    assert result["parsed"]["validation"]["program_header_count"] == 2
    assert result["parsed"]["validation"]["note_segment_count"] == 1
    assert result["parsed"]["validation"]["load_segment_count"] == 1
    assert result["parsed"]["validation"]["recognized_core_note_count"] == 1
    assert result["parsed"]["validation"]["material_load_segment_count"] == 1
    assert result["parsed"]["source"] == "gdb-cli"
    cli_call = next(call for call in gdb.calls if call[0] == "cli")
    assert cli_call[1] == f"generate-core-file {output}"
    assert os.path.getsize(output) == len(minimal_elf64_core())


def test_core_generation_rejects_cli_command_delimiters(tmp_path):
    triage, _ = adapter(stop_record())

    with pytest.raises(ValueError, match="forbidden control character"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core\nmaintenance packet on"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )


def test_core_generation_rejects_invalid_elf(tmp_path):
    triage, gdb = adapter(stop_record())
    gdb.core_payload = b"not-a-core"

    with pytest.raises(RuntimeError, match="not an ELF"):
        triage.generate_core_file(
            "alpha:testgen:1:1", str(tmp_path / "core"), max_size_bytes=4096
        )


def test_core_estimate_is_bounded_mapping_sum():
    triage, _ = adapter(stop_record())

    result = triage.estimate_core_size()

    assert result["success"] is True
    assert result["mapping_count"] == 2
    assert result["estimated_upper_bound_bytes"] == 1024 * 1024 + 0x2000 + 0x3000
    assert result["estimate_saturated"] is False


def test_core_estimate_saturates_instead_of_dropping_huge_mapping():
    triage, _ = adapter(stop_record())
    triage.inspection.memory_maps = lambda: {
        "success": True,
        "parsed": {
            "mappings": [
                {
                    "start": "0x0",
                    "end": hex(MAX_CONFIGURED_CORE_DUMP_BYTES + 0x1000),
                    "readable": True,
                }
            ]
        },
        "raw_responses": [],
    }

    result = triage.estimate_core_size()

    assert result["success"] is True
    assert result["mapping_count"] == 1
    assert result["estimated_upper_bound_bytes"] == MAX_CONFIGURED_CORE_DUMP_BYTES
    assert result["estimate_saturated"] is True


def test_core_estimate_includes_non_readable_mapping_as_upper_bound():
    triage, _ = adapter(stop_record())
    triage.inspection.memory_maps = lambda: {
        "success": True,
        "parsed": {
            "mappings": [
                {
                    "start": "0x1000",
                    "end": "0x5000",
                    "readable": False,
                    "writable": False,
                    "executable": True,
                }
            ]
        },
        "raw_responses": [],
    }

    result = triage.estimate_core_size()

    assert result["mapping_count"] == 1
    assert result["estimated_upper_bound_bytes"] == 1024 * 1024 + 0x4000


def test_run_until_crash_does_not_treat_stale_exit_as_crash():
    # A prior crash record survives in the adapter but is no longer live (the
    # inferior has gone: no pid, record marked non-live).  Even though the record
    # classifies as a fatal signal, it must not be reported as a current crash.
    triage, gdb = adapter(stop_record(live=False))
    gdb.state = "stopped"
    gdb.pid = None

    result = triage.run_until_crash(mode="wait")

    assert result["success"] is True
    assert result["parsed"]["crashed"] is False
    assert any("not attached to a live" in w for w in result.get("warnings", []))
    # It must never have tried to snapshot a dead inferior.
    assert not any(call[0] == "read" for call in gdb.calls)


def test_validate_elf_core_rejects_truncated_stub(tmp_path):
    triage, gdb = adapter(stop_record())
    # Header-only ET_CORE with no program-header table.
    stub = bytearray(64)
    stub[:4] = b"\x7fELF"
    stub[4] = 2
    stub[5] = 1
    stub[6] = 1
    stub[16:18] = (4).to_bytes(2, "little")
    stub[20:24] = (1).to_bytes(4, "little")
    stub[32:40] = (64).to_bytes(8, "little")
    stub[52:54] = (64).to_bytes(2, "little")
    stub[54:56] = (56).to_bytes(2, "little")
    stub[56:58] = (1).to_bytes(2, "little")
    gdb.core_payload = bytes(stub)

    with pytest.raises(RuntimeError, match="truncated"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )


def test_validate_elf_core_rejects_segment_outside_file(tmp_path):
    triage, gdb = adapter(stop_record())
    payload = bytearray(minimal_elf64_core())
    # First PT_NOTE starts at file offset 64; p_offset is bytes 8..16.
    payload[72:80] = (len(payload) + 4096).to_bytes(8, "little")
    gdb.core_payload = bytes(payload)

    with pytest.raises(RuntimeError, match="points outside"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )


def test_validate_elf_core_rejects_unrecognized_or_empty_notes(tmp_path):
    triage, gdb = adapter(stop_record())
    payload = bytearray(minimal_elf64_core())
    # Change NT_PRSTATUS to an unknown note type.
    note_offset = 64 + 56 * 2
    payload[note_offset + 8 : note_offset + 12] = (0xFFFF).to_bytes(4, "little")
    gdb.core_payload = bytes(payload)

    with pytest.raises(RuntimeError, match="NT_PRSTATUS"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )


def test_validate_elf_core_rejects_empty_load_segment(tmp_path):
    triage, gdb = adapter(stop_record())
    payload = bytearray(minimal_elf64_core())
    load_header = 64 + 56
    payload[load_header + 32 : load_header + 48] = b"\x00" * 16
    gdb.core_payload = bytes(payload)

    with pytest.raises(RuntimeError, match="non-empty PT_LOAD"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )


def test_validate_elf_core_bounds_note_entry_work(tmp_path, monkeypatch):
    triage, gdb = adapter(stop_record())
    monkeypatch.setattr(crash_core_module, "MAX_ELF_CORE_NOTE_ENTRIES", 4)
    prefix = bytearray(minimal_elf64_core())[: 64 + 56 * 2]
    note = (
        (1).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + (0xFFFF).to_bytes(4, "little")
        + b"A"
        + b"\x00" * 3
    )
    notes = note * 5
    load_bytes = b"\x00" * 64
    note_offset = len(prefix)
    load_offset = note_offset + len(notes)
    prefix[64 + 8 : 64 + 16] = note_offset.to_bytes(8, "little")
    prefix[64 + 32 : 64 + 40] = len(notes).to_bytes(8, "little")
    prefix[64 + 40 : 64 + 48] = len(notes).to_bytes(8, "little")
    load_header = 64 + 56
    prefix[load_header + 8 : load_header + 16] = load_offset.to_bytes(8, "little")
    prefix[load_header + 32 : load_header + 40] = len(load_bytes).to_bytes(8, "little")
    prefix[load_header + 40 : load_header + 48] = len(load_bytes).to_bytes(8, "little")
    gdb.core_payload = bytes(prefix) + notes + load_bytes

    with pytest.raises(RuntimeError, match="too many ELF note entries"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=2 * 1024 * 1024,
            min_free_bytes=0,
        )


def test_cyclic_offset_big_endian_labels_low_high_by_value_not_position():
    # 0x6161616261616163 big-endian -> bytes b'aaabaaac'.
    # Low-order n bytes are the TRAILING four (b'aaac'); high-order are LEADING.
    result = CrashTriage.find_cyclic_offset("0x6161616261616163", n=4, endian="big")
    matches = {m["kind"]: m for m in result["parsed"]["matches"]}
    assert matches["low_order_n"]["query_hex"] == b"aaac".hex()
    assert matches["high_order_n"]["query_hex"] == b"aaab".hex()
    # Position labels are always present and unambiguous.
    assert matches["leading_n"]["query_hex"] == b"aaab".hex()
    assert matches["trailing_n"]["query_hex"] == b"aaac".hex()


def test_cyclic_offset_little_endian_low_order_is_leading():
    result = CrashTriage.find_cyclic_offset("0x6161616261616163", n=4, endian="little")
    matches = {m["kind"]: m for m in result["parsed"]["matches"]}
    # little-endian in-memory bytes = b'caaabaaa'; low-order n is the leading four.
    assert matches["low_order_n"]["query_hex"] == matches["leading_n"]["query_hex"]


def test_core_generation_hard_kills_writer_on_timeout(tmp_path):
    triage, gdb = adapter(stop_record())

    events = []

    def timed_out_execute(command, timeout_sec=7.0):
        gdb.calls.append(("cli", command, timeout_sec))
        # Simulate gdb blocking past the timeout mid-write: create a partial file
        # and report a timeout instead of completing.
        output_path = command.removeprefix("generate-core-file ")
        with open(output_path, "wb") as destination:
            destination.write(b"\x00" * 32)
        return {
            "success": False,
            "state": "running",
            "timed_out": True,
            "responses": [],
        }

    def signal_interrupt():
        events.append("signal")
        return {"success": True}

    def wait_for_writer_quiescence(timeout_sec=2.0, poll_interval=0.02):
        events.append("wait")
        return False  # gdb did NOT stop cooperatively

    def kill_process_group():
        events.append("kill")
        return {"success": True, "pid": 123, "killed_group": True}

    gdb.execute_command = timed_out_execute
    gdb.signal_interrupt = signal_interrupt
    gdb.wait_for_writer_quiescence = wait_for_writer_quiescence
    gdb.kill_process_group = kill_process_group

    with pytest.raises(RuntimeError, match="timed out"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    # The writer must be signalled and, since it did not quiesce, hard-killed
    # BEFORE the exception propagates to any staging cleanup.
    assert "kill" in events
    assert events.index("wait") < events.index("kill")


def test_core_generation_prefers_clean_stop_when_signal_succeeds(tmp_path):
    triage, gdb = adapter(stop_record())
    events = []
    signalled = threading.Event()

    def oversized_execute(command, timeout_sec=7.0):
        output_path = command.removeprefix("generate-core-file ")
        with open(output_path, "wb") as destination:
            destination.write(b"\x00" * 8192)
        # Block until the size monitor notices the overshoot and signals.
        signalled.wait(timeout=5)
        return {
            "success": False,
            "state": "running",
            "timed_out": True,
            "responses": [],
        }

    def signal_interrupt():
        events.append("signal")
        signalled.set()
        return {"success": True, "state": "running", "responses": []}

    def wait_for_writer_quiescence(timeout_sec=2.0, poll_interval=0.02):
        events.append("wait")
        return True  # gdb stopped cleanly after the signal

    def kill_process_group():
        events.append("kill")
        return {"success": True, "pid": 123, "killed_group": True}

    gdb.execute_command = oversized_execute
    gdb.signal_interrupt = signal_interrupt
    gdb.wait_for_writer_quiescence = wait_for_writer_quiescence
    gdb.kill_process_group = kill_process_group

    with pytest.raises(RuntimeError, match="exceeded"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    # A successful signal is the clean stop: the bounded quiescence wait runs,
    # and the hard kill stays a fallback that must not fire.
    assert "signal" in events
    assert events.index("signal") < events.index("wait")
    assert "kill" not in events


def test_core_generation_kills_writer_when_signal_interrupt_is_unsupported(
    tmp_path,
):
    triage, gdb = adapter(stop_record())
    events = []
    signalled = threading.Event()

    def oversized_execute(command, timeout_sec=7.0):
        output_path = command.removeprefix("generate-core-file ")
        with open(output_path, "wb") as destination:
            destination.write(b"\x00" * 8192)
        signalled.wait(timeout=5)
        return {
            "success": False,
            "state": "running",
            "timed_out": True,
            "responses": [],
        }

    def signal_interrupt():
        events.append("signal")
        signalled.set()
        return {
            "success": False,
            "state": "running",
            "responses": [],
            "error": "Signal-only interrupt requires POSIX process signalling",
            "type": "SignalInterruptUnsupported",
        }

    def wait_for_writer_quiescence(timeout_sec=2.0, poll_interval=0.02):
        events.append("wait")
        return False  # no cooperative stop without a signal

    def kill_process_group():
        events.append("kill")
        return {"success": True, "pid": 123, "confirmed_dead": True}

    gdb.execute_command = oversized_execute
    gdb.signal_interrupt = signal_interrupt
    gdb.wait_for_writer_quiescence = wait_for_writer_quiescence
    gdb.kill_process_group = kill_process_group

    with pytest.raises(RuntimeError):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    # The unsupported signal is recorded as a violation and the process-group
    # kill escalation is the fallback that actually quiesces the writer.
    assert events.index("signal") < events.index("wait") < events.index("kill")


def test_custom_fatal_signal_policy_survives_analysis_refresh():
    triage, _ = adapter(stop_record(signal="SIGTRAP"))

    initial = triage.capture_crash_snapshot(
        "alpha:testgen:1:1", fatal_signals=["sigtrap"]
    )
    refreshed = triage.analyze_crash("alpha:testgen:1:1", refresh=True)

    assert initial["success"] is True
    assert initial["parsed"]["crash_snapshot"]["fatal_signals"] == ["SIGTRAP"]
    assert refreshed["success"] is True
    assert refreshed["parsed"]["crash_snapshot"]["fatal_signals"] == ["SIGTRAP"]
    assert refreshed["parsed"]["crash_snapshot"]["stop"]["crashed"] is True
    assert refreshed["parsed"]["analysis"]["category"] == "signal_crash"


def test_custom_fatal_signal_policy_flows_into_core_capture(tmp_path):
    triage, _ = adapter(stop_record(signal="SIGTRAP"))
    triage.capture_crash_snapshot("alpha:testgen:1:1", fatal_signals=["SIGTRAP"])

    result = triage.generate_core_file(
        "alpha:testgen:1:1",
        str(tmp_path / "core"),
        max_size_bytes=4096,
        min_free_bytes=0,
    )

    assert result["success"] is True


def test_analyze_crash_accepts_custom_fatal_signal_without_cached_snapshot():
    triage, _ = adapter(stop_record(signal="SIGTRAP"))

    result = triage.analyze_crash(
        "alpha:testgen:1:1", refresh=True, fatal_signals=["SIGTRAP"]
    )

    assert result["success"] is True
    assert result["parsed"]["crash_snapshot"]["fatal_signals"] == ["SIGTRAP"]


def test_analysis_policy_change_refreshes_instead_of_claiming_cached():
    triage, _ = adapter(stop_record(signal="SIGTRAP"))
    triage.capture_crash_snapshot("alpha:testgen:1:1", fatal_signals=["SIGTRAP"])

    result = triage.analyze_crash(
        "alpha:testgen:1:1",
        fatal_signals=["SIGTRAP", "SIGTERM"],
    )

    assert result["success"] is True
    assert result["parsed"]["cached"] is False
    assert result["parsed"]["crash_snapshot"]["fatal_signals"] == [
        "SIGTRAP",
        "SIGTERM",
    ]


def test_analysis_capture_option_change_refreshes_cache():
    triage, _ = adapter(stop_record())
    triage.capture_crash_snapshot("alpha:testgen:1:1")

    result = triage.analyze_crash(
        "alpha:testgen:1:1",
        stack_bytes=512,
        disassembly_bytes=128,
        include_mappings=False,
    )

    assert result["success"] is True
    assert result["parsed"]["cached"] is False
    assert result["parsed"]["crash_snapshot"]["capture_options"] == {
        "stack_bytes": 512,
        "disassembly_bytes": 128,
        "include_mappings": False,
    }
    assert result["parsed"]["crash_snapshot"]["stack"]["requested_bytes"] == 512


def test_cancellation_drains_late_stop_before_propagating():
    triage, gdb = adapter()
    drain_count = 0
    cancellation_checks = 0

    def late_stop_drain(timeout_sec=0.0, max_rounds=10):
        nonlocal drain_count
        drain_count += 1
        gdb.calls.append(("drain", timeout_sec, max_rounds))
        if drain_count == 1:
            return {"success": True, "state": "running", "responses": []}
        gdb.record = gdb.pending_record
        gdb.state = "stopped"
        return {
            "success": True,
            "state": "stopped",
            "responses": [
                {
                    "type": "notify",
                    "message": "stopped",
                    "payload": gdb.pending_record["payload"],
                }
            ],
        }

    def cancellation_check():
        nonlocal cancellation_checks
        cancellation_checks += 1
        if cancellation_checks >= 2:
            raise RuntimeError("cancelled")

    gdb.drain_async = late_stop_drain

    with pytest.raises(RuntimeError, match="cancelled"):
        triage.run_until_crash(mode="run", cancellation_check=cancellation_check)

    assert drain_count >= 2
    assert gdb.state == "stopped"
    assert getattr(gdb, "_destroyed", False) is False


def test_run_timeout_keeps_session_only_after_synchronized_stop():
    triage, gdb = adapter()

    def no_stop_drain(timeout_sec=0.0, max_rounds=10):
        gdb.calls.append(("drain", timeout_sec, max_rounds))
        return {"success": True, "state": "running", "responses": []}

    def synchronized_interrupt(timeout_sec=1.0):
        gdb.record = gdb.pending_record
        gdb.state = "stopped"
        return {"success": True, "state": "stopped", "responses": []}

    gdb.drain_async = no_stop_drain
    gdb.interrupt = synchronized_interrupt

    result = triage.run_until_crash(mode="run", timeout_seconds=0.001)

    assert result["timed_out"] is True
    assert result["parsed"]["interrupt_synchronized"] is True
    assert result["parsed"]["session_destroyed"] is False
    assert gdb._destroyed is False


def test_run_timeout_invalidates_unsynchronized_session():
    triage, gdb = adapter()
    kills = []

    def no_stop_drain(timeout_sec=0.0, max_rounds=10):
        return {"success": True, "state": "running", "responses": []}

    def unsynchronized_interrupt(timeout_sec=1.0):
        return {"success": False, "state": "running", "responses": []}

    def kill_process_group():
        kills.append(True)
        return {"success": True, "confirmed_dead": True}

    gdb.drain_async = no_stop_drain
    gdb.interrupt = unsynchronized_interrupt
    gdb.kill_process_group = kill_process_group

    result = triage.run_until_crash(mode="run", timeout_seconds=0.001)

    assert result["timed_out"] is True
    assert result["parsed"]["interrupt_synchronized"] is False
    assert result["parsed"]["session_destroyed"] is True
    assert kills == [True]
    assert gdb._destroyed is True


def test_run_until_crash_surfaces_backend_failure_without_burning_budget():
    triage, gdb = adapter()

    def failing_drain(timeout_sec=0.0, max_rounds=10):
        gdb.calls.append(("drain", timeout_sec, max_rounds))
        return {
            "success": False,
            "state": "running",
            "responses": [],
            "backend_error": True,
            "type": "OSError",
            "error": "pipe is closed",
        }

    gdb.drain_async = failing_drain

    result = triage.run_until_crash(mode="run", timeout_seconds=30)

    assert result["success"] is False
    assert result["parsed"]["backend_error"] is True
    assert result["parsed"]["crashed"] is False
    assert result["type"] == "OSError"
    assert "pipe is closed" in result["error"]
    assert gdb._destroyed is True


def test_unconfirmed_core_writer_poison_session_before_cleanup(tmp_path):
    triage, gdb = adapter(stop_record())

    def failed_execute(command, timeout_sec=7.0):
        output_path = command.removeprefix("generate-core-file ")
        with open(output_path, "wb") as destination:
            destination.write(b"partial")
        raise RuntimeError("transport failed")

    gdb.execute_command = failed_execute
    gdb.wait_for_writer_quiescence = lambda **kwargs: False
    gdb.kill_process_group = lambda: {
        "success": False,
        "confirmed_dead": False,
        "error": "still alive",
    }

    with pytest.raises(WriterNotQuiescentError, match="Could not confirm"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    assert getattr(gdb, "_destroyed", False) is True


def test_missing_quiescence_capability_forces_quarantine_error(tmp_path):
    triage, gdb = adapter(stop_record())
    gdb.execute_command = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("transport failed")
    )

    with pytest.raises(WriterNotQuiescentError, match="cannot confirm"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    assert gdb._destroyed is True


def test_core_generation_reports_actual_rlimit_apply_and_restore(tmp_path):
    triage, gdb = adapter(stop_record())
    events = []
    original_execute = gdb.execute_command

    def apply_limit(cap):
        events.append(("apply", cap))
        return {
            "applied": True,
            "previous_soft": 8192,
            "previous_hard": 16384,
            "applied_soft": cap,
        }

    def execute(command, timeout_sec=7.0):
        events.append(("execute",))
        return original_execute(command, timeout_sec)

    def restore_limit(previous):
        events.append(("restore", previous["previous_soft"]))
        return True

    gdb.apply_fsize_limit = apply_limit
    gdb.execute_command = execute
    gdb.restore_fsize_limit = restore_limit

    result = triage.generate_core_file(
        "alpha:testgen:1:1",
        str(tmp_path / "core"),
        max_size_bytes=4096,
        min_free_bytes=0,
    )

    enforcement = result["parsed"]["limit_enforcement"]
    assert enforcement["kernel_rlimit_applied"] is True
    assert enforcement["kernel_rlimit_bytes"] == 4096
    assert enforcement["kernel_rlimit_restored"] is True
    assert enforcement["fallback"] is None
    assert events == [("apply", 4096), ("execute",), ("restore", 8192)]


def test_failed_rlimit_restore_invalidates_session_and_surfaces_warning(tmp_path):
    triage, gdb = adapter(stop_record())
    gdb.apply_fsize_limit = lambda cap: {
        "applied": True,
        "previous_soft": 8192,
        "previous_hard": 16384,
    }
    gdb.restore_fsize_limit = lambda previous: False

    with pytest.raises(RuntimeError, match="Failed to restore"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    assert getattr(gdb, "_destroyed", False) is True


def test_core_generation_refuses_polling_only_limit_fallback(tmp_path):
    triage, gdb = adapter(stop_record())
    gdb.apply_fsize_limit = lambda cap: {
        "applied": False,
        "error": "prlimit unavailable",
    }

    with pytest.raises(RuntimeError, match="refused.*kernel file-size limit"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    assert not any(call[0] == "cli" for call in gdb.calls)


def test_restore_exception_invalidates_session(tmp_path):
    triage, gdb = adapter(stop_record())

    def restore_raises(previous):
        raise OSError("stale pid")

    gdb.restore_fsize_limit = restore_raises

    with pytest.raises(RuntimeError, match="stale pid"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    assert gdb._destroyed is True


def test_primary_generation_error_survives_restore_exception(tmp_path):
    triage, gdb = adapter(stop_record())
    gdb.execute_command = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("primary transport failure")
    )
    gdb.wait_for_writer_quiescence = lambda **kwargs: True
    gdb.restore_fsize_limit = lambda previous: (_ for _ in ()).throw(
        OSError("restore also failed")
    )

    with pytest.raises(RuntimeError, match="primary transport failure"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    assert gdb._destroyed is True


def test_command_failure_is_not_masked_by_restore_failure(tmp_path):
    triage, gdb = adapter(stop_record())
    gdb.execute_command = lambda *args, **kwargs: {
        "success": False,
        "state": "stopped",
        "error": "core command failed",
        "responses": [],
    }
    gdb.restore_fsize_limit = lambda previous: (_ for _ in ()).throw(
        OSError("restore also failed")
    )

    with pytest.raises(RuntimeError, match="core command failed"):
        triage.generate_core_file(
            "alpha:testgen:1:1",
            str(tmp_path / "core"),
            max_size_bytes=4096,
            min_free_bytes=0,
        )

    assert gdb._destroyed is True


def test_core_reference_requires_exact_crash_provenance():
    metadata = ArtifactMetadata(
        artifact_id="a" * 32,
        tool="capture_core_dump",
        session_id="alpha",
        content_type="application/x-coredump",
        encoding="binary",
        size_bytes=100,
        sha256="b" * 64,
        created_at="2026-07-23T00:00:00+00:00",
        expires_at=None,
        provenance={
            "schema": "pao.core-provenance.v1",
            "crash_id": "alpha:testgen:1:1",
            "session_id": "alpha",
        },
    )

    reference = _core_reference_for_crash(
        metadata,
        expected_crash_id="alpha:testgen:1:1",
        session_id="alpha",
        verification={"valid": True, "artifact": metadata.to_dict()},
    )

    assert reference["artifact_id"] == "a" * 32
    with pytest.raises(ValueError, match="was not captured"):
        _core_reference_for_crash(
            metadata,
            expected_crash_id="alpha:testgen:2:1",
            session_id="alpha",
            verification={"valid": True, "artifact": metadata.to_dict()},
        )


def test_core_reference_rejects_legacy_artifact_without_provenance():
    metadata = ArtifactMetadata(
        artifact_id="a" * 32,
        tool="capture_core_dump",
        session_id="alpha",
        content_type="application/x-coredump",
        encoding="binary",
        size_bytes=100,
        sha256="b" * 64,
        created_at="2026-07-23T00:00:00+00:00",
        expires_at=None,
    )

    with pytest.raises(ValueError, match="capture a new core"):
        _core_reference_for_crash(
            metadata,
            expected_crash_id="alpha:testgen:1:1",
            session_id="alpha",
            verification={"valid": True, "artifact": metadata.to_dict()},
        )


def test_core_reference_rejects_failed_payload_integrity():
    metadata = ArtifactMetadata(
        artifact_id="a" * 32,
        tool="capture_core_dump",
        session_id="alpha",
        content_type="application/x-coredump",
        encoding="binary",
        size_bytes=100,
        sha256="b" * 64,
        created_at="2026-07-23T00:00:00+00:00",
        expires_at=None,
        provenance={
            "schema": "pao.core-provenance.v1",
            "crash_id": "alpha:testgen:1:1",
            "session_id": "alpha",
        },
    )

    with pytest.raises(ValueError, match="integrity verification"):
        _core_reference_for_crash(
            metadata,
            expected_crash_id="alpha:testgen:1:1",
            session_id="alpha",
            verification={"valid": False, "artifact": metadata.to_dict()},
        )


def test_module_identity_keeps_posix_container_paths_verbatim():
    identity = CrashTriage._module_identity({"file": "/workspace/target"})
    assert identity["kind"] == "file"
    assert identity["path"] == "/workspace/target"
    assert identity["name"] == "target"
    assert identity["stable"] is True
