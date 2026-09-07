"""Round-2 regression tests: leftover review items (compare, si_code, macho, btf, ida, watch)."""

import struct

import pytest

from paomcp.tools.backends.crash import CrashTriage


class TestFaultDirection:
    @pytest.mark.parametrize(
        ("instruction", "expected"),
        [
            ("mov [rax], 1", "write"),
            ("mov QWORD PTR [rbp-0x8], rdx", "write"),
            ("mov rax, [rdi]", "read"),
            ("mov eax, dword ptr [rcx+4]", "read"),
            ("push rbp", "write"),
            ("pop rbp", "read"),
            ("call QWORD PTR [rax+0x10]", "read"),
            ("ret", "read"),
            ("nop", None),
            ("mov rax, rbx", None),
            ("", None),
            (None, None),
        ],
    )
    def test_memory_direction(self, instruction, expected):
        assert CrashTriage._memory_direction(instruction) == expected

    def test_execute_wins_over_mnemonic(self):
        out = CrashTriage._fault_access(0x401000, 0x401000, "mov [rax], 1", 1)
        assert out["access"] == "execute"

    def test_store_is_write_with_evidence(self):
        out = CrashTriage._fault_access(0x0, 0x401000, "mov [rax], rdx", 1)
        assert out["access"] == "write"
        assert out["si_code"] == 1
        assert out["mnemonic"] == "mov"

    def test_unknown_stays_unknown(self):
        out = CrashTriage._fault_access(0x0, 0x401000, "nop", None)
        assert out["access"] == "unknown"


class TestCompareStackTop5:
    def _snapshots(self):
        from tests.test_crash_triage import adapter, stop_record

        triage, _ = adapter(stop_record())
        captured = triage.capture_crash_snapshot("alpha:testgen:1:1")
        return triage, captured["parsed"]["crash_snapshot"]

    def test_field_evidence_has_stack_top5(self):
        import copy

        triage, base = self._snapshots()
        for suffix in ("1:2", "1:3"):
            snapshot = copy.deepcopy(base)
            snapshot["crash_id"] = f"alpha:testgen:{suffix}"
            snapshot.pop("classification", None)
            snapshot.pop("signature", None)
            classification = CrashTriage._classification_for_snapshot(snapshot)
            snapshot["classification"] = classification
            snapshot["signature"] = CrashTriage._fingerprint_for_snapshot(
                snapshot, classification
            )
            triage._remember_snapshot(snapshot)
        compared = triage.compare_crashes("alpha:testgen:1:1", "alpha:testgen:1:2")
        fields = {item["field"] for item in compared["parsed"]["field_evidence"]}
        assert "stack_top5" in fields
        assert "top_frame" not in fields

    def test_divergent_tail_splits_family(self):
        import copy

        triage, base = self._snapshots()
        other = copy.deepcopy(base)
        other["crash_id"] = "alpha:testgen:1:9"
        frames = other["debugger"]["frames"]
        assert isinstance(frames, list) and frames, "need a live frame to diverge"
        frames[0] = dict(frames[0])
        frames[0]["function"] = "totally_different_caller"
        other.pop("classification", None)
        other.pop("signature", None)
        classification = CrashTriage._classification_for_snapshot(other)
        other["classification"] = classification
        other["signature"] = CrashTriage._fingerprint_for_snapshot(
            other, classification
        )
        triage._remember_snapshot(other)
        compared = triage.compare_crashes("alpha:testgen:1:1", "alpha:testgen:1:9")
        statuses = {
            item["field"]: item["status"]
            for item in compared["parsed"]["field_evidence"]
        }
        assert statuses.get("stack_top5") == "different"


class TestMachoSlices:
    def _thin_arm64(self) -> bytes:
        return struct.pack("<8I", 0xFEEDFACF, 0x0100000C, 0, 2, 0, 0, 0, 0)

    def _fat(self, entries) -> bytes:
        blob = struct.pack(">II", 0xCAFEBABE, len(entries))
        offset = 8 + 20 * len(entries)
        body = b""
        for cputype, size in entries:
            blob += struct.pack(">IIIII", cputype, 0, offset, size, 12)
            body += self._thin_arm64()[:size].ljust(size, b"\x00")
            offset += size
        return blob + body

    def test_cigam_thin_parses_big_endian(self):
        from paomcp.tools.backends.macho import parse_macho

        # MH_MAGIC_64 stored big-endian: bytes FE ED FA CF.
        image = struct.pack(">8I", 0xFEEDFACF, 0x0100000C, 0, 2, 0, 0, 0, 0)
        macho = parse_macho(image)
        assert macho.is_64 is True
        assert macho.magic == 0xFEEDFACF
        assert macho.cpu_name == "arm64"

    def test_fat_inventory_and_first_slice(self):
        from paomcp.tools.backends.macho import parse_fat_macho, parse_macho_slice

        blob = self._fat([(0x0100000C, 32), (0x01000007, 32)])
        slices = parse_fat_macho(blob)
        assert len(slices) == 2
        assert slices[0].cpu_name == "arm64"
        assert slices[1].cpu_name == "x86_64"
        first = parse_macho_slice(blob, 0)
        assert first.cpu_name == "arm64"

    def test_fat_overrun_rejected(self):
        from paomcp.tools.backends.macho import FatMachOError, parse_fat_macho

        blob = struct.pack(">II", 0xCAFEBABE, 1) + struct.pack(
            ">IIIII", 0x0100000C, 0, 0x1000, 0x2000, 12
        )
        with pytest.raises(FatMachOError):
            parse_fat_macho(blob)

    def test_load_cap(self, tmp_path, monkeypatch):
        from paomcp.tools.backends import macho as macho_module
        from paomcp.tools.backends.macho import MachOError, load_macho

        path = tmp_path / "huge"
        path.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 32)
        monkeypatch.setattr(macho_module.os.path, "getsize", lambda p: 10**12)
        with pytest.raises(MachOError, match="cap"):
            load_macho(str(path))


class TestBtfCaps:
    def _blob(self, *, kind: int, vlen: int, type_len: int, str_len: int = 8) -> bytes:
        import struct as _struct

        header = _struct.pack(
            "<HBBIIIII", 0xEB9F, 1, 0, 24, 0, type_len, type_len, str_len
        )
        info = (kind << 24) | vlen
        rec = _struct.pack("<III", 0, info, 8)
        return header + rec + b"\x00" * 64 + b"\x00" * str_len

    def test_absurd_vlen_rejected(self):
        from paomcp.tools.backends.btf import BtfError, BtfFile

        # kind=STRUCT, vlen=0xFFFF
        with pytest.raises(BtfError, match="vlen"):
            BtfFile.parse(self._blob(kind=4, vlen=0xFFFF, type_len=64))

    def test_record_overrun_rejected(self):
        from paomcp.tools.backends.btf import BtfError, BtfFile

        # kind=ENUM vlen=8 needs 8*8=64 extra bytes but section holds 12
        with pytest.raises(BtfError, match="overruns"):
            BtfFile.parse(self._blob(kind=6, vlen=8, type_len=12))


class TestIdaQuota:
    def test_mutating_quota_rejects_past_cap(self):
        from paomcp.tools.backends.ida import IdaProMCP, MAX_IDA_MUTATING_CALLS_PER_DB

        ida = IdaProMCP()
        ida._mutating_counts["db1"] = MAX_IDA_MUTATING_CALLS_PER_DB
        rejected = ida._call("py_eval", {"code": "1+1", "database": "db1"})
        assert rejected["success"] is False
        assert "quota" in str(rejected.get("error") or "")

    def test_read_calls_uncapped_and_status_reports_usage(self):
        from paomcp.tools.backends.ida import IdaProMCP

        ida = IdaProMCP()
        assert ida._check_mutating_quota("decompile", {"addr": 1}) is None
        assert ida.mutating_usage() == {}


class TestWatchRebaseline:
    def test_second_check_is_quiet_after_change(self):
        from tests.test_regions import build_regions

        heap, gdb = build_regions()
        gdb.place(0x300000, b"\x90" * 256)
        armed = heap.arm_region_watch(address=hex(0x300000), length=256)
        watch_id = armed["parsed"]["watch_id"]
        gdb.place(0x300000, b"\x90" * 128 + b"\xcc" * 128)
        first = heap.check_region_watches()["parsed"]
        assert first["changed_count"] == 1
        assert first["watches"][0]["rebaselined"] is True
        second = heap.check_region_watches()["parsed"]
        assert second["changed_count"] == 0
        assert heap.disarm_region_watch(watch_id=watch_id)["parsed"]["removed"] is True


class TestBundleOverBudget:
    def test_huge_artifact_skipped_not_loaded(self, tmp_path):
        from tests.test_research_bundle import _FakeArtifactStore

        from paomcp.tools.backends.research import ResearchRegistry
        from paomcp.tools.backends.research_bundle import ResearchBundle

        registry = ResearchRegistry()
        store = _FakeArtifactStore(tmp_path / "store")
        (tmp_path / "store").mkdir(exist_ok=True)
        sid = registry.create_session_response(name="over-budget")["parsed"]["session"][
            "session_id"
        ]
        store.add("a" * 32, b"x" * 100)
        registry.link_evidence_response(session_id=sid, kind="artifact", ref="a" * 32)
        # Cap below the 100-byte artifact: the collector must skip it with
        # an over_budget warning *without reading the payload off disk*
        # (prove it by deleting the file first).
        (tmp_path / "store" / ("a" * 32 + ".dat")).unlink()
        bundle = ResearchBundle(registry, artifact_store=store, max_bundle_bytes=50)
        session = registry.get_session(sid)
        index, blobs, warnings = bundle._collect_artifact_payloads(session)
        assert blobs == {}
        assert index[0]["resolved"] is False
        assert index[0]["reason"] == "over_budget"
        assert any("over_budget" in w for w in warnings)


class TestRopExportGuards:
    def test_variable_name_injection_rejected(self):
        from paomcp.tools.backends.rop_chain import ChainValidator, ChainValidatorError

        with pytest.raises(ChainValidatorError, match="identifier"):
            ChainValidator().export_chain(
                chain=[{"kind": "target", "address": 0x401000}],
                format="python",
                variable_name="chain\nimport os",
            )

    def test_sequence_steps_use_distinct_gadgets(self):
        import os
        import tempfile

        import tests.test_rop_gadgets as helpers
        from paomcp.tools.backends.rop_gadgets import GadgetIndex

        records = [
            helpers._record(0x400100, ["pop rdi", "ret"]),
            helpers._record(0x400200, ["pop rdi", "nop", "ret"]),
        ]
        idx = GadgetIndex(loader=helpers._fake_loader(records))
        handle = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        handle.write(b"BIN")
        handle.close()
        try:
            parsed = idx.find_gadget_sequence(
                binary_path=handle.name, steps=["pop rdi", "pop rdi"]
            )["parsed"]
        finally:
            os.unlink(handle.name)
        assert parsed["verdict"] == "complete"
        addrs = [step["gadget"]["address"] for step in parsed["steps"]]
        assert addrs == [0x400100, 0x400200]
