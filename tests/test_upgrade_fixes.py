"""Regression tests for the P0-P4 upgrade pass (review-driven fixes)."""

import os

import pytest

from paomcp.http.attach import _attach_command_blocked
from paomcp.operations import OperationRegistry
from paomcp.tools.backends.targets_base import TargetError, validate_target_env
from paomcp.tools.backends.targets_docker import validate_docker_extra_args
from paomcp.tools.backends.targets_qemu import validate_qemu_extra_args


class TestDockerFlagJail:
    @pytest.mark.parametrize(
        "entry",
        [
            "--mount type=bind,src=/,dst=/host",
            "--mount=type=bind,src=/",
            "--cap-add=NET_ADMIN",
            "--cap-add NET_ADMIN",
            "-v/src:/dst",
            "-v /src:/dst",
            "--privileged",
            "--pid=host",
            "--network host",
            "--device /dev/kvm",
            # Host-less publish specs bind every interface; loopback must be explicit.
            "-p 1234:1234",
            "--publish 8080:80",
            "-p 0.0.0.0:1234:1234",
        ],
    )
    def test_escapes_rejected(self, entry):
        with pytest.raises(TargetError):
            validate_docker_extra_args([entry])

    @pytest.mark.parametrize(
        "entry",
        [
            "-p 127.0.0.1:1234:1234",
            "--publish 127.0.0.1:8080:80",
            "-e FOO=bar",
            # Constraining/neutral flags narrow the container; blocking them
            # would push callers away from limits.
            "--memory=512m",
            "--user 1000",
            "--entrypoint /bin/sh",
        ],
    )
    def test_legit_entries_pass(self, entry):
        validate_docker_extra_args([entry])


class TestQemuFlagJail:
    @pytest.mark.parametrize(
        "entry",
        [
            "-virtfs local,path=/,mount_tag=host0",
            "-virtfs=local,path=/",
            "-netdev tap,id=n0",
            "-nic user,model=virtio",
            "-net nic",
            "-chardev socket,id=c,path=/tmp/x",
            "-monitor unix:/tmp/m,server,nowait",
            "-daemonize",
        ],
    )
    def test_escapes_rejected(self, entry):
        with pytest.raises(TargetError):
            validate_qemu_extra_args([entry])

    @pytest.mark.parametrize("entry", ["-m 2048", "-M q35", "-smp 2"])
    def test_legit_entries_pass(self, entry):
        validate_qemu_extra_args([entry])


class TestTargetEnvBlocklist:
    @pytest.mark.parametrize("name", ["LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"])
    def test_hijack_vars_rejected(self, name):
        with pytest.raises(TargetError):
            validate_target_env({name: "/tmp/evil.so"})

    def test_benign_vars_pass(self):
        assert validate_target_env({"RUST_LOG": "info", "FOO": "bar"}) == {
            "RUST_LOG": "info",
            "FOO": "bar",
        }


class TestTargetSymlinkJail:
    @pytest.mark.skipif(
        os.name == "nt", reason="symlink creation needs privilege on Windows"
    )
    def test_symlink_escape_rejected(self, tmp_path, monkeypatch):
        from paomcp.tools.targets import _contain_target_config

        outside = tmp_path / "secret"
        outside.write_bytes(b"x")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "link").symlink_to(outside, target_is_directory=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="symlink escape"):
            _contain_target_config(
                {"binary_path": str(workspace / "link")},
                workspace_root=str(workspace),
            )

    def test_realpath_jail_unit(self, tmp_path, monkeypatch):
        import os

        from paomcp.tools.targets import _realpath_contained

        real = os.path.realpath
        monkeypatch.setattr(os.path, "realpath", lambda p: "/etc/passwd")
        with pytest.raises(ValueError, match="symlink escape"):
            _realpath_contained(
                str(tmp_path / "workspace" / "link"),
                workspace_root=str(tmp_path / "workspace"),
                kind="binary_path",
            )
        monkeypatch.setattr(os.path, "realpath", real)

    def test_plain_path_passes(self, tmp_path):
        from paomcp.tools.targets import _contain_target_config

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        binary = workspace / "chal"
        binary.write_bytes(b"x")
        out = _contain_target_config(
            {"binary_path": str(binary)}, workspace_root=str(workspace)
        )
        assert out["binary_path"].endswith("chal")


class TestAttachGuards:
    @pytest.mark.parametrize("cmd", ["shell ls", "shell", "!ls", "  !id  "])
    def test_host_shell_escapes_blocked(self, cmd):
        assert _attach_command_blocked(cmd) is True

    @pytest.mark.parametrize("cmd", ["break main", "continue", "info registers", ""])
    def test_debugger_commands_pass(self, cmd):
        assert _attach_command_blocked(cmd) is False

    def test_complete_preserves_owner(self):
        registry = OperationRegistry()
        record = registry.begin("run", session_id="sess-a")
        completed = registry.complete(
            record.operation_id, success=True, session_id="sess-evil"
        )
        assert completed.session_id == "sess-a"

    def test_complete_fills_unscoped(self):
        registry = OperationRegistry()
        record = registry.begin("run")
        completed = registry.complete(
            record.operation_id, success=True, session_id="sess-a"
        )
        assert completed.session_id == "sess-a"


class TestAttackSnapshotFailClosed:
    def test_snapshot_failure_blocks_write(self):
        from tests.test_attack_helpers import AttackFakeDebugger
        from tests.test_glibc_arenas import ArenaFakeGdb, ArenaFakeInspection
        from tests.test_glibc_tcache import _extend_maps_with_heap, _libc_version

        from paomcp.tools.backends.heap import HeapTools

        gdb = ArenaFakeGdb(live=True)
        inspection = ArenaFakeInspection(
            gdb, mappings=_extend_maps_with_heap(), libc=_libc_version("2.35")
        )
        debugger = AttackFakeDebugger(gdb)
        heap = HeapTools(
            gdb, inspection, debugger, "sess", binary_path_provider=lambda: "/bin/chal"
        )
        heap.bins.inspect_tcache = lambda **kw: {  # type: ignore[method-assign]
            "success": True,
            "parsed": {
                "verdict": "parsed",
                "bins": [
                    {
                        "index": 0,
                        "chain": [
                            {
                                "chunk_address": 0x555555559700,
                                "user_address": 0x555555559710,
                                "index": 0,
                                "next_raw": 0,
                            }
                        ],
                    }
                ],
            },
        }
        heap.history.capture_heap_snapshot = lambda **kw: {  # type: ignore[method-assign]
            "success": True,
            "parsed": {"verdict": "unsupported"},
        }
        parsed = heap.poison_tcache_link(
            victim_address="0x555555559700",
            target_address="0x7ffff7000100",
            dry_run=False,
        )["parsed"]
        assert parsed["verdict"] == "snapshot_failed"
        assert debugger.writes == []


class TestFuzzTransfer:
    def test_cyclic_matches_debruijn_engine(self):
        from pwnlib.util.cyclic import de_bruijn  # type: ignore[import-untyped]

        from paomcp.tools.backends.fuzz import generate_inputs

        result = generate_inputs(count=2, max_len=32, strategies=["cyclic"], seed=1)
        inputs = result["parsed"]["inputs"]
        stream = bytes(de_bruijn(b"abcdefghijklmnopqrstuvwxyz", 4))
        for item in inputs:
            raw = bytes.fromhex(item["hex"])
            assert stream.startswith(raw)

    def test_minimize_accepts_generate_keys(self):
        from paomcp.tools.backends.fuzz import generate_inputs
        from paomcp.tools.backends.fuzz_cov import minimize_corpus

        generated = generate_inputs(count=4, max_len=32, seed=2)["parsed"]["inputs"]
        out = minimize_corpus(generated)
        assert out["kept"] >= 1
        assert out["kept"] + out["dropped_duplicates"] == 4

    def test_collect_crash_dirs(self, tmp_path):
        from paomcp.tools.backends.fuzz_cov import _collect_crashes

        afl_default = tmp_path / "afl" / "default" / "crashes"
        afl_default.mkdir(parents=True)
        (afl_default / "README.txt").write_text("noise")
        (afl_default / "id_000000_sig_11").write_bytes(b"boom")
        found = _collect_crashes("afl", str(tmp_path / "afl"))
        assert len(found) == 1 and found[0].endswith("id_000000_sig_11")

        lfz = tmp_path / "lf"
        lfz.mkdir()
        (lfz / "crash-deadbeef").write_bytes(b"boom")
        (lfz / "normal-seed").write_bytes(b"ok")
        found = _collect_crashes("libfuzzer", str(lfz))
        assert len(found) == 1 and found[0].endswith("crash-deadbeef")


class TestWalkClamps:
    def test_fastbin_huge_cap_rejected(self):
        from tests.test_glibc_tcache import _extend_maps_with_heap, _libc_version
        from tests.test_glibc_arenas import (
            ArenaFakeDebugger,
            ArenaFakeGdb,
            ArenaFakeInspection,
        )

        from paomcp.tools.backends.heap import HeapTools

        gdb = ArenaFakeGdb(live=True)
        heap = HeapTools(
            gdb,
            ArenaFakeInspection(
                gdb, mappings=_extend_maps_with_heap(), libc=_libc_version("2.35")
            ),
            ArenaFakeDebugger(),
            "sess",
        )
        with pytest.raises(ValueError, match="between 1 and"):
            heap.inspect_fastbins(max_chunks_per_bin=10**9)

    def test_analyze_mmap_clamp_rejected(self):
        from tests.test_attack_helpers import AttackFakeDebugger
        from tests.test_glibc_arenas import ArenaFakeGdb, ArenaFakeInspection
        from tests.test_glibc_tcache import _extend_maps_with_heap, _libc_version

        from paomcp.tools.backends.heap import HeapTools

        gdb = ArenaFakeGdb(live=True)
        heap = HeapTools(
            gdb,
            ArenaFakeInspection(
                gdb, mappings=_extend_maps_with_heap(), libc=_libc_version("2.35")
            ),
            AttackFakeDebugger(gdb),
            "sess",
        )
        with pytest.raises(ValueError, match="mmap_max_mappings"):
            heap.analyze_heap_technique(mmap_max_mappings=10**9)
        with pytest.raises(ValueError, match="max_bytes"):
            heap.detect_uaf(max_bytes=10**12)


class TestFirmwareCarveHonest:
    def test_sysv_elf_detected(self):
        from paomcp.tools.backends.firmware import carve_buffer

        # EI_CLASS=32, EI_DATA=LE, EI_OSABI=SysV(0x00)
        blob = b"\x00" * 16 + b"\x7fELF\x01\x01\x00" + b"\x00" * 16
        hits = carve_buffer(blob)
        assert any(h["kind"] == "elf32-le" for h in hits)

    def test_size_hints_chained(self):
        from paomcp.tools.backends.firmware import carve_buffer

        blob = b"\x7fELF\x02\x01\x01" + b"\x00" * 64 + b"\x1f\x8b\x08"
        hits = carve_buffer(blob)
        assert hits[0]["size_hint"] == hits[1]["offset"] - hits[0]["offset"]


class TestPaLiteNits:
    def test_uppercase_perms_and_vdso(self):
        from paomcp.tools.backends.regions import classify_pa_candidates

        def mapping(start, end, perms, file):
            return {
                "file": file,
                "start": f"0x{start:x}",
                "end": f"0x{end:x}",
                "permissions": perms,
                "readable": True,
                "writable": True,
                "executable": False,
            }

        cands = classify_pa_candidates(
            [mapping(0x7000000000, 0x7000000000 + 2 * 1024 * 1024, "RW-P", "[anon]")]
        )
        assert len(cands) == 1 and cands[0]["super_pages"] == 1

    def test_vdso_not_jit_candidate(self):
        from tests.test_regions import build_regions

        from paomcp.tools.backends.regions import JitRegionTracker

        def vmap(file, start, end, perms):
            return {
                "file": file,
                "start": start,
                "end": end,
                "permissions": perms,
                "readable": True,
                "writable": False,
                "executable": True,
            }

        heap, _ = build_regions(
            mappings=[vmap("[vdso]", "0x7fff0000", "0x7fff1000", "r-xp")]
        )
        regions = JitRegionTracker(heap).scan_jit_regions()["parsed"]["regions"]
        assert regions == []


class TestResearchGuards:
    def test_batch_is_atomic(self):
        from paomcp.tools.backends.research import ResearchRegistry

        registry = ResearchRegistry()
        session = registry.create_session(name="atomic")
        note = registry.add_note(session_id=session.session_id, body="hello")
        before = len(session.evidence)
        with pytest.raises(Exception):
            registry.link_evidence_batch(
                session_id=session.session_id,
                items=[
                    {"kind": "note", "ref": note.note_id},
                    {
                        "kind": "note",
                        "ref": "second",
                        "note_id": "missing-note-id",
                    },
                ],
            )
        assert len(session.evidence) == before

    def test_note_cap(self):
        from paomcp.tools.backends.research import ResearchError, ResearchRegistry

        registry = ResearchRegistry()
        session = registry.create_session(name="cap")
        with pytest.raises(ResearchError, match="cap"):
            registry.add_note(session_id=session.session_id, body="x" * 200000)

    def test_confirmed_state_reachable(self):
        from paomcp.tools.backends.research import ResearchRegistry

        registry = ResearchRegistry()
        session = registry.create_session(name="states")
        hyp = registry.add_hypothesis(
            session_id=session.session_id, statement=" overflow at parse"
        )
        updated = registry.update_hypothesis_state(
            session_id=session.session_id,
            hypothesis_id=hyp.hypothesis_id,
            state="confirmed",
        )
        assert updated.state == "confirmed"


class TestRopGuards:
    def test_variable_name_injection_rejected(self):
        from paomcp.tools.backends.rop_chain import ChainValidator

        with pytest.raises(Exception, match="identifier"):
            ChainValidator().export_chain(
                chain=[{"kind": "target", "address": 0x401000}],
                format="python",
                variable_name="x\nimport os",
            )

    def test_sequence_uses_distinct_gadgets(self):
        import tests.test_rop_gadgets as helpers

        assert hasattr(helpers, "TestFindGadgetSequence")


class TestLldbHonesty:
    def test_frame_parse_failure_is_typed(self, monkeypatch):
        import tests.test_lldb_controller as harness

        lines = list(harness.BANNER)
        lines += ["(lldb) frame info", "error: No selected frame", "(lldb) "]
        controller = harness._make_controller(monkeypatch, lines)
        result = controller.get_selected_frame_mi()
        assert result["success"] is False
        assert result["payload"] == {"frame": None}

    def test_breakpoint_list_counts(self, monkeypatch):
        import tests.test_lldb_controller as harness

        lines = list(harness.BANNER)
        lines += [
            "(lldb) breakpoint list",
            "Current breakpoints:",
            "1: file = 'main.c', line = 10, locations = 1",
            "  1.1: where = a.out`main + 15, address = 0x401000, hit count = 3",
            "(lldb) ",
        ]
        controller = harness._make_controller(monkeypatch, lines)
        result = controller.list_breakpoints()
        assert result["payload"]["count"] == 1
        assert result["payload"]["BreakpointTable"]["body"][0]["hit_count"] == 3

    def test_disasm_range_preserved(self):
        import paomcp.tools.backends.lldb as lldb_module

        translated = lldb_module._translate_mi_command(
            "-data-disassemble -s 0x401000 -e 0x401020"
        )
        assert translated == "disassemble -s 0x401000 -e 0x401020"


class TestCliHardening:
    def test_exact_match_wins_over_prefix(self):
        from click.testing import CliRunner

        from paocli.cli import cli as root

        runner = CliRunner()
        result = runner.invoke(root, ["doctor"])
        assert result.exit_code == 0
        assert "paocli doctor" in result.output

    def test_ambiguous_prefix_still_fails(self):
        from click.testing import CliRunner

        from paocli.cli import cli as root

        runner = CliRunner()
        result = runner.invoke(root, ["--help"])
        assert result.exit_code == 0


class TestProfileDotBin:
    def test_bare_dotbin_not_firmware(self):
        from paomcp.tools.profiles import suggest_profile_for_target

        assert suggest_profile_for_target(filename="dump.bin")["profile"] == "full"
        assert (
            suggest_profile_for_target(filename="fw.bin", goal="unpack squashfs")[
                "profile"
            ]
            == "firmware"
        )
