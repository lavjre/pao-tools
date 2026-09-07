"""P2/P3 completion tests: fuzz bridge, firmware, syzkaller, PA-lite, profiles."""

import os

import pytest

from paomcp.tools.backends.firmware import (
    carve_buffer,
    extract_firmware,
    extract_strings,
)
from paomcp.tools.backends.fuzz_cov import minimize_corpus, run_coverage_batch
from paomcp.tools.backends.regions import classify_pa_candidates
from paomcp.tools.backends.syzkaller import generate_syz_config, ingest_syz_crash
from paomcp.tools.profiles import suggest_profile_for_target


class TestCorpusMinimize:
    def test_dedups_and_caps(self):
        inputs = [
            {"data_hex": "41" * 64},
            {"data_hex": "41" * 64},
            {"data_hex": "42" * 16},
        ]
        out = minimize_corpus(inputs, max_files=10, max_bytes=1 << 20)
        assert out["kept"] == 2
        assert out["dropped_duplicates"] == 1
        assert len(out["sha256"]) == 2

    def test_budget_drops_largest(self):
        inputs = [{"data_hex": "41" * 100}, {"data_hex": "42" * 10}]
        out = minimize_corpus(inputs, max_files=10, max_bytes=50)
        assert out["kept"] == 1
        assert out["total_bytes"] == 10


class TestCoverageBatch:
    def test_unavailable_without_engine(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        result = run_coverage_batch(
            engine="afl",
            target_path=str(tmp_path / "t"),
            corpus_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            time_seconds=5,
        )
        assert result["parsed"]["verdict"] == "unavailable"

    def test_fake_runner_reports_crashes(self, tmp_path):
        def _fake(**kw):
            assert kw["engine"] == "afl"
            return {"crashes": ["/out/crashes/id:000000"]}

        result = run_coverage_batch(
            engine="afl",
            target_path="/workspace/chal",
            corpus_dir=str(tmp_path / "in"),
            output_dir=str(tmp_path / "out"),
            time_seconds=5,
            runner=_fake,
        )
        parsed = result["parsed"]
        assert parsed["verdict"] == "completed"
        assert parsed["crash_count"] == 1


class TestFirmwareCarve:
    def test_carve_finds_elf_and_gzip(self):
        blob = b"\x00" * 64 + b"\x7fELF\x02\x01\x01" + b"\x00" * 64 + b"\x1f\x8b\x08"
        hits = carve_buffer(blob)
        kinds = {h["kind"] for h in hits}
        assert "elf64-le" in kinds
        assert "gzip" in kinds
        assert hits == sorted(hits, key=lambda h: h["offset"])

    def test_strings_capped(self):
        blob = b"hello-world-123\x00short\x00abcdefgh\n"
        assert "hello-world-123" in extract_strings(blob)

    def test_extract_firmware_carve_only(self, tmp_path):
        img = tmp_path / "fw.bin"
        img.write_bytes(
            b"\x00" * 128 + b"\x7fELF\x01\x01\x01" + b"A" * 256 + b"version-9.9.9\x00"
        )
        out = extract_firmware(
            image_path=str(img), output_dir=str(tmp_path / "x"), timeout_seconds=60
        )
        parsed = out["parsed"]
        assert parsed["verdict"] == "extracted"
        assert parsed["elf_count"] >= 1
        assert parsed["image"]["sha256"]


class TestSyzkaller:
    def _paths(self, tmp_path):
        kernel_src = tmp_path / "src"
        kernel_src.mkdir()
        image = tmp_path / "bull.img"
        image.write_bytes(b"fake-image")
        key = tmp_path / "key"
        key.write_bytes(b"fake-key")
        work = tmp_path / "work"
        work.mkdir()
        return {
            "kernel_src": str(kernel_src),
            "image_path": str(image),
            "ssh_key_path": str(key),
            "workdir": str(work),
        }

    def test_config_renders(self, tmp_path):
        result = generate_syz_config(**self._paths(tmp_path))
        parsed = result["parsed"]
        assert parsed["verdict"] == "rendered"
        assert parsed["config"]["target"] == "linux/amd64"
        assert "syz-manager" in parsed["next"]

    def test_config_rejects_host_escape(self, tmp_path):
        with pytest.raises(ValueError):
            generate_syz_config(
                **self._paths(tmp_path), qemu_args="-virtfs local,path=/"
            )

    def test_config_rejects_bad_arch_and_missing_paths(self, tmp_path):
        paths = self._paths(tmp_path)
        with pytest.raises(ValueError):
            generate_syz_config(**{**paths, "target_arch": "mips"})
        with pytest.raises(ValueError):
            generate_syz_config(**{**paths, "image_path": str(tmp_path / "nope")})

    def test_ingest_null_deref(self):
        from tests.test_kernel_oops import NULL_DEREF

        result = ingest_syz_crash(log_text=NULL_DEREF)
        parsed = result["parsed"]
        assert parsed["fault_kind"] == "null_deref"
        assert parsed["fingerprint"]
        assert parsed["frames"] == ["kmem_cache_alloc", "vuln_ioctl"]

    def test_ingest_propagates_triage_failure(self):
        result = ingest_syz_crash(log_text="   ")
        assert result["parsed"]["verdict"] == "unresolvable"
        assert result["parsed"]["supported"] is False


class TestPaLite:
    def _mapping(
        self, start, end, perms="rw-p", file="[anon]", r=True, w=True, x=False
    ):
        return {
            "file": file,
            "start": f"0x{start:x}",
            "end": f"0x{end:x}",
            "permissions": perms,
            "readable": r,
            "writable": w,
            "executable": x,
        }

    def test_flags_super_page(self):
        maps = [self._mapping(0x7000000000, 0x7000000000 + 2 * 1024 * 1024)]
        cands = classify_pa_candidates(maps)
        assert len(cands) == 1
        assert cands[0]["super_pages"] == 1

    def test_skips_executable_and_tiny(self):
        maps = [
            self._mapping(0x1000, 0x2000),
            self._mapping(
                0x7000000000, 0x7000020000, perms="r-xp", r=True, w=False, x=True
            ),
        ]
        assert classify_pa_candidates(maps) == []

    def test_chrome_marker_boosts_confidence(self):
        maps = [
            self._mapping(
                0x7000000001, 0x7000000001 + 2 * 1024 * 1024, file="[anon:chrome]"
            ),
        ]
        cands = classify_pa_candidates(maps)
        assert cands and cands[0]["confidence"] == "medium"


class TestSuggestProfile:
    @pytest.mark.parametrize(
        "hints,expected",
        [
            ({"filename": "fw.bin", "goal": "unpack squashfs"}, "firmware"),
            ({"filename": "libSystem.B.dylib"}, "rev"),
            ({"filename": "vmlinux", "goal": "syzkaller"}, "kernel"),
            ({"filename": "chrome", "goal": "jit exploit"}, "chrome"),
            ({"goal": "bounty report cve"}, "bounty"),
            ({"filename": "chal", "goal": "flag"}, "ctf"),
            ({"goal": "tcache poison pwn"}, "pwn"),
            ({"goal": "something random"}, "full"),
        ],
    )
    def test_rules(self, hints, expected):
        assert suggest_profile_for_target(**hints)["profile"] == expected


class TestNewToolsRegistered:
    @pytest.mark.asyncio
    async def test_p2p3_tools_present(self):
        from paomcp.server import create_mcp

        mcp = create_mcp()
        tools = await mcp.list_tools(run_middleware=False)
        names = {t.name for t in tools}
        for expected in (
            "run_afl_batch",
            "minimize_corpus",
            "extract_firmware",
            "generate_syz_config",
            "ingest_syz_crash",
            "classify_pa_regions",
            "paocli_link_evidence",
            "suggest_profile",
        ):
            assert expected in names, expected
