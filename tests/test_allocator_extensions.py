"""Tests for allocator detection extensions (mallocng, scudo, plugin contract).

The glibc parser stack is complete; technique analysis extends the
:class:`AllocatorDiscovery` classifier to distinguish musl mallocng from
oldmalloc, recognizes Scudo, and locks in the plugin contract every
glibc-specific inspector already follows -- return an ``unsupported``
response with the honest allocator identity attached, never guess.
"""

from typing import Any, Dict

from paomcp.tools.backends import heap as heap_module
from paomcp.tools.backends.heap import (
    ALLOCATOR_RUNTIME_COMMANDS,
    MUSL_MALLOCNG_MIN_VERSION,
    TYPED_INSPECTION_FAMILIES,
    AllocatorDiscovery,
    HeapHistory,
)

from tests.test_heap_snapshots import (
    StubArenas,
    StubBins,
    StubDiscovery,
    StubGdb,
    _profile_64,
)


def mapping(
    file: str, start: str = "0x555555554000", end: str = "0x555555556000"
) -> Dict[str, Any]:
    return {
        "file": file,
        "start": start,
        "end": end,
        "permissions": "r-xp",
        "readable": True,
        "writable": False,
        "executable": True,
    }


# ---------------------------------------------------------------------------
# Scudo
# ---------------------------------------------------------------------------
class TestScudoDetection:
    def test_standalone_scudo_runtime_is_recognized(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/usr/lib/x86_64-linux-gnu/libclang_rt.scudo-x86_64.so")],
            None,
        )
        assert result["family"] == "scudo"
        assert result["safe_linking"] == "not_applicable"
        assert result["variant"] is None

    def test_libscudo_alternate_name_is_recognized(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/system/lib64/libscudo.so")], None
        )
        assert result["family"] == "scudo"

    def test_bionic_libc_is_treated_as_scudo(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/system/apex/com.android.runtime/bionic/lib64/libc.so")],
            None,
        )
        assert result["family"] == "scudo"

    def test_plain_libc_still_classified_as_glibc(self) -> None:
        """A bare ``libc.so`` under /lib is glibc-shaped and must not become scudo."""
        result = AllocatorDiscovery.classify(
            [mapping("/lib/x86_64-linux-gnu/libc.so.6")],
            {"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"},
        )
        assert result["family"] == "glibc"

    def test_runtime_commands_registry_includes_scudo(self) -> None:
        assert "scudo" in ALLOCATOR_RUNTIME_COMMANDS
        assert ALLOCATOR_RUNTIME_COMMANDS["scudo"] == ()


# ---------------------------------------------------------------------------
# Musl mallocng vs oldmalloc
# ---------------------------------------------------------------------------
class TestMuslVariant:
    def test_mallocng_gate_is_1_2(self) -> None:
        # documents the version gate this file relies on
        assert MUSL_MALLOCNG_MIN_VERSION == (1, 2)

    def test_musl_1_2_reports_mallocng(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/lib/ld-musl-x86_64.so.1")],
            {"libc": "/lib/ld-musl-x86_64.so.1", "version": "1.2.4"},
        )
        assert result["family"] == "musl"
        assert result["variant"] == "mallocng"

    def test_musl_pre_1_2_reports_oldmalloc(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/lib/ld-musl-x86_64.so.1")],
            {"libc": "/lib/ld-musl-x86_64.so.1", "version": "1.1.24"},
        )
        assert result["family"] == "musl"
        assert result["variant"] == "oldmalloc"

    def test_musl_without_version_reports_no_variant(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/lib/ld-musl-x86_64.so.1")], None
        )
        assert result["family"] == "musl"
        assert result["variant"] is None
        assert result["safe_linking"] == "not_applicable"

    def test_musl_exact_gate_version_is_mallocng(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/lib/ld-musl-x86_64.so.1")],
            {"libc": "/lib/ld-musl-x86_64.so.1", "version": "1.2.0"},
        )
        assert result["variant"] == "mallocng"


class TestGlibcVariant:
    def test_glibc_variant_is_ptmalloc2(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/lib/x86_64-linux-gnu/libc.so.6")],
            {"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"},
        )
        assert result["variant"] == "ptmalloc2"

    def test_glibc_variant_reported_without_version(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/lib/x86_64-linux-gnu/libc.so.6")], None
        )
        # ptmalloc2 is the only shipping glibc allocator today, so version
        # absence does not force ``variant`` to None.
        assert result["variant"] == "ptmalloc2"


class TestJemallocClassification:
    def test_jemalloc_versioned_object_matches(self) -> None:
        result = AllocatorDiscovery.classify(
            [
                mapping("/bin/chal"),
                mapping("/usr/lib/libjemalloc-5.3.0.so"),
            ],
            None,
        )
        assert result["family"] == "jemalloc"
        assert result["variant"] is None
        assert result["safe_linking"] == "not_applicable"


class TestLibmallocClassification:
    def test_libsystem_malloc_runtime_is_recognized(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/usr/lib/system/libsystem_malloc.dylib")], None
        )
        assert result["family"] == "libmalloc"
        assert result["variant"] is None

    def test_libsystem_malloc_versioned_dylib_is_recognized(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/usr/lib/system/libsystem_malloc.dylib.1")], None
        )
        assert result["family"] == "libmalloc"

    def test_libsystem_umbrella_is_recognized(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/usr/lib/libSystem.B.dylib")], None
        )
        assert result["family"] == "libmalloc"

    def test_libsystem_umbrella_versioned_is_recognized(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/usr/lib/libSystem.B.dylib.2")], None
        )
        assert result["family"] == "libmalloc"

    def test_plain_elf_libc_is_not_libmalloc(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/lib/x86_64-linux-gnu/libc.so.6")], None
        )
        assert result["family"] == "glibc"

    def test_runtime_commands_registry_includes_libmalloc(self) -> None:
        assert "libmalloc" in ALLOCATOR_RUNTIME_COMMANDS
        assert ALLOCATOR_RUNTIME_COMMANDS["libmalloc"] == ()

    def test_libmalloc_is_in_typed_inspection_families(self) -> None:
        assert "libmalloc" in TYPED_INSPECTION_FAMILIES


class TestUnknownClassification:
    def test_unknown_family_has_no_variant(self) -> None:
        result = AllocatorDiscovery.classify(
            [mapping("/bin/chal")], None  # no libc-bearing mapping at all
        )
        assert result["family"] == "unknown"
        assert result["variant"] is None


# ---------------------------------------------------------------------------
# Plugin contract: every glibc-only inspector must return an honest
# ``unsupported`` response when the allocator is not glibc.  Snapshots capture
# the same contract by storing an ``unsupported`` snapshot rather than
# guessing an empty ledger.  This locks that behavior in for scudo.
# ---------------------------------------------------------------------------
class TestPluginContract:
    def test_scudo_snapshot_is_unsupported_not_empty(self, monkeypatch) -> None:
        scudo_classification = {
            "family": "scudo",
            "variant": None,
            "version": None,
            "confidence": "likely",
            "safe_linking": "not_applicable",
        }
        monkeypatch.setattr(
            heap_module.AllocatorDiscovery,
            "classify",
            staticmethod(lambda mappings, libcinfo: dict(scudo_classification)),
        )
        history = HeapHistory(
            StubGdb(live=True),
            None,
            None,
            "sess",
            discovery=StubDiscovery(scudo_classification, _profile_64()),
            arenas=StubArenas(),
            bins=StubBins(),
        )
        response = history.capture_heap_snapshot(name="scudo-baseline")
        parsed = response["parsed"]
        # Contract: an unsupported allocator produces an "unsupported"
        # verdict with an empty ledger AND surfaces the honest family
        # identity -- the tool must not silently claim the heap is empty.
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"
        assert parsed["allocator"]["family"] == "scudo"
        assert parsed["allocator"]["variant"] is None
        assert parsed["chunk_ledger"] == []
        # The unsupported snapshot is still retained under the given name
        # so a subsequent diff request produces a typed unsupported
        # response instead of a hard "snapshot not found" error.
        listing = history.list_heap_snapshots()["parsed"]
        assert listing["snapshots"][0]["allocator_family"] == "scudo"
