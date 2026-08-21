"""Tests for the pure SLUB freelist decode helpers (Stage 6.10).

The helpers are memory-agnostic: a ``reader`` callable supplies raw encoded
freepointers, which lets every walk/termination/detection path be exercised
without a live target.
"""

import pytest

from paomcp.tools.backends.slub import (
    SLUB_ENCODING_LEGACY,
    SLUB_ENCODING_SWAB,
    SLUB_HEAD_ENCODED,
    SLUB_HEAD_RAW,
    slub_decode_ptr,
    slub_detect_chain,
    slub_swab,
    slub_walk_chain,
)

RANDOM = 0xDEADBEEFCAFEBABE
OFFSET = 0x10
PB = 64

SLAB_MEM = 0x10000
SLAB_BYTES = 0x1000
SLAB_END = SLAB_MEM + SLAB_BYTES

OBJ0, OBJ1, OBJ2 = 0x10000, 0x10040, 0x10080


def enc(
    ptr: int, store: int, *, random: int = RANDOM, encoding: str = SLUB_ENCODING_SWAB
) -> int:
    """Encode a freepointer the way the kernel does (test-side)."""
    if encoding == SLUB_ENCODING_SWAB:
        return ptr ^ random ^ slub_swab(store, PB)
    if encoding == SLUB_ENCODING_LEGACY:
        return ptr ^ random ^ store
    return ptr


def make_reader(mem):
    return lambda addr: mem.get(addr)


def place_chain(
    mem,
    objects,
    *,
    random: int = RANDOM,
    encoding: str = SLUB_ENCODING_SWAB,
    offset: int = OFFSET,
):
    """Store the per-object links for ``objects`` ending in NULL."""
    for i, obj in enumerate(objects):
        nxt = objects[i + 1] if i + 1 < len(objects) else 0
        mem[obj + offset] = enc(nxt, obj + offset, random=random, encoding=encoding)


# ---------------------------------------------------------------------------
class TestSwab:
    def test_swab64_known_vector(self):
        assert slub_swab(0x1122334455667788, 64) == 0x8877665544332211

    def test_swab32_known_vector(self):
        assert slub_swab(0x11223344, 32) == 0x44332211

    def test_swab_is_involution(self):
        for value in (0, 1, 0x1122334455667788, 0xFFFFFFFFFFFFFFFF):
            assert slub_swab(slub_swab(value, 64), 64) == value

    def test_swab_round_trip_decode(self):
        store = OBJ0 + OFFSET
        encoded = enc(OBJ1, store)
        assert (
            slub_decode_ptr(
                encoded, RANDOM, store, PB, hardened=True, encoding=SLUB_ENCODING_SWAB
            )
            == OBJ1
        )

    def test_legacy_round_trip_decode(self):
        store = OBJ0 + OFFSET
        encoded = enc(OBJ1, store, encoding=SLUB_ENCODING_LEGACY)
        assert (
            slub_decode_ptr(
                encoded, RANDOM, store, PB, hardened=True, encoding=SLUB_ENCODING_LEGACY
            )
            == OBJ1
        )

    def test_unhardened_is_identity(self):
        assert (
            slub_decode_ptr(OBJ1, RANDOM, OBJ0, PB, hardened=False, encoding=None)
            == OBJ1
        )

    def test_decode_masks_to_pointer_width(self):
        # A 32-bit decode never leaks high garbage bits.
        r32 = 0xDEADBEEF
        store = 0x1000
        encoded = enc(0x10040, store, random=r32, encoding=SLUB_ENCODING_LEGACY)
        decoded = slub_decode_ptr(
            encoded, r32, store, 32, hardened=True, encoding=SLUB_ENCODING_LEGACY
        )
        assert decoded == 0x10040
        assert decoded < 2**32

    def test_unknown_encoding_raises(self):
        with pytest.raises(ValueError, match="unknown SLUB encoding"):
            slub_decode_ptr(1, RANDOM, 2, PB, hardened=True, encoding="bogus")


# ---------------------------------------------------------------------------
class TestWalk:
    def _walk(self, head, mem, *, head_is_encoded=False, **kw):
        args = dict(
            head=head,
            head_is_encoded=head_is_encoded,
            head_store_addr=0,
            cache_offset=OFFSET,
            cache_random=RANDOM,
            hardened=True,
            encoding=SLUB_ENCODING_SWAB,
            slab_mem=SLAB_MEM,
            slab_bytes=SLAB_BYTES,
            max_objects=64,
            pointer_bits=PB,
            reader=make_reader(mem),
        )
        args.update(kw)
        return slub_walk_chain(**args)

    def test_walk_full_chain_terminates_null(self):
        mem = {}
        place_chain(mem, [OBJ0, OBJ1, OBJ2])
        chain = self._walk(OBJ0, mem)
        assert chain.objects == [OBJ0, OBJ1, OBJ2]
        assert chain.termination == "null"
        assert chain.count == 3
        assert chain.chain[0].freepointer_address == OBJ0 + OFFSET
        assert chain.chain[0].decoded_next == OBJ1
        assert chain.chain[-1].decoded_next == 0

    def test_walk_null_head(self):
        chain = self._walk(0, {})
        assert chain.objects == []
        assert chain.termination == "null"

    def test_walk_cycle_detected(self):
        mem = {OBJ0 + OFFSET: enc(OBJ0, OBJ0 + OFFSET)}  # self-loop
        chain = self._walk(OBJ0, mem)
        assert chain.termination == "cycle"
        assert "loop" in chain.note

    def test_walk_max_objects(self):
        mem = {}
        place_chain(mem, [OBJ0, OBJ1, OBJ2])
        chain = slub_walk_chain(
            head=OBJ0,
            head_is_encoded=False,
            head_store_addr=0,
            cache_offset=OFFSET,
            cache_random=RANDOM,
            hardened=True,
            encoding=SLUB_ENCODING_SWAB,
            slab_mem=SLAB_MEM,
            slab_bytes=SLAB_BYTES,
            max_objects=2,
            pointer_bits=PB,
            reader=make_reader(mem),
        )
        assert chain.objects == [OBJ0, OBJ1]
        assert chain.termination == "max_objects"

    def test_walk_out_of_range_head(self):
        chain = self._walk(SLAB_END + 0x10, {})
        assert chain.objects == []
        assert chain.termination == "out_of_range"

    def test_walk_unreadable_slot(self):
        mem = {OBJ0 + OFFSET: enc(OBJ1, OBJ0 + OFFSET)}  # no OBJ1 slot
        chain = self._walk(OBJ0, mem)
        assert chain.objects == [OBJ0]
        assert chain.termination == "unreadable"

    def test_walk_encoded_head(self):
        mem = {}
        place_chain(mem, [OBJ1, OBJ2])
        chain = self._walk(
            enc(OBJ1, OBJ0), mem, head_is_encoded=True, head_store_addr=OBJ0
        )
        assert chain.objects == [OBJ1, OBJ2]
        assert chain.head_mode == SLUB_HEAD_ENCODED


# ---------------------------------------------------------------------------
class TestDetect:
    def _detect(self, mem, *, head=OBJ0, hardened=True, **kw):
        return slub_detect_chain(
            head=head,
            head_store_addr=0,
            cache_offset=OFFSET,
            cache_random=RANDOM,
            hardened=hardened,
            slab_mem=SLAB_MEM,
            slab_bytes=SLAB_BYTES,
            slab_objects=4,
            max_objects=64,
            pointer_bits=PB,
            reader=make_reader(mem),
            **kw,
        )

    def test_detect_swab_wins_over_legacy(self):
        mem = {}
        place_chain(mem, [OBJ0, OBJ1, OBJ2], encoding=SLUB_ENCODING_SWAB)
        best, attempts = self._detect(mem)
        assert best is not None
        assert best.head_mode == SLUB_HEAD_RAW
        assert best.encoding == SLUB_ENCODING_SWAB
        assert best.objects == [OBJ0, OBJ1, OBJ2]
        assert len(attempts) == 4
        # Attempts are evaluated in a deterministic raw-then-encoded, swab-
        # then-legacy order so the evidence list is stable across runs.
        assert [a.head_mode for a in attempts] == [
            SLUB_HEAD_RAW,
            SLUB_HEAD_RAW,
            SLUB_HEAD_ENCODED,
            SLUB_HEAD_ENCODED,
        ]
        assert [a.encoding for a in attempts] == [
            SLUB_ENCODING_SWAB,
            SLUB_ENCODING_LEGACY,
            SLUB_ENCODING_SWAB,
            SLUB_ENCODING_LEGACY,
        ]

    def test_detect_legacy_wins(self):
        mem = {}
        place_chain(mem, [OBJ0, OBJ1, OBJ2], encoding=SLUB_ENCODING_LEGACY)
        best, attempts = self._detect(mem)
        assert best.encoding == SLUB_ENCODING_LEGACY
        assert best.objects == [OBJ0, OBJ1, OBJ2]

    def test_detect_raw_head_with_per_object_swab(self):
        # 6.x model: raw decoded head, obfuscated object slots.
        mem = {}
        place_chain(mem, [OBJ1, OBJ2], encoding=SLUB_ENCODING_SWAB)
        best, _ = self._detect(mem, head=OBJ1)
        assert best.head_mode == SLUB_HEAD_RAW
        assert best.objects == [OBJ1, OBJ2]

    def test_detect_no_candidate_unresolved(self):
        mem = {0x11000: 0x12345678}  # nothing decodes inside the window
        best, attempts = self._detect(mem, head=0x11000)
        assert best is None
        assert len(attempts) == 4
        assert all(a.termination == "out_of_range" for a in attempts)

    def test_detect_unhardened_single_candidate(self):
        mem = {OBJ0 + OFFSET: OBJ1, OBJ1 + OFFSET: OBJ2, OBJ2 + OFFSET: 0}
        best, attempts = self._detect(mem, hardened=False)
        assert best is not None
        assert best.hardened is False
        assert best.encoding is None
        assert best.objects == [OBJ0, OBJ1, OBJ2]
        assert len(attempts) == 1

    def test_detect_tie_prefers_raw_over_encoded(self):
        # Both raw candidates decode head=OBJ0 to the same chain; the encoded
        # candidates decode to garbage.  The winner must be a raw candidate.
        mem = {}
        place_chain(mem, [OBJ0, OBJ1, OBJ2], encoding=SLUB_ENCODING_SWAB)
        best, _ = self._detect(mem, head=OBJ0)
        assert best.head_mode == SLUB_HEAD_RAW
        assert best.objects == [OBJ0, OBJ1, OBJ2]
