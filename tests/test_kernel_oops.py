from paomcp.tools.backends.kernel_oops import analyze_kernel_oops

NULL_DEREF = """\
[  123.456789] BUG: unable to handle kernel NULL pointer dereference at 0000000000000000
[  123.456790] #PF: supervisor write access in kernel mode
[  123.456791] #PF: error_code(0x0002) - not-present page
[  123.456792] Oops: 0002 [#1] SMP PTI
[  123.456793] CPU: 2 PID: 1234 Comm: vuln_mod Tainted: G           OE
[  123.456794] RIP: 0010:slab_alloc_node+0x3a4/0x4b0
[  123.456795] Call Trace:
[  123.456796]  kmem_cache_alloc+0x120/0x200
[  123.456797]  vuln_ioctl+0x45/0x90 [vuln]
"""

KASAN_UAF = """\
[   10.000001] BUG: KASAN: use-after-free in kfree+0x90/0x120
[   10.000002] Write of size 8 at addr ffff888012345678 by task syz-exec/42
[   10.000003] CPU: 0 PID: 42 Comm: syz-exec Tainted: G        W
[   10.000004] Call Trace:
[   10.000005]  kfree+0x90/0x120
[   10.000006]  timer_callback+0x22/0x60
"""


def test_null_deref_triage():
    result = analyze_kernel_oops(NULL_DEREF)
    assert result["success"] is True
    parsed = result["parsed"]
    assert parsed["fault_kind"] == "null_deref"
    assert parsed["fault_address"] == "0000000000000000"
    assert parsed["rip"] == {"symbol": "slab_alloc_node", "offset": "3a4"}
    assert parsed["call_trace"][:2] == ["kmem_cache_alloc", "vuln_ioctl"]
    assert parsed["cpu"] == {"cpu": "2", "pid": "1234", "comm": "vuln_mod"}
    assert parsed["oops_code"] == "0002"
    assert len(parsed["fingerprint"]) == 16


def test_kasan_uaf_triage():
    result = analyze_kernel_oops(KASAN_UAF)
    assert result["success"] is True
    parsed = result["parsed"]
    assert parsed["fault_kind"] == "kasan_uaf"
    assert parsed["fault_address"] == "ffff888012345678"
    assert parsed["kasan_access"] == {"kind": "write", "size": 8}
    assert "G" in (parsed["tainted"] or "")
    assert "W" in (parsed["tainted"] or "")


def test_fingerprint_stable_across_timestamps_and_aslr():
    first = analyze_kernel_oops(NULL_DEREF)["parsed"]["fingerprint"]
    shifted = NULL_DEREF.replace("123.456789", "999.000001").replace(
        "slab_alloc_node+0x3a4", "slab_alloc_node+0x3a4"
    )
    second = analyze_kernel_oops(shifted)["parsed"]["fingerprint"]
    assert first == second
    other = analyze_kernel_oops(KASAN_UAF)["parsed"]["fingerprint"]
    assert other != first


def test_unknown_log_still_fingerprints():
    result = analyze_kernel_oops("some random kernel line\nanother line\n")
    assert result["success"] is True
    assert result["parsed"]["fault_kind"] == "unknown"
    assert result["warnings"]


def test_rejects_empty_and_oversize():
    bad = analyze_kernel_oops("   ")
    assert bad["success"] is False
    big = analyze_kernel_oops("x" * (1024 * 1024 + 1))
    assert big["success"] is False
