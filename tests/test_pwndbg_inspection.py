import pytest

from pwnomcp.tools.backends.inspection import PwndbgInspection

CHECKSEC_OUTPUT = """\
File:     /workspace/chal
Arch:     amd64
RELRO:      Full RELRO
Stack:      Canary found
NX:         NX enabled
PIE:        PIE enabled
FORTIFY:    Disabled
SHSTK:      Enabled
IBT:        Enabled
"""

SECTIONS_OUTPUT = """\
             Start                End     Perm       Size  Name
             0x318              0x334      R--       0x1c  .interp
            0x4020             0x46b0      R-X      0x690  .plt
           0x22c38            0x23000      RW-      0x3c8  .got
"""

GOT_OUTPUT = """\
State of the GOT of /workspace/chal:
GOT protection: Full RELRO | Found 2 GOT entries passing the filter
[0x555555557fc0] puts@GLIBC_2.2.5 -> 0x7ffff7e2a980
[0x555555557fc8] free@GLIBC_2.2.5 -> 0
"""

PLT_OUTPUT = """\
Section .plt 0x1020 - 0x1040:
0x1030: puts@plt
Section .plt.sec 0x1040 - 0x1050:
0x1040: free@plt
"""

VMMAP_OUTPUT = """\
               Start                End Perm     Size  Offset File
►     0x555555554000     0x555555555000 r--p     1000       0 /workspace/chal
►     0x555555555000     0x555555556000 r-xp     1000    1000 /workspace/chal
      0x7ffffffde000     0x7ffffffff000 rw-p    21000       0 [stack]
"""

THREADS_OUTPUT = """\
  global_num  name      status    pc              symbol
------------  --------  --------  --------------  -------------------
           1  chal      stopped   0x555555555149  main+4
           2  worker    running   0x7ffff7e7a000  clone+32

Showing 2 of 2 threads.
"""

PROCINFO_OUTPUT = """\
exe        '/workspace/chal'
cmdline    /workspace/chal --mode test
cwd        '/workspace'
pid        42
tid        42
ppid       1
uid        [1000, 1000, 1000, 1000]
gid        [1000, 1000, 1000, 1000]
groups     [1000, 27]
fd[0]      /dev/null
fd[1]      pipe:[1234]
"""

LIBCINFO_OUTPUT = """\
libc: glibc
libc version: 2.39
linked: dynamically
URLs:
    project homepage:       https://sourceware.org/glibc/
    read the source:        not available
Mappings:
    libc is at:             0x7ffff7c00000
           /usr/lib/x86_64-linux-gnu/libc.so.6
    ld is at:               0x7ffff7fc0000
           /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
Symbolication:
    has exported symbols:  yes
    has internal symbols:  no
    has debug info:        yes
"""

LINKMAP_OUTPUT = """\
Node           Objfile                                Load Bias       Dynamic Segment
0x7ffff7ffd2e0 <Unknown, likely /workspace/chal>       0x555555554000  0x555555557dd8
0x7ffff7ffd8c0 /usr/lib/x86_64-linux-gnu/libc.so.6     0x7ffff7c00000  0x7ffff7dd1b60
"""

XINFO_OUTPUT = """\
Extended information for virtual address 0x555555555149:

  Containing mapping:
    0x555555555000     0x555555556000 r-xp     1000    1000 /workspace/chal

  Offset information:
         Mapped Area 0x555555555149 = 0x555555555000 + 0x149
         File (Base) 0x555555555149 = 0x555555554000 + 0x1149
         File (Disk) 0x555555555149 = /workspace/chal + 0x1149

 Containing ELF sections:
               .text 0x555555555149 = 0x555555555040 + 0x109
"""


class FakeGdb:
    def __init__(self, failures=None):
        self.calls = []
        self.timeouts = []
        self.failures = set(failures or [])
        self.outputs = {
            'checksec -f "/workspace/chal"': CHECKSEC_OUTPUT,
            "elfsections --no-rebase": SECTIONS_OUTPUT,
            "got --show-readonly --all puts": GOT_OUTPUT,
            "got": GOT_OUTPUT,
            "plt --all-symbols": PLT_OUTPUT,
            "plt": PLT_OUTPUT,
            "vmmap --writable /workspace/chal": VMMAP_OUTPUT,
            "threads 2": THREADS_OUTPUT,
            "procinfo": PROCINFO_OUTPUT,
            "libcinfo": LIBCINFO_OUTPUT,
            "linkmap": LINKMAP_OUTPUT,
            "xinfo $pc": XINFO_OUTPUT,
        }

    def execute_command(self, command, timeout_sec=7.0):
        self.calls.append(command)
        self.timeouts.append(timeout_sec)
        success = command in self.outputs and command not in self.failures
        output = self.outputs.get(command, f"Undefined command: {command}\n")
        return {
            "success": success,
            "state": "stopped",
            "responses": [{"type": "console", "payload": output}],
        }

    def get_state(self):
        return "stopped"


def test_checksec_returns_normalized_mitigations_and_raw_evidence():
    adapter = PwndbgInspection(FakeGdb())

    result = adapter.checksec("/workspace/chal")

    assert result["success"] is True
    assert result["parsed"]["arch"] == "amd64"
    assert result["parsed"]["mitigations"]["relro"]["level"] == "full"
    assert result["parsed"]["mitigations"]["canary"]["enabled"] is True
    assert result["parsed"]["mitigations"]["fortify"]["enabled"] is False
    assert "Full RELRO" in result["raw_output"]
    assert result["timeout_seconds"] == 60
    assert adapter.gdb.timeouts == [60]


def test_elf_sections_preserve_hex_addresses_and_permissions():
    adapter = PwndbgInspection(FakeGdb())

    result = adapter.elf_sections()

    assert result["parsed"]["count"] == 3
    text = result["parsed"]["sections"][1]
    assert text == {
        "name": ".plt",
        "start": "0x4020",
        "end": "0x46b0",
        "size": "0x690",
        "permissions": "R-X",
        "readable": True,
        "writable": False,
        "executable": True,
    }


def test_got_plt_aggregate_parses_entries_and_options():
    gdb = FakeGdb()
    adapter = PwndbgInspection(gdb)

    result = adapter.inspect_got_plt(
        symbol_filter="puts",
        include_readonly=True,
        all_objects=True,
        all_plt_symbols=True,
    )

    assert result["success"] is True
    assert result["complete"] is True
    assert result["parsed"]["got"]["count"] == 2
    assert result["parsed"]["got"]["entries"][0]["symbol"] == "puts@GLIBC_2.2.5"
    assert result["parsed"]["plt"]["entry_count"] == 1
    assert result["parsed"]["plt"]["sections"][0]["entries"][0]["symbol"] == "puts@plt"
    assert gdb.calls == ["got --show-readonly --all puts", "plt --all-symbols"]


def test_got_plt_returns_partial_result_when_runtime_got_is_unavailable():
    adapter = PwndbgInspection(FakeGdb(failures={"got"}))

    result = adapter.inspect_got_plt()

    assert result["success"] is True
    assert result["complete"] is False
    assert result["component_status"] == {"got": False, "plt": True}
    assert "got" in result["warnings"][0]


def test_vmmap_parses_permissions_and_validates_filter():
    adapter = PwndbgInspection(FakeGdb())

    result = adapter.memory_maps("/workspace/chal", writable=True)

    assert result["parsed"]["count"] == 3
    assert result["parsed"]["mappings"][1]["executable"] is True
    assert result["parsed"]["mappings"][2]["file"] == "[stack]"
    with pytest.raises(ValueError):
        adapter.memory_maps("$pc\nquit")


def test_threads_and_process_info_return_typed_records():
    adapter = PwndbgInspection(FakeGdb())

    threads = adapter.threads(limit=2)
    process = adapter.process_info()

    assert threads["parsed"]["total"] == 2
    assert threads["parsed"]["threads"][0]["symbol"] == "main+4"
    assert process["parsed"]["pid"] == 42
    assert process["parsed"]["uid"] == [1000, 1000, 1000, 1000]
    assert process["parsed"]["file_descriptors"]["1"] == "pipe:[1234]"


def test_libc_info_and_link_map_are_aggregated():
    adapter = PwndbgInspection(FakeGdb())

    result = adapter.inspect_libc()

    libc = result["parsed"]["libc"]
    assert libc["libc"] == "glibc"
    assert libc["version"] == "2.39"
    assert libc["mappings"]["libc"]["base"] == "0x7ffff7c00000"
    assert libc["symbolication"]["has_debug_info"] is True
    assert result["parsed"]["link_map"]["count"] == 2
    assert (
        "likely /workspace/chal"
        in result["parsed"]["link_map"]["entries"][0]["objfile"]
    )


def test_xinfo_parses_mapping_offsets_and_sections():
    adapter = PwndbgInspection(FakeGdb())

    result = adapter.address_info("$pc")

    parsed = result["parsed"]
    assert parsed["address"] == "0x555555555149"
    assert parsed["mapping"]["file"] == "/workspace/chal"
    assert parsed["offsets"]["file_(disk)"]["offset"] == "0x1149"
    assert parsed["sections"][0]["name"] == ".text"


def test_inspection_rejects_command_injection_and_invalid_limits():
    adapter = PwndbgInspection(FakeGdb())

    with pytest.raises(ValueError):
        adapter.got("puts; shell id")
    with pytest.raises(ValueError):
        adapter.address_info("$pc;quit")
    with pytest.raises(ValueError):
        adapter.threads(limit=0)
