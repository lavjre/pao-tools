"""MCP tools for the kernel SLUB slab analyzer.

Resolves ``struct kmem_cache`` / ``struct slab`` field offsets from BTF and
decodes the (optionally obfuscated) SLUB freelist.  Both tools are read-only:
they only ever read target memory -- nothing is written, so there is no
``idempotency_key`` and no mutation-set entry.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.operations import current_operation_id
from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_session_action,
)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def inspect_slub_cache(
        session_id: str,
        cache_address: str,
        btf_path: Optional[str] = None,
        page_size: int = 4096,
        byteorder: str = "little",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Inspect a kernel ``struct kmem_cache``: name, object size,
        freepointer offset, hardening, and slab geometry.

        ``cache_address`` is the address of a ``struct kmem_cache`` (e.g. a
        ``kmalloc-*`` cache) in a live kernel target.  Field offsets are
        resolved from BTF: by default ``/sys/kernel/btf/vmlinux`` on the
        host, or pass ``btf_path`` for a BTF blob file.  Read-only.  Returns
        the cache's size class, ``random`` value, hardening state, order, and
        objects-per-slab, plus the BTF source and resolved fields.
        ``byteorder`` is the target memory byte order (``"little"`` default;
        pass ``"big"`` for big-endian kernels like s390x / ppc64).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.inspect_slub_cache(
                cache_address=cache_address,
                btf_path=btf_path,
                page_size=page_size,
                byteorder=byteorder,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def decode_slub_freelist(
        session_id: str,
        cache_address: str,
        slab_address: str,
        slab_mem_address: Optional[str] = None,
        btf_path: Optional[str] = None,
        page_size: int = 4096,
        max_objects: Optional[int] = None,
        byteorder: str = "little",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Decode a SLUB slab's free-object freelist.

        ``cache_address`` is a ``struct kmem_cache``; ``slab_address`` is the
        address of the slab's ``struct slab`` (or ``struct page`` on older
        kernels) header.  ``slab_mem_address`` optionally names the base of
        the slab's page window; when omitted it is derived from the slab
        address, aligned to the slab's full extent (order-aware).  Object
        slots are XOR/swap-obfuscated when the cache is hardened
        (``CONFIG_SLAB_FREELIST_HARDENED``); both the 5.x legacy and >=5.7
        ``swab`` encodings and both raw and encoded head interpretations are
        auto-detected.  ``byteorder`` is the target memory byte order
        (``"little"`` default; ``"big"`` for big-endian kernels).  Returns the
        decoded object addresses, per-link evidence, and every candidate
        attempt.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.decode_slub_freelist(
                cache_address=cache_address,
                slab_address=slab_address,
                slab_mem_address=slab_mem_address,
                btf_path=btf_path,
                page_size=page_size,
                max_objects=max_objects,
                byteorder=byteorder,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def generate_syz_config(
        kernel_src: str,
        image_path: str,
        ssh_key_path: str,
        workdir: str,
        kernel_obj: Optional[str] = None,
        syzkaller_path: str = "/syzkaller",
        target_arch: str = "amd64",
        qemu_args: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Render a syz-manager.cfg for sidecar syzkaller runs (pure, no VM boot).

        Args:
            kernel_src: Kernel source checkout path.
            image_path: Bootable QEMU image path.
            ssh_key_path: Syzkaller SSH key path.
            workdir: Syzkaller workdir path.
            kernel_obj: Kernel build obj dir (default kernel_src).
            syzkaller_path: Syzkaller checkout path.
            target_arch: Syz target arch (default amd64).
            qemu_args: Extra QEMU args (host-escape flags rejected).
        """
        from paomcp.tools.backends.syzkaller import generate_syz_config as _gen
        from paomcp.tools.common import run_blocking as _blocking

        _ = ctx

        def _run() -> Dict[str, Any]:
            return _gen(
                kernel_src=kernel_src,
                kernel_obj=kernel_obj,
                image_path=image_path,
                ssh_key_path=ssh_key_path,
                workdir=workdir,
                syzkaller_path=syzkaller_path,
                target_arch=target_arch,
                qemu_args=qemu_args,
            )

        return await _blocking(_run)

    @mcp.tool()
    @catch_errors()
    async def ingest_syz_crash(
        log_text: Optional[str] = None,
        log_path: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Triage a syzkaller crash log via the kernel-oops pipeline (no debugger needed).

        Args:
            log_text: Crash log text (exactly one of log_text / log_path).
            log_path: Crash log file under the configured workspace.
        """
        from paomcp.tools.backends.syzkaller import ingest_syz_crash as _ingest
        from paomcp.tools.common import get_services as _services
        from paomcp.tools.common import run_blocking as _blocking

        services = _services(ctx)
        resolved_log = None
        if log_path is not None:
            from paomcp.utils.paths import resolve_workspace_file as _resolve_file

            resolved_log = _resolve_file(
                log_path, services.runtime_paths.workspace_root, kind="log_path"
            )

        def _run() -> Dict[str, Any]:
            return _ingest(log_text=log_text, log_path=resolved_log)

        return await _blocking(_run)


__all__ = ["register"]
