"""MCP tools for lossless local ELF static analysis."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.backends.static_analysis import StaticBinaryAnalyzer
from paomcp.tools.common import catch_errors, get_services, run_blocking
from paomcp.operations import current_operation_id
from paomcp.services import AppServices
from paomcp.utils.paths import resolve_workspace_file, stable_file_snapshot

_ANALYZER = StaticBinaryAnalyzer()


def _stream_all_strings(
    services: AppServices,
    binary_path: str,
    *,
    min_length: int,
    refresh: bool,
    tool: str,
) -> Dict[str, Any]:
    descriptor, output_path = tempfile.mkstemp(
        prefix="static-strings-",
        suffix=".jsonl",
        dir=services.runtime_paths.runtime_root,
    )
    os.close(descriptor)
    operation_id = current_operation_id()
    cancellation_event = (
        services.operation_registry.cancellation_event(operation_id)
        if operation_id is not None
        else None
    )
    try:
        streamed = _ANALYZER.stream_strings_to_file(
            binary_path,
            output_path,
            min_length=min_length,
            cancellation_event=cancellation_event,
            minimum_free_bytes=services.artifact_store.min_free_bytes,
        )
        size_bytes = int(streamed["output_size_bytes"])
        if size_bytes <= services.artifact_store.inline_response_bytes:
            with open(output_path, "r", encoding="utf-8") as source:
                entries: Any = [json.loads(line) for line in source if line.strip()]
        else:
            artifact = services.artifact_store.save_file(
                output_path,
                tool=tool,
                content_type="application/x-ndjson",
                encoding="utf-8",
            )
            entries = {
                **artifact.to_dict(),
                "format": "ndjson",
                "record_schema": "pao.static-string.v1",
            }
        total_count = int(streamed["total_count"])
        public_streamed = {
            key: value for key, value in streamed.items() if key != "output_path"
        }
        return {
            **public_streamed,
            "cache": {
                **streamed["cache"],
                "refreshed": refresh,
            },
            "strings": {
                "min_length": min_length,
                "total_count": total_count,
                "returned_count": total_count,
                "limited_by_caller": False,
                "limit": None,
                "entries": entries,
            },
        }
    finally:
        try:
            os.unlink(output_path)
        except FileNotFoundError:
            pass


def _analyze_with_streamed_strings(
    services: AppServices,
    binary_path: str,
    *,
    min_string_length: int,
    refresh: bool,
    cancellation_event: Any = None,
) -> Dict[str, Any]:
    result = _ANALYZER.analyze(
        binary_path,
        include_strings=False,
        min_string_length=min_string_length,
        refresh=refresh,
        cancellation_event=cancellation_event,
    )
    streamed = _stream_all_strings(
        services,
        binary_path,
        min_length=min_string_length,
        refresh=refresh,
        tool="analyze_static_binary",
    )
    if streamed["fingerprint"]["sha256"] != result["fingerprint"]["sha256"]:
        raise RuntimeError("binary_path changed between static analysis passes")
    result["parsed"]["strings"] = streamed["strings"]
    result["cache"]["strings_hit"] = False
    result["cache"]["strings_streamed"] = True
    return result


def _resolve_workspace_file(raw_path: str, workspace_root: str, kind: str) -> str:
    """Compatibility wrapper around the shared regular-file resolver."""
    return resolve_workspace_file(raw_path, workspace_root, kind=kind)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def analyze_static_binary(
        binary_path: str,
        include_strings: bool = True,
        min_string_length: int = 4,
        string_limit: Optional[int] = None,
        refresh: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Build a typed, lossless static model for a workspace ELF binary.

        Args:
            binary_path: ELF under the configured workspace; relative paths resolve
                from the workspace root.
            include_strings: Include printable ASCII strings and exact file offsets.
            min_string_length: Minimum printable string length, from 1 to 65536.
            string_limit: Optional caller-selected result limit. Results are complete
                by default; when limited, counts explicitly report omitted entries.
            refresh: Bypass and replace immutable SHA-256 analysis cache entries.

        The result includes fingerprints, ELF/program/section headers, symbols,
        imports, exports, relocations, dynamic linkage, build ID, debug link, DWARF
        metadata, and optional strings. Large responses are delivered as artifacts.
        """
        services = get_services(ctx)
        resolved = _resolve_workspace_file(
            binary_path,
            services.runtime_paths.workspace_root,
            "binary_path",
        )

        def analyze_snapshot() -> Dict[str, Any]:
            operation_id = current_operation_id()
            cancellation_event = (
                services.operation_registry.cancellation_event(operation_id)
                if operation_id is not None
                else None
            )
            with stable_file_snapshot(
                resolved,
                services.runtime_paths.runtime_root,
                workspace_root=services.runtime_paths.workspace_root,
                minimum_free_bytes=services.artifact_store.min_free_bytes,
                cancellation_requested=(
                    cancellation_event.is_set
                    if cancellation_event is not None
                    else None
                ),
            ) as snapshot_path:
                if include_strings and string_limit is None:
                    result = _analyze_with_streamed_strings(
                        services,
                        snapshot_path,
                        min_string_length=min_string_length,
                        refresh=refresh,
                        cancellation_event=cancellation_event,
                    )
                else:
                    result = _ANALYZER.analyze(
                        snapshot_path,
                        include_strings=include_strings,
                        min_string_length=min_string_length,
                        string_limit=string_limit,
                        refresh=refresh,
                        cancellation_event=cancellation_event,
                    )
                result["binary_path"] = resolved
                return result

        return await run_blocking(analyze_snapshot)

    @mcp.tool()
    @catch_errors()
    async def extract_binary_strings(
        binary_path: str,
        min_length: int = 4,
        limit: Optional[int] = None,
        refresh: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Extract printable ASCII strings from an ELF with exact file offsets.

        Args:
            binary_path: ELF under the configured workspace.
            min_length: Minimum printable string length, from 1 to 65536.
            limit: Optional caller-selected limit; omitted means every match.
            refresh: Bypass the immutable SHA-256 string cache.

        Even when `limit` is supplied, the response reports the complete match count
        and marks `limited_by_caller`; no server-side truncation is silent.
        """
        services = get_services(ctx)
        resolved = _resolve_workspace_file(
            binary_path,
            services.runtime_paths.workspace_root,
            "binary_path",
        )

        def extract_snapshot() -> Dict[str, Any]:
            operation_id = current_operation_id()
            cancellation_event = (
                services.operation_registry.cancellation_event(operation_id)
                if operation_id is not None
                else None
            )
            with stable_file_snapshot(
                resolved,
                services.runtime_paths.runtime_root,
                workspace_root=services.runtime_paths.workspace_root,
                minimum_free_bytes=services.artifact_store.min_free_bytes,
                cancellation_requested=(
                    cancellation_event.is_set
                    if cancellation_event is not None
                    else None
                ),
            ) as snapshot_path:
                if limit is None:
                    result = _stream_all_strings(
                        services,
                        snapshot_path,
                        min_length=min_length,
                        refresh=refresh,
                        tool="extract_binary_strings",
                    )
                else:
                    result = _ANALYZER.extract_strings(
                        snapshot_path,
                        min_length=min_length,
                        limit=limit,
                        refresh=refresh,
                        cancellation_event=cancellation_event,
                    )
                result["binary_path"] = resolved
                return result

        return await run_blocking(extract_snapshot)

    @mcp.tool()
    @catch_errors()
    async def compare_binaries(
        left_path: str,
        right_path: str,
        refresh: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Compare two workspace ELF binaries using complete typed models.

        Args:
            left_path: Baseline ELF under the configured workspace.
            right_path: Comparison ELF under the configured workspace.
            refresh: Bypass cached models for both binaries.

        Differences cover hashes, headers, sections, imports, exports, dynamic
        dependencies, RPATH/RUNPATH, build IDs, DWARF presence, and record counts.
        """
        services = get_services(ctx)
        workspace_root = services.runtime_paths.workspace_root
        left = _resolve_workspace_file(left_path, workspace_root, "left_path")
        right = _resolve_workspace_file(right_path, workspace_root, "right_path")

        def compare_snapshots() -> Dict[str, Any]:
            operation_id = current_operation_id()
            cancellation_event = (
                services.operation_registry.cancellation_event(operation_id)
                if operation_id is not None
                else None
            )
            cancellation_requested = (
                cancellation_event.is_set if cancellation_event is not None else None
            )
            with stable_file_snapshot(
                left,
                services.runtime_paths.runtime_root,
                workspace_root=services.runtime_paths.workspace_root,
                minimum_free_bytes=services.artifact_store.min_free_bytes,
                cancellation_requested=cancellation_requested,
            ) as left_snapshot:
                with stable_file_snapshot(
                    right,
                    services.runtime_paths.runtime_root,
                    workspace_root=services.runtime_paths.workspace_root,
                    minimum_free_bytes=services.artifact_store.min_free_bytes,
                    cancellation_requested=cancellation_requested,
                ) as right_snapshot:
                    result = _ANALYZER.compare(
                        left_snapshot,
                        right_snapshot,
                        refresh=refresh,
                        cancellation_event=cancellation_event,
                    )
                    result["left"]["binary_path"] = left
                    result["right"]["binary_path"] = right
                    return result

        return await run_blocking(compare_snapshots)

    @mcp.tool()
    @catch_errors()
    async def extract_firmware(
        image_path: str,
        output_dir: Optional[str] = None,
        timeout_seconds: int = 600,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Extract + carve a firmware image (unblob/binwalk when present, pure carve always).

        Args:
            image_path: Firmware image under the configured workspace.
            output_dir: Extraction dir under the workspace (default auto).
            timeout_seconds: Overall budget (max 3600).
        """
        from paomcp.tools.backends.firmware import extract_firmware as _extract

        services = get_services(ctx)
        workspace_root = services.runtime_paths.workspace_root
        resolved_image = _resolve_workspace_file(
            image_path, workspace_root, "image_path"
        )
        resolved_out = None
        if output_dir is not None:
            from paomcp.utils.paths import resolve_workspace_path as _resolve_dir

            resolved_out = _resolve_dir(
                output_dir,
                workspace_root=workspace_root,
                require_exists=False,
                kind="corpus_dir",
            )

        def _run() -> Dict[str, Any]:
            return _extract(
                image_path=resolved_image,
                output_dir=resolved_out,
                timeout_seconds=timeout_seconds,
            )

        return await run_blocking(_run)
