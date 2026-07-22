"""Manual Linux smoke test for descriptor-safe workspace snapshots."""

import json
from pathlib import Path
import tempfile

from paomcp.utils.paths import resolve_workspace_file, stable_file_snapshot


def _expect_snapshot_rejected(source: str, runtime: str, workspace: str) -> None:
    try:
        with stable_file_snapshot(
            source,
            runtime,
            workspace_root=workspace,
        ):
            pass
    except (OSError, RuntimeError, ValueError):
        return
    raise RuntimeError("workspace-escaping replacement was accepted")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pao-path-smoke-") as temp_dir:
        root = Path(temp_dir)
        workspace = root / "workspace"
        runtime = root / "runtime"
        nested = workspace / "nested"
        outside = root / "outside"
        nested.mkdir(parents=True)
        outside.mkdir()
        source = nested / "target.elf"
        source.write_bytes(b"inside")
        (outside / "target.elf").write_bytes(b"outside")
        resolved = resolve_workspace_file(str(source), str(workspace))

        source.unlink()
        nested.rmdir()
        nested.symlink_to(outside, target_is_directory=True)
        _expect_snapshot_rejected(resolved, str(runtime), str(workspace))

        print(
            json.dumps(
                {
                    "success": True,
                    "parent_symlink_rejected": True,
                    "open_mode": "openat-no-follow",
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
