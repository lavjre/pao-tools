"""Manual smoke test for the pinned Ghidra runtime in the Docker image."""

import json
from pathlib import Path
import subprocess
import tempfile

from paomcp.artifacts import ArtifactStore
from paomcp.tools.backends.decompiler import GhidraDecompiler


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pao-ghidra-smoke-") as job_dir:
        root = Path(job_dir)
        source_path = root / "smoke.c"
        binary_path = root / "smoke"
        source_path.write_text(
            "int helper(int value) { return value + 1; }\n"
            "int main(void) { return helper(41) == 42 ? 0 : 1; }\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["gcc", "-g", "-O0", str(source_path), "-o", str(binary_path)],
            check=True,
        )

        artifact_store = ArtifactStore(
            str(root / "artifacts"),
            quota_bytes=0,
            min_free_bytes=0,
            cleanup_interval_seconds=0,
        )
        backend = GhidraDecompiler(
            runtime_dir=str(root / "runtime"),
            artifact_store=artifact_store,
            max_cache_entries=2,
        )
        try:
            first = backend.decompile(
                str(binary_path), function="main", timeout=180, function_timeout=30
            )
            if not first.get("success"):
                raise RuntimeError(f"Ghidra decompilation failed: {first}")
            model = first.get("model")
            if not isinstance(model, dict) or not model.get("functions"):
                raise RuntimeError(f"Structured function model is missing: {first}")
            second = backend.decompile(
                str(binary_path), function="main", timeout=180, function_timeout=30
            )
            if not second.get("cache", {}).get("hit"):
                raise RuntimeError(f"Ghidra content cache did not replay: {second}")
            print(
                json.dumps(
                    {
                        "success": True,
                        "function_count": first["function_count"],
                        "schema_version": model["schema_version"],
                        "cache_replayed": True,
                    },
                    sort_keys=True,
                )
            )
        finally:
            artifact_store.close()


if __name__ == "__main__":
    main()
