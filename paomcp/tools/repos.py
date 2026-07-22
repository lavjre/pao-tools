import os
from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import catch_errors, get_services, run_blocking
from paomcp.utils.paths import resolve_workspace_path


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def fetch_repo(
        repo_url: str,
        version: Optional[str] = None,
        target_dir: Optional[str] = None,
        shallow: bool = True,
        overwrite: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Fetch a git repository into /workspace.

        Args:
            repo_url: Repository URL (https or ssh).
            version: Branch/tag/commit to checkout (None = default branch).
            target_dir: Optional specific directory; defaults to a name derived from
                the URL. When provided, it must be inside the container under
                /workspace; relative paths resolve under /workspace.
            shallow: Whether to clone shallowly.
            overwrite: Replace an existing target directory when True. Defaults to False
                to protect workspace data.
        """
        services = get_services(ctx)
        tools = services.git_tools
        workspace_root = services.runtime_paths.workspace_root
        if target_dir:
            target_dir = resolve_workspace_path(
                target_dir,
                workspace_root=workspace_root,
                require_exists=False,
                kind="target_dir",
            )
        else:
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            target_dir = os.path.join(workspace_root, repo_name)
        result = await run_blocking(
            lambda: tools.fetch_repo(
                repo_url, version, target_dir, shallow, overwrite=overwrite
            )
        )
        return result
