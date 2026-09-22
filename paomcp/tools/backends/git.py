"""
Git repository management tools for Pao Tools

Provides tools for fetching specific versions of git repositories.
Useful for analyzing specific vulnerable versions of software or libraries.
"""

import logging
import os
import shutil
import tempfile
from typing import Dict, Any, Optional
import subprocess

logger = logging.getLogger(__name__)


def _is_strict_descendant(root: str, candidate: str) -> bool:
    """Whether ``candidate`` is inside ``root`` but is not ``root`` itself."""
    try:
        if os.path.commonpath([root, candidate]) != root:
            return False
    except ValueError:
        return False
    return candidate != root


def _redact_repo_url(repo_url: str) -> str:
    """Strip embedded credentials before a URL reaches an INFO log."""
    scheme, separator, remainder = repo_url.partition("://")
    if not separator:
        return repo_url
    userinfo, at_sign, host = remainder.rpartition("@")
    if not at_sign or not userinfo:
        return repo_url
    return f"{scheme}://***@{host}"


class GitTools:
    """Tools for git repository operations"""

    def __init__(self, workspace_dir: Optional[str] = None):
        """
        Initialize git tools.

        Sets up a workspace directory for cloned repositories.
        """
        self.workspace_dir = workspace_dir or os.path.join(
            tempfile.gettempdir(), "pao", "repos"
        )
        os.makedirs(self.workspace_dir, exist_ok=True)
        logger.info(f"Git workspace initialized at: {self.workspace_dir}")

    ALLOWED_SCHEMES = (
        "https://",
        "http://",
        "git@github.com:",
        "git@gitlab.com:",
        "ssh://git@",
    )

    def _validate_repo_url(self, repo_url: str) -> Optional[Dict[str, Any]]:
        lowered = repo_url.strip().lower()
        if lowered.startswith(("ext::", "fd::", "file://", "ftp://", "ftps://")):
            return {
                "success": False,
                "error": f"Blocked unsafe git URL scheme in '{repo_url}'; use https:// or explicit ssh",
                "type": "ValueError",
            }
        if not lowered.startswith(self.ALLOWED_SCHEMES):
            return {
                "success": False,
                "error": f"repo_url must start with one of {self.ALLOWED_SCHEMES}; got '{repo_url}'",
                "type": "ValueError",
            }
        return None

    def fetch_repo(
        self,
        repo_url: str,
        version: Optional[str] = None,
        target_dir: Optional[str] = None,
        shallow: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch a specific version of a git repository.

        :param repo_url: Git repository URL (https or ssh)
        :param version: Specific version to checkout (branch/tag/commit hash). If None, uses default branch
        :param target_dir: Target directory name. If None, derives from repo URL
        :param shallow: Whether to perform shallow clone (faster for large repos)
        :param overwrite: Whether to replace an existing target directory
        :returns: Dictionary with fetch results including path and status
        """
        rejected = self._validate_repo_url(repo_url)
        if rejected is not None:
            return rejected
        try:
            # Extract repo name if target_dir not specified
            if not target_dir:
                repo_name = repo_url.rstrip("/").split("/")[-1]
                if repo_name.endswith(".git"):
                    repo_name = repo_name[:-4]
                target_dir = repo_name

            # Containment: the repository must land inside the workspace so
            # overwrite=True can never rmtree caller-owned content outside it.
            # Use realpath so a symlinked workspace path cannot smuggle a
            # target outside the root, and require a *strict* descendant so
            # target_dir='.' (or a URL ending in '/.') cannot resolve to the
            # workspace root itself.
            workspace_root = os.path.realpath(self.workspace_dir)
            candidate = (
                os.path.realpath(os.path.join(workspace_root, target_dir))
                if not os.path.isabs(target_dir)
                else os.path.realpath(target_dir)
            )
            if not _is_strict_descendant(workspace_root, candidate):
                return {
                    "success": False,
                    "error": (
                        f"target_dir escapes the git workspace '{workspace_root}'"
                    ),
                    "path": candidate,
                    "type": "ValueError",
                }
            repo_path = candidate

            # Never destroy caller-owned workspace content implicitly.
            if os.path.exists(repo_path):
                if not overwrite:
                    return {
                        "success": False,
                        "error": "Target directory already exists",
                        "path": repo_path,
                        "type": "FileExistsError",
                    }
                # Re-check with a fresh realpath immediately before the
                # destructive call (mirrors the create_target pattern), so a
                # symlink swapped in after the first check cannot redirect the
                # deletion outside the workspace or back onto the root.
                verified = os.path.realpath(repo_path)
                if not _is_strict_descendant(workspace_root, verified):
                    return {
                        "success": False,
                        "error": (
                            f"target_dir escapes the git workspace "
                            f"'{workspace_root}' after symlink resolution"
                        ),
                        "path": verified,
                        "type": "ValueError",
                    }
                logger.info("Replacing repository at %s", verified)
                shutil.rmtree(verified)

            # Clone the repository.  A shallow clone has no history to
            # satisfy an arbitrary version ref, so when a version is
            # requested we always fetch the full history before checkout.
            clone_cmd = ["git", "clone"]
            if shallow and not version:
                clone_cmd.extend(["--depth", "1"])
            clone_cmd.extend([repo_url, repo_path])

            logger.info(f"Cloning repository: {_redact_repo_url(repo_url)}")
            logger.debug("Cloning repository (full URL): %s", repo_url)
            result = subprocess.run(
                clone_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for large repos
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": "Clone failed",
                    "stderr": result.stderr,
                    "command": " ".join(clone_cmd),
                }

            # Checkout specific version if requested
            checkout_info = None
            if version:
                logger.info(f"Checking out version: {version}")

                # A depth-1 clone cannot fetch arbitrary tags/SHAs, so
                # unshallow before fetching the specific ref. Every git
                # invocation carries a timeout so a wedged remote cannot
                # pin a worker slot forever.
                if shallow:
                    subprocess.run(
                        ["git", "fetch", "--unshallow", "origin"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                # First, fetch the specific ref if needed. The explicit --
                # separator blocks leading-dash option injection via version.
                fetch_result = subprocess.run(
                    ["git", "fetch", "origin", "--", version],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                # Try to checkout the version
                checkout_result = subprocess.run(
                    ["git", "checkout", "--", version],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if checkout_result.returncode != 0:
                    # Try as a remote branch
                    checkout_result = subprocess.run(
                        ["git", "checkout", "-b", version, "--", f"origin/{version}"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )

                if checkout_result.returncode != 0:
                    return {
                        "success": False,
                        "error": "Checkout failed",
                        "stderr": checkout_result.stderr,
                        "clone_success": True,
                        "path": repo_path,
                    }

                checkout_info = checkout_result.stdout.strip()

            # Get current commit info
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            current_commit = (
                commit_result.stdout.strip()
                if commit_result.returncode == 0
                else "unknown"
            )

            # Get branch info
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            current_branch = (
                branch_result.stdout.strip()
                if branch_result.returncode == 0
                else "unknown"
            )

            return {
                "success": True,
                "path": repo_path,
                "repo_url": repo_url,
                "version": version or "default",
                "current_commit": current_commit,
                "current_branch": current_branch,
                "checkout_info": checkout_info,
                "workspace": self.workspace_dir,
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Operation timed out", "timeout": 300}
        except Exception as e:
            logger.error(f"Git fetch error: {str(e)}")
            return {"success": False, "error": str(e), "type": type(e).__name__}

    def cleanup_workspace(self) -> Dict[str, Any]:
        """
        Clean up the git workspace directory.

        :returns: Dictionary with cleanup status
        """
        try:
            if os.path.exists(self.workspace_dir):
                shutil.rmtree(self.workspace_dir)
                os.makedirs(self.workspace_dir, exist_ok=True)
                return {
                    "success": True,
                    "message": "Workspace cleaned",
                    "path": self.workspace_dir,
                }
            else:
                return {
                    "success": True,
                    "message": "Workspace already clean",
                    "path": self.workspace_dir,
                }
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            return {"success": False, "error": str(e)}
