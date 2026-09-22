import inspect
from pathlib import Path
import sys

from paomcp.__main__ import _parse_args
from paomcp.runtime import run_http

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE5_MCP_TOOLS = {
    "run_until_crash",
    "capture_crash_snapshot",
    "analyze_crash",
    "classify_crash",
    "compare_crashes",
    "find_cyclic_offset",
    "capture_core_dump",
    "generate_crash_report",
}


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_stage5_smoke_is_present_in_the_docker_context():
    dockerignore = _read(".dockerignore")
    dockerfile = _read("Dockerfile")

    assert "!tests/docker_stage5_smoke.py" in dockerignore
    assert (
        "COPY tests/docker_stage5_smoke.py ./tests/docker_stage5_smoke.py" in dockerfile
    )


def test_stage6_9_smoke_and_one_gadget_are_pinned_in_the_docker_image():
    dockerignore = _read(".dockerignore")
    dockerfile = _read("Dockerfile")

    assert "!tests/docker_stage6_9_smoke.py" in dockerignore
    assert (
        "COPY tests/docker_stage6_9_smoke.py ./tests/docker_stage6_9_smoke.py"
        in dockerfile
    )
    assert "ARG ONE_GADGET_VERSION=1.10.0" in dockerfile
    assert (
        'gem install --no-document one_gadget -v "${ONE_GADGET_VERSION}"' in dockerfile
    )
    assert "one_gadget --version" in dockerfile


def test_pwndbg_and_pwninit_downloads_are_pinned_and_verified():
    dockerfile = _read("Dockerfile")

    assert "ARG PWNDBG_VERSION=2026.02.18" in dockerfile
    assert (
        'ARG PWNDBG_SHA256_X86_64="eeb93972d7910bf8233abf296b00577efb7137d94655502985566a328e5cecce"'
        in dockerfile
    )
    assert (
        'ARG PWNDBG_SHA256_ARM64="8d74a4be037281aaa37b51a046a625d3055a87ebb2397553292bcae321be169f"'
        in dockerfile
    )
    assert "${PWNDBG_SHA256}  /tmp/pwndbg.tar.xz" in dockerfile
    assert "ARG PWNINIT_VERSION=3.3.1" in dockerfile
    assert 'ARG PWNINIT_SHA256="' in dockerfile
    assert "releases/download/${PWNINIT_VERSION}/pwninit" in dockerfile
    assert "${PWNINIT_SHA256}  /usr/local/bin/pwninit" in dockerfile
    assert "pwninit --version" in dockerfile


def test_helper_venv_is_baked_outside_the_runtime_volume():
    dockerfile = _read("Dockerfile")
    service = _read("paomcp.service")

    assert "PAO_PYTHON_VENV=/opt/pao-python/shared_venv" in dockerfile
    assert "PAO_PYTHON_VENV=/opt/pao-python/shared_venv" in service
    assert "/var/lib/pao/python/shared_venv" not in dockerfile


def test_local_http_defaults_to_loopback(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["paomcp"])

    assert _parse_args().host == "127.0.0.1"
    assert inspect.signature(run_http).parameters["host"].default == "127.0.0.1"


def test_container_explicitly_binds_internally_for_port_forwarding():
    dockerfile = _read("Dockerfile")

    assert 'CMD ["--host", "0.0.0.0"]' in dockerfile


def test_compose_is_loopback_only_and_checks_http_readiness():
    compose = _read("docker-compose.yml")

    assert not compose.lstrip().startswith("version:")
    assert "platform: linux/amd64" in compose
    assert '"127.0.0.1:5500:5500"' in compose
    assert "http://127.0.0.1:5500/healthz" in compose
    assert '["CMD", "pgrep"' not in compose
    # Compose boots the image entrypoint on 0.0.0.0, so the fail-closed
    # remote-bind gate needs both env values or the container exits.
    assert "PAO_ALLOW_REMOTE=1" in compose
    assert "PAO_MCP_TOKEN=${PAO_MCP_TOKEN:?" in compose


def test_systemd_unit_supplies_remote_bind_env_from_a_token_file():
    service = _read("paomcp.service")
    dockerfile = _read("Dockerfile")

    assert "Environment=PAO_ALLOW_REMOTE=1" in service
    assert "EnvironmentFile=/etc/pao/pao.env" in service
    # The image must create that file, otherwise the unit crash-loops on
    # boot as shipped.
    assert "/etc/pao/pao.env" in dockerfile
    assert "PAO_MCP_TOKEN=" in dockerfile
    assert "chmod 600 /etc/pao/pao.env" in dockerfile


def test_publish_serializes_repository_and_pushes_the_smoke_tested_image():
    workflow = _read(".github/workflows/publish-ghcr.yml")

    assert "group: publish-ghcr-${{ github.repository }}" in workflow
    assert "group: publish-ghcr-${{ github.ref }}" not in workflow
    assert "cancel-in-progress: false" in workflow
    build_index = workflow.index("Build exact image for smoke and publication")
    boot_index = workflow.index("Boot default entrypoint and probe health")
    stage5_index = workflow.index("Run stage5 smoke test")
    stage6_9_index = workflow.index("Run stage6-9 smoke test")
    login_index = workflow.index("Log in to GHCR")
    push_index = workflow.index("Push the smoke-tested image")
    assert (
        build_index
        < boot_index
        < stage5_index
        < stage6_9_index
        < login_index
        < push_index
    )
    assert "docker push --all-tags" in workflow


def test_cloud_build_gates_publication_on_verify_boot_and_stage5_smoke():
    cloudbuild = _read("cloudbuild.yaml")

    verify_index = cloudbuild.index("id: verify-python")
    build_index = cloudbuild.index("id: build-candidate")
    boot_index = cloudbuild.index("id: boot-default-entrypoint")
    stage5_index = cloudbuild.index("id: stage5-runtime-smoke")
    stage6_9_index = cloudbuild.index("id: stage6-9-runtime-smoke")
    tag_index = cloudbuild.index("id: tag-verified-image")
    images_index = cloudbuild.index("\nimages:")

    assert (
        verify_index
        < build_index
        < boot_index
        < stage5_index
        < stage6_9_index
        < tag_index
    )
    assert tag_index < images_index
    assert "uv run pytest -v" in cloudbuild
    assert "--platform=linux/amd64" in cloudbuild
    assert "pao-tools:cloudbuild-$BUILD_ID" in cloudbuild
    assert "curl -fsS http://127.0.0.1:5500/healthz" in cloudbuild
    assert '"http://127.0.0.1:5500/mcp"' in cloudbuild
    assert "tools = await client.list_tools()" in cloudbuild
    assert 'docker exec -i "$${BOOT_CONTAINER}"' in cloudbuild
    # The container binds 0.0.0.0, so the boot step supplies the gate env
    # and the in-container smoke client authenticates the /mcp call.
    assert "-e PAO_ALLOW_REMOTE=1" in cloudbuild
    assert "-e PAO_MCP_TOKEN=pao-cloudbuild-smoke-token" in cloudbuild
    assert "from fastmcp.client.auth import BearerAuth" in cloudbuild
    assert 'auth=BearerAuth(os.environ["PAO_MCP_TOKEN"])' in cloudbuild
    for tool_name in STAGE5_MCP_TOOLS:
        assert f'"{tool_name}"' in cloudbuild
    assert "/app/tests/docker_stage5_smoke.py" in cloudbuild
    assert "/app/tests/docker_stage6_9_smoke.py" in cloudbuild
    assert "trap cleanup EXIT" in cloudbuild
    assert "pao-tools-cloud-boot-$BUILD_ID" in cloudbuild
    assert "pao-cloud-stage5-runtime-$BUILD_ID" in cloudbuild
    assert "pao-cloud-stage6-9-runtime-$BUILD_ID" in cloudbuild


def test_local_build_helper_only_advertises_loopback_http():
    helper = _read("docker-build.local.sh")

    assert "docker run -p 127.0.0.1:5500:5500" in helper
    assert "docker run -p 5500:5500" not in helper


def test_legacy_public_deploy_fails_closed_without_explicit_acknowledgement():
    deploy = _read("deploy.sh")

    guard_index = deploy.index("PAO_ACK_LEGACY_UNSAFE_PUBLIC_DEPLOY")
    exit_index = deploy.index("exit 64")
    publish_index = deploy.index("--type=LoadBalancer")

    assert guard_index < exit_index < publish_index
    assert "I_UNDERSTAND_PAO_LEGACY_PUBLIC_DEPLOY_IS_UNSAFE" in deploy
    assert "deprecated, unsafe legacy deployment path" in deploy
    assert "does not configure the debugger privileges/isolation" in deploy
    assert "SYS_PTRACE/SYS_ADMIN" in deploy


def test_both_docker_workflows_boot_the_default_entrypoint():
    for relative_path in (
        ".github/workflows/build.yml",
        ".github/workflows/publish-ghcr.yml",
    ):
        workflow = _read(relative_path)
        assert "Boot default entrypoint and probe health" in workflow
        assert "curl -fsS http://127.0.0.1:5500/healthz" in workflow
        boot_section = workflow[
            workflow.index("Boot default entrypoint and probe health") : workflow.index(
                "Remove entrypoint smoke resources"
            )
        ]
        assert "--entrypoint" not in boot_section
        assert 'docker exec -i "$BOOT_CONTAINER"' in boot_section
        assert '"http://127.0.0.1:5500/mcp"' in boot_section
        assert "tools = await client.list_tools()" in boot_section
        for tool_name in STAGE5_MCP_TOOLS:
            assert f'"{tool_name}"' in boot_section
        # A boot without the remote-bind env exits and the health probe goes
        # red, and a tokenless /mcp call gets 401; both are regression bait.
        assert "-e PAO_ALLOW_REMOTE=1" in boot_section
        assert "-e PAO_MCP_TOKEN=" in boot_section
        assert 'PAO_ALLOW_REMOTE: "1"' in boot_section
        assert "from fastmcp.client.auth import BearerAuth" in boot_section
        assert 'auth=BearerAuth(os.environ["PAO_MCP_TOKEN"])' in boot_section
        assert "${{ github.run_id }}-${{ github.run_attempt }}" in workflow


def test_stage5_smoke_uses_real_isolated_services():
    smoke = _read("tests/docker_stage5_smoke.py")

    assert "create_services(" in smoke
    assert 'runtime_root=os.path.join(runtime_dir, "runtime")' in smoke
    assert "cast(Any, None)" not in smoke
    assert '"gdb_features": {"generate_core_file": {"available": True}}' not in smoke


def test_stage6_9_smoke_uses_real_runtime_boundaries():
    smoke = _read("tests/docker_stage6_9_smoke.py")

    assert "create_services(" in smoke
    assert "Client(create_mcp(services))" in smoke
    assert "STAGE6_TOOLS" in smoke
    assert "STAGE7_TOOLS" in smoke
    assert "STAGE8_TOOLS" in smoke
    assert "STAGE9_TOOLS" in smoke
    assert '"detect_allocator"' in smoke
    assert '"find_one_gadgets"' in smoke
    assert '"local_process"' in smoke
    assert '"verify_research_bundle"' in smoke
    assert "/lib/x86_64-linux-gnu/libc.so.6" in smoke
