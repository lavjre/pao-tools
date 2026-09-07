#!/usr/bin/env bash
# Native pao-tools setup for Linux and macOS (no Docker).
#
#   ./scripts/setup-native.sh [--with-ghidra] [--with-pwninit] [--with-fuzz] [--with-firmware] [--all] [--yes]
#
# Installs system deps, pwndbg (Linux), uv python deps, the helper venv, and
# writes .env.native. Ends with `paocli doctor`. macOS uses the lldb backend
# (Mach-O + static); Linux ELF live-debugging on macOS still needs Docker/VM.
set -euo pipefail

WITH_GHIDRA=0; WITH_PWNINIT=0; WITH_FUZZ=0; WITH_FIRMWARE=0; ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --with-ghidra) WITH_GHIDRA=1 ;;
    --with-pwninit) WITH_PWNINIT=1 ;;
    --with-fuzz) WITH_FUZZ=1 ;;
    --with-firmware) WITH_FIRMWARE=1 ;;
    --all) WITH_GHIDRA=1; WITH_PWNINIT=1; WITH_FUZZ=1; WITH_FIRMWARE=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAO_HOME="${HOME}/.local/share/pao"
BIN_HOME="${HOME}/.local/bin"
GHIDRA_VERSION="12.1.2"
GHIDRA_DATE="20260605"
GHIDRA_SHA256="b62e81a0390618466c019c60d8c2f796ced2509c4c1aea4a37644a77272cf99d"
PWNINIT_VERSION="3.3.1"
PWNINIT_SHA256="1b72653be59f8f13ea934d1153e8d845eac0dbac62a8c4d9dc4a25577f009362"

OS="$(uname -s)"
SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

confirm() {
  if [ "$ASSUME_YES" -eq 1 ]; then return 0; fi
  read -r -p "$1 [y/N] " ans
  [ "${ans:-N}" = "y" ] || [ "${ans:-N}" = "Y" ]
}

install_linux() {
  if [ -f /etc/os-release ]; then . /etc/os-release; else ID="unknown"; fi
  case "${ID:-unknown}" in
    ubuntu|debian|pop|linuxmint)
      $SUDO apt-get update
      $SUDO apt-get install -y --no-install-recommends \
        build-essential gdb gdbserver gdb-multiarch lldb git curl file \
        patchelf ruby ruby-dev python3-venv pipx \
        qemu-system-x86 qemu-user openjdk-21-jre-headless
      [ "$WITH_FUZZ" -eq 1 ] && $SUDO apt-get install -y --no-install-recommends afl++
      [ "$WITH_FIRMWARE" -eq 1 ] && $SUDO apt-get install -y --no-install-recommends binwalk squashfs-tools ;;
    fedora|rhel|centos|rocky|alma)
      $SUDO dnf install -y \
        gcc gcc-c++ make gdb gdb-gdbserver lldb git curl file \
        patchelf ruby ruby-devel python3-pip pipx \
        qemu-system-x86 qemu-user java-21-openjdk-headless
      [ "$WITH_FUZZ" -eq 1 ] && $SUDO dnf install -y afl++ || true
      [ "$WITH_FIRMWARE" -eq 1 ] && $SUDO dnf install -y binwalk || true ;;
    arch|manjaro|endeavouros)
      $SUDO pacman -Sy --noconfirm --needed \
        base-devel gdb lldb git curl file patchelf ruby \
        qemu-system-x86 qemu-user jre21-openjdk-headless python-pipx
      [ "$WITH_FUZZ" -eq 1 ] && $SUDO pacman -S --noconfirm --needed afl++ || true ;;
    *) echo "unsupported linux distro ($ID): install gdb/pwndbg/uv manually, then re-run." >&2; exit 1 ;;
  esac
  PWNDBG_PIN="2026.02.18"
  if ! command -v pwndbg >/dev/null 2>&1; then
    pipx install "pwndbg==${PWNDBG_PIN}" || pipx install pwndbg \
      || python3 -m pip install --user "pwndbg==${PWNDBG_PIN}" \
      || python3 -m pip install --user pwndbg
  fi
}

install_macos() {
  command -v brew >/dev/null 2>&1 || { echo "install Homebrew first: https://brew.sh" >&2; exit 1; }
  xcode-select -p >/dev/null 2>&1 || xcode-select --install || true
  brew install python@3.12 uv git pipx
  [ "$WITH_FUZZ" -eq 1 ] && brew install afl-fuzz || true
  [ "$WITH_FIRMWARE" -eq 1 ] && brew install binwalk squashfs || true
  echo "macOS note: gdb/pwndbg skipped (unsigned gdb cannot debug). Backend defaults to lldb below."
}

if [ "$OS" = "Darwin" ]; then install_macos; else install_linux; fi

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
mkdir -p "$PAO_HOME" "$BIN_HOME"
cd "$REPO_ROOT"
uv sync --extra dev

if [ "$WITH_GHIDRA" -eq 1 ]; then
  if [ ! -x "$PAO_HOME/ghidra/support/analyzeHeadless" ]; then
    ASSET="ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_DATE}.zip"
    curl --proto '=https' --tlsv1.2 -fsSL --retry 5 \
      "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/${ASSET}" \
      -o /tmp/ghidra-native.zip
    echo "${GHIDRA_SHA256}  /tmp/ghidra-native.zip" | sha256sum -c - 2>/dev/null \
      || shasum -a 256 -c <(echo "${GHIDRA_SHA256}  /tmp/ghidra-native.zip")
    rm -rf "$PAO_HOME/ghidra" && mkdir -p "$PAO_HOME"
    unzip -q /tmp/ghidra-native.zip -d "$PAO_HOME"
    mv "$PAO_HOME/ghidra_${GHIDRA_VERSION}_PUBLIC" "$PAO_HOME/ghidra"
    rm /tmp/ghidra-native.zip
  fi
fi

if [ "$WITH_PWNINIT" -eq 1 ] && [ "$OS" != "Darwin" ]; then
  if [ ! -x "$BIN_HOME/pwninit" ]; then
    curl --proto '=https' --tlsv1.2 -fsSL --retry 5 \
      "https://github.com/io12/pwninit/releases/download/${PWNINIT_VERSION}/pwninit" \
      -o "$BIN_HOME/pwninit"
    echo "${PWNINIT_SHA256}  $BIN_HOME/pwninit" | sha256sum -c -
    chmod +x "$BIN_HOME/pwninit"
  fi
fi

HELPER_PKGS="requests numpy ipython hexdump click pwntools ropper"
[ "$WITH_FIRMWARE" -eq 1 ] && HELPER_PKGS="$HELPER_PKGS unblob"
uv venv "$PAO_HOME/python-venv" 2>/dev/null || true
uv pip install --python "$PAO_HOME/python-venv" $HELPER_PKGS

BACKEND="auto"
[ "$OS" = "Darwin" ] && BACKEND="lldb"
GHIDRA_LINE="# PAO_GHIDRA_HEADLESS=$PAO_HOME/ghidra/support/analyzeHeadless"
[ "$WITH_GHIDRA" -eq 1 ] && GHIDRA_LINE="PAO_GHIDRA_HEADLESS=$PAO_HOME/ghidra/support/analyzeHeadless"
cat > "$REPO_ROOT/.env.native" <<EOF
# Generated by scripts/setup-native.sh — source it or copy what you need.
PAO_RUNTIME_ROOT=$PAO_HOME/runtime
PAO_PYTHON_VENV=$PAO_HOME/python-venv
PAO_DEBUGGER_BACKEND=$BACKEND
PAO_ALLOW_REMOTE=0
# PAO_MCP_TOKEN=fill-only-if-binding-non-loopback
$GHIDRA_LINE
EOF

export PATH="$BIN_HOME:$PATH"
uv run paocli doctor || true
echo
echo "done. next:"
echo "  source .env.native   # or: export PAO_RUNTIME_ROOT=\$HOME/.local/share/pao/runtime"
echo "  uv run python -m paomcp            # HTTP on 127.0.0.1:5500/mcp"
echo "  uv run python -m paomcp --stdio    # stdio transport"
