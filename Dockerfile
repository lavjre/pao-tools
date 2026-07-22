FROM ghcr.io/astral-sh/uv:latest AS uv
FROM ubuntu:24.04 AS base

ARG TARGETARCH
ARG PWNDBG_VERSION=2026.02.18
ARG GHIDRA_VERSION=12.1.2
ARG GHIDRA_RELEASE_DATE=20260605
ARG GHIDRA_SHA256=b62e81a0390618466c019c60d8c2f796ced2509c4c1aea4a37644a77272cf99d

ENV DEBIAN_FRONTEND=noninteractive

RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      curl wget git vim nano file tmux sudo unzip xz-utils ca-certificates \
      build-essential gcc g++ clang make cmake bison flex gcc-multilib \
      gdb gdbserver gdb-multiarch lldb strace ltrace patchelf elfutils libc6-dbg \
      qemu-system-x86 qemu-user qemu-user-binfmt \
      python3 python3-pip python3-venv python3-dev python3-setuptools python3-poetry \
      openjdk-21-jdk-headless \
      libasan8 libubsan1 liblsan0 libtsan2 \
      libc6:i386 libstdc++6:i386 libgcc-s1:i386 zlib1g:i386 lib32z1 \
      libc6-dbg libc6-dbg:i386 \
      libssl3:i386 libncurses6:i386 libreadline8:i386 libtinfo6:i386 \
      libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev 

COPY --from=uv /uv /uvx /bin/

RUN set -eux; \
    case "$TARGETARCH" in \
      amd64) PWNDBG_ARCH=x86_64 ;; \
      arm64) PWNDBG_ARCH=arm64 ;; \
      *) echo "Unsupported pwndbg portable architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    PWNDBG_ASSET="pwndbg_${PWNDBG_VERSION}_${PWNDBG_ARCH}-portable.tar.xz"; \
    curl --proto '=https' --tlsv1.2 -fsSL \
      --retry 5 --retry-delay 2 --retry-all-errors \
      "https://github.com/pwndbg/pwndbg/releases/download/${PWNDBG_VERSION}/${PWNDBG_ASSET}" \
      -o /tmp/pwndbg.tar.xz; \
    mkdir -p /opt/pwndbg; \
    tar -xJf /tmp/pwndbg.tar.xz -C /opt/pwndbg --strip-components=2; \
    test -x /opt/pwndbg/bin/pwndbg; \
    ln -s /opt/pwndbg/bin/pwndbg /usr/local/bin/pwndbg; \
    rm /tmp/pwndbg.tar.xz

ENV PAO_PWNDBG_EXPECTED_VERSION=${PWNDBG_VERSION}
ENV PWNDBG_NO_AUTOUPDATE=1

RUN pwndbg --batch --quiet -ex "version" -ex "quit"

# Pin and verify the official Ghidra release used for local headless decompilation.
RUN set -eux; \
    GHIDRA_ASSET="ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_RELEASE_DATE}.zip"; \
    curl --proto '=https' --tlsv1.2 -fsSL \
      --retry 5 --retry-delay 2 --retry-all-errors \
      "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/${GHIDRA_ASSET}" \
      -o /tmp/ghidra.zip; \
    echo "${GHIDRA_SHA256}  /tmp/ghidra.zip" | sha256sum -c -; \
    unzip -q /tmp/ghidra.zip -d /opt; \
    mv "/opt/ghidra_${GHIDRA_VERSION}_PUBLIC" /opt/ghidra; \
    test -x /opt/ghidra/support/analyzeHeadless; \
    ln -s /opt/ghidra/support/analyzeHeadless /usr/local/bin/analyzeHeadless; \
    rm /tmp/ghidra.zip

ENV PAO_GHIDRA_VERSION=${GHIDRA_VERSION}
ENV PAO_GHIDRA_HEADLESS=/opt/ghidra/support/analyzeHeadless
ENV PAO_GHIDRA_MAXMEM=2G
ENV PAO_GHIDRA_MAX_CONCURRENT=1

# Install systemd so the VM can boot it as PID 1 under Ignite
RUN apt-get update && apt-get install -y systemd systemd-sysv && \
    ln -sf /lib/systemd/systemd /sbin/init

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY paomcp ./paomcp
COPY paocli ./paocli
COPY paomcp.service /etc/systemd/system/paomcp.service

# Create workspace directory for command execution
RUN mkdir -p /workspace

RUN useradd -m -s /bin/bash pao && \
    chown -R pao:pao /app && \
    chown -R pao:pao /workspace

RUN wget https://github.com/io12/pwninit/releases/download/3.3.1/pwninit -O /usr/local/bin/pwninit && \
    chmod +x /usr/local/bin/pwninit

# Enable paomcp systemd service for VM boot (firecrackers)
RUN mkdir -p /etc/systemd/system/multi-user.target.wants && \
    chmod 644 /etc/systemd/system/paomcp.service && \
    ln -sf /etc/systemd/system/paomcp.service /etc/systemd/system/multi-user.target.wants/paomcp.service

USER pao

ENV PYTHONPATH=/app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN uv sync

# paocli is vendored in this project and installed by `uv sync`.

# Pre-create the shared PythonTools venv used by Pao Tools to avoid runtime setup
RUN mkdir -p /tmp/pao/python && \
    uv venv /tmp/pao/python/shared_venv && \
    uv pip install --python /tmp/pao/python/shared_venv \
      requests numpy ipython hexdump click pwntools ropper

ENV PROD=true

EXPOSE 5500
# For Firecracker VM: systemd boots as PID 1 and starts paomcp.service
ENTRYPOINT ["/app/.venv/bin/python", "-m", "paomcp"]
