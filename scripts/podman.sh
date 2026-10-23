#!/usr/bin/env bash
# Rootless, user-level podman confined to the project directory (no sudo, nothing in ~/.config or ~/.local).
#   ./scripts/podman.sh install        # download the static podman bundle into tools/podman-linux-amd64
#   ./scripts/podman.sh <podman args>  # e.g. build -t portfolio-rag . / run ... / images
# Without newuidmap/newgidmap (setuid helpers from the `uidmap` package, which need root to install)
# podman falls back to a *single* UID mapping (container root = your user). Image layers that chown
# to other ids are handled by `ignore_chown_errors`; processes inside the container must run as root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE="$ROOT/tools/podman-linux-amd64"
STATE="$ROOT/tools/podman-state"
VERSION="${PODMAN_STATIC_VERSION:-latest}"

if [[ "${1:-}" == "install" ]]; then
  mkdir -p "$ROOT/tools"
  url="https://github.com/mgoltzsche/podman-static/releases/${VERSION/latest/latest\/download}/podman-linux-amd64.tar.gz"
  [[ "$VERSION" != latest ]] && url="https://github.com/mgoltzsche/podman-static/releases/download/$VERSION/podman-linux-amd64.tar.gz"
  curl -fsSL "$url" | tar -xz -C "$ROOT/tools"
  "$BUNDLE/usr/local/bin/podman" --version
  exit 0
fi
[[ -x "$BUNDLE/usr/local/bin/podman" ]] || { echo "podman not installed: run $0 install" >&2; exit 1; }

export HOME="$STATE/home" XDG_RUNTIME_DIR="$STATE/run" XDG_CONFIG_HOME="$STATE/home/.config" XDG_DATA_HOME="$STATE/data"
export PATH="$BUNDLE/usr/local/bin:$PATH" TMPDIR="$STATE/tmp"  # buildah contexts stay inside the project
# podman looks up sub-id ranges for $USER; the ranges exist in /etc/subuid but cannot be used without the
# setuid newuidmap helper, so ask for the documented single-mapping mode instead of failing.
export USER="single-mapping-$(id -u)"
CONF="$XDG_CONFIG_HOME/containers"
mkdir -p "$CONF" "$XDG_RUNTIME_DIR" "$XDG_DATA_HOME" "$TMPDIR" && chmod 700 "$XDG_RUNTIME_DIR"
cat > "$CONF/containers.conf" <<CONFEOF
[containers]
cgroups = "disabled"
[engine]
cgroup_manager = "cgroupfs"
events_logger = "file"
runtime = "crun"
conmon_path = ["$BUNDLE/usr/local/lib/podman/conmon"]
helper_binaries_dir = ["$BUNDLE/usr/local/lib/podman", "$BUNDLE/usr/local/libexec/podman", "$BUNDLE/usr/local/bin"]
[engine.runtimes]
crun = ["$BUNDLE/usr/local/bin/crun"]
[network]
default_rootless_network_cmd = "pasta"
CONFEOF
cat > "$CONF/storage.conf" <<CONFEOF
[storage]
driver = "overlay"
graphroot = "$XDG_DATA_HOME/storage"
runroot = "$XDG_RUNTIME_DIR/storage"
[storage.options.overlay]
mount_program = "$BUNDLE/usr/local/bin/fuse-overlayfs"
ignore_chown_errors = "true"
CONFEOF
cp -n "$BUNDLE/etc/containers/policy.json" "$BUNDLE/etc/containers/registries.conf" "$CONF/" 2>/dev/null || true
exec "$BUNDLE/usr/local/bin/podman" "$@"
