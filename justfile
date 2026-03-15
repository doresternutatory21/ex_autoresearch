# --- Lifecycle ---

# Start the full application (main node + CUDA worker if available)
start:
    #!/usr/bin/env bash
    set -euo pipefail
    SNAME="$(basename "$(pwd)")"
    export PORT="${PORT:-$(phx-port)}"
    export GPU_TARGET="${GPU_TARGET:-rocm}"

    # Ensure the ROCm XLA extension is in place for the main node
    just _snapshot-xla dev rocm

    echo "Starting $SNAME on GPU_TARGET=$GPU_TARGET, port $PORT"
    elixir --sname "$SNAME" --cookie devcookie -S mix phx.server > run.log 2>&1 &
    scripts/dev_node.sh await
    echo "Main node $SNAME is up"

# Stop the full application (main node + CUDA worker)
stop:
    scripts/dev_node.sh stop

# Start in foreground (visible output, for debugging)
start-fg:
    #!/usr/bin/env bash
    SNAME="$(basename "$(pwd)")"
    export PORT="${PORT:-$(phx-port)}"
    export GPU_TARGET="${GPU_TARGET:-rocm}"
    just _snapshot-xla dev rocm
    echo "Starting $SNAME on GPU_TARGET=$GPU_TARGET, port $PORT"
    exec elixir --sname "$SNAME" --cookie devcookie -S mix phx.server 2>&1 | tee run.log

# --- Build ---

# Compile the default (ROCm) build and snapshot the XLA extension
compile:
    #!/usr/bin/env bash
    mix compile
    just _snapshot-xla dev rocm

# Compile the CUDA build variant (one-time, downloads CUDA XLA archive)
compile-cuda:
    #!/usr/bin/env bash
    echo "Compiling CUDA build (XLA_TARGET=cuda)..."
    XLA_TARGET=cuda GPU_TARGET=cuda MIX_BUILD_PATH=_build/cuda mix compile
    just _snapshot-xla cuda cuda

# Snapshot the XLA extension for a build variant so it doesn't get
# clobbered when the other variant compiles. Both _build/dev and
# _build/cuda create a symlink priv/xla_extension → deps/exla/cache/
# which is shared. This replaces the symlink with a real copy.
_snapshot-xla BUILD TARGET:
    #!/usr/bin/env bash
    PRIV="_build/{{BUILD}}/lib/exla/priv"
    XLA_DIR="$PRIV/xla_extension"
    SNAPSHOT="_build/{{BUILD}}_xla_snapshot"

    # If we already have a snapshot, restore it
    if [ -d "$SNAPSHOT" ] && [ -L "$XLA_DIR/lib" -o ! -d "$XLA_DIR" ]; then
        rm -rf "$XLA_DIR"
        cp -a "$SNAPSHOT" "$XLA_DIR"
        echo "[{{TARGET}}] Restored XLA extension from snapshot"
        exit 0
    fi

    # If xla_extension exists and is/contains symlinks, snapshot a real copy
    if [ -d "$XLA_DIR" ]; then
        rm -rf "$SNAPSHOT"
        cp -aL "$XLA_DIR" "$SNAPSHOT"  # -L dereferences symlinks
        # Replace symlinked dir with real copy
        rm -rf "$XLA_DIR"
        cp -a "$SNAPSHOT" "$XLA_DIR"
        echo "[{{TARGET}}] Snapshotted XLA extension"
    fi

# --- Utilities ---

# Open the app in a browser
open:
    #!/usr/bin/env bash
    if ! scripts/dev_node.sh status > /dev/null 2>&1; then
        echo "Node not running, starting..."
        just start
    fi
    phx-port open

# Check if the BEAM node is running
status:
    #!/usr/bin/env bash
    scripts/dev_node.sh status
    echo "---"
    epmd -names 2>&1 | grep -v "^epmd:"

# Execute an expression on the running BEAM node
rpc EXPR:
    scripts/dev_node.sh rpc "{{EXPR}}"

# Show GPU status
gpu:
    #!/usr/bin/env bash
    echo "=== NVIDIA ==="
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU"
    echo ""
    echo "=== ROCm ==="
    rocm-smi --showuse 2>/dev/null | head -10 || echo "No ROCm GPU"
