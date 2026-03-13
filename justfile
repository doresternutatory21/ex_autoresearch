# Start the main node on ROCm iGPU (visible output, logs to run.log)
start:
    #!/usr/bin/env bash
    SNAME="$(basename "$(pwd)")"
    export PORT="${PORT:-$(phx-port)}"
    export GPU_TARGET="${GPU_TARGET:-rocm}"
    echo "Starting $SNAME on GPU_TARGET=$GPU_TARGET, port $PORT"
    exec elixir --sname "$SNAME" --cookie devcookie -S mix phx.server 2>&1 | tee run.log

# Start the main node in background (logs to run.log only)
start-bg:
    #!/usr/bin/env bash
    SNAME="$(basename "$(pwd)")"
    export PORT="${PORT:-$(phx-port)}"
    export GPU_TARGET="${GPU_TARGET:-rocm}"
    echo "Starting $SNAME (bg) on GPU_TARGET=$GPU_TARGET, port $PORT"
    exec elixir --sname "$SNAME" --cookie devcookie -S mix phx.server > run.log 2>&1

# Start a CUDA worker node (separate BEAM, same machine)
# Requires: just compile-cuda (one-time)
start-cuda NAME="cuda_worker":
    #!/usr/bin/env bash
    echo "Starting CUDA worker '{{NAME}}'..."
    WORKER_ONLY=1 GPU_TARGET=cuda MIX_BUILD_PATH=_build/cuda \
      elixir --sname {{NAME}} --cookie devcookie \
      -S mix run --no-halt 2>&1 | tee run_{{NAME}}.log

# Start a CUDA worker in background
start-cuda-bg NAME="cuda_worker":
    #!/usr/bin/env bash
    echo "Starting CUDA worker '{{NAME}}' (bg)..."
    WORKER_ONLY=1 GPU_TARGET=cuda MIX_BUILD_PATH=_build/cuda \
      elixir --sname {{NAME}} --cookie devcookie \
      -S mix run --no-halt > run_{{NAME}}.log 2>&1 &
    echo "Worker started, check run_{{NAME}}.log"

# Compile the default (ROCm) build
compile:
    mix compile

# Compile a separate CUDA build (one-time, takes a while for XLA download)
compile-cuda:
    #!/usr/bin/env bash
    echo "Compiling CUDA build (XLA_TARGET=cuda)..."
    echo "This downloads the CUDA XLA archive on first run."
    XLA_TARGET=cuda MIX_BUILD_PATH=_build/cuda mix compile

# Open the app in a browser (starts the server if not running)
open:
    #!/usr/bin/env bash
    SNAME="$(basename "$(pwd)")"
    if ! scripts/dev_node.sh status > /dev/null 2>&1; then
        echo "Node $SNAME not running, starting in background..."
        export PORT="${PORT:-$(phx-port)}"
        export GPU_TARGET="${GPU_TARGET:-rocm}"
        elixir --sname "$SNAME" --cookie devcookie -S mix phx.server > run.log 2>&1 &
        scripts/dev_node.sh await
    fi
    phx-port open

# Stop the running BEAM node gracefully
stop:
    scripts/dev_node.sh rpc "System.halt()"

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
