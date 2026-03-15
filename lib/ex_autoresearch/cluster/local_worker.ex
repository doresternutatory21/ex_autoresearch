defmodule ExAutoresearch.Cluster.LocalWorker do
  @moduledoc """
  Spawns a local worker BEAM node with a different GPU backend.

  Used to run CUDA alongside ROCm on the same machine. The child is a
  fully independent BEAM instance started with WORKER_ONLY=1, which skips
  Phoenix and only runs the training infrastructure.

  Ported from basileus.

  ## Prerequisites

      # One-time: compile for CUDA target
      just compile-cuda

  ## Usage

      # From iex on the main node:
      {:ok, pid} = ExAutoresearch.Cluster.LocalWorker.start_link(
        name: "cuda_worker",
        gpu_target: "cuda",
        build_path: "_build/cuda"
      )
  """

  use GenServer, restart: :temporary

  require Logger

  defstruct [:name, :gpu_target, :build_path, :port, :os_pid]

  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: via(name))
  end

  def stop(name), do: GenServer.stop(via(name))

  defp via(name), do: {:global, {__MODULE__, name}}

  @impl true
  def init(opts) do
    name = Keyword.fetch!(opts, :name)
    gpu_target = Keyword.get(opts, :gpu_target, "cuda")
    build_path = Keyword.get(opts, :build_path, "_build/cuda")
    cookie = Atom.to_string(:erlang.get_cookie())
    project_dir = File.cwd!()

    mix_env = if function_exported?(Mix, :env, 0), do: Atom.to_string(Mix.env()), else: "dev"

    # Ensure the XLA extension is a real copy, not a symlink to shared cache
    restore_xla_snapshot(project_dir, build_path)

    # Hermetic CUDA 12.8 runtime libraries (extracted from Docker build cache).
    # Required because system CUDA 13 compat symlinks don't satisfy versioned symbols.
    cuda_libs_dir = Path.join(project_dir, "_build/cuda_libs")

    ld_path =
      [cuda_libs_dir, System.get_env("LD_LIBRARY_PATH")]
      |> Enum.reject(&is_nil/1)
      |> Enum.join(":")

    env = [
      {~c"WORKER_ONLY", ~c"1"},
      {~c"GPU_TARGET", String.to_charlist(gpu_target)},
      {~c"XLA_TARGET", String.to_charlist(xla_target(gpu_target))},
      {~c"MIX_ENV", String.to_charlist(mix_env)},
      {~c"MIX_BUILD_PATH", String.to_charlist(build_path)},
      {~c"LD_LIBRARY_PATH", String.to_charlist(ld_path)}
    ]

    {name_flag, full_node_name} =
      case Atom.to_string(node()) do
        name_str ->
          host_part = name_str |> String.split("@") |> List.last()

          if String.contains?(host_part, ".") or String.match?(host_part, ~r/^\d+\.\d+/) do
            {"--name", "#{name}@#{host_part}"}
          else
            {"--sname", "#{name}@#{host_part}"}
          end
      end

    args = [
      name_flag,
      full_node_name,
      "--cookie",
      cookie,
      "--erl",
      "-kernel inet_dist_listen_min 9000 inet_dist_listen_max 9100",
      "-S",
      "mix",
      "run",
      "--no-halt"
    ]

    Logger.info("Starting local worker: #{full_node_name} (#{gpu_target})")

    port =
      Port.open({:spawn_executable, System.find_executable("elixir")}, [
        :binary,
        :exit_status,
        :stderr_to_stdout,
        args: args,
        cd: project_dir,
        env: env
      ])

    {:os_pid, os_pid} = Port.info(port, :os_pid)

    state = %__MODULE__{
      name: full_node_name,
      gpu_target: gpu_target,
      build_path: build_path,
      port: port,
      os_pid: os_pid
    }

    Process.send_after(self(), :await_connection, 2_000)

    {:ok, state}
  end

  @impl true
  def handle_info(:await_connection, state) do
    target = String.to_atom(state.name)

    case Node.connect(target) do
      true ->
        Logger.info("Local worker connected: #{target}")

        ExAutoresearch.Cluster.register_capabilities(%{
          "gpu_target" => state.gpu_target,
          "role" => "worker"
        })

      _ ->
        Logger.debug("Waiting for worker #{state.name}...")
        Process.send_after(self(), :await_connection, 2_000)
    end

    {:noreply, state}
  end

  def handle_info({port, {:data, data}}, %{port: port} = state) do
    for line <- String.split(data, "\n", trim: true) do
      Logger.debug("[#{state.name}] #{line}")
    end

    {:noreply, state}
  end

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.warning("Local worker #{state.name} exited (#{status})")
    {:stop, {:worker_exited, status}, state}
  end

  @impl true
  def terminate(_reason, state) do
    if state.port && Port.info(state.port), do: Port.close(state.port)
    :ok
  end

  defp xla_target("cuda"), do: "cuda12"
  defp xla_target("rocm"), do: "rocm"
  defp xla_target(_), do: "host"

  # Restore the XLA extension from snapshot if the priv dir contains a
  # symlink (which would point to the shared deps/exla/cache/ and may
  # have been overwritten by a different build variant).
  defp restore_xla_snapshot(project_dir, build_path) do
    build_name = build_path |> String.replace("_build/", "")
    xla_dir = Path.join([project_dir, build_path, "lib/exla/priv/xla_extension"])
    snapshot = Path.join(project_dir, "_build/#{build_name}_xla_snapshot")
    lib_link = Path.join(xla_dir, "lib")

    cond do
      File.dir?(snapshot) and (not File.dir?(xla_dir) or symlink?(lib_link)) ->
        File.rm_rf!(xla_dir)
        File.cp_r!(snapshot, xla_dir)
        Logger.info("Restored XLA extension from snapshot for #{build_name}")

      File.dir?(xla_dir) and symlink?(lib_link) ->
        # No snapshot yet — create one by dereferencing the symlinks
        File.rm_rf!(snapshot)
        # Copy with dereferencing: read through symlinks
        {_, 0} = System.cmd("cp", ["-aL", xla_dir, snapshot])
        File.rm_rf!(xla_dir)
        File.cp_r!(snapshot, xla_dir)
        Logger.info("Created XLA snapshot for #{build_name}")

      true ->
        :ok
    end
  rescue
    e -> Logger.warning("XLA snapshot restore failed: #{Exception.message(e)}")
  end

  defp symlink?(path) do
    case File.lstat(path) do
      {:ok, %{type: :symlink}} -> true
      _ -> false
    end
  end
end
