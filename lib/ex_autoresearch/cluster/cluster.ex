defmodule ExAutoresearch.Cluster do
  @moduledoc """
  Multi-machine cluster coordination.

  Tracks node capabilities (GPU type, VRAM, RAM), assigns work based on
  hardware, and reports cluster status. Ported from basileus.

  Each node registers its capabilities on join. The coordinator tracks
  which nodes are available and routes training tasks accordingly.
  """

  use GenServer

  require Logger

  @capabilities_table __MODULE__.Capabilities

  defstruct [:node_id, :capabilities, :status, :joined_at, :last_heartbeat]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Register the current node's capabilities."
  def register_capabilities(capabilities) when is_map(capabilities) do
    GenServer.call(__MODULE__, {:register, node(), capabilities})
  end

  @doc "Get all registered nodes with their capabilities."
  def list_nodes do
    GenServer.call(__MODULE__, :list_nodes)
  end

  @doc "Get cluster status summary."
  def status do
    GenServer.call(__MODULE__, :status)
  end

  @doc "Find the best node for a given task type (:train, :eval)."
  def best_node_for(task_type) when is_atom(task_type) do
    GenServer.call(__MODULE__, {:best_node_for, task_type})
  end

  @impl true
  def init(_opts) do
    :ets.new(@capabilities_table, [:named_table, :set, :public, read_concurrency: true])

    if Node.alive?(), do: :net_kernel.monitor_nodes(true)

    register_local_capabilities()
    Process.send_after(self(), :check_heartbeats, 30_000)

    {:ok, %{heartbeat_interval: 30_000, stale_threshold: 90_000}}
  end

  @impl true
  def handle_call({:register, node_name, capabilities}, _from, state) do
    entry = %__MODULE__{
      node_id: node_name,
      capabilities: capabilities,
      status: :online,
      joined_at: DateTime.utc_now(),
      last_heartbeat: DateTime.utc_now()
    }

    :ets.insert(@capabilities_table, {node_name, entry})
    Logger.info("Node registered: #{node_name} with #{inspect(capabilities)}")

    Phoenix.PubSub.broadcast(
      ExAutoresearch.PubSub,
      "cluster:events",
      {:node_joined, %{node: node_name, capabilities: capabilities}}
    )

    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:list_nodes, _from, state) do
    nodes =
      :ets.tab2list(@capabilities_table)
      |> Enum.map(fn {_key, entry} -> entry end)
      |> Enum.sort_by(& &1.node_id)

    {:reply, nodes, state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    nodes =
      :ets.tab2list(@capabilities_table)
      |> Enum.map(fn {_key, entry} -> entry end)

    status = %{
      total_nodes: length(nodes),
      online: Enum.count(nodes, &(&1.status == :online)),
      offline: Enum.count(nodes, &(&1.status == :offline)),
      gpu_nodes: Enum.count(nodes, fn n -> n.status == :online and has_gpu?(n.capabilities) end),
      nodes: Enum.map(nodes, &node_summary/1)
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call({:best_node_for, :train}, _from, state) do
    nodes =
      :ets.tab2list(@capabilities_table)
      |> Enum.map(fn {_key, entry} -> entry end)
      |> Enum.filter(&(&1.status == :online))

    best =
      nodes
      |> Enum.sort_by(fn n ->
        # Prefer: discrete GPU > iGPU, then by VRAM, then by RAM
        {if(has_gpu?(n.capabilities), do: 0, else: 1), -Map.get(n.capabilities, "vram_mb", 0),
         -Map.get(n.capabilities, "memory_gb", 0)}
      end)
      |> List.first()

    result = if best, do: {:ok, best.node_id}, else: {:error, :no_available_node}
    {:reply, result, state}
  end

  @impl true
  def handle_info({:nodeup, node_name}, state) do
    Logger.info("Node joined cluster: #{node_name}")
    {:noreply, state}
  end

  @impl true
  def handle_info({:nodedown, node_name}, state) do
    Logger.warning("Node left cluster: #{node_name}")

    case :ets.lookup(@capabilities_table, node_name) do
      [{^node_name, entry}] ->
        :ets.insert(@capabilities_table, {node_name, %{entry | status: :offline}})

        Phoenix.PubSub.broadcast(
          ExAutoresearch.PubSub,
          "cluster:events",
          {:node_left, %{node: node_name}}
        )

      [] ->
        :ok
    end

    {:noreply, state}
  end

  @impl true
  def handle_info(:check_heartbeats, state) do
    now = DateTime.utc_now()
    local = node()

    case :ets.lookup(@capabilities_table, local) do
      [{^local, entry}] ->
        :ets.insert(@capabilities_table, {local, %{entry | last_heartbeat: now}})

      [] ->
        :ok
    end

    :ets.tab2list(@capabilities_table)
    |> Enum.each(fn {node_name, entry} ->
      if node_name != local and entry.status == :online and entry.last_heartbeat do
        age_ms = DateTime.diff(now, entry.last_heartbeat, :millisecond)

        if age_ms > state.stale_threshold do
          Logger.warning("Node #{node_name} marked stale (#{div(age_ms, 1000)}s)")
          :ets.insert(@capabilities_table, {node_name, %{entry | status: :offline}})
        end
      end
    end)

    Process.send_after(self(), :check_heartbeats, state.heartbeat_interval)
    {:noreply, state}
  end

  # Private helpers

  defp register_local_capabilities do
    :ets.insert(
      @capabilities_table,
      {node(),
       %__MODULE__{
         node_id: node(),
         capabilities: detect_local_capabilities(),
         status: :online,
         joined_at: DateTime.utc_now(),
         last_heartbeat: DateTime.utc_now()
       }}
    )
  end

  defp detect_local_capabilities do
    %{
      "cpu_count" => System.schedulers_online(),
      "memory_gb" => total_memory_gb(),
      "gpu" => detect_gpu(),
      "gpu_target" => System.get_env("GPU_TARGET", "host")
    }
  end

  defp total_memory_gb do
    if Code.ensure_loaded?(:memsup) and function_exported?(:memsup, :get_system_memory_data, 0) do
      case apply(:memsup, :get_system_memory_data, []) do
        data when is_list(data) ->
          total = Keyword.get(data, :total_memory, 0)
          Float.round(total / (1024 * 1024 * 1024), 1)

        _ ->
          0.0
      end
    else
      0.0
    end
  rescue
    _ -> 0.0
  end

  defp detect_gpu do
    case System.cmd("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"],
           stderr_to_stdout: true
         ) do
      {output, 0} ->
        output
        |> String.trim()
        |> String.split("\n")
        |> Enum.map(&String.trim/1)
        |> Enum.reject(&(&1 == ""))

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp has_gpu?(capabilities) do
    case Map.get(capabilities, "gpu", []) do
      gpus when is_list(gpus) and gpus != [] -> true
      _ -> false
    end
  end

  defp node_summary(n) do
    %{
      node_id: to_string(n.node_id),
      status: n.status,
      capabilities: n.capabilities,
      last_heartbeat: n.last_heartbeat && DateTime.to_iso8601(n.last_heartbeat)
    }
  end
end
