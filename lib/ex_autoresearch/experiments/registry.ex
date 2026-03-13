defmodule ExAutoresearch.Experiments.Registry do
  @moduledoc """
  Version registry for hot-loaded experiment modules.

  Stores all experiment versions in ETS with their source code, loss,
  and metadata. Modules live in the BEAM simultaneously — no restarts needed.
  """

  use GenServer

  require Logger

  @table __MODULE__

  defstruct [:version_id, :module, :code, :description, :parent_id,
             :loss, :steps, :training_seconds, :status, :kept, :loaded_at]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  # --- Reads (direct ETS, never block) ---

  def all do
    :ets.tab2list(@table)
    |> Enum.map(fn {_id, entry} -> entry end)
    |> Enum.sort_by(& &1.loaded_at, DateTime)
  rescue
    ArgumentError -> []
  end

  def get(version_id) do
    case :ets.lookup(@table, version_id) do
      [{^version_id, entry}] -> {:ok, entry}
      [] -> :error
    end
  rescue
    ArgumentError -> :error
  end

  def best do
    all()
    |> Enum.filter(& &1.kept)
    |> Enum.filter(& &1.loss)
    |> Enum.min_by(& &1.loss, fn -> nil end)
  end

  def count do
    :ets.info(@table, :size)
  rescue
    ArgumentError -> 0
  end

  # --- Writes ---

  def register(version_id, attrs) do
    GenServer.call(__MODULE__, {:register, version_id, attrs})
  end

  def update(version_id, attrs) do
    GenServer.call(__MODULE__, {:update, version_id, attrs})
  end

  # --- Server ---

  @impl true
  def init(_opts) do
    :ets.new(@table, [:named_table, :set, :public, read_concurrency: true])
    {:ok, %{}}
  end

  @impl true
  def handle_call({:register, version_id, attrs}, _from, state) do
    entry = struct!(%__MODULE__{}, Map.put(attrs, :version_id, version_id))
    :ets.insert(@table, {version_id, entry})
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:update, version_id, attrs}, _from, state) do
    case :ets.lookup(@table, version_id) do
      [{^version_id, entry}] ->
        updated = struct!(entry, attrs)
        :ets.insert(@table, {version_id, updated})
        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end
end
