defmodule ExAutoresearch.Training.Trainer do
  @moduledoc """
  Training loop GenServer.

  Manages the time-budgeted training loop:
  1. Initialize model and optimizer
  2. Run forward/backward passes with gradient accumulation
  3. Apply LR schedule based on wall-clock progress
  4. Stop after time_budget seconds
  5. Evaluate BPB on validation set
  6. Broadcast results via PubSub
  """

  use GenServer

  require Logger

  alias ExAutoresearch.Model.Config

  defstruct [
    :config,
    :model,
    :params,
    :optimizer_state,
    :start_time,
    :step,
    :total_loss,
    status: :idle
  ]

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Start a training run with the given config."
  def train(%Config{} = config) do
    GenServer.call(__MODULE__, {:train, config}, :infinity)
  end

  @doc "Get current training status."
  def status do
    GenServer.call(__MODULE__, :status)
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_call({:train, config}, _from, _state) do
    Logger.info("Starting training with config: #{inspect(config, limit: 5)}")

    # Build model
    model = ExAutoresearch.Model.GPT.build(config)

    # Initialize parameters
    n_embd = Config.n_embd(config)
    template = %{"input_ids" => Nx.template({config.device_batch_size, config.sequence_len}, :s64)}
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(template, %{})

    # Setup optimizer (AdamW with per-group LRs)
    optimizer = Polaris.Optimizers.adamw(
      learning_rate: config.matrix_lr,
      b1: config.adam_beta1,
      b2: config.adam_beta2,
      decay: config.weight_decay
    )

    {optimizer_init, _optimizer_update} = optimizer
    optimizer_state = optimizer_init.(params)

    new_state = %__MODULE__{
      config: config,
      model: model,
      params: params,
      optimizer_state: optimizer_state,
      start_time: System.monotonic_time(:millisecond),
      step: 0,
      total_loss: 0.0,
      status: :training
    }

    # For now, return immediately — actual training loop will be
    # implemented when the dataloader is ready
    Logger.info("Model initialized: #{n_embd}-dim, #{config.n_layer} layers")

    result = %{
      status: :initialized,
      n_params: count_params(params),
      n_embd: n_embd,
      n_layer: config.n_layer
    }

    {:reply, {:ok, result}, new_state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    status = %{
      status: state.status,
      step: state.step,
      elapsed_ms: elapsed_ms(state)
    }

    {:reply, status, state}
  end

  defp elapsed_ms(%{start_time: nil}), do: 0
  defp elapsed_ms(%{start_time: t}), do: System.monotonic_time(:millisecond) - t

  defp count_params(params) do
    params
    |> Enum.flat_map(fn {_name, layer_params} ->
      Enum.map(layer_params, fn {_key, tensor} -> Nx.size(tensor) end)
    end)
    |> Enum.sum()
  end
end
