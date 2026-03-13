defmodule ExAutoresearch.Agent.Researcher do
  @moduledoc """
  Autonomous experiment loop.

  Orchestrates the research cycle:
  1. Ask LLM for next experiment (or run baseline)
  2. Apply config changes
  3. Run training with time budget
  4. Evaluate results
  5. Keep or discard
  6. Repeat forever
  """

  use GenServer

  require Logger

  alias ExAutoresearch.Model.Config
  alias ExAutoresearch.Training.Trainer
  alias ExAutoresearch.Agent.{LLM, Program}

  defstruct [
    :current_config,
    :baseline_loss,
    experiments: [],
    status: :idle,
    model: "claude-sonnet-4"
  ]

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Start the autonomous experiment loop."
  def start_research(opts \\ []) do
    GenServer.cast(__MODULE__, {:start_research, opts})
  end

  @doc "Stop the experiment loop after current experiment finishes."
  def stop_research do
    GenServer.cast(__MODULE__, :stop_research)
  end

  @doc "Get current status."
  def status do
    GenServer.call(__MODULE__, :status)
  end

  @doc "Get experiment history."
  def experiments do
    GenServer.call(__MODULE__, :experiments)
  end

  # Server

  @impl true
  def init(opts) do
    config = Keyword.get(opts, :config, default_config())
    model = Keyword.get(opts, :model, "claude-sonnet-4")
    {:ok, %__MODULE__{current_config: config, model: model}}
  end

  @impl true
  def handle_cast({:start_research, opts}, state) do
    config = Keyword.get(opts, :config, state.current_config)
    model = Keyword.get(opts, :model, state.model)

    new_state = %{state | current_config: config, model: model, status: :running}

    broadcast(:status_changed, %{status: :running})

    # Start the loop in a separate task to not block the GenServer
    self_pid = self()
    Task.start(fn -> experiment_loop(self_pid) end)

    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:stop_research, state) do
    Logger.info("Research stop requested — will stop after current experiment")
    {:noreply, %{state | status: :stopping}}
  end

  @impl true
  def handle_call(:status, _from, state) do
    {:reply,
     %{
       status: state.status,
       experiment_count: length(state.experiments),
       baseline_loss: state.baseline_loss,
       current_config: state.current_config
     }, state}
  end

  @impl true
  def handle_call(:experiments, _from, state) do
    {:reply, state.experiments, state}
  end

  @impl true
  def handle_call({:run_experiment, config, description}, _from, state) do
    experiment_id = generate_id()
    Logger.info("[#{experiment_id}] Running: #{description}")

    broadcast(:experiment_started, %{
      experiment_id: experiment_id,
      description: description,
      config: config
    })

    # Run training
    result =
      case Trainer.train(config, experiment_id: experiment_id) do
        {:ok, train_result} ->
          Map.merge(train_result, %{description: description})

        {:error, reason} ->
          %{
            experiment_id: experiment_id,
            status: :crashed,
            description: description,
            error: inspect(reason),
            final_loss: nil
          }
      end

    # Decide: keep or discard
    {kept, new_baseline, new_config} = decide(result, state)

    full_result = Map.merge(result, %{kept: kept})

    new_state = %{state |
      experiments: state.experiments ++ [full_result],
      baseline_loss: new_baseline,
      current_config: new_config
    }

    broadcast(:experiment_completed, full_result)

    {:reply, full_result, new_state}
  end

  @impl true
  def handle_call(:should_continue?, _from, state) do
    {:reply, state.status == :running, state}
  end

  @impl true
  def handle_call({:update_status, new_status}, _from, state) do
    {:reply, :ok, %{state | status: new_status}}
  end

  # Experiment loop (runs in a Task)

  defp experiment_loop(agent_pid) do
    # Ensure Trainer is running
    case GenServer.whereis(Trainer) do
      nil -> Trainer.start_link()
      _pid -> :ok
    end

    # Run baseline first
    state = GenServer.call(agent_pid, :status)
    if state.experiment_count == 0 do
      Logger.info("Running baseline experiment...")
      config = state.current_config
      GenServer.call(agent_pid, {:run_experiment, config, "baseline"}, :infinity)
    end

    # Loop
    loop(agent_pid)
  end

  defp loop(agent_pid) do
    if GenServer.call(agent_pid, :should_continue?) do
      case propose_next_experiment(agent_pid) do
        {:ok, config, description} ->
          GenServer.call(agent_pid, {:run_experiment, config, description}, :infinity)
          loop(agent_pid)

        {:error, reason} ->
          Logger.error("Failed to propose experiment: #{inspect(reason)}")
          # Wait and retry
          Process.sleep(5_000)
          loop(agent_pid)
      end
    else
      Logger.info("Research loop stopped")
      GenServer.call(agent_pid, {:update_status, :idle})
      broadcast(:status_changed, %{status: :idle})
    end
  end

  defp propose_next_experiment(agent_pid) do
    state = GenServer.call(agent_pid, :status)
    experiments = GenServer.call(agent_pid, :experiments)

    prompt_text = Program.format_proposal_request(experiments, state.current_config)

    Logger.info("Asking LLM for next experiment...")
    broadcast(:agent_thinking, %{prompt: String.slice(prompt_text, 0, 200) <> "..."})

    case LLM.prompt(prompt_text, system: Program.system_prompt()) do
      {:ok, response} ->
        Logger.info("LLM response: #{String.slice(response, 0, 200)}")
        broadcast(:agent_responded, %{response: response})
        parse_proposal(response, state.current_config)

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp parse_proposal(response, current_config) do
    # Extract JSON from response (may be wrapped in markdown code blocks)
    json_str =
      case Regex.run(~r/```(?:json)?\s*\n(.*?)\n```/s, response) do
        [_, json] -> json
        _ -> response
      end

    case Jason.decode(json_str) do
      {:ok, %{"changes" => changes, "description" => desc}} ->
        new_config = apply_changes(current_config, changes)
        {:ok, new_config, desc}

      {:ok, %{"changes" => changes}} ->
        new_config = apply_changes(current_config, changes)
        {:ok, new_config, "LLM experiment"}

      {:error, _} ->
        # Try to parse the whole response as config changes
        Logger.warning("Could not parse LLM response as JSON, using baseline config")
        {:ok, current_config, "retry (parse error)"}
    end
  end

  defp apply_changes(config, changes) when is_map(changes) do
    Enum.reduce(changes, config, fn {key, value}, acc ->
      atom_key = String.to_existing_atom(key)
      if Map.has_key?(acc, atom_key) do
        Map.put(acc, atom_key, value)
      else
        Logger.warning("Unknown config key: #{key}")
        acc
      end
    end)
  rescue
    _ -> config
  end

  defp decide(result, state) do
    case {result[:final_loss], state.baseline_loss} do
      {nil, baseline} ->
        # Crash — discard
        {false, baseline, state.current_config}

      {loss, nil} ->
        # First run — this becomes the baseline
        Logger.info("Baseline established: loss=#{loss}")
        {true, loss, state.current_config}

      {loss, baseline} when loss < baseline ->
        improvement = Float.round(baseline - loss, 6)
        Logger.info("✅ Improvement! #{baseline} → #{loss} (Δ#{improvement})")
        {true, loss, result[:config] || state.current_config}

      {loss, baseline} ->
        Logger.info("❌ No improvement: #{loss} >= #{baseline}")
        {false, baseline, state.current_config}
    end
  end

  defp default_config do
    %Config{
      n_layer: 2,
      aspect_ratio: 16,
      n_head: 2,
      n_kv_head: 2,
      head_dim: 16,
      vocab_size: 256,
      sequence_len: 32,
      device_batch_size: 4,
      time_budget: 15,
      matrix_lr: 0.01
    }
  end

  defp broadcast(event, payload) do
    Phoenix.PubSub.broadcast(ExAutoresearch.PubSub, "agent:events", {event, payload})
  rescue
    _ -> :ok
  end

  defp generate_id do
    :crypto.strong_rand_bytes(4) |> Base.hex_encode32(case: :lower, padding: false)
  end
end
