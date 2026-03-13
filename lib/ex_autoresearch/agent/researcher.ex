defmodule ExAutoresearch.Agent.Researcher do
  @moduledoc """
  Autonomous experiment loop with hot-loaded versioned modules.

  The LLM generates complete Elixir modules that define GPT models.
  Each version is compiled and loaded into the running BEAM.
  The best versions survive; worse ones are discarded but preserved.
  """

  use GenServer

  require Logger

  alias ExAutoresearch.Experiments.{Registry, Loader, Runner}
  alias ExAutoresearch.Agent.{LLM, Prompts}

  @status_table __MODULE__.Status

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def start_research(opts \\ []), do: GenServer.cast(__MODULE__, {:start_research, opts})
  def stop_research, do: put_status(:status, :stopping)

  def status do
    best = Registry.best()
    %{
      status: get_status(:status, :idle),
      experiment_count: Registry.count(),
      best_loss: best && best.loss,
      best_version: best && best.version_id,
      model: get_status(:model, "claude-sonnet-4")
    }
  end

  # Server

  @impl true
  def init(opts) do
    :ets.new(@status_table, [:named_table, :set, :public, read_concurrency: true])
    put_status(:status, :idle)
    put_status(:model, Keyword.get(opts, :model, "claude-sonnet-4"))
    put_status(:time_budget, Keyword.get(opts, :time_budget, 15))
    {:ok, %{task: nil}}
  end

  @impl true
  def handle_cast({:start_research, opts}, state) do
    if opts[:model], do: put_status(:model, opts[:model])
    if opts[:time_budget], do: put_status(:time_budget, opts[:time_budget])

    put_status(:status, :running)
    broadcast(:status_changed, %{status: :running})

    task = Task.async(fn -> experiment_loop() end)
    {:noreply, %{state | task: task}}
  end

  @impl true
  def handle_info({ref, _result}, %{task: %Task{ref: ref}} = state) do
    Process.demonitor(ref, [:flush])
    put_status(:status, :idle)
    broadcast(:status_changed, %{status: :idle})
    {:noreply, %{state | task: nil}}
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, %{task: %Task{ref: ref}} = state) do
    Logger.error("Experiment loop crashed: #{inspect(reason, limit: 3)}")
    put_status(:status, :idle)
    broadcast(:status_changed, %{status: :idle})
    {:noreply, %{state | task: nil}}
  end

  @impl true
  def handle_info(_, state), do: {:noreply, state}

  # --- Experiment loop ---

  defp experiment_loop do
    # Run baseline if no experiments yet
    if Registry.count() == 0 do
      Logger.info("Loading baseline module...")
      run_baseline()
    end

    loop()
  end

  defp loop do
    if get_status(:status) == :running do
      case propose_and_run() do
        :ok -> loop()
        {:error, reason} ->
          Logger.error("Experiment failed: #{inspect(reason, limit: 3)}")
          Process.sleep(3_000)
          loop()
      end
    else
      Logger.info("Research loop stopped")
    end
  end

  defp run_baseline do
    version_id = gen_id()
    template = Prompts.read("template.md")

    # Extract the code block from the template
    code = case Regex.run(~r/```elixir\n(.*?)```/s, template) do
      [_, elixir_code] -> elixir_code
      _ -> template
    end

    code = Loader.inject_version_id(code, version_id)

    case Loader.load(version_id, code) do
      {:ok, module} ->
        Registry.register(version_id, %{
          module: module,
          code: code,
          description: "baseline",
          parent_id: nil,
          status: :running,
          kept: false,
          loaded_at: DateTime.utc_now()
        })

        broadcast(:experiment_started, %{version_id: version_id, description: "baseline"})

        result = Runner.run(module, version_id: version_id, time_budget: get_status(:time_budget, 15))

        Registry.update(version_id, %{
          loss: result[:loss],
          steps: result[:steps],
          training_seconds: result[:training_seconds],
          status: result[:status],
          kept: true
        })

        broadcast(:experiment_completed, Map.merge(result, %{description: "baseline", kept: true}))

        Logger.info("Baseline established: loss=#{result[:loss]}")

      {:error, reason} ->
        Logger.error("Baseline failed to load: #{inspect(reason)}")
    end
  end

  defp propose_and_run do
    version_id = gen_id()
    best = Registry.best()
    history = Registry.all()

    # Build prompt with full context
    prompt = Prompts.build_proposal_prompt(history, best, version_id)

    Logger.info("Asking LLM for experiment v_#{version_id}...")
    broadcast(:agent_thinking, %{version_id: version_id})

    case LLM.prompt(prompt, system: Prompts.system_prompt(), model: get_status(:model)) do
      {:ok, response} ->
        # Extract code and reasoning from response
        {code, description, reasoning} = parse_response(response, version_id)

        broadcast(:agent_responded, %{reasoning: reasoning, description: description})

        # Load and run
        code = Loader.inject_version_id(code, version_id)

        case Loader.load(version_id, code) do
          {:ok, module} ->
            parent_id = best && best.version_id

            Registry.register(version_id, %{
              module: module,
              code: code,
              description: description,
              parent_id: parent_id,
              status: :running,
              kept: false,
              loaded_at: DateTime.utc_now()
            })

            broadcast(:experiment_started, %{version_id: version_id, description: description})

            result = Runner.run(module, version_id: version_id, time_budget: get_status(:time_budget, 15))

            # Decide
            {kept, _} = decide(result[:loss], best && best.loss)

            Registry.update(version_id, %{
              loss: result[:loss],
              steps: result[:steps],
              training_seconds: result[:training_seconds],
              status: result[:status],
              kept: kept
            })

            broadcast(:experiment_completed, Map.merge(result, %{description: description, kept: kept}))

            :ok

          {:error, reason} ->
            Logger.error("Module v_#{version_id} failed to load: #{inspect(reason)}")

            Registry.register(version_id, %{
              module: nil,
              code: code,
              description: description,
              parent_id: best && best.version_id,
              status: :crashed,
              kept: false,
              loaded_at: DateTime.utc_now()
            })

            broadcast(:experiment_completed, %{
              version_id: version_id, description: description,
              kept: false, status: :crashed, loss: nil, steps: 0,
              error: inspect(reason)
            })

            :ok
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp parse_response(response, version_id) do
    # Try to extract code block
    code = case Regex.run(~r/```elixir\n(.*?)```/s, response) do
      [_, elixir_code] -> elixir_code
      _ -> response
    end

    # Try to extract reasoning (may be outside code block)
    reasoning = case Regex.run(~r/(?:reasoning|rationale|why)[:\s]*(.+?)(?:\n\n|```)/si, response) do
      [_, r] -> String.trim(r)
      _ ->
        # Try JSON format
        case Regex.run(~r/"reasoning"\s*:\s*"([^"]+)"/s, response) do
          [_, r] -> r
          _ -> String.slice(response, 0, 200)
        end
    end

    # Try to extract description from @moduledoc
    description = case Regex.run(~r/@moduledoc\s+"([^"]+)"/s, code) do
      [_, d] -> d
      _ -> "LLM experiment v_#{version_id}"
    end

    {code, description, reasoning}
  end

  defp decide(nil, _baseline), do: {false, nil}
  defp decide(loss, nil) do
    Logger.info("✅ Baseline: loss=#{loss}")
    {true, loss}
  end
  defp decide(loss, baseline) when loss < baseline do
    Logger.info("✅ Improvement! #{baseline} → #{loss} (Δ#{Float.round(baseline - loss, 6)})")
    {true, loss}
  end
  defp decide(loss, baseline) do
    Logger.info("❌ No improvement: #{loss} >= #{baseline}")
    {false, baseline}
  end

  # --- Helpers ---

  defp put_status(key, value), do: :ets.insert(@status_table, {key, value})
  defp get_status(key, default \\ nil) do
    case :ets.lookup(@status_table, key) do
      [{^key, value}] -> value
      [] -> default
    end
  rescue
    ArgumentError -> default
  end

  defp broadcast(event, payload) do
    Phoenix.PubSub.broadcast(ExAutoresearch.PubSub, "agent:events", {event, payload})
  rescue
    _ -> :ok
  end

  defp gen_id, do: :crypto.strong_rand_bytes(4) |> Base.hex_encode32(case: :lower, padding: false)
end
