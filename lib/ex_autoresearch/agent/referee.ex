defmodule ExAutoresearch.Agent.Referee do
  @moduledoc """
  Monitors concurrent training trials and kills losing ones early.

  Subscribes to PubSub step events. When multiple trials are in-flight,
  compares their loss at common step checkpoints. A trial is killed if:

  1. Both trials have reached the comparison checkpoint (50% of step_budget)
  2. One trial's loss is >20% worse than the other at the same step count
  3. OR a trial's loss is rising (last 1000 steps trending upward)

  Killing frees the GPU to start the next experiment immediately.
  """

  use GenServer

  require Logger

  alias ExAutoresearch.Experiments.Runner

  defstruct [:step_budget, trials: %{}, killed: MapSet.new()]

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end

  @impl true
  def init(opts) do
    step_budget = Keyword.fetch!(opts, :step_budget)
    Phoenix.PubSub.subscribe(ExAutoresearch.PubSub, "agent:events")
    {:ok, %__MODULE__{step_budget: step_budget}}
  end

  @impl true
  def handle_info({:trial_started, %{version_id: vid}}, state) do
    {:noreply, %{state | trials: Map.put(state.trials, vid, %{points: []})}}
  end

  @impl true
  def handle_info({:trial_completed, %{version_id: vid}}, state) do
    {:noreply, %{state |
      trials: Map.delete(state.trials, vid),
      killed: MapSet.delete(state.killed, vid)
    }}
  end

  @impl true
  def handle_info({:step, %{version_id: vid, step: step, loss: loss}}, state) when is_number(loss) do
    # Ignore events from already-killed trials
    if MapSet.member?(state.killed, vid) do
      {:noreply, state}
    else
      state = update_in(state.trials[vid], fn
        nil -> %{points: [{step, loss}]}
        trial -> %{trial | points: [{step, loss} | Enum.take(trial.points, 99)]}
      end)

      active = state.trials |> Enum.filter(fn {_, t} -> length(t.points) > 0 end)

      state =
        if length(active) >= 2 do
          maybe_kill_loser(state, active)
        else
          state
        end

      {:noreply, state}
    end
  end

  @impl true
  def handle_info({:queue_migration, winner_vid}, state) do
    # Get the winner's checkpoint from the node that ran it
    checkpoint =
      Runner.get_checkpoint(winner_vid) ||
        Enum.find_value(Node.list(), fn node ->
          case :rpc.call(node, Runner, :get_checkpoint, [winner_vid], 5_000) do
            nil -> nil
            {:badrpc, _} -> nil
            ckpt -> ckpt
          end
        end)

    if checkpoint do
      case ExAutoresearch.Experiments.Registry.get_trial(winner_vid) do
        {:ok, trial} when not is_nil(trial) and not is_nil(trial.code) ->
          # Find the fastest connected worker node
          fast_node =
            Node.list()
            |> Enum.find(fn n ->
              name = Atom.to_string(n)
              String.contains?(name, "cuda") or String.contains?(name, "worker")
            end)
            |> Kernel.||(node())

          migration = %{version_id: winner_vid, code: trial.code, checkpoint: checkpoint}
          ExAutoresearch.Agent.Researcher.queue_migration(fast_node, migration)
          Logger.info("🔄 Migration queued: v_#{winner_vid} → #{fast_node} (#{div(byte_size(checkpoint), 1024)} KB)")

        _ ->
          Logger.warning("Migration failed: no trial code for v_#{winner_vid}")
      end
    else
      Logger.warning("Migration failed: no checkpoint for v_#{winner_vid} — winner continues on slow GPU")
    end

    {:noreply, state}
  end

  @impl true
  def handle_info(_, state), do: {:noreply, state}

  # Compare trials at their most recent common step range.
  # Kill the loser. If the winner is on a slower GPU, halt it too and
  # queue a migration to the faster (now-freed) GPU.
  defp maybe_kill_loser(state, active) do
    checkpoint = div(state.step_budget, 2)

    # Only act when all trials are past the checkpoint
    all_past_checkpoint? = Enum.all?(active, fn {_, t} ->
      {latest_step, _} = hd(t.points)
      latest_step >= checkpoint
    end)

    if all_past_checkpoint? do
      with_loss =
        Enum.map(active, fn {vid, t} ->
          {_step, loss} = Enum.min_by(t.points, fn {s, _} -> abs(s - checkpoint) end)
          {latest_step, _} = hd(t.points)
          {vid, loss, loss_trending_up?(t.points), latest_step}
        end)

      {best_vid, best_loss, _, _} = Enum.min_by(with_loss, fn {_, loss, _, _} -> loss end)
      {worst_vid, worst_loss, _, _} = Enum.max_by(with_loss, fn {_, loss, _, _} -> loss end)

      if best_vid != worst_vid do
        ratio = worst_loss / best_loss
        {_, _, trending_up?, _} = Enum.find(with_loss, fn {vid, _, _, _} -> vid == worst_vid end)

        should_kill? = ratio > 1.2 or trending_up?

        if should_kill? do
          {_, _, _, best_steps} = Enum.find(with_loss, fn {vid, _, _, _} -> vid == best_vid end)
          {_, _, _, worst_steps} = Enum.find(with_loss, fn {vid, _, _, _} -> vid == worst_vid end)
          winner_is_slower? = best_steps < worst_steps

          reason = if trending_up?, do: "loss trending upward", else: "#{Float.round((ratio - 1) * 100, 1)}% worse"
          Logger.info("🏁 Referee: halting v_#{worst_vid} (#{reason} vs v_#{best_vid})")
          kill_trial(worst_vid)

          if winner_is_slower? do
            Logger.info("🔄 Referee: migrating winner v_#{best_vid} to faster GPU")
            kill_trial(best_vid)
            Process.send_after(self(), {:queue_migration, best_vid}, 3_000)

            %{state |
              trials: state.trials |> Map.delete(worst_vid) |> Map.delete(best_vid),
              killed: state.killed |> MapSet.put(worst_vid) |> MapSet.put(best_vid)
            }
          else
            %{state |
              trials: Map.delete(state.trials, worst_vid),
              killed: MapSet.put(state.killed, worst_vid)
            }
          end
        else
          state
        end
      else
        state
      end
    else
      state
    end
  end

  # Check if loss is trending upward over the last ~20 recorded points
  defp loss_trending_up?(points) when length(points) < 10, do: false

  defp loss_trending_up?(points) do
    recent = Enum.take(points, 10)
    {_, newest_loss} = hd(recent)
    {_, oldest_loss} = List.last(recent)
    # Loss is rising if newest > oldest by >5%
    newest_loss > oldest_loss * 1.05
  end

  defp kill_trial(version_id) do
    # Signal local halt
    Runner.halt(version_id)

    # Also signal on all connected nodes (for remote trials)
    for node <- Node.list() do
      :rpc.cast(node, Runner, :halt, [version_id])
    end
  end
end
