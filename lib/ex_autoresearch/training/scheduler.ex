defmodule ExAutoresearch.Training.Scheduler do
  @moduledoc """
  Progress-based learning rate scheduler with warmup and warmdown.

  Unlike step-based schedulers, this uses wall-clock time progress
  (elapsed / time_budget) to compute the LR multiplier, making
  experiments with different batch sizes directly comparable.
  """

  alias ExAutoresearch.Model.Config

  @doc """
  Compute the LR multiplier given training progress (0.0 to 1.0).

  Returns a float in [final_lr_frac, 1.0].
  """
  def lr_multiplier(progress, %Config{} = config) do
    cond do
      # Warmup phase: linear ramp from 0 to 1
      config.warmup_ratio > 0 and progress < config.warmup_ratio ->
        progress / config.warmup_ratio

      # Warmdown phase: linear decay from 1 to final_lr_frac
      config.warmdown_ratio > 0 and progress > (1.0 - config.warmdown_ratio) ->
        warmdown_progress = (progress - (1.0 - config.warmdown_ratio)) / config.warmdown_ratio
        1.0 - warmdown_progress * (1.0 - config.final_lr_frac)

      # Steady state
      true ->
        1.0
    end
  end

  @doc """
  Compute weight decay multiplier: linearly decays to 0 as training progresses.
  """
  def weight_decay_multiplier(progress) do
    max(0.0, 1.0 - progress)
  end
end
