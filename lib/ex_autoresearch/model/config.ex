defmodule ExAutoresearch.Model.Config do
  @moduledoc """
  GPT model configuration struct.

  All hyperparameters that the agent can tune live here.
  The `n_embd` field is derived: `n_layer * aspect_ratio`.
  """

  @enforce_keys [:n_layer]
  defstruct [
    # Architecture
    n_layer: 8,
    n_head: 6,
    n_kv_head: 6,
    aspect_ratio: 64,
    head_dim: 128,
    window_pattern: "SSSL",

    # Fixed (from tokenizer / data)
    vocab_size: 8192,
    sequence_len: 2048,

    # Training
    total_batch_size: 524_288,
    device_batch_size: 128,
    embedding_lr: 0.6,
    unembedding_lr: 0.004,
    matrix_lr: 0.04,
    scalar_lr: 0.5,
    weight_decay: 0.2,
    adam_beta1: 0.8,
    adam_beta2: 0.95,
    warmup_ratio: 0.0,
    warmdown_ratio: 0.5,
    final_lr_frac: 0.0,
    time_budget: 300,
    softcap: 15.0
  ]

  @type t :: %__MODULE__{}

  @doc "Compute embedding dimension from depth and aspect ratio."
  def n_embd(%__MODULE__{n_layer: n_layer, aspect_ratio: aspect_ratio}) do
    n_layer * aspect_ratio
  end

  @doc "Number of gradient accumulation steps."
  def grad_accum_steps(%__MODULE__{} = config) do
    tokens_per_micro = config.device_batch_size * config.sequence_len
    div(config.total_batch_size, tokens_per_micro)
  end

  @doc "Parse window pattern string into per-layer window sizes."
  def window_sizes(%__MODULE__{} = config) do
    half_seq = div(config.sequence_len, 2)
    full_seq = config.sequence_len

    pattern =
      config.window_pattern
      |> String.graphemes()
      |> Enum.map(fn
        "S" -> half_seq
        "L" -> full_seq
        other -> raise "Unknown window pattern character: #{other}"
      end)

    # Repeat pattern to fill n_layer, last layer always full
    Stream.cycle(pattern)
    |> Enum.take(config.n_layer)
    |> List.replace_at(config.n_layer - 1, full_seq)
  end
end
