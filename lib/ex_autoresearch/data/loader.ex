defmodule ExAutoresearch.Data.Loader do
  @moduledoc """
  Data loader with best-fit packing.

  Packs variable-length documents into fixed-length sequences with minimal waste.
  Prepends BOS token per document. Returns {input, target} Nx tensors
  as an infinite stream of batches.

  For initial development, generates synthetic data so training can proceed
  without the full HuggingFace download.
  """

  alias ExAutoresearch.Model.Config

  @doc """
  Create an infinite stream of {input_map, targets} batches for Axon.Loop.

  Each batch is:
  - input: %{"input_ids" => tensor {batch_size, seq_len} of s64}
  - target: tensor {batch_size, seq_len, vocab_size} of f32 (one-hot)
  """
  def stream(%Config{} = config, opts \\ []) do
    mode = Keyword.get(opts, :mode, :synthetic)
    batch_size = Keyword.get(opts, :batch_size, config.device_batch_size)
    seq_len = config.sequence_len
    vocab_size = config.vocab_size

    case mode do
      :synthetic -> synthetic_stream(batch_size, seq_len, vocab_size)
      :file -> file_stream(config, opts)
    end
  end

  @doc """
  Create a synthetic data stream for testing.

  Generates sequences where targets are input shifted by 1 (next-token prediction).
  Uses a simple repeating pattern so the model can learn something meaningful.
  """
  def synthetic_stream(batch_size, seq_len, vocab_size) do
    Stream.repeatedly(fn ->
      # Generate patterned sequences the model can learn
      # Pattern: incrementing tokens with wrapping
      base = :rand.uniform(vocab_size) - 1
      input = Nx.iota({batch_size, seq_len}, type: :s64)
             |> Nx.add(base)
             |> Nx.remainder(vocab_size)

      targets = Nx.add(input, 1) |> Nx.remainder(vocab_size)

      # One-hot encode targets for cross-entropy loss
      targets_oh =
        Nx.equal(
          Nx.iota({batch_size * seq_len, vocab_size}, axis: 1),
          Nx.reshape(targets, {batch_size * seq_len, 1})
        )
        |> Nx.reshape({batch_size, seq_len, vocab_size})
        |> Nx.as_type(:f32)

      {%{"input_ids" => input}, targets_oh}
    end)
  end

  @doc """
  Create a file-backed data stream from downloaded parquet shards.

  Placeholder — will read parquet files and apply best-fit packing
  when the full data pipeline is implemented.
  """
  def file_stream(%Config{} = config, _opts) do
    # TODO: implement parquet reading + best-fit packing
    # For now, fall back to synthetic
    synthetic_stream(config.device_batch_size, config.sequence_len, config.vocab_size)
  end
end
