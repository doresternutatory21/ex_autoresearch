defmodule ExAutoresearch.Training.Metrics do
  @moduledoc """
  Training evaluation metrics, primarily Bits Per Byte (BPB).

  BPB is a vocabulary-size-independent metric:
    BPB = total_cross_entropy_nats / (ln(2) × total_utf8_bytes)

  Lower is better. It measures compression efficiency in bits per byte of text.
  """

  import Nx.Defn

  @doc """
  Compute bits per byte from per-token losses and byte counts.

  loss_per_token: {n_tokens} — cross-entropy loss per token (nats)
  bytes_per_token: {n_tokens} — UTF-8 byte count per token
  """
  defn compute_bpb(loss_per_token, bytes_per_token) do
    mask = Nx.greater(bytes_per_token, 0)
    total_nats = Nx.sum(loss_per_token * mask)
    total_bytes = Nx.sum(bytes_per_token * mask)

    total_nats / (Nx.log(Nx.tensor(2.0)) * total_bytes)
  end
end
