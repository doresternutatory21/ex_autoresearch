defmodule ExAutoresearch.Model.Attention do
  @moduledoc """
  Causal self-attention with Rotary Position Embeddings (RoPE).

  Implements standard O(n²) dot-product attention with:
  - Grouped Query Attention (GQA): n_kv_head ≤ n_head
  - Rotary embeddings for position encoding
  - Configurable window size for sliding window attention
  - Causal masking
  """

  import Nx.Defn

  @rope_base 10_000.0

  @doc """
  Precompute cos/sin frequency tables for RoPE.

  Returns {cos, sin} tensors of shape {max_seq_len, head_dim}.
  """
  def precompute_rope_frequencies(head_dim, max_seq_len) do
    half_dim = div(head_dim, 2)
    exponents = Nx.iota({half_dim}) |> Nx.as_type(:f32)
    freqs = Nx.exp(Nx.negate(exponents) * (Nx.log(Nx.tensor(@rope_base)) / half_dim))

    positions = Nx.iota({max_seq_len}) |> Nx.as_type(:f32)

    angles = Nx.outer(positions, freqs)

    cos = Nx.concatenate([Nx.cos(angles), Nx.cos(angles)], axis: -1)
    sin = Nx.concatenate([Nx.sin(angles), Nx.sin(angles)], axis: -1)

    {cos, sin}
  end

  @doc """
  Apply rotary position embeddings to Q or K tensor.

  x: {batch, seq_len, n_heads, head_dim}
  cos, sin: {seq_len, head_dim} (precomputed)
  """
  defn apply_rotary_emb(x, cos, sin) do
    {_batch, seq_len, _heads, head_dim} = Nx.shape(x)
    half = div(head_dim, 2)

    cos = cos[0..(seq_len - 1)//1] |> Nx.reshape({1, seq_len, 1, head_dim})
    sin = sin[0..(seq_len - 1)//1] |> Nx.reshape({1, seq_len, 1, head_dim})

    x_first = x[[.., .., .., 0..(half - 1)//1]]
    x_second = x[[.., .., .., half..(head_dim - 1)//1]]

    rotated = Nx.concatenate([Nx.negate(x_second), x_first], axis: -1)

    x * cos + rotated * sin
  end

  @doc """
  Build a causal attention mask with optional window size.

  Returns a mask of shape {1, 1, seq_len, seq_len} where
  allowed positions are 0.0 and masked positions are -infinity.
  """
  def causal_mask(seq_len, opts \\ []) do
    window_size = Keyword.get(opts, :window_size, seq_len)

    rows = Nx.iota({seq_len, 1})
    cols = Nx.iota({1, seq_len})

    causal = Nx.greater(cols, rows)
    window = Nx.greater(Nx.subtract(rows, cols), window_size)
    masked = Nx.logical_or(causal, window)

    mask = Nx.select(masked, Nx.Constants.neg_infinity(:f32), 0.0)
    Nx.reshape(mask, {1, 1, seq_len, seq_len})
  end
end
