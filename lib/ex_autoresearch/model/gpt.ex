defmodule ExAutoresearch.Model.GPT do
  @moduledoc """
  GPT language model built with Axon.

  Assembles the full model from:
  - Token embeddings
  - N transformer blocks (RMS Norm → Attention → Residual → RMS Norm → MLP → Residual)
  - Final RMS Norm
  - LM Head (linear projection to vocab)
  - Logit softcap for numerical stability

  The model is constructed as an Axon graph and compiled via EXLA for GPU execution.
  """

  alias ExAutoresearch.Model.Config

  import Nx.Defn

  @doc "Logit softcap: softcap * tanh(logits / softcap)"
  defn softcap(logits, cap) do
    cap * Nx.tanh(logits / cap)
  end

  @doc "RMS normalization."
  defn rms_norm(x) do
    variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
    x * Nx.rsqrt(variance + 1.0e-6)
  end

  @doc """
  Build the GPT model as an Axon graph.

  Returns an Axon model that takes token IDs {batch, seq_len}
  and produces logits {batch, seq_len, vocab_size}.
  """
  def build(%Config{} = config) do
    n_embd = Config.n_embd(config)

    input = Axon.input("input_ids", shape: {nil, config.sequence_len})

    # Token embedding
    x = Axon.embedding(input, config.vocab_size, n_embd, name: "token_emb")

    # Transformer blocks
    x =
      Enum.reduce(0..(config.n_layer - 1), x, fn i, acc ->
        transformer_block(acc, config, n_embd, i)
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm", epsilon: 1.0e-6)

    # LM head (project to vocab)
    Axon.dense(x, config.vocab_size, use_bias: false, name: "lm_head")
  end

  defp transformer_block(x, config, n_embd, layer_idx) do
    # Pre-norm → Attention → Residual
    normed = Axon.layer_norm(x, name: "block_#{layer_idx}_norm1", epsilon: 1.0e-6)

    attn =
      normed
      |> Axon.dense(n_embd * 3, use_bias: false, name: "block_#{layer_idx}_qkv")
      |> Axon.nx(fn qkv ->
        {q, k, v} = split_qkv(qkv, config.n_head, n_embd)
        scaled_dot_product_attention(q, k, v)
      end)
      |> Axon.dense(n_embd, use_bias: false, name: "block_#{layer_idx}_attn_out")

    x = Axon.add(x, attn)

    # Pre-norm → MLP → Residual
    normed = Axon.layer_norm(x, name: "block_#{layer_idx}_norm2", epsilon: 1.0e-6)

    mlp =
      normed
      |> Axon.dense(4 * n_embd, use_bias: false, name: "block_#{layer_idx}_mlp_up")
      |> Axon.activation(:relu)
      |> Axon.nx(fn x -> Nx.pow(x, 2) end)
      |> Axon.dense(n_embd, use_bias: false, name: "block_#{layer_idx}_mlp_down")

    Axon.add(x, mlp)
  end

  defp split_qkv(qkv, n_head, n_embd) do
    head_dim = div(n_embd, n_head)
    {batch, seq_len, _} = Nx.shape(qkv)

    q = qkv[[.., .., 0..(n_embd - 1)//1]]
    k = qkv[[.., .., n_embd..(2 * n_embd - 1)//1]]
    v = qkv[[.., .., (2 * n_embd)..(3 * n_embd - 1)//1]]

    q = Nx.reshape(q, {batch, seq_len, n_head, head_dim})
    k = Nx.reshape(k, {batch, seq_len, n_head, head_dim})
    v = Nx.reshape(v, {batch, seq_len, n_head, head_dim})

    {q, k, v}
  end

  defp scaled_dot_product_attention(q, k, v) do
    {batch, seq_len, n_head, head_dim} = Nx.shape(q)
    scale = Nx.rsqrt(Nx.tensor(head_dim, type: :f32))

    # Transpose to {batch, n_head, seq, head_dim}
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    # Attention scores: {batch, n_head, seq, seq}
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)

    # Causal mask
    mask = ExAutoresearch.Model.Attention.causal_mask(seq_len)
    scores = Nx.add(scores, mask)

    # Softmax
    weights = Axon.Activations.softmax(scores, axis: -1)

    # Weighted sum
    out = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Transpose back and reshape: {batch, seq, n_embd}
    out
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, n_head * head_dim})
  end
end
