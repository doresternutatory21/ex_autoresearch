## Module Template

```elixir
defmodule ExAutoresearch.Experiments.V_VERSION_ID do
  @moduledoc "DESCRIPTION OF WHAT THIS VERSION CHANGES"

  import Nx.Defn

  def config do
    %{
      n_layer: 2,
      n_embd: 32,
      n_head: 2,
      n_kv_head: 2,
      head_dim: 16,
      vocab_size: 256,
      sequence_len: 32,
      batch_size: 4
    }
  end

  def build do
    config = config()
    input = Axon.input("input_ids", shape: {nil, config.sequence_len})

    x = Axon.embedding(input, config.vocab_size, config.n_embd, name: "token_emb")

    x =
      Enum.reduce(0..(config.n_layer - 1), x, fn i, acc ->
        transformer_block(acc, config, i)
      end)

    x = Axon.layer_norm(x, name: "final_norm", epsilon: 1.0e-6)
    Axon.dense(x, config.vocab_size, use_bias: false, name: "lm_head")
  end

  defp transformer_block(x, config, i) do
    # Pre-norm → Attention → Residual
    normed = Axon.layer_norm(x, name: "b#{i}_norm1", epsilon: 1.0e-6)

    attn =
      normed
      |> Axon.dense(config.n_embd * 3, use_bias: false, name: "b#{i}_qkv")
      |> Axon.nx(fn qkv -> simple_causal_attention(qkv, config.n_head, config.n_embd) end)
      |> Axon.dense(config.n_embd, use_bias: false, name: "b#{i}_attn_out")

    x = Axon.add(x, attn)

    # Pre-norm → MLP → Residual
    normed = Axon.layer_norm(x, name: "b#{i}_norm2", epsilon: 1.0e-6)

    mlp =
      normed
      |> Axon.dense(4 * config.n_embd, use_bias: false, name: "b#{i}_mlp_up")
      |> Axon.activation(:relu)
      |> Axon.nx(fn x -> Nx.pow(x, 2) end)
      |> Axon.dense(config.n_embd, use_bias: false, name: "b#{i}_mlp_down")

    Axon.add(x, mlp)
  end

  defp simple_causal_attention(qkv, n_head, n_embd) do
    head_dim = div(n_embd, n_head)
    {batch, seq_len, _} = Nx.shape(qkv)

    q = qkv[[.., .., 0..(n_embd - 1)//1]] |> Nx.reshape({batch, seq_len, n_head, head_dim})
    k = qkv[[.., .., n_embd..(2 * n_embd - 1)//1]] |> Nx.reshape({batch, seq_len, n_head, head_dim})
    v = qkv[[.., .., (2 * n_embd)..(3 * n_embd - 1)//1]] |> Nx.reshape({batch, seq_len, n_head, head_dim})

    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    scale = Nx.rsqrt(Nx.tensor(head_dim, type: :f32))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)

    mask = Nx.iota({seq_len, seq_len})
    causal = Nx.greater(Nx.iota({1, seq_len}), Nx.iota({seq_len, 1}))
    scores = Nx.select(causal, Nx.Constants.neg_infinity(:f32), scores)

    weights = Axon.Activations.softmax(scores, axis: -1)
    out = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    out |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({batch, seq_len, n_head * head_dim})
  end

  def optimizer do
    Polaris.Optimizers.adamw(learning_rate: 0.01)
  end
end
```
