defmodule ExAutoresearch.Model.MLP do
  @moduledoc """
  MLP block with ReLU² activation.

  Architecture: Linear(n_embd → 4*n_embd) → ReLU² → Linear(4*n_embd → n_embd)

  ReLU² = relu(x)² provides sharper gating than standard ReLU or GELU,
  which empirically improves training efficiency for small models.
  """

  import Nx.Defn

  @doc "ReLU² activation: relu(x)²"
  defn relu_squared(x) do
    x |> Nx.max(0) |> Nx.pow(2)
  end
end
