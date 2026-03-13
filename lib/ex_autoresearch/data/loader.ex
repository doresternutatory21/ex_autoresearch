defmodule ExAutoresearch.Data.Loader do
  @moduledoc """
  Data loader with best-fit packing.

  Packs variable-length documents into fixed-length sequences with 0% padding.
  Prepends BOS token per document. Returns {x, y} Nx tensors (input, target)
  as an infinite stream of batches.
  """

  # Placeholder — will be implemented when parquet reading is available
end
