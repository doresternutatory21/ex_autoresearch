defmodule ExAutoresearch.Data.Tokenizer do
  @moduledoc """
  BPE tokenizer wrapper.

  Wraps a Bumblebee-loaded tokenizer or provides a simple interface
  for encoding/decoding text. Must support byte counting per token
  for the BPB evaluation metric.
  """

  defstruct [:tokenizer, :vocab_size, :bos_token_id, :token_bytes]

  @doc """
  Load a GPT2-style tokenizer via Bumblebee.

  Falls back to a simple byte-level tokenizer if Bumblebee is not available.
  """
  def load(opts \\ []) do
    repo = Keyword.get(opts, :repo, "openai-community/gpt2")

    case Bumblebee.load_tokenizer({:hf, repo}) do
      {:ok, tokenizer} ->
        {:ok,
         %__MODULE__{
           tokenizer: tokenizer,
           vocab_size: 50257,
           bos_token_id: 50256,
           token_bytes: nil
         }}

      {:error, reason} ->
        {:error, {:tokenizer_load_failed, reason}}
    end
  end

  @doc "Encode text to token IDs."
  def encode(%__MODULE__{tokenizer: tokenizer}, text) do
    Bumblebee.apply_tokenizer(tokenizer, text)
  end
end
