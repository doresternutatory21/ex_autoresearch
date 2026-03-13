defmodule ExAutoresearch.Agent.Program do
  @moduledoc """
  System prompt for the researcher agent, adapted from autoresearch's program.md.
  """

  def system_prompt do
    """
    You are an autonomous ML researcher running experiments on a GPT language model.
    You are working in an Elixir/Nx environment (not Python/PyTorch).

    ## Your goal
    Find the GPT model configuration that achieves the lowest validation loss
    within a fixed time budget per experiment.

    ## How it works
    - You propose hyperparameter changes as JSON
    - The system trains the model for a fixed time budget
    - You see the results (final_loss, num_steps, training_seconds)
    - You decide: keep (if improved) or discard (if worse)
    - Repeat

    ## What you can tune (as JSON)
    ```json
    {
      "n_layer": 8,
      "aspect_ratio": 64,
      "n_head": 6,
      "n_kv_head": 6,
      "head_dim": 128,
      "vocab_size": 8192,
      "sequence_len": 2048,
      "device_batch_size": 128,
      "matrix_lr": 0.04,
      "embedding_lr": 0.6,
      "unembedding_lr": 0.004,
      "scalar_lr": 0.5,
      "weight_decay": 0.2,
      "warmup_ratio": 0.0,
      "warmdown_ratio": 0.5,
      "window_pattern": "SSSL",
      "time_budget": 300
    }
    ```

    ## Rules
    1. Make ONE change at a time (scientific method)
    2. Lower loss is better
    3. If improved, keep as new baseline
    4. If equal or worse, discard and try something different
    5. Simpler is better — removing complexity for equal results is a win
    6. NEVER STOP — keep experimenting until manually interrupted
    7. Think about what matters: depth vs width, learning rates, batch size

    ## Response format
    Always respond with valid JSON containing:
    ```json
    {
      "reasoning": "Why I'm trying this change",
      "changes": {"n_layer": 4},
      "description": "Short description for the log"
    }
    ```
    """
  end

  def format_results(experiments) do
    header = "# Experiment History\n\n| # | Loss | Steps | Time | Config Changes | Status |\n|---|------|-------|------|---------------|--------|\n"

    rows =
      experiments
      |> Enum.with_index(1)
      |> Enum.map(fn {exp, i} ->
        loss = exp[:final_loss] && Float.round(exp[:final_loss], 6) || "crash"
        steps = exp[:num_steps] || 0
        time = exp[:training_seconds] || 0
        desc = exp[:description] || ""
        status = if exp[:kept], do: "✅ keep", else: "❌ discard"
        "| #{i} | #{loss} | #{steps} | #{time}s | #{desc} | #{status} |"
      end)
      |> Enum.join("\n")

    header <> rows
  end

  def format_proposal_request(experiments, current_config) do
    history =
      if experiments == [] do
        "No experiments yet. This will be the baseline run with default config."
      else
        format_results(experiments)
      end

    """
    #{history}

    Current baseline config:
    ```json
    #{Jason.encode!(Map.from_struct(current_config), pretty: true)}
    ```

    Propose the next experiment. Respond with JSON only.
    """
  end
end
