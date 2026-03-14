defmodule ExAutoresearch.Agent.Prompts do
  @moduledoc """
  Reads and composes prompts from priv/prompts/*.md files.

  Prompts are editable at runtime - changes are picked up on next read.
  """

  @prompts_dir "priv/prompts"

  def read(filename) do
    path = Path.join([@prompts_dir, filename])
    case File.read(path) do
      {:ok, content} -> content
      {:error, _} -> "# #{filename} not found"
    end
  end

  def system_prompt do
    [read("system.md"), read("strategy.md"), read("constraints.md")]
    |> Enum.join("\n\n")
  end

  def build_proposal_prompt(history, best, kept_versions, version_id) do
    template_code = read("template.md")

    best_section =
      if best do
        """
        ## Current best version (v_#{best.version_id}, loss: #{safe_round(best.final_loss, 6)})

        ```elixir
        #{best.code}
        ```
        """
      else
        "No experiments yet. Generate the baseline using the template below."
      end

    history_section =
      if history != [] do
        recent = Enum.take(history, -20)
        total = length(history)
        kept_count = Enum.count(history, & &1.kept)

        rows =
          recent
          |> Enum.map(fn e ->
            loss = safe_round(e.final_loss, 6) || "crash"
            status = if e.kept, do: "✅ kept", else: "❌ discarded"
            desc = String.slice(e.description || "", 0, 80)
            model_tag = short_model(e.model)
            "| v_#{e.version_id} | #{loss} | #{e.num_steps || 0} | #{model_tag} | #{desc} | #{status} |"
          end)
          |> Enum.join("\n")

        """
        ## Experiment history (#{total} total, #{kept_count} kept, showing last 20)

        | Version | Loss | Steps | Model | Description | Status |
        |---------|------|-------|-------|-------------|--------|
        #{rows}
        """
      else
        ""
      end

    notable_section =
      kept_versions
      |> Enum.filter(& &1.code)
      |> Enum.take(3)
      |> Enum.map(fn e ->
        """
        ### v_#{e.version_id} (loss: #{safe_round(e.final_loss, 6)}) - #{e.description}

        ```elixir
        #{e.code}
        ```
        """
      end)
      |> Enum.join("\n")

    notable_header = if notable_section != "", do: "## Notable kept versions (source code)\n\n#{notable_section}", else: ""

    """
    #{best_section}

    #{history_section}

    #{notable_header}

    ## Your task

    Generate a NEW experiment version. Your module must be named ExAutoresearch.Experiments.V_#{version_id}.

    Output ONLY the complete defmodule block - no explanation outside the code.
    Put your reasoning in the @moduledoc string.

    #{template_code}
    """
  end

  defp safe_round(val, decimals) when is_float(val), do: Float.round(val, decimals)
  defp safe_round(val, _decimals) when is_integer(val), do: val / 1
  defp safe_round(_, _), do: nil

  defp short_model(nil), do: "-"
  defp short_model(m) do
    m
    |> String.replace("claude-", "")
    |> String.replace("gpt-", "gpt")
    |> String.replace("-preview", "")
  end
end
