defmodule ExAutoresearch.Agent.Prompts do
  @moduledoc """
  Reads and composes prompts from priv/prompts/*.md files.

  Prompts are editable at runtime — changes are picked up on next read.
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

  def build_proposal_prompt(history, best, version_id) do
    template_code = read("template.md")

    best_section =
      if best do
        """
        ## Current best version (v_#{best.version_id}, loss: #{best.loss && Float.round(best.loss, 6)})

        ```elixir
        #{best.code}
        ```
        """
      else
        "No experiments yet. Generate the baseline using the template below."
      end

    history_section =
      if history != [] do
        rows =
          history
          |> Enum.map(fn e ->
            loss = e.loss && Float.round(e.loss, 6) || "crash"
            status = if e.kept, do: "✅ kept", else: "❌ discarded"
            "| v_#{e.version_id} | #{loss} | #{e.steps || 0} | #{e.description || ""} | #{status} |"
          end)
          |> Enum.join("\n")

        """
        ## Experiment history

        | Version | Loss | Steps | Description | Status |
        |---------|------|-------|-------------|--------|
        #{rows}
        """
      else
        ""
      end

    # Show source code of top 3 kept versions for combining ideas
    notable_section =
      history
      |> Enum.filter(& &1.kept)
      |> Enum.filter(& &1.code)
      |> Enum.sort_by(& &1.loss)
      |> Enum.take(3)
      |> Enum.map(fn e ->
        """
        ### v_#{e.version_id} (loss: #{e.loss && Float.round(e.loss, 6)}) — #{e.description}

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

    Generate a NEW experiment version. Your module must be named `ExAutoresearch.Experiments.V_#{version_id}`.

    Output ONLY the complete `defmodule` block — no explanation outside the code.
    Put your reasoning in the @moduledoc string.

    #{template_code}
    """
  end
end
