alias ExAutoresearch.Experiments.{Registry, Loader}

# Find the best trial across all campaigns
campaigns = Registry.list_campaigns()

best =
  Enum.flat_map(campaigns, fn c ->
    Registry.all_trials(c.id)
    |> Enum.filter(fn t -> t.kept && t.final_loss != nil && t.code != nil end)
    |> Enum.map(fn t -> {c, t} end)
  end)
  |> Enum.min_by(fn {_, t} -> t.final_loss end, fn -> nil end)

case best do
  nil ->
    IO.puts("No kept trials found.")
    System.halt(1)

  {campaign, trial} ->
    # Load and compile module
    code = Loader.inject_version_id(trial.code, trial.version_id)
    {module, mermaid} =
      case Loader.load(trial.version_id, code) do
        {:ok, mod} ->
          mmd = try do
            mod.build() |> ExAutoresearch.Model.Display.as_mermaid()
          rescue
            _ -> nil
          end
          {mod, mmd}
        _ ->
          {nil, nil}
      end

    config = if module, do: (try do module.config() rescue _ -> %{} end), else: %{}

    # Build markdown
    md = """
    # 🏆 Best Experiment: v_#{trial.version_id}

    | Metric | Value |
    |--------|-------|
    | **Loss** | #{trial.final_loss} |
    | **Steps** | #{trial.num_steps} |
    | **Training time** | #{trial.training_seconds}s |
    | **Campaign** | #{campaign.tag} |
    | **GPU** | #{trial.gpu || "unknown"} |
    | **Model** | #{trial.model} |
    | **Step budget** | #{campaign.step_budget || "time-based"} |

    ## Configuration

    ```elixir
    #{inspect(config, pretty: true, limit: :infinity)}
    ```

    ## Description

    #{trial.description}

    #{if trial.reasoning && trial.reasoning != trial.description, do: "## Reasoning\n\n#{trial.reasoning}\n", else: ""}

    ## Source Code

    ```elixir
    #{trial.code}
    ```

    #{if mermaid do """
    ## Architecture Diagram

    ```mermaid
    #{mermaid}
    ```
    """ else "" end}
    """

    File.write!("winner.md", md)
    IO.puts("=== Best: v_#{trial.version_id} (#{campaign.tag}) ===")
    IO.puts("Loss: #{trial.final_loss}")
    IO.puts("Steps: #{trial.num_steps} in #{trial.training_seconds}s")
    IO.puts("GPU: #{trial.gpu || "unknown"}")
    IO.puts("Description: #{String.slice(trial.description || "", 0, 120)}")
    IO.puts("\nWritten to winner.md")
end
