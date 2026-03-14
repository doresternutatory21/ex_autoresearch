defmodule ExAutoresearch.Repo.Migrations.PersistentRuns do
  use Ecto.Migration

  def change do
    create table(:runs, primary_key: false) do
      add :id, :binary_id, null: false, primary_key: true
      add :tag, :string, null: false
      add :status, :string, default: "running"
      add :model, :string, default: "claude-sonnet-4"
      add :time_budget, :integer, default: 15
      add :base_config, :map
      add :best_experiment_id, :binary_id

      timestamps(type: :utc_datetime)
    end

    create unique_index(:runs, [:tag])

    # Recreate experiments with run_id + new fields
    drop_if_exists table(:experiments)

    create table(:experiments, primary_key: false) do
      add :id, :binary_id, null: false, primary_key: true
      add :run_id, references(:runs, type: :binary_id, on_delete: :delete_all), null: false
      add :version_id, :string, null: false
      add :status, :string, default: "pending"
      add :code, :text
      add :description, :text
      add :reasoning, :text
      add :parent_id, :binary_id
      add :model, :string
      add :config, :map
      add :final_loss, :float
      add :training_seconds, :float
      add :num_steps, :integer
      add :kept, :boolean, default: false
      add :error, :text

      timestamps(type: :utc_datetime)
    end

    create index(:experiments, [:run_id])
    create index(:experiments, [:version_id])
    create index(:experiments, [:run_id, :kept])
  end
end
