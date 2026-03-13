defmodule ExAutoresearch.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      ExAutoresearchWeb.Telemetry,
      ExAutoresearch.Repo,
      {Ecto.Migrator,
       repos: Application.fetch_env!(:ex_autoresearch, :ecto_repos), skip: skip_migrations?()},
      {Oban,
       AshOban.config(
         Application.fetch_env!(:ex_autoresearch, :ash_domains),
         Application.fetch_env!(:ex_autoresearch, Oban)
       )},
      # Start a worker by calling: ExAutoresearch.Worker.start_link(arg)
      # {ExAutoresearch.Worker, arg},
      # Start to serve requests, typically the last entry
      {DNSCluster, query: Application.get_env(:ex_autoresearch, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: ExAutoresearch.PubSub},
      ExAutoresearch.Training.Trainer,
      ExAutoresearch.Agent.Researcher,
      ExAutoresearchWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: ExAutoresearch.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    ExAutoresearchWeb.Endpoint.config_change(changed, removed)
    :ok
  end

  defp skip_migrations?() do
    # By default, sqlite migrations are run when using a release
    System.get_env("RELEASE_NAME") == nil
  end
end
