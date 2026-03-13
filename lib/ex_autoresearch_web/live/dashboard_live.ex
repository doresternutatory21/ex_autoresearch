defmodule ExAutoresearchWeb.DashboardLive do
  use ExAutoresearchWeb, :live_view

  alias ExAutoresearch.Agent.Researcher
  alias ExAutoresearch.Experiments.Registry

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(ExAutoresearch.PubSub, "agent:events")
      :timer.send_interval(2000, self(), :tick)
    end

    agent = Researcher.status()
    experiments = Registry.all() |> Enum.map(&Map.put(&1, :id, &1.version_id))

    socket =
      socket
      |> assign(:agent_status, agent.status)
      |> assign(:best_loss, agent.best_loss)
      |> assign(:best_version, agent.best_version)
      |> assign(:experiment_count, agent.experiment_count)
      |> assign(:current_step, nil)
      |> assign(:current_progress, nil)
      |> assign(:agent_log, [])
      |> stream(:experiments, experiments)

    {:ok, socket}
  end

  @impl true
  def handle_event("start_research", _params, socket) do
    Researcher.start_research()
    {:noreply, assign(socket, :agent_status, :running)}
  end

  @impl true
  def handle_event("stop_research", _params, socket) do
    Researcher.stop_research()
    {:noreply, assign(socket, :agent_status, :stopping)}
  end

  @impl true
  def handle_info(:tick, socket) do
    agent = Researcher.status()

    socket =
      socket
      |> assign(:agent_status, agent.status)
      |> assign(:best_loss, agent.best_loss)
      |> assign(:best_version, agent.best_version)
      |> assign(:experiment_count, agent.experiment_count)

    {:noreply, socket}
  end

  @impl true
  def handle_info({:experiment_started, payload}, socket) do
    socket =
      socket
      |> assign(:current_step, 0)
      |> assign(:current_progress, 0)
      |> add_log("🧪 Started: #{payload[:description]} (v_#{payload[:version_id]})")

    {:noreply, socket}
  end

  @impl true
  def handle_info({:experiment_completed, result}, socket) do
    status = if result[:kept], do: "✅ kept", else: "❌ discarded"
    loss_str = result[:loss] && Float.round(result[:loss], 6) || "crash"

    entry = %{
      id: result[:version_id],
      version_id: result[:version_id],
      loss: result[:loss],
      steps: result[:steps],
      training_seconds: result[:training_seconds],
      description: result[:description],
      kept: result[:kept],
      status: result[:status]
    }

    socket =
      socket
      |> stream_insert(:experiments, entry)
      |> assign(:current_step, nil)
      |> assign(:current_progress, nil)
      |> add_log("#{status} v_#{result[:version_id]} loss=#{loss_str} — #{result[:description]}")

    {:noreply, socket}
  end

  @impl true
  def handle_info({:step, payload}, socket) do
    socket =
      socket
      |> assign(:current_step, payload[:step])
      |> assign(:current_progress, payload[:progress])

    {:noreply, socket}
  end

  @impl true
  def handle_info({:agent_thinking, _payload}, socket) do
    {:noreply, add_log(socket, "🤔 Thinking...")}
  end

  @impl true
  def handle_info({:agent_responded, payload}, socket) do
    reasoning = payload[:reasoning] || "..."
    {:noreply, add_log(socket, "💡 #{reasoning}")}
  end

  @impl true
  def handle_info({:status_changed, payload}, socket) do
    {:noreply, assign(socket, :agent_status, payload[:status])}
  end

  @impl true
  def handle_info(_, socket), do: {:noreply, socket}

  defp add_log(socket, message) do
    ts = Calendar.strftime(DateTime.utc_now(), "%H:%M:%S")
    logs = ["#{ts} #{message}" | socket.assigns.agent_log] |> Enum.take(100)
    assign(socket, :agent_log, logs)
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash}>
      <div class="max-w-7xl mx-auto p-6 space-y-6">
        <%!-- Header --%>
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-2xl font-bold text-zinc-100">🔬 ex_autoresearch</h1>
            <p class="text-zinc-400 text-sm">BEAM-native autonomous GPT research · Hot-loaded versioned modules</p>
          </div>
          <div class="flex items-center gap-3">
            <span class={["px-3 py-1 rounded-full text-sm font-medium", status_class(@agent_status)]}>
              {@agent_status}
            </span>
            <%= if @agent_status in [:idle, :stopping] do %>
              <button phx-click="start_research"
                class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition">
                ▶ Start Research
              </button>
            <% else %>
              <button phx-click="stop_research"
                class="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition">
                ⏹ Stop
              </button>
            <% end %>
          </div>
        </div>

        <%!-- Stats --%>
        <div class="grid grid-cols-5 gap-4">
          <.stat label="Versions" value={@experiment_count} />
          <.stat label="Best Loss" value={@best_loss && Float.round(@best_loss, 6) || "—"} />
          <.stat label="Best Version" value={@best_version && "v_#{@best_version}" || "—"} />
          <.stat label="Current Step" value={@current_step || "—"} />
          <.stat label="Progress" value={@current_progress && "#{@current_progress}%" || "—"} />
        </div>

        <div class="grid grid-cols-5 gap-6">
          <%!-- Experiment versions (wider) --%>
          <div class="col-span-3 bg-zinc-900 rounded-xl border border-zinc-800 p-4">
            <h2 class="text-lg font-semibold text-zinc-200 mb-3">📊 Experiment Versions</h2>
            <div class="overflow-y-auto max-h-[32rem]">
              <table class="w-full text-sm">
                <thead>
                  <tr class="text-zinc-500 border-b border-zinc-800">
                    <th class="text-left py-2 px-2">Version</th>
                    <th class="text-left py-2 px-2">Loss</th>
                    <th class="text-right py-2 px-2">Steps</th>
                    <th class="text-right py-2 px-2">Time</th>
                    <th class="text-left py-2 px-2">Description</th>
                    <th class="text-center py-2 px-2">Status</th>
                  </tr>
                </thead>
                <tbody id="experiments" phx-update="stream">
                  <tr :for={{dom_id, exp} <- @streams.experiments} id={dom_id}
                    class={["border-b border-zinc-800/50 hover:bg-zinc-800/30",
                            exp[:kept] && "bg-emerald-950/20"]}>
                    <td class="py-2 px-2 font-mono text-xs text-zinc-400">
                      {"v_#{exp[:version_id] || "?"}"}
                    </td>
                    <td class="py-2 px-2 font-mono text-zinc-300">
                      {fmt_loss(exp[:loss])}
                    </td>
                    <td class="py-2 px-2 text-right text-zinc-400">
                      {exp[:steps] || "—"}
                    </td>
                    <td class="py-2 px-2 text-right text-zinc-400">
                      {exp[:training_seconds] && "#{exp[:training_seconds]}s" || "—"}
                    </td>
                    <td class="py-2 px-2 text-zinc-300 max-w-xs truncate">
                      {exp[:description] || ""}
                    </td>
                    <td class="py-2 px-2 text-center">
                      {if exp[:kept], do: "✅", else: if(exp[:status] == :crashed, do: "💥", else: "❌")}
                    </td>
                  </tr>
                </tbody>
              </table>
              <div class="hidden only:block text-zinc-600 text-center py-8">
                No experiments yet. Hit ▶ Start Research.
              </div>
            </div>
          </div>

          <%!-- Agent log --%>
          <div class="col-span-2 bg-zinc-900 rounded-xl border border-zinc-800 p-4">
            <h2 class="text-lg font-semibold text-zinc-200 mb-3">🤖 Agent Reasoning</h2>
            <div class="overflow-y-auto max-h-[32rem] space-y-1">
              <%= if @agent_log == [] do %>
                <p class="text-zinc-600 text-center py-8">Waiting for agent activity...</p>
              <% else %>
                <div :for={entry <- @agent_log}
                  class="text-sm font-mono text-zinc-400 py-0.5 leading-relaxed">
                  {entry}
                </div>
              <% end %>
            </div>
          </div>
        </div>
      </div>
    </Layouts.app>
    """
  end

  defp status_class(:running), do: "bg-emerald-900 text-emerald-300"
  defp status_class(:stopping), do: "bg-amber-900 text-amber-300"
  defp status_class(_), do: "bg-zinc-800 text-zinc-400"

  defp fmt_loss(nil), do: "—"
  defp fmt_loss(l) when is_float(l), do: :erlang.float_to_binary(l, decimals: 6)
  defp fmt_loss(l), do: to_string(l)

  attr :label, :string, required: true
  attr :value, :any, required: true
  defp stat(assigns) do
    ~H"""
    <div class="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
      <div class="text-zinc-500 text-xs uppercase tracking-wider">{@label}</div>
      <div class="text-2xl font-bold text-zinc-100 mt-1">{@value}</div>
    </div>
    """
  end
end
