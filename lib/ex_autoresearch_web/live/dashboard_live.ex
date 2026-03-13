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
    experiments =
      Registry.all()
      |> Enum.map(fn exp -> exp |> Map.from_struct() |> Map.put(:id, exp.version_id) end)

    # Auto-select the best version
    best = Registry.best()
    selected = if best, do: load_version_code(best.version_id), else: nil

    socket =
      socket
      |> assign(:agent_status, agent.status)
      |> assign(:best_loss, agent.best_loss)
      |> assign(:best_version, agent.best_version)
      |> assign(:experiment_count, agent.experiment_count)
      |> assign(:current_step, nil)
      |> assign(:current_progress, nil)
      |> assign(:agent_log, [])
      |> assign(:selected_version, selected)
      |> stream(:experiments, experiments)

    {:ok, socket}
  end

  # --- Events ---

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
  def handle_event("select_version", %{"version" => version_id}, socket) do
    selected = load_version_code(version_id)
    {:noreply, assign(socket, :selected_version, selected)}
  end

  @impl true
  def handle_event("select_best", _params, socket) do
    best = Registry.best()
    selected = if best, do: load_version_code(best.version_id), else: nil
    {:noreply, assign(socket, :selected_version, selected)}
  end

  # --- PubSub ---

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
    loss_str = safe_fmt(result[:loss])

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

    # Auto-select new best
    socket =
      if result[:kept] do
        assign(socket, :selected_version, load_version_code(result[:version_id]))
      else
        socket
      end

    {:noreply, socket}
  end

  @impl true
  def handle_info({:step, payload}, socket) do
    {:noreply, socket |> assign(:current_step, payload[:step]) |> assign(:current_progress, payload[:progress])}
  end

  @impl true
  def handle_info({:agent_thinking, _payload}, socket), do: {:noreply, add_log(socket, "🤔 Thinking...")}

  @impl true
  def handle_info({:agent_responded, payload}, socket) do
    {:noreply, add_log(socket, "💡 #{payload[:reasoning] || "..."}")}
  end

  @impl true
  def handle_info({:status_changed, payload}, socket) do
    {:noreply, assign(socket, :agent_status, payload[:status])}
  end

  @impl true
  def handle_info(_, socket), do: {:noreply, socket}

  # --- Helpers ---

  defp add_log(socket, message) do
    ts = Calendar.strftime(DateTime.utc_now(), "%H:%M:%S")
    logs = ["#{ts} #{message}" | socket.assigns.agent_log] |> Enum.take(100)
    assign(socket, :agent_log, logs)
  end

  defp load_version_code(version_id) do
    case Registry.get(version_id) do
      {:ok, exp} ->
        %{
          version_id: exp.version_id,
          code: exp.code || "(no source code)",
          loss: exp.loss,
          steps: exp.steps,
          description: exp.description,
          kept: exp.kept,
          status: exp.status
        }
      :error ->
        nil
    end
  end

  defp safe_fmt(nil), do: "—"
  defp safe_fmt(l) when is_float(l), do: :erlang.float_to_binary(l, decimals: 6)
  defp safe_fmt(l), do: to_string(l)

  # --- Render ---

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash}>
      <div class="max-w-[90rem] mx-auto p-6 space-y-5">
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
          <.stat label="Best Loss" value={safe_fmt(@best_loss)} />
          <.stat label="Best Version" value={@best_version && "v_#{@best_version}" || "—"} />
          <.stat label="Current Step" value={@current_step || "—"} />
          <.stat label="Progress" value={@current_progress && "#{@current_progress}%" || "—"} />
        </div>

        <div class="grid grid-cols-12 gap-5">
          <%!-- Experiment versions table --%>
          <div class="col-span-5 bg-zinc-900 rounded-xl border border-zinc-800 p-4">
            <h2 class="text-lg font-semibold text-zinc-200 mb-3">📊 Experiments</h2>
            <div class="overflow-y-auto max-h-[28rem]">
              <table class="w-full text-sm">
                <thead>
                  <tr class="text-zinc-500 border-b border-zinc-800">
                    <th class="text-left py-1.5 px-2">Version</th>
                    <th class="text-left py-1.5 px-2">Loss</th>
                    <th class="text-right py-1.5 px-2">Steps</th>
                    <th class="text-center py-1.5 px-2"></th>
                  </tr>
                </thead>
                <tbody id="experiments" phx-update="stream">
                  <tr :for={{dom_id, exp} <- @streams.experiments} id={dom_id}
                    phx-click="select_version"
                    phx-value-version={exp[:version_id]}
                    class={[
                      "border-b border-zinc-800/50 cursor-pointer transition",
                      exp[:kept] && "bg-emerald-950/20",
                      @selected_version && @selected_version[:version_id] == exp[:version_id] && "ring-1 ring-indigo-500 bg-indigo-950/20",
                      "hover:bg-zinc-800/40"
                    ]}>
                    <td class="py-1.5 px-2">
                      <div class="font-mono text-xs text-zinc-400">{"v_#{exp[:version_id] || "?"}"}</div>
                      <div class="text-xs text-zinc-500 truncate max-w-[12rem]">{exp[:description] || ""}</div>
                    </td>
                    <td class="py-1.5 px-2 font-mono text-zinc-300 text-sm">
                      {safe_fmt(exp[:loss])}
                    </td>
                    <td class="py-1.5 px-2 text-right text-zinc-500 text-xs">
                      {exp[:steps] || "—"}
                    </td>
                    <td class="py-1.5 px-2 text-center">
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

          <%!-- Code viewer --%>
          <div class="col-span-4 bg-zinc-900 rounded-xl border border-zinc-800 p-4">
            <div class="flex items-center justify-between mb-3">
              <h2 class="text-lg font-semibold text-zinc-200">📝 Module Source</h2>
              <%= if @best_version do %>
                <button phx-click="select_best"
                  class="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 transition">
                  Show Best
                </button>
              <% end %>
            </div>
            <%= if @selected_version do %>
              <div class="mb-3 flex items-center gap-3">
                <span class="font-mono text-sm text-indigo-400">v_{@selected_version[:version_id]}</span>
                <span class="text-xs text-zinc-500">loss: {safe_fmt(@selected_version[:loss])}</span>
                <span class={[
                  "text-xs px-1.5 py-0.5 rounded",
                  if(@selected_version[:kept], do: "bg-emerald-900/50 text-emerald-400", else: "bg-zinc-800 text-zinc-500")
                ]}>
                  {if @selected_version[:kept], do: "kept", else: if(@selected_version[:status] == :crashed, do: "crashed", else: "discarded")}
                </span>
              </div>
              <div class="text-xs text-zinc-500 mb-2 italic">{@selected_version[:description]}</div>
              <div class="overflow-y-auto max-h-[22rem] bg-zinc-950 rounded-lg p-3 border border-zinc-800">
                <pre class="text-xs font-mono text-zinc-300 whitespace-pre-wrap" phx-no-curly-interpolation><code>{@selected_version[:code]}</code></pre>
              </div>
            <% else %>
              <div class="text-zinc-600 text-center py-12">
                Click an experiment to view its source code
              </div>
            <% end %>
          </div>

          <%!-- Agent log --%>
          <div class="col-span-3 bg-zinc-900 rounded-xl border border-zinc-800 p-4">
            <h2 class="text-lg font-semibold text-zinc-200 mb-3">🤖 Agent Log</h2>
            <div class="overflow-y-auto max-h-[28rem] space-y-1">
              <%= if @agent_log == [] do %>
                <p class="text-zinc-600 text-center py-8">Waiting for agent activity...</p>
              <% else %>
                <div :for={entry <- @agent_log}
                  class="text-xs font-mono text-zinc-400 py-0.5 leading-relaxed">
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
