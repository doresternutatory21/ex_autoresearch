defmodule ExAutoresearch.Agent.LLM.CopilotBackend do
  @moduledoc """
  GitHub Copilot backend via jido_ghcopilot Server protocol.

  Maintains a persistent connection with session management.
  Text is accumulated from streaming chunks until the turn completes.
  """

  use GenServer

  require Logger

  alias Jido.GHCopilot.Server.Connection

  defstruct [:conn, :session_id, :model, status: :connecting, buffer: "", caller: nil]

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts)
  end

  @impl true
  def init(opts) do
    model = Keyword.get(opts, :model, "claude-sonnet-4")
    {:ok, %__MODULE__{model: model}, {:continue, :connect}}
  end

  @impl true
  def handle_continue(:connect, state) do
    cli_args = ["--allow-all-tools", "--allow-all-paths", "--allow-all-urls"]

    case Connection.start_link(cli_args: cli_args, cwd: File.cwd!()) do
      {:ok, conn} ->
        case Connection.create_session(conn, %{model: state.model}) do
          {:ok, session_id} ->
            :ok = Connection.subscribe(conn, session_id)
            Logger.info("Copilot backend connected: model=#{state.model}")
            {:noreply, %{state | conn: conn, session_id: session_id, status: :idle}}

          {:error, reason} ->
            Logger.error("Copilot session failed: #{inspect(reason)}")
            Process.send_after(self(), :reconnect, 5_000)
            {:noreply, %{state | conn: conn, status: :error}}
        end

      {:error, reason} ->
        Logger.error("Copilot connection failed: #{inspect(reason)}")
        Process.send_after(self(), :reconnect, 5_000)
        {:noreply, %{state | status: :error}}
    end
  end

  @impl true
  def handle_call({:prompt, text, requested_model}, from, %{status: :idle} = state) do
    state =
      if requested_model && requested_model != state.model do
        case switch_model(state, requested_model) do
          {:ok, new_state} ->
            Logger.info("Copilot switched to: #{requested_model}")
            new_state

          {:error, _} ->
            state
        end
      else
        state
      end

    Logger.debug("Copilot prompt (#{String.length(text)} chars, model: #{state.model})")

    case Connection.send_prompt(state.conn, state.session_id, text) do
      {:ok, _msg_id} ->
        {:noreply, %{state | status: :waiting, buffer: "", caller: from}}

      {:error, reason} ->
        # Session may have died — try to recreate it once
        Logger.warning("Copilot send failed: #{inspect(reason)}. Recreating session...")

        case recreate_session(state) do
          {:ok, new_state} ->
            case Connection.send_prompt(new_state.conn, new_state.session_id, text) do
              {:ok, _msg_id} ->
                {:noreply, %{new_state | status: :waiting, buffer: "", caller: from}}

              {:error, reason2} ->
                {:reply, {:error, reason2}, new_state}
            end

          {:error, _} ->
            {:reply, {:error, reason}, state}
        end
    end
  end

  def handle_call({:prompt, _text, _model}, _from, state) do
    {:reply, {:error, {:not_ready, state.status}}, state}
  end

  @impl true
  def handle_info({:server_event, %{type: type, data: data}}, state) do
    case type do
      "assistant.message" ->
        content = data["content"] || ""
        {:noreply, %{state | buffer: state.buffer <> content}}

      "assistant.message.chunk" ->
        chunk = data["chunkContent"] || data["content"] || ""
        {:noreply, %{state | buffer: state.buffer <> chunk}}

      "session.idle" when not is_nil(state.caller) ->
        response = String.trim(state.buffer)
        GenServer.reply(state.caller, {:ok, response})
        {:noreply, %{state | status: :idle, buffer: "", caller: nil}}

      _ ->
        {:noreply, state}
    end
  end

  def handle_info({:server_tool_call, tool_call}, state) do
    Connection.respond_to_tool_call(state.conn, tool_call.request_id, %{
      "error" => "No tools available. Output code as text in your response."
    })

    {:noreply, state}
  end

  def handle_info(:reconnect, state) do
    Logger.info("Copilot reconnecting...")
    {:noreply, state, {:continue, :connect}}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(_reason, %{conn: conn}) when not is_nil(conn), do: Connection.stop(conn)
  def terminate(_, _), do: :ok

  defp recreate_session(%{conn: conn, session_id: old_sid, model: model} = state) do
    if old_sid, do: catch_unsubscribe(conn, old_sid)

    case Connection.create_session(conn, %{model: model}) do
      {:ok, new_sid} ->
        :ok = Connection.subscribe(conn, new_sid)
        Logger.info("Copilot session recreated: #{new_sid}")
        {:ok, %{state | session_id: new_sid, status: :idle}}

      {:error, reason} ->
        Logger.error("Failed to recreate Copilot session: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp catch_unsubscribe(conn, sid) do
    Connection.unsubscribe(conn, sid)
  rescue
    _ -> :ok
  catch
    _, _ -> :ok
  end

  defp switch_model(%{conn: conn, session_id: old_sid} = state, new_model) do
    if old_sid, do: Connection.unsubscribe(conn, old_sid)

    case Connection.create_session(conn, %{model: new_model}) do
      {:ok, new_sid} ->
        :ok = Connection.subscribe(conn, new_sid)
        {:ok, %{state | session_id: new_sid, model: new_model}}

      {:error, reason} ->
        if old_sid, do: Connection.subscribe(conn, old_sid)
        {:error, reason}
    end
  end
end
