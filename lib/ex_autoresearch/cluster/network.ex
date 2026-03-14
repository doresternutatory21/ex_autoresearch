defmodule ExAutoresearch.Cluster.Network do
  @moduledoc """
  Network utility for detecting the best local IP for clustering.
  Prefers wired ethernet over wifi, excludes virtual interfaces.
  Ported from basileus.
  """

  @excluded_prefixes ~w(wlan docker br- veth wg tun tap virbr)

  @doc "Returns the best local IP address for Erlang distribution."
  def local_ip do
    System.get_env("EX_AUTORESEARCH_IP") || detect_ip()
  end

  @doc "Returns the local IP that routes to a specific remote host."
  def local_ip_for(remote_host) do
    case System.cmd("ip", ["route", "get", remote_host], stderr_to_stdout: true) do
      {output, 0} ->
        case Regex.run(~r/src (\S+)/, output) do
          [_, ip] -> ip
          _ -> local_ip()
        end

      _ ->
        local_ip()
    end
  end

  defp detect_ip do
    case System.cmd("ip", ["-4", "-o", "addr", "show", "scope", "global"], stderr_to_stdout: true) do
      {output, 0} ->
        lines = String.split(output, "\n", trim: true)

        wired =
          lines
          |> Enum.reject(fn line ->
            Enum.any?(@excluded_prefixes, &String.contains?(line, &1))
          end)
          |> Enum.flat_map(fn line ->
            case Regex.run(~r/inet (\d+\.\d+\.\d+\.\d+)/, line) do
              [_, ip] -> [ip]
              _ -> []
            end
          end)

        case wired do
          [ip | _] -> ip
          [] -> fallback_ip(lines)
        end

      _ ->
        "127.0.0.1"
    end
  end

  defp fallback_ip(lines) do
    lines
    |> Enum.flat_map(fn line ->
      case Regex.run(~r/inet (\d+\.\d+\.\d+\.\d+)/, line) do
        [_, ip] -> [ip]
        _ -> []
      end
    end)
    |> List.first("127.0.0.1")
  end
end
