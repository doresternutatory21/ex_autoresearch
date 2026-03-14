defmodule ExAutoresearch.Data.Downloader do
  @moduledoc """
  Downloads parquet data shards from HuggingFace.

  Uses Req for HTTP streaming with exponential backoff retries.
  Stores in ~/.cache/ex_autoresearch/data/.
  """

  require Logger

  @base_url "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
  @max_shard 6542
  @val_shard @max_shard

  @doc "Cache directory for downloaded data."
  def cache_dir do
    Path.join([System.user_home!(), ".cache", "ex_autoresearch", "data"])
  end

  @doc "Download N training shards + the pinned validation shard."
  def download(num_shards, opts \\ []) do
    dir = cache_dir()
    File.mkdir_p!(dir)

    concurrency = Keyword.get(opts, :concurrency, 4)

    # Always include the validation shard
    shard_indices = Enum.to_list(0..(num_shards - 1)) ++ [@val_shard]
    shard_indices = Enum.uniq(shard_indices)

    Logger.info("Downloading #{length(shard_indices)} shards to #{dir}")

    shard_indices
    |> Task.async_stream(&download_shard(&1, dir),
      max_concurrency: concurrency,
      timeout: :infinity
    )
    |> Enum.reduce({0, 0}, fn
      {:ok, :ok}, {ok, err} ->
        {ok + 1, err}

      {:ok, :exists}, {ok, err} ->
        {ok + 1, err}

      {:ok, {:error, reason}}, {ok, err} ->
        Logger.error("Shard download failed: #{inspect(reason)}")
        {ok, err + 1}

      {:exit, reason}, {ok, err} ->
        Logger.error("Shard download crashed: #{inspect(reason)}")
        {ok, err + 1}
    end)
  end

  @doc "Path to a specific shard file."
  def shard_path(index) do
    name = :io_lib.format("shard_~5..0B.parquet", [index]) |> IO.iodata_to_binary()
    Path.join(cache_dir(), name)
  end

  @doc "Path to the validation shard."
  def val_shard_path, do: shard_path(@val_shard)

  defp download_shard(index, dir) do
    name = :io_lib.format("shard_~5..0B.parquet", [index]) |> IO.iodata_to_binary()
    path = Path.join(dir, name)
    tmp_path = path <> ".tmp"

    if File.exists?(path) do
      :exists
    else
      url = "#{@base_url}/#{name}"
      Logger.info("Downloading #{name}...")

      case Req.get(url, into: File.stream!(tmp_path), retry: :transient, max_retries: 5) do
        {:ok, %{status: 200}} ->
          File.rename!(tmp_path, path)
          :ok

        {:ok, %{status: status}} ->
          File.rm(tmp_path)
          {:error, {:http_status, status}}

        {:error, reason} ->
          File.rm(tmp_path)
          {:error, reason}
      end
    end
  end
end
