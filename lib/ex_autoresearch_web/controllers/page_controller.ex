defmodule ExAutoresearchWeb.PageController do
  use ExAutoresearchWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
