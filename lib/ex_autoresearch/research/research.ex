defmodule ExAutoresearch.Research do
  use Ash.Domain

  resources do
    resource ExAutoresearch.Research.Run
    resource ExAutoresearch.Research.Experiment
  end
end
