## Constraints

- **OUTPUT FORMAT: Print the complete defmodule as a fenced elixir code block in your response. Do NOT use any tools to create or write files. Just output the code as text.**
- Your module MUST be named `ExAutoresearch.Experiments.V_<version_id>` (the system fills in the version_id)
- Your module MUST implement these callbacks:
  - `config/0` — returns a map with model hyperparameters
  - `build/0` — returns an Axon model that takes `"input_ids"` {batch, seq_len} and outputs {batch, seq_len, vocab_size}
  - `optimizer/0` — returns a Polaris optimizer tuple
- Optional callbacks:
  - `loss_fn/2` — custom loss function (default: categorical_cross_entropy)
- Available libraries: Axon, Nx, Nx.Defn, Polaris, EXLA
- The model will be trained on next-token prediction with synthetic data
- Training runs for a fixed step count or time budget

## API compatibility (Polaris 0.1.0 / Axon 0.8.1)

- **DO NOT use Polaris.Schedules** (cosine_decay, exponential_decay, etc.) as learning rate — they crash with `Nx.LazyContainer not implemented for List` due to a Polaris/Nx incompatibility. Use a constant learning rate instead.
- **DO NOT use Axon.param/4** — it is undefined/private in Axon 0.8.1.
- **DO NOT pass Axon graph nodes to Axon.nx/3** — the first argument must be an Axon layer output, not a raw Axon struct or parameter.
- **DO NOT use Axon.multiply/2 with non-layer arguments** — both arguments must be Axon layers.
- **DO NOT use Axon.Layers.swish** — use `Axon.activation(:silu)` instead.
- Use `Polaris.Optimizers.adamw(learning_rate: 0.01)` with a float, not a schedule function.
