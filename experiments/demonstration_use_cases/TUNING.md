# SAC tuning

Use development configurations and separate result directories for tuning.
Return to the [run guide](README.md) for the frozen campaign.

For tuning, set `protocol.wandb_mode: offline` in a development configuration.
Training diagnostics and evaluation summaries are retained under each run's
`wandb/` directory. `online` is also supported when W&B credentials are configured.
Each RL job runs in a separate process, with detailed output in `worker.log`.

When changing SAC parallelism, audit replay capacity and learning warmup as well
as minibatch size. At 4,096 environments, a one-million-transition buffer retains
only about 244 simulation steps per lane; a collection chunk of eight steps
already exceeds the default 10,000-transition learning threshold. SAC logs
`replay_history_steps` and `replay_samples_per_transition` to make this scaling
visible. These are diagnostics, not automatic quality guarantees. Keep tuning
runs separate from the frozen campaign until held-out evaluation supports them.
