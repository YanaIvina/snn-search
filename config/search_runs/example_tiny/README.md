# Example Tiny Search Run

This is a small, commit-friendly example of the experiment directory layout used
by `config/run_experiment_model.py`.

It intentionally contains only metadata and architecture configs:

- `run_manifest.json` - default training parameters for the run.
- `history.jsonl` - one JSON object per attempt.
- `attempts/attempt_0001/config.json` - architecture config for codegen.
- `attempts/attempt_0001/attempt_summary.json` - compact result summary.

No checkpoints, datasets, generated `model.py` files, logs, or diagrams are
stored here. They are regenerated locally when needed.

Try:

```bash
python config/run_experiment_model.py --run-dir config/search_runs/example_tiny --list
python config/run_experiment_model.py --run-dir config/search_runs/example_tiny best --dry-run
```
