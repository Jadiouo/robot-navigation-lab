# Included checkpoint

`ppo_tracking_seed0_best.pt` is the compact, CPU-loadable PPO checkpoint
selected at validation step 40,960 in the recorded three-seed run.  It is the
validation-selected snapshot, rather than the last checkpoint, and was not
chosen by held-out test performance.
The payload embeds its 40-dimensional observation-schema version, route-set
description, environment contract, source digest, RNG state, optimiser state,
and training configuration. Loading rejects incompatible observation schemas
or action dimensions; the source digest records provenance rather than
requiring an exact reporting-code match.

The three validation-selected seed checkpoints used by the held-out variance
report are included as `ppo_tracking_seed{0,1,2}_best.pt`. The full raw
evaluation-only record can be regenerated with
[`reproduce_heldout.py`](../reproduce_heldout.py).

Recreate 20 held-out episode artifacts with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[rl]'
.venv/bin/navlab evaluate \
  --checkpoint examples/checkpoints/ppo_tracking_seed0_best.pt \
  --split test --episodes 20 --output outputs/ppo-seed0-test
```

The archived successful example is held-out test episode 6 in
[`docs/results`](../../docs/results/README.md).  The complete final comparison
is also retained there.
