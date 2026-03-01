# minimal-rl-nnx

Minimal RL implementations in [Flax NNX](https://flax.readthedocs.io/en/latest/index.html), inspired by [minimalRL](https://github.com/seungeunrho/minimalRL). All trained on [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

## Quick Start

```bash
pip install -r requirements.txt   # Python >= 3.12
wandb login                       # for experiment logging
```

## Algorithms

| Algorithm | Lines | Command | Training time (MacBook Air M2)|
|-----------|-------|---------|---------------|
| [PPO](ppo.py) | 228 | `python ppo.py` | ~40 sec |
| [PPO_LSTM](ppo_lstm.py) | 278 | `python ppo_lstm.py` | ~5 mins |
| [PPO_TrXL](ppo_trxl.py) | 421 | `python ppo_trxl.py` | 1+ hour (recommend GPU training) |
| [PPO_GTrXL](ppo_gtrxl.py) | 456 | `python ppo_gtrxl.py` | 1+ hour (recommend GPU training) |
| [A2C](a2c.py) | 180 | `python a2c.py` | ~100 sec |
| [Impala](impala.py) ([cleanba](https://github.com/vwxyzjn/cleanba) style)| 263 | `python impala.py` | ~100 sec |


If you'd like to see a specific algorithm implemented, feel free to open an [issue](../../issues).

## Tuning Tips

- Training failed with `gamma=0.97`. Setting it to `0.99` was critical for learning.
- Increasing hidden dim from 128 to 256 improved both convergence speed and final performance.
- For A2C, updating the actor with `V` instead of `G - V` (advantage) caused training to fail.
- TrXL appears to be highly sensitive to hyperparameter tuning. For example, increasing `trxl_dim` from 128 to 256 (and `trxl-num-heads` from 2 to 4) caused training to fail.
- In contrast, GTrXL was more stable and still trained well when increasing `trxl_dim` to 256.

## GTrXL Implementation Summary

[impl_resources/gtrxl.py](impl_resources/gtrxl.py) includes a compact GTrXL backbone with GRU-style gating:

- `GRUGate`: Implements the GRU-like gate from Parisotto et al. (2019), replacing plain residual addition with gated mixing.
- `GTrXLLayer`: Pre-norm attention + FFN block with two gates (`gate_attn`, `gate_ffn`) and causal/memory masking.
- `GTrXLBackbone`:
  - Maintains Transformer-XL style rolling memory via `TrXLState(memory, valid_len, pos)`.
  - Supports both `step` (single-timestep inference) and `unroll` (sequence training).
  - Supports absolute sinusoidal or learned positional encodings.
  - Uses stop-gradient on memory writes (`jax.lax.stop_gradient`) for stable recurrent training.


## Performance graph

<img src="assets/performance_graph.png" width="300" />
