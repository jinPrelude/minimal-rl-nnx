# minimal-rl-nnx

Minimal RL implementations in [Flax NNX](https://flax.readthedocs.io/en/latest/index.html), inspired by [minimalRL](https://github.com/seungeunrho/minimalRL). All trained on [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

## Quick Start

```bash
pip install -r requirements.txt   # Python >= 3.12
wandb login                       # for experiment logging
```

## Algorithms

| Algorithm | Lines | Command | Training time (MacBook Air M2) | Environment |
|-----------|-------|---------|--------------------------------|-------------|
| [PPO](ppo.py) | 228 | `python ppo.py` | ~40 sec | [LunarLander-v3](ppo.py) |
| [PPO_LSTM](ppo_lstm.py) | 278 | `python ppo_lstm.py` | ~5 mins | [LunarLander-v3](ppo_lstm.py) |
| [PPO_TrXL](ppo_trxl.py) (very unstable. use GTrXL) | 498 | `python ppo_trxl.py` | 1+ hour (recommend GPU training) | [LunarLander-v3](ppo_trxl.py) |
| [PPO_GTrXL](ppo_gtrxl.py) | 531 | `python ppo_gtrxl.py` | 1+ hour (recommend GPU training) | [LunarLander-v3](ppo_gtrxl.py), [ALE/Breakout-v5](ppo_gtrxl_atari.py) |
| [A2C](a2c.py) | 180 | `python a2c.py` | ~100 sec | [LunarLander-v3](a2c.py) |
| [Impala](impala.py) ([cleanba](https://github.com/vwxyzjn/cleanba) style)| 263 | `python impala.py` | ~100 sec | [LunarLander-v3](impala.py) |
| [Impala_LSTM](impala_lstm.py) | 294 | `python impala_lstm.py` | | [LunarLander-v3](impala_lstm.py) |


If you'd like to see a specific algorithm implemented, feel free to open an [issue](../../issues).

## Tuning Tips

- Training failed with `gamma=0.97`. Setting it to `0.99` was critical for learning.
- Increasing hidden dim from 128 to 256 improved both convergence speed and final performance.
- For A2C, updating the actor with `V` instead of `G - V` (advantage) caused training to fail.
- TrXL appears to be highly sensitive to hyperparameter tuning. For example, increasing `trxl_dim` from 128 to 256 (and `trxl-num-heads` from 2 to 4) caused training to fail.
- In contrast, GTrXL was more stable and still trained well when increasing `trxl_dim` to 256.

# Notes
- For TrXL and GTrXL, to keep the code simple, I removed the logic that prevents Transformer-XL from attending to the previous episode's memory when a new episode starts. Convergence was slightly slower, but it does not affect learning. A GTrXL version with proper `done` masking is available at `code_archive/ppo_gtrxl_with_done_mask.py`.

## Performance graph

<img src="assets/performance_graph.png" width="300" />
