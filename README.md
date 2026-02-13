# minimal_flaxrl
minimal rl implementation using [flax nnx](https://flax.readthedocs.io/en/latest/index.html), inspired by [minimal_rl](https://github.com/seungeunrho/minimalRL).

## installation
```bash
# python >= 3.12 
pip install -r requirements.txt
wandb login # login wandb for logging
```

## Train
### PPO (215 lines)
```bash
python ppo.py
```