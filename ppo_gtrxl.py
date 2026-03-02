import os
import time
from argparse import ArgumentParser

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from flax import struct
import flax.nnx as nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import wandb


MODEL_DTYPE = jnp.bfloat16
PARAM_DTYPE = jnp.bfloat16

OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EPISODE_STEPS = 300


class ReplayBuffer:
    """Fixed-size rollout buffer for PPO."""

    def __init__(self, num_steps: int, num_envs: int, obs_shape):
        self.num_steps = num_steps
        self.num_envs = num_envs
        # Keep rollout tensors in float32 for precision; but not strictly tested.
        self.obs = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.reset()

    def reset(self):
        self.size = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        if self.size >= self.num_steps:
            raise ValueError("ReplayBuffer is full. Call reset() before adding new data.")

        t = self.size
        self.obs[t] = np.asarray(obs, dtype=np.float32)
        self.actions[t] = np.asarray(actions, dtype=np.int32)
        self.log_probs[t] = np.asarray(log_probs, dtype=np.float32)
        self.rewards[t] = np.asarray(rewards, dtype=np.float32)
        self.dones[t] = np.asarray(dones, dtype=np.float32)
        self.values[t] = np.asarray(values, dtype=np.float32)
        self.size += 1

    def as_jax(self):
        if self.size != self.num_steps:
            raise ValueError(f"ReplayBuffer not full: expected {self.num_steps}, got {self.size}")

        return (
            jnp.asarray(self.obs),
            jnp.asarray(self.actions),
            jnp.asarray(self.log_probs),
            jnp.asarray(self.rewards),
            jnp.asarray(self.dones),
            jnp.asarray(self.values),
        )


class TrXLState(struct.PyTreeNode):
    memory: jax.Array       # [B, M, L, D]
    valid_len: jax.Array    # [B]
    pos: jax.Array          # [B]


def detach_state(state: TrXLState) -> TrXLState:
    return TrXLState(
        memory=jax.lax.stop_gradient(state.memory),
        valid_len=jax.lax.stop_gradient(state.valid_len),
        pos=jax.lax.stop_gradient(state.pos),
    )


class GRUGate(nnx.Module):
    def __init__(self, dim: int, bias_init: float = 2.0, *, rngs: nnx.Rngs):
        self.w_r = nnx.Linear(dim, dim, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.u_r = nnx.Linear(dim, dim, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.w_z = nnx.Linear(dim, dim, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.u_z = nnx.Linear(dim, dim, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.w_g = nnx.Linear(dim, dim, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.u_g = nnx.Linear(dim, dim, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.b_g = nnx.Param(jnp.full((dim,), bias_init, dtype=MODEL_DTYPE))

    def __call__(self, x, y):
        r = jax.nn.sigmoid(self.w_r(y) + self.u_r(x))
        z = jax.nn.sigmoid(self.w_z(y) + self.u_z(x) - self.b_g[...])
        h_hat = jnp.tanh(self.w_g(y) + self.u_g(r * x))
        return (1.0 - z) * x + z * h_hat


class GTrXLBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, gate_bias_init: float, *, rngs: nnx.Rngs):
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")

        self.attn_norm = nnx.LayerNorm(num_features=dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(num_features=dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)

        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            dtype=MODEL_DTYPE,
            param_dtype=PARAM_DTYPE,
            dropout_rate=0.0,
            decode=False,
            use_bias=False,
            rngs=rngs,
        )
        self.ffn = nnx.Linear(dim, dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)

        self.attn_gate = GRUGate(dim, bias_init=gate_bias_init, rngs=rngs)
        self.ffn_gate = GRUGate(dim, bias_init=gate_bias_init, rngs=rngs)

    def __call__(self, memory, query, mask):
        if mask.ndim == 2:
            attn_mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            attn_mask = mask[:, None, :, :]
        else:
            raise ValueError(f"memory_mask must be rank-2 or rank-3, got {mask.ndim}")

        mem = self.attn_norm(memory)
        q = self.attn_norm(query)

        attn_out = self.attn(q, mem, mem, mask=attn_mask, deterministic=True)
        x = self.attn_gate(query, nnx.relu(attn_out))

        ffn_out = self.ffn(self.ffn_norm(x))
        x = self.ffn_gate(x, nnx.relu(ffn_out))
        return x


class PPOGTrXL(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        trxl_dim: int,
        trxl_num_layers: int,
        trxl_num_heads: int,
        trxl_memory_length: int,
        gtrxl_gate_bias_init: float,
        *,
        rngs: nnx.Rngs,
    ):
        if trxl_dim % 2 != 0:
            raise ValueError(f"trxl_dim must be even for sinusoidal encoding, got {trxl_dim}")

        self.hidden_dim = trxl_dim
        self.num_layers = trxl_num_layers
        self.memory_len = trxl_memory_length

        self.encoder = nnx.Linear(obs_dim, trxl_dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.layers = nnx.List([
            GTrXLBlock(trxl_dim, trxl_num_heads, gtrxl_gate_bias_init, rngs=rngs)
            for _ in range(trxl_num_layers)
        ])
        self.post_trxl = nnx.Linear(trxl_dim, trxl_dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.policy_head = nnx.Linear(trxl_dim, num_actions, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.value_head = nnx.Linear(trxl_dim, 1, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)

        freqs = jnp.arange(0, trxl_dim, 2, dtype=MODEL_DTYPE)
        self.inv_freq = 10_000.0 ** (-freqs / trxl_dim)

    def init_state(self, batch_size: int) -> TrXLState:
        return TrXLState(
            memory=jnp.zeros((batch_size, self.memory_len, self.num_layers, self.hidden_dim), dtype=MODEL_DTYPE),
            valid_len=jnp.zeros((batch_size,), dtype=jnp.int32),
            pos=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def _encode_obs(self, obs):
        return self.encoder(jnp.asarray(obs, dtype=MODEL_DTYPE))

    def _pos_emb(self, positions):
        positions = jnp.maximum(positions, 0).astype(MODEL_DTYPE)
        sinusoid = positions[..., None] * self.inv_freq[None, :]
        return jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)

    def _build_unroll_mask(self, valid_len, num_steps: int):
        batch_size = valid_len.shape[0]

        memory_idx = jnp.arange(self.memory_len, dtype=jnp.int32)[None, :]
        query_idx = jnp.arange(num_steps, dtype=jnp.int32)[None, :]
        key_idx = jnp.arange(self.memory_len + num_steps, dtype=jnp.int32)[None, :]

        memory_valid = memory_idx >= (self.memory_len - valid_len[:, None])
        query_valid = jnp.ones((batch_size, num_steps), dtype=jnp.bool_)
        key_valid = jnp.concatenate([memory_valid, query_valid], axis=1)

        query_key_idx = self.memory_len + query_idx
        within_window = key_idx[:, None, :] >= (query_key_idx - self.memory_len)[:, :, None]
        strictly_past = key_idx[:, None, :] < query_key_idx[:, :, None]
        return key_valid[:, None, :] & within_window & strictly_past

    def _step_core(self, state: TrXLState, x_t):
        memory_idx = jnp.arange(self.memory_len, dtype=jnp.int32)[None, :]
        memory_mask = memory_idx >= (self.memory_len - state.valid_len[:, None])

        memory_pos = state.pos[:, None] + memory_idx - self.memory_len
        memories = state.memory + self._pos_emb(memory_pos)[:, :, None, :]

        x = x_t
        layer_inputs = []
        for i, layer in enumerate(self.layers):
            layer_inputs.append(x)
            x = layer(memories[:, :, i], x[:, None, :], memory_mask)
            x = x.squeeze(1)

        new_tokens = jnp.stack(layer_inputs, axis=1)
        new_memory = jnp.concatenate([state.memory[:, 1:], new_tokens[:, None, :, :]], axis=1)

        new_state = TrXLState(
            memory=new_memory,
            valid_len=jnp.minimum(state.valid_len + 1, self.memory_len),
            pos=state.pos + 1,
        )
        return new_state, x

    def step(self, obs, state: TrXLState, done):
        del done  # kept for API compatibility
        x = self._encode_obs(obs)
        state, hidden = self._step_core(state, x)

        hidden = nnx.relu(self.post_trxl(hidden))
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value, state

    def unroll(self, obs_seq, done_seq, init_state: TrXLState):
        del done_seq  # kept for API compatibility

        x = jnp.swapaxes(self._encode_obs(obs_seq), 0, 1)  # [B, T, D]
        num_steps = x.shape[1]

        time_idx = jnp.arange(num_steps, dtype=jnp.int32)[None, :]
        memory_idx = jnp.arange(self.memory_len, dtype=jnp.int32)[None, :]

        query_pos = init_state.pos[:, None] + time_idx
        memory_pos = init_state.pos[:, None] + memory_idx - self.memory_len

        query_pos_emb = self._pos_emb(query_pos)
        memory_pos_emb = self._pos_emb(memory_pos)
        attn_mask = self._build_unroll_mask(init_state.valid_len, num_steps)

        layer_inputs = []
        for i, layer in enumerate(self.layers):
            layer_inputs.append(x)
            kv = jnp.concatenate(
                [
                    init_state.memory[:, :, i] + memory_pos_emb,
                    x + query_pos_emb,
                ],
                axis=1,
            )
            x = layer(kv, x, attn_mask)

        hidden = nnx.relu(self.post_trxl(x))
        logits = jnp.swapaxes(self.policy_head(hidden), 0, 1)
        values = jnp.swapaxes(self.value_head(hidden).squeeze(-1), 0, 1)

        final_layers = []
        for i in range(self.num_layers):
            tokens = jnp.concatenate([init_state.memory[:, :, i], layer_inputs[i]], axis=1)
            final_layers.append(tokens[:, -self.memory_len :, :])

        final_state = TrXLState(
            memory=jnp.stack(final_layers, axis=2),
            valid_len=jnp.minimum(init_state.valid_len + num_steps, self.memory_len),
            pos=init_state.pos + num_steps,
        )
        return logits, values, final_state


@nnx.jit
def sample_action(model, obs, state, done, rngs):
    logits, value, new_state = model.step(obs, state, done)
    # Keep sampling numerics in float32 for precision; but not strictly tested.
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = rngs.categorical(logits, axis=-1)
    sampled_log_prob = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    return sampled_log_prob, actions, value, new_state


@nnx.jit
def bootstrap_value(model, obs, state, done):
    _, value, _ = model.step(obs, state, done)
    # Keep bootstrap values in float32 for precision; but not strictly tested.
    return value.astype(jnp.float32)


def calculate_gae(rewards, values, dones, next_value, next_done, gamma: float, lmbda: float):
    next_values = jnp.concatenate([values[1:], next_value[None, :]], axis=0)
    next_nonterminal = 1.0 - jnp.concatenate([dones[1:], next_done[None, :]], axis=0)
    deltas = rewards + gamma * next_values * next_nonterminal - values

    def scan_step(last_advantage, inputs):
        delta_t, nonterminal_t = inputs
        advantage = delta_t + gamma * lmbda * nonterminal_t * last_advantage
        return advantage, advantage

    init_advantage = jnp.zeros_like(next_value)
    _, advantages = jax.lax.scan(scan_step, init_advantage, (deltas, next_nonterminal), reverse=True)
    returns = advantages + values
    return advantages, returns


def loss_fn(model, batch, clip_eps, ent_coef):
    obs, dones, actions, old_log_probs, advantages, returns, init_state = batch
    logits, values, final_state = model.unroll(obs, dones, init_state)

    old_log_probs = old_log_probs.astype(MODEL_DTYPE)
    advantages = advantages.astype(MODEL_DTYPE)
    returns = returns.astype(MODEL_DTYPE)
    clip_eps = jnp.asarray(clip_eps, dtype=MODEL_DTYPE)
    ent_coef = jnp.asarray(ent_coef, dtype=MODEL_DTYPE)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)

    ratio = jnp.exp(selected_log_probs - old_log_probs)
    actor_loss = rlax.clipped_surrogate_pg_loss(ratio.reshape(-1), advantages.reshape(-1), clip_eps).mean()
    critic_loss = optax.huber_loss(values, jax.lax.stop_gradient(returns)).mean()
    entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * log_probs, axis=-1).mean()

    total_loss = actor_loss + jnp.asarray(0.5, dtype=MODEL_DTYPE) * critic_loss - ent_coef * entropy
    return total_loss, (actor_loss, critic_loss, entropy, final_state)


def make_minibatches(batch, initial_state: TrXLState, env_indices, envs_per_batch: int, segment_length: int):
    obs, actions, old_log_probs, _, dones, _, advantages, returns = batch
    num_segments = obs.shape[0] // segment_length

    env_ids = jnp.asarray(env_indices, dtype=jnp.int32).reshape(-1, envs_per_batch)

    def split_time_and_env(x):
        x = jnp.take(x, env_ids, axis=1)     # [T, MB, E, ...]
        x = jnp.swapaxes(x, 0, 1)            # [MB, T, E, ...]
        return x.reshape(x.shape[0], num_segments, segment_length, *x.shape[2:])

    return (
        split_time_and_env(obs),
        split_time_and_env(dones),
        split_time_and_env(actions),
        split_time_and_env(old_log_probs),
        split_time_and_env(advantages),
        split_time_and_env(returns),
        TrXLState(
            memory=jnp.take(initial_state.memory, env_ids, axis=0),
            valid_len=jnp.take(initial_state.valid_len, env_ids, axis=0),
            pos=jnp.take(initial_state.pos, env_ids, axis=0),
        ),
    )


@nnx.jit
def update_ppo(model, optimizer, minibatches, metrics, clip_eps=0.2, ent_coef=0.0001):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def train_minibatch(carry, minibatch):
        model, optimizer, metrics = carry
        obs_segments, dones_segments, actions_segments, old_log_probs_segments, advantages_segments, returns_segments, init_state = minibatch

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def train_segment(carry, segment):
            model, optimizer, metrics, state = carry
            obs, dones, actions, old_log_probs, advantages, returns = segment

            (_, (actor_loss, critic_loss, entropy, next_state)), grad = grad_fn(
                model,
                (obs, dones, actions, old_log_probs, advantages, returns, state),
                clip_eps,
                ent_coef,
            )
            optimizer.update(model, grad)
            metrics.update(actor_loss=actor_loss, critic_loss=critic_loss, entropy=entropy)
            return model, optimizer, metrics, detach_state(next_state)

        model, optimizer, metrics, _ = train_segment(
            (model, optimizer, metrics, init_state),
            (
                obs_segments,
                dones_segments,
                actions_segments,
                old_log_probs_segments,
                advantages_segments,
                returns_segments,
            ),
        )
        return model, optimizer, metrics

    train_minibatch((model, optimizer, metrics), minibatches)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--segment-length", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-minibatch", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=3)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)

    parser.add_argument("--trxl-dim", type=int, default=128)
    parser.add_argument("--trxl-num-layers", type=int, default=3)
    parser.add_argument("--trxl-num-heads", type=int, default=2)
    parser.add_argument("--trxl-memory-length", type=int, default=64)
    parser.add_argument("--gtrxl-gate-bias-init", type=float, default=2.0)
    return parser.parse_args()


def validate_args(args):
    assert args.env_name == "LunarLander-v3", "This minimal implementation supports only LunarLander-v3."
    assert args.num_minibatch >= 1
    assert args.num_envs % args.num_minibatch == 0
    assert args.segment_length > 0
    assert args.num_steps % args.segment_length == 0
    assert args.trxl_dim % args.trxl_num_heads == 0
    assert args.trxl_memory_length > 0


def main():
    args = parse_args()
    validate_args(args)

    envs_per_batch = args.num_envs // args.num_minibatch
    rngs = nnx.Rngs(args.seed)

    # Init model, optimizer, and metrics
    memory_len = min(args.trxl_memory_length, MAX_EPISODE_STEPS)
    model = PPOGTrXL(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        trxl_dim=args.trxl_dim,
        trxl_num_layers=args.trxl_num_layers,
        trxl_num_heads=args.trxl_num_heads,
        trxl_memory_length=memory_len,
        gtrxl_gate_bias_init=args.gtrxl_gate_bias_init,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adamw(args.learning_rate), wrt=nnx.Param)
    metrics = nnx.metrics.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        actor_loss=nnx.metrics.Average("actor_loss"),
        entropy=nnx.metrics.Average("entropy"),
    )

    # Init environment
    envs = gym.make_vec(
        args.env_name,
        num_envs=args.num_envs,
        vectorization_mode="sync",
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    assert envs.single_observation_space.shape == (OBS_DIM,)
    assert envs.single_action_space.n == NUM_ACTIONS

    wandb.init(
        project="minimal-flaxrl",
        name=f"ppo_gtrxl_{args.env_name}",
        config={**vars(args), "trxl_memory_length": memory_len},
    )

    # reset environment & init replay buffer
    obs, _ = envs.reset(seed=args.seed)
    state = model.init_state(args.num_envs)
    done = np.zeros(args.num_envs, dtype=np.float32)
    replay = ReplayBuffer(args.num_steps, args.num_envs, envs.single_observation_space.shape)

    global_env_step = 0
    start_time = time.time()
    for iteration in range(args.num_iter):
        rollout_rewards = []
        rollout_lengths = []
        initial_state = TrXLState(memory=state.memory, valid_len=state.valid_len, pos=state.pos)

        for _ in range(args.num_steps):
            log_prob, action, value, state = sample_action(model, obs, state, done, rngs)

            next_obs, reward, terminated, truncated, info = envs.step(np.asarray(action))
            next_done = np.maximum(terminated, truncated).astype(np.float32)

            replay.add(obs, action, log_prob, reward, done, value)
            global_env_step += args.num_envs

            if "_episode" in info:
                for i, finished in enumerate(info["_episode"]):
                    if finished:
                        rollout_rewards.append(float(info["episode"]["r"][i]))
                        rollout_lengths.append(int(info["episode"]["l"][i]))

            obs = next_obs
            done = next_done

        obs_batch, actions_batch, log_probs_batch, rewards_batch, dones_batch, values_batch = replay.as_jax()

        next_value = bootstrap_value(model, obs, state, done)
        advantages, returns = calculate_gae(
            rewards_batch,
            values_batch,
            dones_batch,
            next_value,
            jnp.asarray(done, dtype=jnp.float32),
            gamma=args.gamma,
            lmbda=args.lmbda,
        )

        train_batch = (
            obs_batch,
            actions_batch,
            log_probs_batch,
            rewards_batch,
            dones_batch,
            values_batch,
            advantages,
            returns,
        )

        for _ in range(args.num_epochs):
            env_indices = np.asarray(jax.random.permutation(rngs(), args.num_envs))
            minibatches = make_minibatches(
                train_batch,
                initial_state,
                env_indices[: args.num_minibatch * envs_per_batch],
                envs_per_batch,
                args.segment_length,
            )
            update_ppo(
                model,
                optimizer,
                minibatches,
                metrics,
                clip_eps=args.clip_eps,
                ent_coef=args.ent_coef,
            )

        metric_values = {k: float(v) for k, v in metrics.compute().items()}
        sps = int(global_env_step / max(time.time() - start_time, 1e-6))

        log_data = {
            "train/iteration": iteration,
            "train/global_env_step": global_env_step,
            "train/sps": sps,
            "train/actor_loss": metric_values["actor_loss"],
            "train/critic_loss": metric_values["critic_loss"],
            "train/entropy": metric_values["entropy"],
        }
        if rollout_rewards:
            log_data["episode/reward_mean"] = float(np.mean(rollout_rewards))
            log_data["episode/reward_max"] = float(np.max(rollout_rewards))
            log_data["episode/length_mean"] = float(np.mean(rollout_lengths))
            log_data["episode/count"] = len(rollout_rewards)
        wandb.log(log_data, step=global_env_step)

        metrics.reset()
        replay.reset()

    envs.close()


if __name__ == "__main__":
    main()
