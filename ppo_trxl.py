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


def clone_state(state: TrXLState) -> TrXLState:
    return TrXLState(
        memory=state.memory,
        valid_len=state.valid_len,
        pos=state.pos,
    )


def reset_done_in_state(state: TrXLState, done_mask) -> TrXLState:
    done = jnp.asarray(done_mask, dtype=jnp.bool_)
    keep = ~done
    return TrXLState(
        memory=state.memory * keep[:, None, None, None].astype(state.memory.dtype),
        valid_len=state.valid_len * keep.astype(state.valid_len.dtype),
        pos=state.pos * keep.astype(state.pos.dtype),
    )


def append_memory_token(state: TrXLState, token, max_episode_steps: int) -> TrXLState:
    if token.ndim != 3:
        raise ValueError(f"`token` must be rank-3 [B, L, D], got shape={token.shape}")
    mem_len = state.memory.shape[1]
    token = jnp.asarray(token, dtype=state.memory.dtype)
    return TrXLState(
        memory=jnp.concatenate([state.memory[:, 1:], token[:, None, :, :]], axis=1),
        valid_len=jnp.minimum(state.valid_len + 1, mem_len),
        pos=jnp.minimum(state.pos + 1, max_episode_steps),
    )


def build_memory_inputs(state: TrXLState, mem_len: int):
    memory_idx = jnp.arange(mem_len, dtype=jnp.int32)[None, :]
    memory_mask = memory_idx >= (mem_len - state.valid_len[:, None])
    return state.memory, memory_mask


def build_parallel_eval_metadata(dones_seq, init_state: TrXLState, mem_len: int, max_episode_steps: int):
    del max_episode_steps
    done = jnp.swapaxes(jnp.asarray(dones_seq, dtype=jnp.bool_), 0, 1)  # [B, T]
    batch_size, seq_len = done.shape

    t = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    m = jnp.arange(mem_len, dtype=jnp.int32)[None, :]
    init_pos = init_state.pos.astype(jnp.int32)
    init_valid_len = init_state.valid_len.astype(jnp.int32)

    episode_ids = jnp.cumsum(done.astype(jnp.int32), axis=1)  # [B, T]
    reset_points = jnp.where(done, jnp.broadcast_to(t, (batch_size, seq_len)), -jnp.ones((batch_size, seq_len), dtype=jnp.int32))
    last_reset = jnp.maximum.accumulate(reset_points, axis=1)
    query_pos_unclamped = jnp.where(episode_ids == 0, init_pos[:, None] + t, t - last_reset)

    memory_valid = m >= (mem_len - init_valid_len[:, None])
    memory_pos_unclamped = init_pos[:, None] + m - mem_len

    memory_episode_ids = jnp.where(memory_valid, jnp.zeros_like(memory_pos_unclamped), -jnp.ones_like(memory_pos_unclamped))
    memory_key_pos = jnp.where(
        memory_valid,
        memory_pos_unclamped,
        jnp.full_like(memory_pos_unclamped, -1_000_000_000),
    )
    key_episode_ids = jnp.concatenate([memory_episode_ids, episode_ids], axis=1)  # [B, M+T]
    key_pos = jnp.concatenate([memory_key_pos, query_pos_unclamped], axis=1)  # [B, M+T]

    query_pos = query_pos_unclamped[:, :, None]  # [B, T, 1]
    return (
        (key_episode_ids[:, None, :] == episode_ids[:, :, None])
        & (key_pos[:, None, :] <= query_pos)
        & (key_pos[:, None, :] >= query_pos - mem_len)
    )


class RelativeMultiHeadAttention(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim}, {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.values = nnx.Linear(self.head_size, self.head_size, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.keys = nnx.Linear(self.head_size, self.head_size, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.queries = nnx.Linear(self.head_size, self.head_size, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.r_proj = nnx.Linear(embed_dim, num_heads * self.head_size, use_bias=False, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.u_bias = nnx.Param(jnp.zeros((num_heads, self.head_size), dtype=MODEL_DTYPE))
        self.v_bias = nnx.Param(jnp.zeros((num_heads, self.head_size), dtype=MODEL_DTYPE))
        self.fc_out = nnx.Linear(num_heads * self.head_size, embed_dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        freqs = jnp.arange(0, embed_dim, 2, dtype=MODEL_DTYPE)
        self.inv_freqs = 10_000.0 ** (-freqs / embed_dim)

    def _relative_sinusoid_embedding(self, num_positions: int, dtype):
        seq = jnp.arange(num_positions, dtype=self.inv_freqs.dtype)[:, None]
        sinusoidal_inp = seq * self.inv_freqs[None, :]
        return jnp.concatenate([jnp.sin(sinusoidal_inp), jnp.cos(sinusoidal_inp)], axis=-1).astype(dtype)

    def _relative_keys(self, query_len: int, key_len: int, query_offset: int | None, dtype):
        if query_offset is None:
            query_offset = key_len
        if query_offset < 0:
            raise ValueError(f"`query_offset` must be >= 0, got {query_offset}")

        max_distance = key_len + query_len
        query_positions = jnp.arange(query_len, dtype=jnp.int32)[:, None] + query_offset
        key_positions = jnp.arange(key_len, dtype=jnp.int32)[None, :]
        relative_positions = jnp.clip(query_positions - key_positions, a_min=0, a_max=max_distance - 1)

        rel_sinusoid = self._relative_sinusoid_embedding(max_distance, dtype=dtype)
        rel_keys = self.r_proj(rel_sinusoid).reshape(max_distance, self.num_heads, self.head_size)
        return rel_keys[relative_positions]

    def __call__(self, values, keys, query, mask, query_offset: int | None = None, self_kv=None):
        batch_size = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(batch_size, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(batch_size, key_len, self.num_heads, self.head_size)
        query = query.reshape(batch_size, query_len, self.num_heads, self.head_size)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        rel_keys = self._relative_keys(query_len, key_len, query_offset, queries.dtype)

        mem_content_energy = jnp.einsum("nqhd,nkhd->nhqk", queries + self.u_bias[...], keys)
        mem_position_energy = jnp.einsum("nqhd,qkhd->nhqk", queries + self.v_bias[...], rel_keys)
        mem_energy = mem_content_energy + mem_position_energy
        scale = jnp.sqrt(jnp.asarray(self.head_size, dtype=queries.dtype))
        mask_fill = jnp.asarray(-1e30, dtype=mem_energy.dtype)

        if self_kv is None:
            if mask is not None:
                mem_energy = jnp.where(mask[:, None, :, :], mem_energy, mask_fill)
            attention = jax.nn.softmax(mem_energy / scale, axis=3)
            out = jnp.einsum("nhql,nlhd->nqhd", attention, values).reshape(batch_size, query_len, self.num_heads * self.head_size)
            return self.fc_out(out), attention

        if query_len != 1:
            raise ValueError("`self_kv` fast path expects query_len == 1")
        if mask is not None and mask.ndim != 2:
            raise ValueError("`self_kv` fast path expects rank-2 mask [B, K]")

        self_kv = self_kv.reshape(batch_size, 1, self.num_heads, self.head_size)
        self_values = self.values(self_kv)
        self_keys = self.keys(self_kv)

        rel0 = self._relative_sinusoid_embedding(key_len + query_len, dtype=queries.dtype)[:1]
        rel0 = self.r_proj(rel0).reshape(1, 1, self.num_heads, self.head_size)

        self_content_energy = jnp.einsum("nqhd,nkhd->nhqk", queries + self.u_bias[...], self_keys)
        self_position_energy = jnp.einsum("nqhd,qkhd->nhqk", queries + self.v_bias[...], rel0)
        self_energy = self_content_energy + self_position_energy

        if mask is not None:
            mem_energy = jnp.where(mask[:, None, None, :], mem_energy, mask_fill)

        energy = jnp.concatenate([mem_energy, self_energy], axis=3)
        attention = jax.nn.softmax(energy / scale, axis=3)
        attn_mem = attention[..., :key_len]
        attn_self = attention[..., key_len:]

        out_mem = jnp.einsum("nhql,nlhd->nqhd", attn_mem, values)
        out_self = jnp.einsum("nhql,nlhd->nqhd", attn_self, self_values)
        out = (out_mem + out_self).reshape(batch_size, query_len, self.num_heads * self.head_size)
        return self.fc_out(out), attention


class TrXLBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")

        self.layer_norm_q = nnx.LayerNorm(num_features=dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(num_features=dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.layer_norm_attn = nnx.LayerNorm(num_features=dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.attention = RelativeMultiHeadAttention(dim, num_heads, rngs=rngs)
        self.fc_projection = nnx.Linear(dim, dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)

    def __call__(self, kv, query, mask, query_offset: int | None = None, self_kv=None):
        query_ = self.layer_norm_q(query)
        kv = self.norm_kv(kv)
        if self_kv is not None:
            self_kv = self.norm_kv(self_kv)
        attn_out, _ = self.attention(kv, kv, query_, mask, query_offset=query_offset, self_kv=self_kv)
        x = attn_out + query
        x_ = self.layer_norm_attn(x)
        forward = nnx.relu(self.fc_projection(x_))
        return forward + x


class PPOTrXL(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        trxl_dim: int,
        trxl_num_layers: int,
        trxl_num_heads: int,
        trxl_memory_length: int,
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
            TrXLBlock(trxl_dim, trxl_num_heads, rngs=rngs)
            for _ in range(trxl_num_layers)
        ])
        self.post_trxl = nnx.Linear(trxl_dim, trxl_dim, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.policy_head = nnx.Linear(trxl_dim, num_actions, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)
        self.value_head = nnx.Linear(trxl_dim, 1, dtype=MODEL_DTYPE, param_dtype=PARAM_DTYPE, rngs=rngs)

    def init_state(self, batch_size: int) -> TrXLState:
        return TrXLState(
            memory=jnp.zeros((batch_size, self.memory_len, self.num_layers, self.hidden_dim), dtype=MODEL_DTYPE),
            valid_len=jnp.zeros((batch_size,), dtype=jnp.int32),
            pos=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def _encode_obs(self, obs):
        return self.encoder(jnp.asarray(obs, dtype=MODEL_DTYPE))

    def _step_core(self, state: TrXLState, x_t):
        memory_window, memory_mask = build_memory_inputs(state, self.memory_len)
        x = x_t
        layer_inputs = []
        for i, layer in enumerate(self.layers):
            layer_inputs.append(x)
            q = x[:, None, :]
            x = layer(
                memory_window[:, :, i],
                q,
                memory_mask,
                query_offset=self.memory_len,
                self_kv=q,
            )
            x = x.squeeze(1)
        return append_memory_token(state, jnp.stack(layer_inputs, axis=1), MAX_EPISODE_STEPS), x

    def step(self, obs, state: TrXLState, done):
        x = self._encode_obs(obs)
        state = reset_done_in_state(state, done)
        state, hidden = self._step_core(state, x)

        hidden = nnx.relu(self.post_trxl(hidden))
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value, state

    def unroll(self, obs_seq, done_seq, init_state: TrXLState):
        x = jnp.swapaxes(self._encode_obs(obs_seq), 0, 1)  # [B, T, D]
        attn_mask = build_parallel_eval_metadata(done_seq, init_state, self.memory_len, MAX_EPISODE_STEPS)
        fallback_kv_token = jnp.zeros((1, 1, x.shape[-1]), dtype=x.dtype)

        for i, layer in enumerate(self.layers):
            kv = jnp.concatenate([init_state.memory[:, :, i], x], axis=1)
            no_key_rows = ~jnp.any(attn_mask, axis=2, keepdims=True)
            dummy_kv = jnp.broadcast_to(fallback_kv_token, (kv.shape[0], 1, kv.shape[2]))
            kv = jnp.concatenate([dummy_kv, kv], axis=1)
            layer_mask = jnp.concatenate([no_key_rows, attn_mask], axis=2)
            x = layer(
                kv,
                x,
                layer_mask,
                query_offset=kv.shape[1] - x.shape[1],
            )

        hidden = nnx.relu(self.post_trxl(x))
        logits = jnp.swapaxes(self.policy_head(hidden), 0, 1)
        values = jnp.swapaxes(self.value_head(hidden).squeeze(-1), 0, 1)
        return logits, values


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
    logits, values = model.unroll(obs, dones, init_state)

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
    return total_loss, (actor_loss, critic_loss, entropy)


def make_minibatches(batch, segment_init_states, env_indices, envs_per_batch: int, segment_length: int):
    obs, actions, old_log_probs, _, dones, _, advantages, returns = batch
    num_segments = obs.shape[0] // segment_length

    env_ids = jnp.asarray(env_indices, dtype=jnp.int32).reshape(-1, envs_per_batch)

    def split_time_and_env(x):
        x = jnp.take(x, env_ids, axis=1)     # [T, MB, E, ...]
        x = jnp.swapaxes(x, 0, 1)            # [MB, T, E, ...]
        return x.reshape(x.shape[0], num_segments, segment_length, *x.shape[2:])

    seg_memory = jnp.stack([s.memory for s in segment_init_states], axis=0)      # [S, B, M, L, D]
    seg_valid_len = jnp.stack([s.valid_len for s in segment_init_states], axis=0)  # [S, B]
    seg_pos = jnp.stack([s.pos for s in segment_init_states], axis=0)            # [S, B]

    return (
        split_time_and_env(obs),
        split_time_and_env(dones),
        split_time_and_env(actions),
        split_time_and_env(old_log_probs),
        split_time_and_env(advantages),
        split_time_and_env(returns),
        TrXLState(
            memory=jnp.swapaxes(jnp.take(seg_memory, env_ids, axis=1), 0, 1),      # [MB, S, E, M, L, D]
            valid_len=jnp.swapaxes(jnp.take(seg_valid_len, env_ids, axis=1), 0, 1),  # [MB, S, E]
            pos=jnp.swapaxes(jnp.take(seg_pos, env_ids, axis=1), 0, 1),            # [MB, S, E]
        ),
    )


@nnx.jit
def update_ppo(model, optimizer, minibatches, metrics, clip_eps=0.2, ent_coef=0.0001):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def train_minibatch(carry, minibatch):
        model, optimizer, metrics = carry
        obs_segments, dones_segments, actions_segments, old_log_probs_segments, advantages_segments, returns_segments, init_states = minibatch

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def train_segment(carry, segment):
            model, optimizer, metrics = carry
            obs, dones, actions, old_log_probs, advantages, returns, init_state = segment

            (_, (actor_loss, critic_loss, entropy)), grad = grad_fn(
                model,
                (obs, dones, actions, old_log_probs, advantages, returns, init_state),
                clip_eps,
                ent_coef,
            )
            optimizer.update(model, grad)
            metrics.update(actor_loss=actor_loss, critic_loss=critic_loss, entropy=entropy)
            return model, optimizer, metrics

        model, optimizer, metrics = train_segment(
            (model, optimizer, metrics),
            (
                obs_segments,
                dones_segments,
                actions_segments,
                old_log_probs_segments,
                advantages_segments,
                returns_segments,
                init_states,
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
    model = PPOTrXL(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        trxl_dim=args.trxl_dim,
        trxl_num_layers=args.trxl_num_layers,
        trxl_num_heads=args.trxl_num_heads,
        trxl_memory_length=memory_len,
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
        name=f"ppo_trxl_{args.env_name}",
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
        segment_init_states = [None] * (args.num_steps // args.segment_length)

        for step in range(args.num_steps):
            state_for_step = reset_done_in_state(state, done)
            if step % args.segment_length == 0:
                segment_init_states[step // args.segment_length] = clone_state(state_for_step)
            log_prob, action, value, state = sample_action(
                model,
                obs,
                state_for_step,
                jnp.zeros((args.num_envs,), dtype=jnp.float32),
                rngs,
            )

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

        bootstrap_state = reset_done_in_state(state, done)
        next_value = bootstrap_value(model, obs, bootstrap_state, jnp.zeros((args.num_envs,), dtype=jnp.float32))
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
        if any(s is None for s in segment_init_states):
            raise ValueError("Some segment initial states were not captured during rollout.")

        for _ in range(args.num_epochs):
            env_indices = np.asarray(jax.random.permutation(rngs(), args.num_envs))
            minibatches = make_minibatches(
                train_batch,
                segment_init_states,
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
