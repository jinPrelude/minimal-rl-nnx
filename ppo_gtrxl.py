import os
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


class ReplayBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_shape):
        self.num_steps = num_steps
        self.num_envs = num_envs
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
        idx = self.size
        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = np.asarray(actions, dtype=np.int32)
        self.log_probs[idx] = np.asarray(log_probs, dtype=np.float32)
        self.rewards[idx] = np.asarray(rewards, dtype=np.float32)
        self.dones[idx] = np.asarray(dones, dtype=np.float32)
        self.values[idx] = np.asarray(values, dtype=np.float32)
        self.size += 1

    def get(self):
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
    memory: jax.Array
    valid_len: jax.Array
    pos: jax.Array


class GRUGate(nnx.Module):
    def __init__(self, dim: int, bias_init: float = 2.0, *, rngs: nnx.Rngs):
        self.w_r = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.u_r = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.w_z = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.u_z = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.w_g = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.u_g = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.b_g = nnx.Param(jnp.full((dim,), bias_init, dtype=jnp.float32))

    def __call__(self, x, y):
        r = jax.nn.sigmoid(self.w_r(y) + self.u_r(x))
        z = jax.nn.sigmoid(self.w_z(y) + self.u_z(x) - self.b_g[...])
        h_hat = jnp.tanh(self.w_g(y) + self.u_g(r * x))
        return (1.0 - z) * x + z * h_hat


class GTrXLBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, gate_bias_init: float, *, rngs: nnx.Rngs):
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")

        self.norm_attn = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.norm_ffn = nnx.LayerNorm(num_features=dim, rngs=rngs)

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            dropout_rate=0.0,
            decode=False,
            use_bias=False,
            rngs=rngs,
        )
        self.fc = nnx.Linear(dim, dim, rngs=rngs)

        self.gate_attn = GRUGate(dim, bias_init=gate_bias_init, rngs=rngs)
        self.gate_ffn = GRUGate(dim, bias_init=gate_bias_init, rngs=rngs)

    def __call__(self, memory, query, memory_mask):
        normed = self.norm_attn(jnp.concatenate([memory, query], axis=1))
        memory_len = memory.shape[1]
        mem_norm = normed[:, :memory_len, :]
        q_norm = normed[:, memory_len:, :]

        attn_out = self.attention(
            q_norm,
            mem_norm,
            mem_norm,
            mask=memory_mask[:, None, :, :],
            deterministic=True,
        )
        x = self.gate_attn(query, nnx.relu(attn_out))

        ffn_out = self.fc(self.norm_ffn(x))
        x = self.gate_ffn(x, nnx.relu(ffn_out))
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
        self.n_layers = trxl_num_layers
        self.memory_len = trxl_memory_length

        self.fc_encoder = nnx.Linear(obs_dim, trxl_dim, rngs=rngs)
        self.layers = nnx.List([
            GTrXLBlock(trxl_dim, trxl_num_heads, gtrxl_gate_bias_init, rngs=rngs)
            for _ in range(trxl_num_layers)
        ])
        self.hidden_post_trxl = nnx.Linear(trxl_dim, trxl_dim, rngs=rngs)
        self.fc_pi = nnx.Linear(trxl_dim, num_actions, rngs=rngs)
        self.fc_v = nnx.Linear(trxl_dim, 1, rngs=rngs)

        freqs = jnp.arange(0, trxl_dim, 2, dtype=jnp.float32)
        self.inv_freq = 10_000.0 ** (-freqs / trxl_dim)

    def init_state(self, batch_size: int):
        return TrXLState(
            memory=jnp.zeros((batch_size, self.memory_len, self.n_layers, self.hidden_dim), dtype=jnp.float32),
            valid_len=jnp.zeros((batch_size,), dtype=jnp.int32),
            pos=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def _encode(self, obs):
        return self.fc_encoder(jnp.asarray(obs, dtype=jnp.float32))

    def _position_embedding(self, positions):
        clipped = jnp.maximum(positions, 0).astype(jnp.float32)
        sinusoid = clipped[..., None] * self.inv_freq[None, :]
        return jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)

    @staticmethod
    def _reset_state_on_done(state: TrXLState, done):
        mask_float = 1.0 - done
        mask_int = 1 - done.astype(jnp.int32)
        return TrXLState(
            memory=state.memory * mask_float[:, None, None, None],
            valid_len=state.valid_len * mask_int,
            pos=state.pos * mask_int,
        )

    def _trxl_step(self, state: TrXLState, x_t):
        memory_idx = jnp.arange(self.memory_len, dtype=jnp.int32)[None, :]
        memory_mask = memory_idx >= (self.memory_len - state.valid_len[:, None])
        memory_positions = state.pos[:, None] + memory_idx - self.memory_len
        memories = state.memory + self._position_embedding(memory_positions)[:, :, None, :]

        x = x_t
        layer_tokens = []
        for i, layer in enumerate(self.layers):
            layer_tokens.append(jax.lax.stop_gradient(x))
            x = layer(memories[:, :, i], x[:, None, :], memory_mask[:, None, :])
            x = x.squeeze(1)

        new_tokens = jnp.stack(layer_tokens, axis=1)
        new_memory = jnp.concatenate([state.memory[:, 1:], new_tokens[:, None, :, :]], axis=1)
        new_state = TrXLState(
            memory=new_memory,
            valid_len=jnp.minimum(state.valid_len + 1, self.memory_len),
            pos=state.pos + 1,
        )
        return new_state, x

    def step(self, obs, state: TrXLState, done):
        x = self._encode(obs)
        state = self._reset_state_on_done(state, done)
        state, hidden = self._trxl_step(state, x)
        hidden = nnx.relu(self.hidden_post_trxl(hidden))
        logits = self.fc_pi(hidden)
        value = self.fc_v(hidden).squeeze(-1)
        return logits, value, state

    def unroll(self, obs_seq, done_seq, init_state: TrXLState):
        x_seq = self._encode(obs_seq)

        def scan_step(state, inputs):
            x_t, done_t = inputs
            state = self._reset_state_on_done(state, done_t)
            state, hidden = self._trxl_step(state, x_t)
            return state, hidden

        final_state, hidden_seq = jax.lax.scan(scan_step, init_state, (x_seq, done_seq))
        hidden_seq = nnx.relu(self.hidden_post_trxl(hidden_seq))
        logits = self.fc_pi(hidden_seq)
        values = self.fc_v(hidden_seq).squeeze(-1)
        return logits, values, final_state


@nnx.jit
def sample_action(model, obs, state, done, rngs):
    logits, value, new_state = model.step(obs, state, done)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = rngs.categorical(logits, axis=-1)
    sampled_log_prob = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    return sampled_log_prob, actions, value, new_state


@nnx.jit
def bootstrap_value(model, obs, state, done):
    _, value, _ = model.step(obs, state, done)
    return value


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


def loss_fn(model, batch, clip_eps):
    obs, dones, actions, old_log_probs, advantages, returns, init_state = batch

    logits, values, _ = model.unroll(obs, dones, init_state)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)

    ratio = jnp.exp(selected_log_probs - old_log_probs)
    actor_loss = rlax.clipped_surrogate_pg_loss(ratio.reshape(-1), advantages.reshape(-1), clip_eps).mean()
    critic_loss = optax.huber_loss(values, jax.lax.stop_gradient(returns)).mean()
    total_loss = actor_loss + 0.5 * critic_loss
    return total_loss, (actor_loss, critic_loss)


@nnx.jit
def update_ppo(model: nnx.Module, optimizer: nnx.Optimizer, minibatches, metrics: nnx.metrics.MultiMetric, clip_eps=0.2):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def scan_step(carry, minibatch):
        model, optimizer, metrics = carry
        (_, (actor_loss, critic_loss)), grad = grad_fn(model, minibatch, clip_eps)
        optimizer.update(model, grad)
        metrics.update(actor_loss=actor_loss, critic_loss=critic_loss)
        return model, optimizer, metrics

    scan_step((model, optimizer, metrics), minibatches)


def make_minibatches(batch, initial_state: TrXLState, env_indices, envs_per_batch: int):
    obs, actions, old_log_probs, _rewards, dones, _values, advantages, returns = batch
    env_ids = jnp.asarray(env_indices, dtype=jnp.int32).reshape(-1, envs_per_batch)

    def select_time_env(x):
        return jnp.swapaxes(jnp.take(x, env_ids, axis=1), 0, 1)

    return (
        select_time_env(obs),
        select_time_env(dones),
        select_time_env(actions),
        select_time_env(old_log_probs),
        select_time_env(advantages),
        select_time_env(returns),
        TrXLState(
            memory=jnp.take(initial_state.memory, env_ids, axis=0),
            valid_len=jnp.take(initial_state.valid_len, env_ids, axis=0),
            pos=jnp.take(initial_state.pos, env_ids, axis=0),
        ),
    )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-minibatch", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--trxl-dim", type=int, default=128)
    parser.add_argument("--trxl-num-layers", type=int, default=3)
    parser.add_argument("--trxl-num-heads", type=int, default=2)
    parser.add_argument("--trxl-memory-length", type=int, default=64)
    parser.add_argument("--gtrxl-gate-bias-init", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_arguments()

    assert args.env_name == "LunarLander-v3", "This minimal implementation supports only LunarLander-v3."
    assert args.num_minibatch >= 1
    assert args.num_envs % args.num_minibatch == 0
    envs_per_batch = args.num_envs // args.num_minibatch
    assert args.trxl_dim % args.trxl_num_heads == 0
    assert args.trxl_memory_length > 0

    max_episode_steps = 300
    memory_length = min(args.trxl_memory_length, max_episode_steps)

    wandb.init(
        project="minimal-flaxrl",
        name=f"ppo_gtrxl_{args.env_name}",
        config={**vars(args), "trxl_memory_length": memory_length},
    )

    rngs = nnx.Rngs(args.seed)
    ppo = PPOGTrXL(
        obs_dim=8,
        num_actions=4,
        trxl_dim=args.trxl_dim,
        trxl_num_layers=args.trxl_num_layers,
        trxl_num_heads=args.trxl_num_heads,
        trxl_memory_length=memory_length,
        gtrxl_gate_bias_init=args.gtrxl_gate_bias_init,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(ppo, optax.adamw(args.learning_rate), wrt=nnx.Param)

    metrics = nnx.metrics.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        actor_loss=nnx.metrics.Average("actor_loss"),
    )

    envs = gym.make_vec(args.env_name, num_envs=args.num_envs, vectorization_mode="sync", max_episode_steps=max_episode_steps)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    assert envs.single_observation_space.shape == (8,)
    assert envs.single_action_space.n == 4

    obs, _ = envs.reset(seed=args.seed)
    replay_buffer = ReplayBuffer(args.num_steps, args.num_envs, envs.single_observation_space.shape)
    done = np.zeros(args.num_envs, dtype=np.float32)
    state = ppo.init_state(args.num_envs)

    global_env_step = 0
    for iteration in range(args.num_iter):
        rollout_rewards = []
        rollout_lengths = []
        initial_state = TrXLState(memory=state.memory, valid_len=state.valid_len, pos=state.pos)

        for _ in range(args.num_steps):
            log_prob, action, value, state = sample_action(ppo, obs, state, done, rngs)

            next_obs, reward, terminated, truncated, info = envs.step(np.asarray(action))
            next_done = np.maximum(terminated, truncated).astype(np.float32)

            replay_buffer.add(obs, action, log_prob, reward, done, value)
            global_env_step += args.num_envs

            if "_episode" in info:
                for idx, finished in enumerate(info["_episode"]):
                    if finished:
                        rollout_rewards.append(float(info["episode"]["r"][idx]))
                        rollout_lengths.append(int(info["episode"]["l"][idx]))

            obs = next_obs
            done = next_done

        rollout = replay_buffer.get()
        obs_batch, actions_batch, log_probs_batch, rewards_batch, dones_batch, values_batch = rollout

        next_value = bootstrap_value(ppo, obs, state, done)
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

        num_minibatches = args.num_minibatch
        for _ in range(args.num_epochs):
            env_indices = np.asarray(jax.random.permutation(rngs(), args.num_envs))
            env_indices = env_indices[: num_minibatches * envs_per_batch]
            minibatches = make_minibatches(train_batch, initial_state, env_indices, envs_per_batch)
            update_ppo(ppo, optimizer, minibatches, metrics, clip_eps=args.clip_eps)

        metric_values = {k: float(v) for k, v in metrics.compute().items()}

        wandb_payload = {
            "train/iteration": iteration,
            "train/global_env_step": global_env_step,
            "train/actor_loss": metric_values["actor_loss"],
            "train/critic_loss": metric_values["critic_loss"],
        }
        if rollout_rewards:
            wandb_payload["episode/reward_mean"] = float(np.mean(rollout_rewards))
            wandb_payload["episode/reward_max"] = float(np.max(rollout_rewards))
            wandb_payload["episode/length_mean"] = float(np.mean(rollout_lengths))
            wandb_payload["episode/count"] = len(rollout_rewards)
        wandb.log(wandb_payload, step=global_env_step)

        metrics.reset()
        replay_buffer.reset()

    envs.close()


if __name__ == "__main__":
    main()
