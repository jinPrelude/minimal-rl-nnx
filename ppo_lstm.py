import os
from argparse import ArgumentParser

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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


class PPOLSTM(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(8, 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, 256, rngs=rngs)
        self.lstm = nnx.LSTMCell(256, 256, rngs=rngs)
        self.fc_pi = nnx.Linear(256, 4, rngs=rngs)
        self.fc_v = nnx.Linear(256, 1, rngs=rngs)

    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        return self.lstm.initialize_carry((batch_size, self.lstm.in_features), rngs=rngs)

    def step(self, obs, carry, done):
        x = nnx.relu(self.fc1(obs))
        x = nnx.relu(self.fc2(x))

        mask = (1.0 - done)[..., None]
        c, h = carry
        carry = (c * mask, h * mask)

        carry, hidden = self.lstm(carry, x)
        logits = self.fc_pi(hidden)
        value = self.fc_v(hidden)
        return logits, value, carry

    def unroll(self, obs_seq, done_seq, init_carry):
        def scan_step(carry, inputs):
            obs_t, done_t = inputs
            logits, value, carry = self.step(obs_t, carry, done_t)
            return carry, (logits, value)

        final_carry, (logits, values) = jax.lax.scan(scan_step, init_carry, (obs_seq, done_seq))
        return logits, values, final_carry


@nnx.jit
def sample_action(model, obs, carry, done, rngs):
    logits, value, next_carry = model.step(obs, carry, done)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = rngs.categorical(logits, axis=-1)
    sampled_log_prob = jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
    return sampled_log_prob, actions, value.squeeze(-1), next_carry


@nnx.jit
def bootstrap_value(model, obs, carry, done):
    _, value, _ = model.step(obs, carry, done)
    return value.squeeze(-1)


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
    obs, dones, actions, old_log_probs, advantages, returns, init_carry = batch

    logits, values, _ = model.unroll(obs, dones, init_carry)
    values = values.squeeze(-1)

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
        (loss, (actor_loss, critic_loss)), grad = grad_fn(model, minibatch, clip_eps)
        optimizer.update(model, grad)
        metrics.update(actor_loss=actor_loss, critic_loss=critic_loss)
        return model, optimizer, metrics

    scan_step((model, optimizer, metrics), minibatches)


def make_minibatches(batch, initial_carry, env_indices, envs_per_batch):
    obs, actions, old_log_probs, rewards, dones, values, advantages, returns = batch
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
        (
            jnp.take(initial_carry[0], env_ids, axis=0),
            jnp.take(initial_carry[1], env_ids, axis=0),
        ),
    )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-minibatch", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_arguments()

    assert args.num_minibatch >= 1
    assert args.num_envs % args.num_minibatch == 0
    envs_per_batch = args.num_envs // args.num_minibatch


    wandb.init(project="minimal-flaxrl", name=f"ppo_lstm_{args.env_name}", config=vars(args))

    rngs = nnx.Rngs(args.seed)
    ppo = PPOLSTM(rngs=rngs)
    optimizer = nnx.Optimizer(ppo, optax.adamw(args.learning_rate), wrt=nnx.Param)

    metrics = nnx.metrics.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        actor_loss=nnx.metrics.Average("actor_loss"),
    )

    envs = gym.make_vec(args.env_name, num_envs=args.num_envs, vectorization_mode="sync", max_episode_steps=300)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    obs, _ = envs.reset(seed=args.seed)
    replay_buffer = ReplayBuffer(args.num_steps, args.num_envs, envs.single_observation_space.shape)
    done = np.zeros(args.num_envs, dtype=np.float32)
    carry = ppo.init_carry(args.num_envs, rngs)

    global_env_step = 0
    for iteration in range(args.num_iter):
        rollout_rewards = []
        rollout_lengths = []
        initial_carry = (carry[0], carry[1])

        for _ in range(args.num_steps):
            log_prob, action, value, carry = sample_action(ppo, obs, carry, done, rngs)

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

        next_value = bootstrap_value(ppo, obs, carry, done)
        advantages, returns = calculate_gae(rewards_batch, values_batch, dones_batch, next_value, jnp.array(done), gamma=args.gamma, lmbda=args.lmbda)

        train_batch = (obs_batch, actions_batch, log_probs_batch, rewards_batch, dones_batch, values_batch, advantages, returns)

        num_minibatches = args.num_minibatch
        for _ in range(args.num_epochs):
            env_indices = np.asarray(jax.random.permutation(rngs(), args.num_envs))
            env_indices = env_indices[: num_minibatches * envs_per_batch]
            minibatches = make_minibatches(train_batch, initial_carry, env_indices, envs_per_batch)
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
