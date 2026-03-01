# Minimal IMPALA (Importance Weighted Actor-Learner Architecture) in JAX/Flax NNX
# Single-device, LunarLander-v3, V-trace off-policy correction
import os
import queue
import threading
import time
from argparse import ArgumentParser
from collections import deque
from typing import List, NamedTuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import flax.nnx as nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import wandb


class IMPALALSTMAgent(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, lstm_hidden_size: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, lstm_hidden_size, rngs=rngs)
        self.lstm = nnx.LSTMCell(lstm_hidden_size, lstm_hidden_size, rngs=rngs)
        self.fc_pi = nnx.Linear(lstm_hidden_size, action_dim, rngs=rngs)
        self.fc_v = nnx.Linear(lstm_hidden_size, 1, rngs=rngs)

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
        value = self.fc_v(hidden).squeeze(-1)
        return logits, value, carry

    def unroll(self, obs_seq, done_seq, init_carry):
        def scan_step(carry, inputs):
            obs_t, done_t = inputs
            logits_t, value_t, carry = self.step(obs_t, carry, done_t)
            return carry, (logits_t, value_t)

        final_carry, (logits, values) = jax.lax.scan(scan_step, init_carry, (obs_seq, done_seq))
        return logits, values, final_carry


class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    logits: list
    rewards: list
    lstm_c: list
    lstm_h: list


def rollout(
    key: jax.Array,
    args,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    graphdef,
):
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("LunarLander-v3", max_episode_steps=300) for _ in range(args.num_envs)]
    )

    @jax.jit
    def get_action(params, obs, carry, done, key):
        obs = jnp.asarray(obs)
        done = jnp.asarray(done)
        agent = nnx.merge(graphdef, params)
        logits, _, next_carry = agent.step(obs, carry, done)
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return obs, action, logits, next_carry, key

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree.map(lambda *xs: jnp.stack(xs), *storage)

    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    returned_episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.float32)
    returned_episode_lengths = np.zeros(args.num_envs, dtype=np.float32)
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = np.zeros(args.num_envs, dtype=np.float32)
    next_carry = None

    storage = []
    params_queue_get_time = deque(maxlen=10)
    for update in range(1, args.num_updates + 2):
        num_steps_with_bootstrap = args.num_steps + 1 + int(len(storage) == 0)

        # Skip params fetch at update 2 to overlap rollout with learner update (1-step stale policy).
        if update != 2:
            params_queue_get_time_start = time.time()
            params = params_queue.get()
            jax.block_until_ready(params)
            params_queue_get_time.append(time.time() - params_queue_get_time_start)
            if next_carry is None:
                agent = nnx.merge(graphdef, params)
                next_carry = agent.init_carry(args.num_envs, nnx.Rngs(args.seed))

        step_increment = 0
        for _ in range(1, num_steps_with_bootstrap):
            obs = next_obs
            done = next_done
            carry = next_carry
            step_increment += args.num_envs

            obs, action, logits, next_carry, key = get_action(params, obs, carry, done, key)
            cpu_action = np.asarray(action)

            next_obs, next_reward_raw, next_term, next_trunc, _ = envs.step(cpu_action)
            next_reward = next_reward_raw.astype(np.float32)
            next_done_bool = next_term | next_trunc
            next_done = next_done_bool.astype(np.float32)

            storage.append(Transition(
                obs=obs, dones=done, actions=action, logits=logits,
                rewards=next_reward,
                lstm_c=carry[0], lstm_h=carry[1],
            ))

            episode_returns += next_reward
            returned_episode_returns = np.where(next_done_bool, episode_returns, returned_episode_returns)
            episode_returns *= 1 - next_done
            episode_lengths += 1
            returned_episode_lengths = np.where(next_done_bool, episode_lengths, returned_episode_lengths)
            episode_lengths *= 1 - next_done

        payload = (
            step_increment,
            jax.device_get(prepare_data(storage)),
            float(np.mean(returned_episode_returns)),
            float(np.max(returned_episode_returns)),
            float(np.mean(returned_episode_lengths)),
            float(np.mean(params_queue_get_time)),
        )
        rollout_queue.put(payload)
        storage = storage[-1:]

    envs.close()


def policy_gradient_loss(logits, *args):
    mean_per_batch = jax.vmap(rlax.policy_gradient_loss, in_axes=1)(logits, *args)
    return jnp.sum(mean_per_batch * logits.shape[0])


def entropy_loss_fn(logits, *args):
    mean_per_batch = jax.vmap(rlax.entropy_loss, in_axes=1)(logits, *args)
    return jnp.sum(mean_per_batch * logits.shape[0])


def impala_loss(agent, obs, actions, behavior_logits, rewards, dones, init_carry, gamma, vf_coef, ent_coef):
    discounts = (1.0 - dones[1:]) * gamma
    policy_logits, values, _ = agent.unroll(obs, dones, init_carry)
    v_tm1, v_t = values[:-1], values[1:]
    policy_logits = policy_logits[:-1]
    behavior_logits = behavior_logits[:-1]
    actions = actions[:-1]
    rewards = rewards[:-1]
    mask = jnp.ones_like(rewards)

    rhos = rlax.categorical_importance_sampling_ratios(policy_logits, behavior_logits, actions)
    vtrace_returns = jax.vmap(rlax.leaky_vtrace_td_error_and_advantage, in_axes=1, out_axes=1)(
        v_tm1, v_t, rewards, discounts, rhos,
    )

    pg_loss = policy_gradient_loss(policy_logits, actions, vtrace_returns.pg_advantage, mask)
    baseline_loss = 0.5 * jnp.sum(jnp.square(vtrace_returns.errors) * mask)
    ent_loss = entropy_loss_fn(policy_logits, mask)

    total_loss = pg_loss + vf_coef * baseline_loss + ent_coef * ent_loss
    return total_loss, (pg_loss, baseline_loss, ent_loss)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=40.0)
    parser.add_argument("--log-frequency", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    batch_size = args.num_envs * args.num_steps
    args.num_updates = args.total_timesteps // batch_size

    wandb.init(project="minimal-flaxrl", name="impala_lstm_LunarLander-v3", config=vars(args))

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    agent = IMPALALSTMAgent(obs_dim=8, action_dim=4, lstm_hidden_size=args.lstm_hidden_size, rngs=nnx.Rngs(args.seed))
    optimizer = nnx.Optimizer(agent, optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.rmsprop(learning_rate=args.learning_rate, decay=0.99, eps=0.01),
    ), wrt=nnx.Param)

    graphdef, _ = nnx.split(agent)

    params_queue = queue.Queue(maxsize=1)
    rollout_queue = queue.Queue(maxsize=1)
    params_queue.put(nnx.state(agent))
    key, thread_key = jax.random.split(key)
    threading.Thread(
        target=rollout,
        args=(thread_key, args, rollout_queue, params_queue, graphdef),
        daemon=True,
    ).start()

    @nnx.jit
    def update(agent, optimizer, storage):
        def loss_fn(agent):
            return impala_loss(
                agent, storage.obs, storage.actions, storage.logits,
                storage.rewards, storage.dones, (storage.lstm_c[0], storage.lstm_h[0]),
                args.gamma, args.vf_coef, args.ent_coef,
            )

        (loss, (pg_loss, v_loss, ent_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
        grad_norm = optax.global_norm(grads)
        optimizer.update(agent, grads)
        return loss, pg_loss, v_loss, ent_loss, grad_norm

    global_step = 0
    start_time = time.time()
    rollout_queue_get_time = deque(maxlen=10)
    for update_idx in range(1, args.num_updates + 1):
        training_time_start = time.time()
        rollout_queue_get_time_start = time.time()
        step_increment, storage, avg_return, max_return, avg_length, avg_params_queue_get_time = rollout_queue.get()
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)

        global_step += step_increment
        loss, pg_loss, v_loss, ent_loss, grad_norm = update(agent, optimizer, storage)

        params_queue.put(nnx.state(agent))

        if update_idx % args.log_frequency == 0:
            train_time = time.time() - training_time_start
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            avg_rollout_queue_get_time = float(np.mean(rollout_queue_get_time))

            print(f"update={update_idx}/{args.num_updates}, " f"loss={float(loss):.4f}, training_time={train_time:.3f}s, sps={sps}")
            print(f"global_step={global_step}, avg_return={avg_return:.2f}, " f"max_return={max_return:.2f}")
            wandb.log(
                {
                    "train/iteration": update_idx,
                    "train/global_env_step": global_step,
                    "train/sps": sps,
                    "train/actor_loss": float(pg_loss),
                    "train/critic_loss": float(v_loss),
                    "train/entropy": float(ent_loss),
                    "train/loss": float(loss),
                    "train/grad_norm": float(grad_norm),
                    "episode/reward_mean": avg_return,
                    "episode/reward_max": max_return,
                    "episode/length_mean": avg_length,
                    "stats/rollout_queue_get_time": avg_rollout_queue_get_time,
                    "stats/params_queue_get_time": avg_params_queue_get_time,
                    "stats/rollout_params_queue_get_time_diff": avg_rollout_queue_get_time - avg_params_queue_get_time,
                },
                step=global_step,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
