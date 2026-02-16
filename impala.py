# Minimal IMPALA (Importance Weighted Actor-Learner Architecture) in JAX/Flax NNX
# Single-device, LunarLander-v3, V-trace off-policy correction
import queue
import threading
import time
from argparse import ArgumentParser
from typing import List, NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from flax import nnx
import wandb


class IMPALAAgent(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(obs_dim, 256, rngs=rngs)
        self.fc_pi = nnx.Linear(256, action_dim, rngs=rngs)
        self.fc_v = nnx.Linear(256, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        logits = self.fc_pi(x)
        value = self.fc_v(x).squeeze(-1)
        return logits, value


class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    logits: list
    rewards: list


def rollout(
    key: jax.random.PRNGKey,
    args,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    graphdef,
):
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("LunarLander-v3", max_episode_steps=300) for _ in range(args.num_envs)]
    )
    global_step = 0

    @jax.jit
    def get_action(params, obs, key):
        obs = jnp.array(obs)
        agent = nnx.merge(graphdef, params)
        logits, _ = agent(obs)
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return obs, action, logits, key

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree.map(lambda *xs: jnp.stack(xs), *storage)

    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    returned_episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.float32)
    returned_episode_lengths = np.zeros(args.num_envs, dtype=np.float32)
    next_obs, _ = envs.reset(seed=args.seed)
    next_reward = np.zeros(args.num_envs, dtype=np.float32)
    next_done = np.zeros(args.num_envs, dtype=np.float32)

    storage = []

    for update in range(1, args.num_updates + 2):
        num_steps_with_bootstrap = args.num_steps + 1 + int(len(storage) == 0)

        params = params_queue.get()
        jax.block_until_ready(params)

        for _ in range(1, num_steps_with_bootstrap):
            obs = next_obs
            done = next_done

            global_step += args.num_envs

            obs, action, logits, key = get_action(params, obs, key)
            cpu_action = np.array(action)

            next_obs, next_reward_raw, next_term, next_trunc, info = envs.step(cpu_action)
            next_reward = next_reward_raw.astype(np.float32)
            next_done_bool = next_term | next_trunc
            next_done = next_done_bool.astype(np.float32)

            storage.append(Transition(
                obs=obs, dones=done, actions=action, logits=logits,
                rewards=next_reward,
            ))

            episode_returns += next_reward
            returned_episode_returns = np.where(next_done_bool, episode_returns, returned_episode_returns)
            episode_returns *= 1 - next_done
            episode_lengths += 1
            returned_episode_lengths = np.where(next_done_bool, episode_lengths, returned_episode_lengths)
            episode_lengths *= 1 - next_done

        avg_episodic_return = np.mean(returned_episode_returns)
        max_episodic_return = np.max(returned_episode_returns)

        payload = (global_step, prepare_data(storage), avg_episodic_return)
        rollout_queue.put(payload)
        storage = storage[-1:]

        if update % args.log_frequency == 0:
            print(f"global_step={global_step}, avg_return={avg_episodic_return:.2f}, max_return={max_episodic_return:.2f}")
            wandb.log({
                "episode/reward_mean": avg_episodic_return,
                "episode/reward_max": max_episodic_return,
                "episode/length_mean": np.mean(returned_episode_lengths),
            }, step=global_step)


def policy_gradient_loss(logits, *args):
    """rlax.policy_gradient_loss, but with sum(loss) and [T, B, ...] inputs."""
    mean_per_batch = jax.vmap(rlax.policy_gradient_loss, in_axes=1)(logits, *args)
    return jnp.sum(mean_per_batch * logits.shape[0])


def entropy_loss_fn(logits, *args):
    """rlax.entropy_loss, but with sum(loss) and [T, B, ...] inputs."""
    mean_per_batch = jax.vmap(rlax.entropy_loss, in_axes=1)(logits, *args)
    return jnp.sum(mean_per_batch * logits.shape[0])


def impala_loss(agent, obs, actions, behavior_logits, rewards, dones, gamma, vf_coef, ent_coef):
    discounts = (1.0 - dones[1:]) * gamma  # dones[t] is for arriving at t; need dones[t+1] for transition t→t+1

    policy_logits, values = agent(obs)

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
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=40.0)
    parser.add_argument("--log-frequency", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_arguments()
    batch_size = args.num_envs * args.num_steps
    args.num_updates = args.total_timesteps // batch_size

    wandb.init(project="minimal-flaxrl", name="impala_LunarLander-v3", config=vars(args))

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    agent = IMPALAAgent(obs_dim=8, action_dim=4, rngs=nnx.Rngs(args.seed))
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
                storage.rewards, storage.dones,
                args.gamma, args.vf_coef, args.ent_coef,
            )

        (loss, (pg_loss, v_loss, ent_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
        optimizer.update(agent, grads)
        return loss, pg_loss, v_loss, ent_loss

    # Learner loop
    for update_idx in range(1, args.num_updates + 1):
        training_time_start = time.time()
        global_step, storage, _ = rollout_queue.get()

        loss, pg_loss, v_loss, ent_loss = update(agent, optimizer, storage)

        params_queue.put(nnx.state(agent))

        if update_idx % args.log_frequency == 0:
            print(f"update={update_idx}/{args.num_updates}, loss={float(loss):.4f}, training_time={time.time() - training_time_start:.3f}s")
            wandb.log({
                "train/iteration": update_idx,
                "train/global_env_step": global_step,
                "train/actor_loss": float(pg_loss),
                "train/critic_loss": float(v_loss),
                "train/entropy": float(ent_loss),
            }, step=global_step)

    wandb.finish()


if __name__ == "__main__":
    main()
