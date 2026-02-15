from argparse import ArgumentParser
import gymnasium as gym
import flax.nnx as nnx
import numpy as np
import jax
from jax import numpy as jnp
import optax
import rlax
import wandb

class ReplayBuffer:
    def __init__(self):
        self._init_data()

    def _init_data(self):
        self.data = [
            [], # obs
            [], # actions
            [], # next_obs
            [], # rewards
            [], # done
        ]

    def add(self, transition):
        for idx, data in enumerate(transition):
            self.data[idx].append(jnp.array(data))

    def get(self):
        obs = jnp.stack(self.data[0], 1)
        actions = jnp.stack(self.data[1], 1)
        next_obs = jnp.stack(self.data[2], 1)
        rewards = jnp.expand_dims(jnp.stack(self.data[3], 1), -1)
        done = jnp.expand_dims(jnp.stack(self.data[4], 1), -1)
        return obs, actions, next_obs, rewards, done


class A2C(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(8, 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, 256, rngs=rngs)
        self.fc_pi = nnx.Linear(256, 4, rngs=rngs)
        self.fc_v = nnx.Linear(256, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        return pi, v

@nnx.jit
def sample_action(model, obs, rngs):
    logits, _ = model(obs)
    actions = rngs.categorical(logits, axis=-1)
    return actions

@nnx.jit
def get_value(model, obs):
    _, v = model(obs)
    return v

def calculate_returns(model, batch, gamma: float = 0.99):
    obs, actions, next_obs, rewards, done = batch
    values = get_value(model, obs)

    # Bootstrap: V(s_{T+1}) for the last timestep
    bootstrap_value = get_value(model, next_obs[:, -1]).squeeze(-1)  # [B]

    # [B, T, 1] -> [T, B] for reverse scan
    rewards_t = rewards.squeeze(-1).transpose(1, 0)
    done_t = done.squeeze(-1).transpose(1, 0)

    def scan_step(carry_G, inputs):
        r_t, d_t = inputs
        G = r_t + gamma * carry_G * (1.0 - d_t)
        return G, G

    _, returns_t = jax.lax.scan(scan_step, bootstrap_value, (rewards_t, done_t), reverse=True)
    returns = jnp.expand_dims(returns_t.transpose(1, 0), -1)  # [B, T, 1]
    advantages = returns - values

    return *batch, advantages, returns

def loss_fn(model, batch):
    obs, actions, next_obs, rewards, done, advantages, returns = batch

    logits, value = model(obs)
    critic_loss = optax.huber_loss(value, jax.lax.stop_gradient(returns)).mean()

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_pi_a = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    actor_loss = -(log_pi_a * jax.lax.stop_gradient(advantages).squeeze(-1)).mean()

    total_loss = actor_loss + 0.5 * critic_loss
    return total_loss, (actor_loss, critic_loss)

@nnx.jit
def update_a2c(model: nnx.Module, optimizer: nnx.Optimizer, batch, metrics: nnx.metrics.MultiMetric):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (actor_loss, critic_loss)), grad = grad_fn(model, batch)
    optimizer.update(model, grad)
    metrics.update(actor_loss=actor_loss, critic_loss=critic_loss)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()

def main():
    args = parse_arguments()
    wandb.init(project="jax-playground", name=f"a2c_{args.env_name}", config=vars(args))

    rngs = nnx.Rngs(args.seed)
    a2c = A2C(rngs=rngs)
    optimizer = nnx.Optimizer(a2c, optax.adam(args.learning_rate), wrt=nnx.Param)
    replay_buffer = ReplayBuffer()

    metrics = nnx.metrics.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        actor_loss=nnx.metrics.Average("actor_loss"),
    )

    envs = gym.make_vec(args.env_name, num_envs=args.num_envs, vectorization_mode='sync', max_episode_steps=300)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    obs, _ = envs.reset()
    global_env_step = 0
    for iteration in range(args.num_iter):
        rollout_rewards = []
        rollout_lengths = []

        for _ in range(args.num_steps):
            sampled_actions = sample_action(a2c, obs, rngs)
            next_obs, r, terminated, truncated, info = envs.step(np.asarray(sampled_actions))
            done = np.maximum(truncated, terminated)
            replay_buffer.add((obs, sampled_actions, next_obs, r, done))
            global_env_step += args.num_envs

            if "_episode" in info:
                for idx, finished in enumerate(info["_episode"]):
                    if finished:
                        episode_reward = float(info["episode"]["r"][idx])
                        episode_length = int(info["episode"]["l"][idx])
                        rollout_rewards.append(episode_reward)
                        rollout_lengths.append(episode_length)

            obs = next_obs

        batch = replay_buffer.get()
        batch = calculate_returns(a2c, batch, gamma=args.gamma)

        # Flatten: [B, T, ...] -> [B*T, ...]
        flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
        update_a2c(a2c, optimizer, flat_batch, metrics)

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
        replay_buffer._init_data()

if __name__=="__main__":
    main()
