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
            [], # log_probs
            [], # next_obs
            [], # rewards
            [], # truncated
            [], # terminated
        ]

    def add(self, transition):
        for idx, data in enumerate(transition):
            self.data[idx].append(jnp.array(data))

    def get(self):
        obs = jnp.stack(self.data[0], 1)
        actions = jnp.stack(self.data[1], 1)
        a_probs = jnp.stack(self.data[2], 1)
        next_obs = jnp.stack(self.data[3], 1)
        rewards = jnp.expand_dims(jnp.stack(self.data[4], 1), -1)
        truncated = jnp.expand_dims(jnp.stack(self.data[5], 1), -1)
        terminated = jnp.expand_dims(jnp.stack(self.data[6], 1), -1)
        return obs, actions, a_probs, next_obs, rewards, truncated, terminated
    

class PPO(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(8, 256, rngs=rngs)
        self.fc_pi = nnx.Linear(256, 4, rngs=rngs)
        self.fc_v = nnx.Linear(256, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        return pi, v

@nnx.jit
def sample_action(model, obs, rngs):
    logits, _ = model(obs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    actions = rngs.categorical(logits, axis=-1)
    log_prob = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    return log_prob, actions

@nnx.jit
def get_value(model, obs):
    _, v = model(obs)
    return v

def calculate_gae(model, batch, gamma: float = 0.97, lmbda: float = 0.97):
    obs, actions, a_probs, next_obs, rewards, truns, terms = batch
    values = get_value(model, obs)
    next_values = get_value(model, next_obs)
    td = (rewards + gamma * next_values * (1.0 - terms) - values).squeeze(-1)
    dones = (truns + terms).squeeze(-1)

    # [batch, seq_len] -> [seq_len, batch] for scan iter
    td_t = td.transpose(1, 0)
    dones_t = dones.transpose(1, 0)

    def scan_step(carry_advantage, inputs):
        td_step, done_step = inputs
        advantage = td_step + gamma * lmbda * carry_advantage * (1.0 - done_step)
        return advantage, advantage

    init_advantage = jnp.zeros(td_t.shape[1], dtype=td.dtype)
    _, advantages_t = jax.lax.scan(scan_step, init_advantage, (td_t, dones_t), reverse=True)
    advantages = jnp.expand_dims(advantages_t.transpose(1, 0), -1)
    return *batch, advantages


def loss_fn(model, batch, gamma=.97):
    obs, actions, old_log_probs, next_obs, rewards, truns, terms, advantages = batch
    # calculate critic loss
    # logits : [8, 20, 2] actions : [8, 20]
    logits, value = model(obs)
    next_value = get_value(model, next_obs)
    target = jax.lax.stop_gradient(rewards + gamma * next_value * (1.0 - terms))
    critic_loss = optax.huber_loss(value, target)
    # ratio
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_pi_a = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    ratio = jnp.exp(log_pi_a - old_log_probs)
    ratio = ratio.reshape(-1)
    advantages = advantages.reshape(-1)
    actor_loss = rlax.clipped_surrogate_pg_loss(ratio, advantages, 0.1)
    total_loss = actor_loss + 0.5 * critic_loss
    return total_loss.mean(), (actor_loss.mean(), critic_loss.mean())

@nnx.jit
def update_ppo(model: nnx.Module, optimizer: nnx.Optimizer, batch, metrics: nnx.metrics.MultiMetric, gamma=0.97):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (actor_loss, critic_loss)), grad = grad_fn(model, batch, gamma)
    optimizer.update(model, grad)
    metrics.update(actor_loss=actor_loss, critic_loss=critic_loss)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lmbda", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()

def main():
    args = parse_arguments()
    wandb.init(project="jax-playground", name=f"ppo_{args.env_name}", config=vars(args))

    rngs = nnx.Rngs(args.seed)
    ppo = PPO(rngs=rngs)
    optimizer = nnx.Optimizer(ppo, optax.adamw(args.learning_rate), wrt=nnx.Param)
    replay_buffer = ReplayBuffer()

    metrics = nnx.metrics.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        actor_loss=nnx.metrics.Average("actor_loss")
    )

    envs = gym.make_vec(args.env_name, num_envs=args.batch_size, vectorization_mode='sync', max_episode_steps=300)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    obs, _ = envs.reset()
    global_env_step = 0
    for iteration in range(args.num_iter):
        rollout_rewards = []
        rollout_lengths = []

        for _ in range(args.num_steps):
            a_probs, sampled_actions = sample_action(ppo, obs, rngs)
            next_obs, r, terminated, truncated, info = envs.step(np.asarray(sampled_actions))
            replay_buffer.add((obs, sampled_actions, a_probs, next_obs, r, truncated, terminated))
            global_env_step += args.batch_size

            if "_episode" in info:
                for idx, finished in enumerate(info["_episode"]):
                    if finished:
                        episode_reward = float(info["episode"]["r"][idx])
                        episode_length = int(info["episode"]["l"][idx])
                        rollout_rewards.append(episode_reward)
                        rollout_lengths.append(episode_length)

            obs = next_obs

        batch = replay_buffer.get()
        batch = calculate_gae(ppo, batch, gamma=args.gamma, lmbda=args.lmbda)

        for _ in range(args.num_epochs):
            update_ppo(ppo, optimizer, batch, metrics, gamma=args.gamma)

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

        # 다음 iteration을 위해 버퍼 초기화
        replay_buffer._init_data()

if __name__=="__main__":
    main()
