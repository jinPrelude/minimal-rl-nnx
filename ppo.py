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

    # [batch, seq_len] -> [seq_len, batch] for scan iter
    td_t = td.transpose(1, 0)
    term_t = terms.squeeze(-1).transpose(1, 0)

    def scan_step(carry_advantage, inputs):
        td_step, done_step = inputs
        advantage = td_step + gamma * lmbda * carry_advantage * (1.0 - done_step)
        return advantage, advantage

    init_advantage = jnp.zeros(td_t.shape[1], dtype=td.dtype)
    _, advantages_t = jax.lax.scan(scan_step, init_advantage, (td_t, term_t), reverse=True)
    advantages = jnp.expand_dims(advantages_t.transpose(1, 0), -1)
    returns = advantages + values

    return *batch, advantages, returns


def loss_fn(model, batch, gamma, clip_eps):
    obs, actions, old_log_probs, next_obs, rewards, truns, terms, advantages, returns = batch

    # logits : [8, 20, 2] actions : [8, 20]
    logits, value = model(obs)
    critic_loss = optax.huber_loss(value, jax.lax.stop_gradient(returns)).mean()
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_pi_a = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    ratio = jnp.exp(log_pi_a - old_log_probs)
    actor_loss = rlax.clipped_surrogate_pg_loss(ratio.reshape(-1), advantages.reshape(-1), clip_eps).mean()

    total_loss = actor_loss + 0.5 * critic_loss
    return total_loss, (actor_loss, critic_loss)

@nnx.jit
def update_ppo(model: nnx.Module, optimizer: nnx.Optimizer, minibatches, metrics: nnx.metrics.MultiMetric, gamma=0.97, clip_eps=0.2):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def scan_step(carry, minibatch):
        model, optimizer, metrics = carry
        (loss, (actor_loss, critic_loss)), grad = grad_fn(model, minibatch, gamma, clip_eps)
        optimizer.update(model, grad)
        metrics.update(actor_loss=actor_loss, critic_loss=critic_loss)
        return model, optimizer, metrics

    scan_step((model, optimizer, metrics), minibatches)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=100000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99) # NOTE: gamma was important for training LunarLander. don't decrease gamma below 0.99
    parser.add_argument("--lmbda", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--clip-eps", type=float, default=0.2)
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

    envs = gym.make_vec(args.env_name, num_envs=args.num_envs, vectorization_mode='sync', max_episode_steps=300)
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
        batch = calculate_gae(ppo, batch, gamma=args.gamma, lmbda=args.lmbda)

        # Flatten: [B, T, ...] -> [B*T, ...]
        flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)

        # Shuffle + minibatch split
        total_samples = args.num_envs * args.num_steps
        perm = jax.random.permutation(rngs(), total_samples)
        shuffled = jax.tree.map(lambda x: x[perm], flat_batch)
        
        num_minibatches = total_samples // args.minibatch_size
        for _ in range(args.num_epochs):
            perm = jax.random.permutation(rngs(), total_samples)
            shuffled = jax.tree.map(lambda x: x[perm], flat_batch)
            minibatches = jax.tree.map(
                lambda x: x[:num_minibatches * args.minibatch_size].reshape(
                    num_minibatches, args.minibatch_size, *x.shape[1:]
                ),
                shuffled
            )
            update_ppo(ppo, optimizer, minibatches, metrics, gamma=args.gamma, clip_eps=args.clip_eps)


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
