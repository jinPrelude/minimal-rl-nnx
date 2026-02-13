import gymnasium as gym
import flax.nnx as nnx
import numpy as np
import jax
from jax import numpy as jnp
import optax
import rlax

class ReplayBuffer:
    def __init__(self):
        self._init_data()

    def _init_data(self):
        self.data = [
            [], # obs
            [], # actions
            [], # logits
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
        self.fc1 = nnx.Linear(4, 256, rngs=rngs)
        self.fc_pi = nnx.Linear(256, 2, rngs=rngs)
        self.fc_v = nnx.Linear(256, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        return pi, v

@nnx.jit
def sample_action(model, obs, rngs):
    logits, _ = model(obs)
    probs = jax.nn.softmax(logits, axis=-1)
    actions = rngs.categorical(logits, axis=-1)
    a_prob = jnp.take_along_axis(probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    return a_prob, actions

@nnx.jit
def get_value(model, obs):
    _, v = model(obs)
    return v

def calculate_gae(model, batch, gamma: float = 0.97, lmbda: float = 0.97):
    obs, actions, a_probs, next_obs, rewards, truns, terms = batch
    values = get_value(model, obs)
    next_values = get_value(model, next_obs)
    td = (rewards + gamma * next_values * (1.0 - terms) - values).squeeze()
    advantage = 0
    advantages_list = []
    for i in reversed(range(td.shape[1])):
        advantage = gamma * lmbda * advantage + td[:, i]
        advantages_list.append(advantage)
    advantages = jnp.expand_dims(jnp.array(advantages_list[::-1]).transpose(), -1)
    return *batch, advantages

# def calculate_gae_scan(model, batch, gamma: float = 0.97, lmbda: float = 0.97):
#     obs, actions, next_obs, rewards, truns, terms = batch # [batch, timestep, 1]
#     values = get_value(model, obs)
#     next_values = get_value(model, next_obs)
#     td = (rewards + gamma * next_values * (1.0 - terms) - values).squeeze()

#     def scan_fn(carry, x):
#         td, terminal = x
#         carry = gamma * lmbda * x * carry * (1 - terminal) + td
#         return carry, carry
    
#     carry = 0
#     xs = 



def sample_minibatch(batch, minibatch_size: int, rngs: nnx.Rngs):
    pass

def get_loss(value, target):
    pass


def update_ppo(model: nnx.Module, optimizer: nnx.Optimizer, batch, gamma=0.97):
    obs, actions, a_probs, next_obs, rewards, truns, terms, advantages = batch

    # calculate critic loss
    # logits : [8, 20, 2] actions : [8, 20]
    logits, value = model(obs)
    next_value = get_value(model, next_obs)
    target = jax.lax.stop_gradient(rewards + gamma * next_value)
    critic_loss = optax.huber_loss(value, target)
    # ratio = jnp.exp(jnp.log())
    # test = rlax.clipped_surrogate_pg_loss()
    probs = jax.nn.softmax(logits, axis=-1)
    pi_a = jnp.take_along_axis(probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    ratio = jnp.exp(jnp.log(pi_a) - jnp.log(a_probs))
    ratio = ratio.reshape(-1)
    advantages = advantages.reshape(-1)
    test = rlax.clipped_surrogate_pg_loss(ratio, advantages, 0.1)
    print("hi")
    # calculate actor loss
    


def main():
    num_iter = 100000
    num_steps = 20
    batch_size = 8
    num_epochs = 4

    rngs = nnx.Rngs(0)
    ppo = PPO(rngs=rngs)
    optimizer = nnx.Optimizer(ppo, optax.adamw(0.001), wrt=nnx.Param)
    replay_buffer = ReplayBuffer()

    envs = gym.make_vec('CartPole-v1', num_envs=batch_size, vectorization_mode='sync')
    for iter in range(num_iter):
        obs, _ = envs.reset()

        for i in range(num_steps):
            a_probs, sampled_actions = sample_action(ppo, obs, rngs)
            next_obs, r, done, truncated, info = envs.step(np.asarray(sampled_actions))
            replay_buffer.add((obs, sampled_actions, a_probs, next_obs, r, done, truncated))
        batch = replay_buffer.get()
        batch = calculate_gae(ppo, batch)
        # batch_v2 = calculate_gae_scan(ppo, batch)
        update_ppo(ppo, optimizer, batch)

if __name__=="__main__":
    main()