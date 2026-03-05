"""
TrXL based on CleanRL implementation, but training logic is changed (segment wise learning).
"""
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import memory_gym  # noqa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from einops import rearrange
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    torch_compile: bool = True
    """if toggled, compile hot paths with torch.compile"""
    torch_compile_mode: str = "reduce-overhead"
    """torch.compile mode (`default`, `reduce-overhead`, `max-autotune`)"""

    # Algorithm specific arguments
    env_id: str = "MortarMayhem-Grid-v0"
    """the id of the environment"""
    total_timesteps: int = 200000000
    """total timesteps of the experiments"""
    init_lr: float = 2.5e-4
    """the initial learning rate of the optimizer"""
    final_lr: float = 1e-5
    """the final learning rate of the optimizer after linearly annealing"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    segment_length: int = 32
    """the number of rollout steps processed in one truncated BPTT segment"""
    anneal_steps: int = 32 * 512 * 10000
    """the number of steps to linearly anneal the learning rate and entropy coefficient from initial to final"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    init_ent_coef: float = 0.0001
    """initial coefficient of the entropy bonus"""
    final_ent_coef: float = 0.000001
    """final coefficient of the entropy bonus after linearly annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer-XL specific arguments
    trxl_num_layers: int = 3
    """the number of transformer layers"""
    trxl_num_heads: int = 4
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 384
    """the dimension of the transformer"""
    trxl_memory_length: int = 88
    """the length of TrXL's sliding memory window"""

    # To be filled on runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_segments: int = 0
    """the number of non-overlapping rollout segments (computed in runtime)"""
    envs_per_minibatch: int = 0
    """the number of environments in one minibatch (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, render_mode="debug_rgb_array"):
    if "MiniGrid" in env_id:
        if render_mode == "debug_rgb_array":
            render_mode = "rgb_array"

    def thunk():
        if "MiniGrid" in env_id:
            env = gym.make(env_id, agent_view_size=3, tile_size=28, render_mode=render_mode)
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=28))
            env = gym.wrappers.TimeLimit(env, 96)
        else:
            env = gym.make(env_id, render_mode=render_mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass
class TrXLState:
    memory: torch.Tensor
    valid_len: torch.Tensor
    pos: torch.Tensor


def init_trxl_state(num_envs, mem_len, num_layers, dim, device):
    return TrXLState(
        memory=torch.zeros((num_envs, mem_len, num_layers, dim), dtype=torch.float32, device=device),
        valid_len=torch.zeros((num_envs,), dtype=torch.long, device=device),
        pos=torch.zeros((num_envs,), dtype=torch.long, device=device),
    )


def clone_trxl_state(state, device=None):
    if device is None:
        device = state.memory.device
    return TrXLState(
        memory=state.memory.to(device).clone(),
        valid_len=state.valid_len.to(device).clone(),
        pos=state.pos.to(device).clone(),
    )


def reset_done_in_state(state, done_mask):
    done_mask = done_mask.to(device=state.memory.device, dtype=torch.bool)
    if done_mask.ndim != 1:
        raise ValueError(f"`done_mask` must be rank-1 [num_envs], got shape={tuple(done_mask.shape)}")

    keep_mask = (~done_mask).view(-1, 1, 1, 1)
    state.memory = torch.where(keep_mask, state.memory, torch.zeros_like(state.memory))
    state.valid_len = torch.where(done_mask, torch.zeros_like(state.valid_len), state.valid_len)
    state.pos = torch.where(done_mask, torch.zeros_like(state.pos), state.pos)
    return state


def append_memory_token(state, token, max_episode_steps):
    if token.ndim != 3:
        raise ValueError(f"`token` must be rank-3 [num_envs, num_layers, dim], got shape={tuple(token.shape)}")
    mem_len = state.memory.shape[1]
    token = token.to(state.memory.device)
    state.memory = torch.cat((state.memory[:, 1:], token.unsqueeze(1)), dim=1)
    state.valid_len = torch.clamp(state.valid_len + 1, max=mem_len)
    state.pos = torch.clamp(state.pos + 1, max=max_episode_steps)
    return state


def build_memory_inputs(state, mem_len, max_episode_steps):
    if state.memory.shape[1] != mem_len:
        raise ValueError(f"State memory length ({state.memory.shape[1]}) does not match `mem_len` ({mem_len})")

    device = state.memory.device
    memory_idx = torch.arange(mem_len, device=device, dtype=torch.long).unsqueeze(0)
    memory_mask = memory_idx >= (mem_len - state.valid_len.unsqueeze(1))
    memory_indices = state.pos.unsqueeze(1) - mem_len + memory_idx
    memory_indices = torch.clamp(memory_indices, min=0, max=max_episode_steps - 1)
    return state.memory, memory_mask, memory_indices


def build_parallel_eval_metadata(dones_seq, init_state, mem_len, max_episode_steps):
    if dones_seq.ndim != 2:
        raise ValueError(f"`dones_seq` must be rank-2 [T, B], got shape={tuple(dones_seq.shape)}")
    if max_episode_steps <= 0:
        raise ValueError(f"`max_episode_steps` must be > 0, got {max_episode_steps}")
    if mem_len <= 0:
        raise ValueError(f"`mem_len` must be > 0, got {mem_len}")

    done = dones_seq.to(dtype=torch.bool, device=init_state.pos.device).transpose(0, 1)  # [B, T]
    batch_size, seq_len = done.shape
    device = done.device

    t = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, T]
    m = torch.arange(mem_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, M]
    init_pos = init_state.pos.to(device=device, dtype=torch.long)
    init_valid_len = init_state.valid_len.to(device=device, dtype=torch.long)

    episode_ids = torch.cumsum(done.to(torch.long), dim=1)  # [B, T]
    reset_points = torch.where(done, t.expand(batch_size, -1), -torch.ones((batch_size, seq_len), dtype=torch.long, device=device))
    last_reset = torch.cummax(reset_points, dim=1).values
    query_pos_unclamped = torch.where(episode_ids == 0, init_pos.unsqueeze(1) + t, t.expand(batch_size, -1) - last_reset)

    memory_valid = m >= (mem_len - init_valid_len.unsqueeze(1))
    memory_pos_unclamped = init_pos.unsqueeze(1) + m - mem_len

    memory_indices = torch.clamp(memory_pos_unclamped, min=0, max=max_episode_steps - 1)
    query_indices = torch.clamp(query_pos_unclamped, min=0, max=max_episode_steps - 1)

    memory_episode_ids = torch.where(memory_valid, torch.zeros_like(memory_pos_unclamped), -torch.ones_like(memory_pos_unclamped))
    memory_key_pos = torch.where(
        memory_valid,
        memory_pos_unclamped,
        torch.full_like(memory_pos_unclamped, fill_value=-1_000_000_000),
    )
    key_episode_ids = torch.cat((memory_episode_ids, episode_ids), dim=1)  # [B, M+T]
    key_pos = torch.cat((memory_key_pos, query_pos_unclamped), dim=1)  # [B, M+T]

    query_pos = query_pos_unclamped.unsqueeze(2)  # [B, T, 1]
    attn_mask = (
        (key_episode_ids.unsqueeze(1) == episode_ids.unsqueeze(2))
        & (key_pos.unsqueeze(1) < query_pos)
        & (key_pos.unsqueeze(1) >= query_pos - mem_len)
    )

    final_valid_len = torch.where(
        episode_ids[:, -1] == 0,
        init_valid_len + seq_len,
        seq_len - last_reset[:, -1],
    )
    final_valid_len = torch.clamp(final_valid_len, min=0, max=mem_len)
    final_pos = torch.clamp(query_pos_unclamped[:, -1] + 1, min=0, max=max_episode_steps)

    return memory_indices, query_indices, attn_mask, final_valid_len, final_pos


class PositionalEncoding(nn.Module):
    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0, device=self.inv_freqs.device, dtype=self.inv_freqs.dtype)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(self.inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb


class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert self.head_size * num_heads == embed_dim, "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Dot-product
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            if mask.ndim == 2:
                attn_mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.ndim == 3:
                attn_mask = mask.unsqueeze(1)
            else:
                raise ValueError(f"`mask` must be rank-2 [B, K] or rank-3 [B, Q, K], got rank={mask.ndim}")
            energy = energy.masked_fill(attn_mask == 0, float("-1e20"))  # -inf causes NaN

        # Normalize energy values and apply softmax to retrieve the attention scores
        attention = torch.softmax(
            energy / (self.embed_dim ** (1 / 2)), dim=3
        )  # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_size)

        return self.fc_out(out), attention


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        # Pre-layer normalization (post-layer normalization is usually less effective)
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value  # K = V -> self-attention
        attention, attention_weights = self.attention(value, key, query_, mask)  # MHA
        x = attention + query  # Skip connection
        x_ = self.layer_norm_attn(x)  # Pre-layer normalization
        forward = self.fc_projection(x_)  # Forward projection
        out = forward + x  # Skip connection
        return out, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, max_episode_steps):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.pos_embedding = PositionalEncoding(dim)
        self.transformer_layers = nn.ModuleList([TransformerLayer(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, memories, mask, memory_indices, detach_new_memory=True):
        # Add absolute positional encoding to every transformer layer input
        pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
        memories = memories + pos_embedding.unsqueeze(2)

        # Forward transformer layers and return new memories (i.e. hidden states)
        out_memories = []
        for i, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach() if detach_new_memory else x)
            x, attention_weights = layer(
                memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask
            )  # args: value, key, query, mask
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x, torch.stack(out_memories, dim=1)

    def forward_sequence(self, x_seq, memories, mask, memory_indices, query_indices, detach_new_memory=True):
        if x_seq.ndim != 3:
            raise ValueError(f"`x_seq` must be rank-3 [B, T, D], got shape={tuple(x_seq.shape)}")

        pos_embedding = self.pos_embedding(self.max_episode_steps)
        memory_pos_embedding = pos_embedding[memory_indices]
        query_pos_embedding = pos_embedding[query_indices]
        fallback_kv_token = pos_embedding[0]

        x = x_seq
        layer_inputs = []
        for i, layer in enumerate(self.transformer_layers):
            layer_inputs.append(x.detach() if detach_new_memory else x)
            kv = torch.cat((memories[:, :, i] + memory_pos_embedding, x + query_pos_embedding), dim=1)
            layer_mask = mask
            if layer_mask is not None:
                no_key_rows = ~layer_mask.any(dim=2, keepdim=True)
                # Always prepend a fallback key to avoid data-dependent graph breaks in torch.compile.
                dummy_kv = fallback_kv_token.to(dtype=kv.dtype, device=kv.device).view(1, 1, -1).expand(
                    kv.shape[0], 1, kv.shape[2]
                )
                kv = torch.cat((dummy_kv, kv), dim=1)
                layer_mask = torch.cat((no_key_rows, layer_mask), dim=2)
            x, attention_weights = layer(kv, kv, x, layer_mask)  # args: value, key, query, mask
            del attention_weights
        return x, torch.stack(layer_inputs, dim=2)


class Agent(nn.Module):
    def __init__(self, args, observation_space, action_dim, max_episode_steps):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps

        if len(self.obs_shape) > 1:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
                nn.ReLU(),
            )
        else:
            self.encoder = layer_init(nn.Linear(observation_space.shape[0], args.trxl_dim))

        self.transformer = Transformer(args.trxl_num_layers, args.trxl_dim, args.trxl_num_heads, self.max_episode_steps)

        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(args.trxl_dim, out_features=action_dim), np.sqrt(0.01))
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), 1)

    def get_value(self, x, memory, memory_mask, memory_indices, detach_new_memory=True):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, _ = self.transformer(x, memory, memory_mask, memory_indices, detach_new_memory=detach_new_memory)
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None, detach_new_memory=True):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, memory = self.transformer(
            x,
            memory,
            memory_mask,
            memory_indices,
            detach_new_memory=detach_new_memory,
        )
        x = self.hidden_post_trxl(x)
        probs = Categorical(logits=self.actor(x))
        if action is None:
            action = probs.sample()
        log_probs = probs.log_prob(action)
        entropies = probs.entropy()
        return action, log_probs, entropies, self.critic(x).flatten(), memory

    def evaluate_segment(self, obs_seq, actions_seq, dones_seq, init_state, max_episode_steps):
        seq_len, batch_size = obs_seq.shape[0], obs_seq.shape[1]
        state = clone_trxl_state(init_state, device=obs_seq.device)
        memory_indices, query_indices, attn_mask, _, _ = build_parallel_eval_metadata(
            dones_seq, state, state.memory.shape[1], max_episode_steps
        )

        if len(self.obs_shape) > 1:
            obs_flat = obs_seq.reshape(seq_len * batch_size, *obs_seq.shape[2:])
            encoded = self.encoder(obs_flat.permute((0, 3, 1, 2)) / 255.0)
        else:
            obs_flat = obs_seq.reshape(seq_len * batch_size, obs_seq.shape[2])
            encoded = self.encoder(obs_flat)

        x_seq = encoded.reshape(seq_len, batch_size, -1).permute((1, 0, 2))
        x_seq, layer_inputs = self.transformer.forward_sequence(
            x_seq,
            state.memory,
            attn_mask,
            memory_indices,
            query_indices,
            detach_new_memory=False,
        )
        del layer_inputs

        x_seq = self.hidden_post_trxl(x_seq)
        x_flat = x_seq.permute((1, 0, 2)).reshape(seq_len * batch_size, -1)

        probs = Categorical(logits=self.actor(x_flat))
        flat_actions = actions_seq.reshape(seq_len * batch_size)
        log_probs = probs.log_prob(flat_actions)
        entropies = probs.entropy().reshape(seq_len, batch_size)
        values = self.critic(x_flat).flatten().reshape(seq_len, batch_size)
        new_log_probs = log_probs.reshape(seq_len, batch_size)

        return new_log_probs, entropies, values


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.segment_length <= 0:
        raise ValueError(f"`segment_length` must be > 0, got {args.segment_length}")
    if args.num_steps % args.segment_length != 0:
        raise ValueError(f"`num_steps` ({args.num_steps}) must be divisible by `segment_length` ({args.segment_length})")
    if args.num_envs % args.num_minibatches != 0:
        raise ValueError(f"`num_envs` ({args.num_envs}) must be divisible by `num_minibatches` ({args.num_minibatches})")
    if args.trxl_memory_length <= 0:
        raise ValueError(f"`trxl_memory_length` must be > 0, got {args.trxl_memory_length}")

    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_segments = int(args.num_steps // args.segment_length)
    args.envs_per_minibatch = int(args.num_envs // args.num_minibatches)
    args.minibatch_size = int(args.envs_per_minibatch * args.segment_length)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Determine the device to be used for training and set the default tensor type
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
    else:
        device = torch.device("cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    observation_space = envs.single_observation_space
    action_dim = envs.single_action_space.n
    # Determine maximum episode steps
    envs.envs[0].reset()  # Memory Gym envs expose max_episode_steps after reset
    max_episode_steps = envs.envs[0].get_wrapper_attr("max_episode_steps")
    if max_episode_steps <= 0:
        max_episode_steps = 1024  # Memory Gym envs have max_episode_steps set to -1
    # Set transformer memory length to max episode steps if greater than max episode steps
    args.trxl_memory_length = min(args.trxl_memory_length, max_episode_steps)
    agent = Agent(args, observation_space, action_dim, max_episode_steps).to(device)
    get_action_and_value = agent.get_action_and_value
    evaluate_segment = agent.evaluate_segment
    if args.torch_compile:
        get_action_and_value = torch.compile(get_action_and_value, mode=args.torch_compile_mode, fullgraph=False)
        evaluate_segment = torch.compile(evaluate_segment, mode=args.torch_compile_mode, fullgraph=False)
    optimizer = optim.AdamW(agent.parameters(), lr=args.init_lr)

    # ALGO Logic: Storage setup
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape, dtype=torch.float32, device=device)
    log_probs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    segment_init_memory = torch.zeros(
        (
            args.num_segments,
            args.num_envs,
            args.trxl_memory_length,
            args.trxl_num_layers,
            args.trxl_dim,
        ),
        dtype=torch.float32,
        device="cpu",
    )
    segment_init_valid_len = torch.zeros((args.num_segments, args.num_envs), dtype=torch.long, device="cpu")
    segment_init_pos = torch.zeros((args.num_segments, args.num_envs), dtype=torch.long, device="cpu")

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)  # Store episode results for monitoring statistics
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    rollout_state = init_trxl_state(
        args.num_envs,
        args.trxl_memory_length,
        args.trxl_num_layers,
        args.trxl_dim,
        device,
    )

    for iteration in range(1, args.num_iterations + 1):
        sampled_episode_infos = []

        # Annealing the learning rate and entropy coefficient if instructed to do so
        do_anneal = args.anneal_steps > 0 and global_step < args.anneal_steps
        frac = 1 - global_step / args.anneal_steps if do_anneal else 0
        lr = (args.init_lr - args.final_lr) * frac + args.final_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        ent_coef = (args.init_ent_coef - args.final_ent_coef) * frac + args.final_ent_coef

        for step in range(args.num_steps):
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                rollout_state = reset_done_in_state(rollout_state, next_done.bool())
                if step % args.segment_length == 0:
                    segment_id = step // args.segment_length
                    segment_init_memory[segment_id] = rollout_state.memory.detach().cpu()
                    segment_init_valid_len[segment_id] = rollout_state.valid_len.detach().cpu()
                    segment_init_pos[segment_id] = rollout_state.pos.detach().cpu()

                obs[step] = next_obs
                dones[step] = next_done
                memory_window, memory_mask, memory_indices = build_memory_inputs(
                    rollout_state, args.trxl_memory_length, max_episode_steps
                )
                action, logprob, _, value, new_memory = get_action_and_value(
                    next_obs,
                    memory_window,
                    memory_mask,
                    memory_indices,
                    detach_new_memory=True,
                )
                rollout_state = append_memory_token(rollout_state, new_memory, max_episode_steps)
                # Store the action, log_prob, and value in the buffer
                actions[step], log_probs[step], values[step] = action, logprob, value

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.maximum(terminations, truncations).astype(np.float32)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done, dtype=torch.float32, device=device)

            if "_episode" in infos and "episode" in infos:
                finished = np.asarray(infos["_episode"], dtype=bool)
                for i in np.where(finished)[0]:
                    sampled_episode_infos.append({k: infos["episode"][k][i] for k in infos["episode"]})

        # Bootstrap value if not done
        with torch.no_grad():
            bootstrap_state = clone_trxl_state(rollout_state, device=device)
            bootstrap_state = reset_done_in_state(bootstrap_state, next_done.bool())
            memory_window, memory_mask, memory_indices = build_memory_inputs(
                bootstrap_state, args.trxl_memory_length, max_episode_steps
            )
            next_value = agent.get_value(
                next_obs,
                memory_window,
                memory_mask,
                memory_indices,
            )
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = torch.zeros((args.num_envs,), dtype=torch.float32, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        segment_obs = obs.reshape(args.num_segments, args.segment_length, args.num_envs, *obs.shape[2:])
        segment_dones = dones.reshape(args.num_segments, args.segment_length, args.num_envs)
        segment_actions = actions.reshape(args.num_segments, args.segment_length, args.num_envs)
        segment_old_log_probs = log_probs.reshape(args.num_segments, args.segment_length, args.num_envs)
        segment_advantages = advantages.reshape(args.num_segments, args.segment_length, args.num_envs)
        segment_returns = returns.reshape(args.num_segments, args.segment_length, args.num_envs)
        segment_old_values = values.reshape(args.num_segments, args.segment_length, args.num_envs)

        # Optimizing the policy and value network
        clipfracs = []
        old_approx_kl_values = []
        approx_kl_values = []
        pg_loss_values = []
        v_loss_values = []
        entropy_values = []
        total_loss_values = []
        early_stop = False
        for epoch in range(args.update_epochs):
            env_perm = torch.randperm(args.num_envs, device="cpu")
            for mb_start in range(0, args.num_envs, args.envs_per_minibatch):
                mb_end = mb_start + args.envs_per_minibatch
                env_ids_cpu = env_perm[mb_start:mb_end]
                env_ids = env_ids_cpu.to(device)

                for seg_id in range(args.num_segments):
                    mb_obs = segment_obs[seg_id, :, env_ids]
                    mb_dones = segment_dones[seg_id, :, env_ids]
                    mb_actions = segment_actions[seg_id, :, env_ids]
                    mb_old_log_probs = segment_old_log_probs[seg_id, :, env_ids]
                    mb_advantages = segment_advantages[seg_id, :, env_ids]
                    mb_returns = segment_returns[seg_id, :, env_ids]
                    mb_old_values = segment_old_values[seg_id, :, env_ids]
                    mb_init_state = TrXLState(
                        memory=segment_init_memory[seg_id, env_ids_cpu].to(device),
                        valid_len=segment_init_valid_len[seg_id, env_ids_cpu].to(device),
                        pos=segment_init_pos[seg_id, env_ids_cpu].to(device),
                    )

                    newlogprob, entropy, newvalue = evaluate_segment(
                        mb_obs, mb_actions, mb_dones, mb_init_state, max_episode_steps
                    )

                    flat_advantages = mb_advantages.reshape(-1)
                    if args.norm_adv:
                        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
                    flat_newlogprob = newlogprob.reshape(-1)
                    flat_old_log_probs = mb_old_log_probs.reshape(-1)
                    logratio = flat_newlogprob - flat_old_log_probs
                    ratio = torch.exp(logratio)
                    pgloss1 = -flat_advantages * ratio
                    pgloss2 = -flat_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss = torch.max(pgloss1, pgloss2).mean()

                    flat_newvalue = newvalue.reshape(-1)
                    flat_returns = mb_returns.reshape(-1)
                    flat_old_values = mb_old_values.reshape(-1)
                    v_loss_unclipped = (flat_newvalue - flat_returns) ** 2
                    if args.clip_vloss:
                        v_loss_clipped = flat_old_values + (flat_newvalue - flat_old_values).clamp(
                            min=-args.clip_coef, max=args.clip_coef
                        )
                        v_loss = torch.max(v_loss_unclipped, (v_loss_clipped - flat_returns) ** 2).mean()
                    else:
                        v_loss = v_loss_unclipped.mean()

                    entropy_loss = entropy.reshape(-1).mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                        old_approx_kl_values.append(old_approx_kl.item())
                        approx_kl_values.append(approx_kl.item())
                        pg_loss_values.append(pg_loss.item())
                        v_loss_values.append(v_loss.item())
                        entropy_values.append(entropy_loss.item())
                        total_loss_values.append(loss.item())

                    if args.target_kl is not None and approx_kl.item() > args.target_kl:
                        early_stop = True
                        break

                if early_stop:
                    break

            if early_stop:
                break

        pg_loss_mean = float(np.mean(pg_loss_values)) if pg_loss_values else 0.0
        v_loss_mean = float(np.mean(v_loss_values)) if v_loss_values else 0.0
        entropy_loss_mean = float(np.mean(entropy_values)) if entropy_values else 0.0
        loss_mean = float(np.mean(total_loss_values)) if total_loss_values else 0.0
        old_approx_kl_mean = float(np.mean(old_approx_kl_values)) if old_approx_kl_values else 0.0
        approx_kl_mean = float(np.mean(approx_kl_values)) if approx_kl_values else 0.0
        clipfrac_mean = float(np.mean(clipfracs)) if clipfracs else 0.0

        y_pred, y_true = values.reshape(-1).cpu().numpy(), returns.reshape(-1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log and monitor training statistics
        episode_infos.extend(sampled_episode_infos)
        episode_result = {}
        if len(episode_infos) > 0:
            for key in episode_infos[0].keys():
                episode_result[key + "_mean"] = np.mean([info[key] for info in episode_infos])
        episode_return_mean = episode_result.get("r_mean", float("nan"))
        episode_length_mean = episode_result.get("l_mean", float("nan"))
        value_mean = float(values.mean().item())
        advantage_mean = float(advantages.mean().item())

        print(
            "{:9} SPS={:4} return={:.2f} length={:.1f} pi_loss={:.3f} v_loss={:.3f} entropy={:.3f} value={:.3f} adv={:.3f}".format(
                iteration,
                int(global_step / (time.time() - start_time)),
                episode_return_mean,
                episode_length_mean,
                pg_loss_mean,
                v_loss_mean,
                entropy_loss_mean,
                value_mean,
                advantage_mean,
            )
        )

        if episode_result:
            for key in episode_result:
                writer.add_scalar("episode/" + key, episode_result[key], global_step)
        writer.add_scalar("episode/value_mean", value_mean, global_step)
        writer.add_scalar("episode/advantage_mean", advantage_mean, global_step)
        writer.add_scalar("charts/learning_rate", lr, global_step)
        writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss_mean, global_step)
        writer.add_scalar("losses/value_loss", v_loss_mean, global_step)
        writer.add_scalar("losses/loss", loss_mean, global_step)
        writer.add_scalar("losses/entropy", entropy_loss_mean, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl_mean, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl_mean, global_step)
        writer.add_scalar("losses/clipfrac", clipfrac_mean, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": agent.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    writer.close()
    envs.close()
