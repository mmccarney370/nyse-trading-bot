# models/policies.py
# Custom policies and PPO subclasses for auxiliary tasks
# =====================================================================
# UPGRADE #6 (Feb 20 2026) — Volatility-target prediction as input feature
# • Aux head now predicts normalized volatility (0-1)
# • Predicted volatility is fed back into latent / next observation (teacher-forcing)
# • Observation dimension is dynamic (includes vol_target column from features.py)
# • Train methods already use volatility_target from env — now fully consistent
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from config import CONFIG
from typing import Optional, Dict, Any, List, Union, Type, Callable, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution
import gymnasium as gym
import math
import logging
logger = logging.getLogger(__name__)

# === Fallback definition for LSTMStates using namedtuple (compatible with library expectations) ===
from collections import namedtuple
LSTMStates = namedtuple('LSTMStates', ['pi', 'vf'])
logger.info("Using local namedtuple fallback for LSTMStates — fully compatible with sb3_contrib RecurrentPPO")

# === Dummy Features Extractor for GTrXL (satisfies base policy without doing anything) ===
class DummyFeaturesExtractor(BaseFeaturesExtractor):
    """
    Minimal pass-through extractor — reports correct features_dim but returns raw observations.
    Used only to satisfy base ActorCriticPolicy init requirements.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int):
        super().__init__(observation_space, features_dim)
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations # Identity — no processing

class AuxMlpPolicy(ActorCriticPolicy):
    """
    Auxiliary policy for non-recurrent PPO.
    Adds a head that predicts normalized volatility (0-1 scale) — used for teacher-forcing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_head = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        aux_pred = self.aux_head(latent_pi)  # volatility target prediction (teacher-forcing)
        return actions, values, log_prob, aux_pred

class AuxRecurrentPolicy(RecurrentActorCriticPolicy):
    """
    COMPATIBLE AUX POLICY FOR RECURRENT PPO (LSTM backbone)
    """
    def __init__(self, *args, **kwargs):
        self._lstm_hidden_size = kwargs.get('lstm_hidden_size', 256)
        super().__init__(*args, **kwargs)
        self.aux_head = nn.Sequential(
            nn.Linear(self._lstm_hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, obs: torch.Tensor, lstm_states, episode_starts, deterministic: bool = False, return_aux: bool = False):
        actions, values, log_prob, new_lstm_states = super().forward(
            obs, lstm_states, episode_starts, deterministic
        )
        if new_lstm_states is None:
            batch_size = obs.shape[0]
            hidden = torch.zeros(batch_size, self._lstm_hidden_size, device=obs.device)
        else:
            hidden = new_lstm_states.pi[0][-1] # last layer
        aux_pred = self.aux_head(hidden)  # volatility target prediction (teacher-forcing)
        if return_aux:
            return actions, values, log_prob, new_lstm_states, aux_pred
        else:
            return actions, values, log_prob, new_lstm_states

class GTrXLLayer(nn.Module):
    """
    Single GTrXL layer: pre-norm (identity map reordering) + GRU-gated residuals.
    Per Parisotto et al. 2020 "Stabilizing Transformers for Reinforcement Learning".

    - Pre-norm attention with XL-style memory concatenation for K/V
    - GRU gates on both attention and FFN residual connections
    - GELU activation in feedforward network
    """
    def __init__(self, hidden_size: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.norm_ff = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )
        # GRU gates stabilize each residual connection independently
        self.gate_attn = nn.GRUCell(hidden_size, hidden_size)
        self.gate_ff = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor = None,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, S, H] current segment
        memory: [B, mem_len, H] previous segment outputs (XL memory), or None
        attn_mask: [S, mem_len+S] causal mask, or None
        Returns: [B, S, H]
        """
        B, S, H = x.shape
        # Pre-norm attention
        x_norm = self.norm_attn(x)
        if memory is not None:
            kv = torch.cat([self.norm_attn(memory), x_norm], dim=1)
        else:
            kv = x_norm
        attn_out, _ = self.attn(x_norm, kv, kv, attn_mask=attn_mask)
        # GRU-gated residual for attention
        gated = self.gate_attn(
            attn_out.reshape(-1, H), x.reshape(-1, H)
        ).reshape(B, S, H)
        # Pre-norm FFN + GRU-gated residual
        ff_out = self.ff(self.norm_ff(gated))
        return self.gate_ff(
            ff_out.reshape(-1, H), gated.reshape(-1, H)
        ).reshape(B, S, H)


class AuxGTrXLPolicy(RecurrentActorCriticPolicy):
    """
    Proper GTrXL (Gated Transformer-XL) Policy — sb3_contrib RecurrentPPO compatible.

    Key architecture features:
    1. Sequence-level attention — transformer sees full chunk context (not single tokens)
    2. XL-style segment memory — previous chunk outputs serve as K/V memory for current chunk
    3. Per-layer GRU gating — stabilizes each transformer layer independently (Parisotto 2020)
    4. Causal masking — prevents future information leakage during training
    5. Pre-norm (identity map reordering) — better gradient flow
    6. Batched evaluate_actions — ~64x speedup over step-by-step processing
    7. Sinusoidal positional encoding with proper offset tracking across XL segments
    8. Memory detachment between chunks for gradient health

    Single-step inference (rollout collection) uses memory-token attention:
    the previous summary vector is prepended as a context token so the transformer
    can attend to both current features and compressed history.

    Training (evaluate_actions) uses full sequence attention with XL memory:
    entire chunks are processed through the transformer at once, with previous
    chunk outputs serving as key/value memory for cross-chunk temporal context.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        hidden_size = CONFIG.get('GTRXL_HIDDEN_SIZE', 256)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=DummyFeaturesExtractor,
            features_extractor_kwargs={"features_dim": hidden_size},
            share_features_extractor=True,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=hidden_size,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,
            lstm_kwargs=lstm_kwargs,
        )
        self.hidden_size = hidden_size
        num_layers = CONFIG.get('GTRXL_NUM_LAYERS', 2)
        num_heads = CONFIG.get('GTRXL_NUM_HEADS', 8)
        self.num_layers = num_layers
        self.memory_length = CONFIG.get('GTRXL_MEMORY_LENGTH', 64)
        self.eval_chunk_size = CONFIG.get('GTRXL_EVAL_CHUNK_SIZE', 64)

        # GTrXL layers with per-layer GRU gating
        self.gtrxl_layers = nn.ModuleList([
            GTrXLLayer(hidden_size, num_heads, hidden_size * 2, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_size)

        # Observation projection
        obs_dim = observation_space.shape[0]
        self.obs_projection = nn.Linear(obs_dim, hidden_size)

        # Sinusoidal positional encoding (not learned — consistent across segments)
        self._build_sinusoidal_pe(hidden_size, max_len=2048)

        # Aux head for volatility prediction
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _build_sinusoidal_pe(self, d_model: int, max_len: int = 2048):
        """Build sinusoidal positional encoding buffer."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('sinusoidal_pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def _make_causal_mask(self, seq_len: int, mem_len: int, device: torch.device) -> torch.Tensor:
        """
        Causal attention mask for XL-style attention.
        Current tokens can attend to all memory tokens + causally to current tokens.
        Shape: [seq_len, mem_len + seq_len]
        """
        # Current-to-current: upper triangular = -inf (can't attend to future)
        causal = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1
        )
        if mem_len > 0:
            # All memory positions are visible to all current positions
            mem_visible = torch.zeros(seq_len, mem_len, device=device)
            return torch.cat([mem_visible, causal], dim=1)
        return causal

    def _gtrxl_forward(self, x: torch.Tensor, layer_memories: list = None,
                       pos_offset: int = 0) -> Tuple[torch.Tensor, list]:
        """
        Forward through all GTrXL layers with XL-style segment memory.

        Args:
            x: [B, S, H] — projected observations for current segment
            layer_memories: list of [B, mem_len, H] per layer, or None
            pos_offset: absolute position offset for positional encoding

        Returns:
            output: [B, S, H] — transformer output for current segment
            new_memories: list of [B, mem_len, H] per layer (detached, for next segment)
        """
        B, S, H = x.shape
        mem_len = layer_memories[0].shape[1] if layer_memories is not None else 0

        # Add sinusoidal positional encoding at correct absolute positions
        x = x + self.sinusoidal_pe[:, pos_offset:pos_offset + S, :H]

        # Causal mask (only needed when sequence length > 1)
        attn_mask = self._make_causal_mask(S, mem_len, x.device) if S > 1 else None

        new_memories = []
        h = x
        for i, layer in enumerate(self.gtrxl_layers):
            mem = layer_memories[i] if layer_memories is not None else None
            h = layer(h, memory=mem, attn_mask=attn_mask)
            # Store this layer's output as XL memory for the next segment (detached)
            new_memories.append(h[:, -self.memory_length:, :].detach())

        return self.output_norm(h), new_memories

    def _extract_memory(self, lstm_states, batch_size: int, device: torch.device) -> torch.Tensor:
        """Extract summary memory vector from LSTMStates (various formats)."""
        if lstm_states is None:
            return torch.zeros(batch_size, self.hidden_size, device=device)
        if hasattr(lstm_states, 'pi'):
            h = lstm_states.pi[0]
            if h is None:
                return torch.zeros(batch_size, self.hidden_size, device=device)
            return h.squeeze(0)
        if lstm_states[0] is None or lstm_states[0].nelement() == 0:
            return torch.zeros(batch_size, self.hidden_size, device=device)
        return lstm_states[0].squeeze(0)

    def _make_lstm_states(self, memory: torch.Tensor) -> LSTMStates:
        """Wrap summary memory into LSTMStates namedtuple for sb3 compatibility."""
        hidden_out = memory.unsqueeze(0)
        cell_out = torch.zeros_like(hidden_out)
        shared_tuple = (hidden_out, cell_out)
        return LSTMStates(pi=shared_tuple, vf=shared_tuple)

    def _single_step_forward(self, obs: torch.Tensor, lstm_states, episode_starts: torch.Tensor) -> torch.Tensor:
        """
        Shared single-step inference: project obs → memory-token attention → output.
        Used by forward(), get_distribution(), predict_values().

        The previous summary vector is prepended as a context token so the transformer
        attends to both current features and compressed history.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if episode_starts is not None and episode_starts.dim() == 0:
            episode_starts = episode_starts.unsqueeze(0)

        B = obs.shape[0]
        features = self.obs_projection(obs).unsqueeze(1)  # [B, 1, H]
        memory = self._extract_memory(lstm_states, B, obs.device)

        # Reset memory on episode boundaries
        if episode_starts is not None and episode_starts.any():
            memory = memory * (1 - episode_starts.float().unsqueeze(-1))

        # Prepend memory as context token — transformer attends to [memory, current]
        mem_token = memory.unsqueeze(1)  # [B, 1, H]
        combined = torch.cat([mem_token, features], dim=1)  # [B, 2, H]

        # Add positional encoding (positions 0 and 1)
        combined = combined + self.sinusoidal_pe[:, :2, :self.hidden_size]

        # Forward through GTrXL layers (no XL memory needed for single-step)
        for layer in self.gtrxl_layers:
            combined = layer(combined)
        combined = self.output_norm(combined)

        # Return last token (current step output)
        return combined[:, -1, :]  # [B, H]

    def forward(self, obs: torch.Tensor, lstm_states: LSTMStates, episode_starts: torch.Tensor,
                deterministic: bool = False, return_aux: bool = False):
        output = self._single_step_forward(obs, lstm_states, episode_starts)
        latent_pi, latent_vf = self.mlp_extractor(output)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        aux_pred = self.aux_head(output)
        new_lstm_states = self._make_lstm_states(output)
        if return_aux:
            return actions, values, log_prob, new_lstm_states, aux_pred
        return actions, values, log_prob, new_lstm_states

    def get_distribution(self, obs: torch.Tensor,
                         lstm_states: Union[LSTMStates, Tuple[torch.Tensor, torch.Tensor]],
                         episode_starts: torch.Tensor) -> Tuple[Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-step inference for backtest/live trading."""
        output = self._single_step_forward(obs, lstm_states, episode_starts)
        latent_pi, _ = self.mlp_extractor(output)
        distribution = self._get_action_dist_from_latent(latent_pi)
        hidden_out = output.unsqueeze(0)
        cell_out = torch.zeros_like(hidden_out)
        return distribution, (hidden_out, cell_out)

    def predict_values(self, obs: torch.Tensor,
                       lstm_states: Tuple[torch.Tensor, torch.Tensor],
                       episode_starts: torch.Tensor) -> torch.Tensor:
        """Value prediction for GAE computation."""
        output = self._single_step_forward(obs, lstm_states, episode_starts)
        _, latent_vf = self.mlp_extractor(output)
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         lstm_states: LSTMStates,
                         episode_starts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched evaluate_actions with proper XL-style segment memory.

        Instead of processing 2048 steps one-at-a-time through the transformer (old approach),
        this processes full chunks (default 64 steps) in a single transformer forward pass.
        Each chunk attends causally within itself AND to the previous chunk's outputs (XL memory).
        Memory is detached between chunks to prevent gradient explosion.

        ~64x speedup over step-by-step processing for the transformer computation.
        """
        device = obs.device
        N = obs.shape[0]

        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        episode_starts = episode_starts.bool()

        # Build segment boundaries from episode starts
        start_indices = torch.where(episode_starts)[0]
        if start_indices.numel() == 0 or start_indices[0] != 0:
            start_indices = torch.cat([torch.tensor([0], device=device), start_indices])
        end_indices = torch.cat([start_indices[1:], torch.tensor([N], device=device)])

        all_values = []
        all_log_probs = []
        all_entropies = []

        for seg_start, seg_end in zip(start_indices, end_indices):
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue

            seg_obs = obs[seg_start:seg_end]
            seg_actions = actions[seg_start:seg_end]

            # Project all observations for this episode segment at once
            features = self.obs_projection(seg_obs).unsqueeze(0)  # [1, seg_len, H]

            # Initialize per-layer XL memories (none for first chunk)
            layer_memories = None
            pos_offset = 0

            # Process in chunks with XL memory between chunks
            for c_start in range(0, seg_len, self.eval_chunk_size):
                c_end = min(c_start + self.eval_chunk_size, seg_len)
                chunk = features[:, c_start:c_end, :]  # [1, chunk_len, H]

                # Forward through full GTrXL stack with XL memory
                chunk_out, layer_memories = self._gtrxl_forward(
                    chunk, layer_memories, pos_offset
                )
                # chunk_out: [1, chunk_len, H] — has gradients for current chunk

                pos_offset += (c_end - c_start)

                # Compute policy outputs for all steps in chunk at once (batched)
                chunk_hidden = chunk_out.squeeze(0)  # [chunk_len, H]
                chunk_acts = seg_actions[c_start:c_end]

                latent_pi, latent_vf = self.mlp_extractor(chunk_hidden)
                dist = self._get_action_dist_from_latent(latent_pi)

                all_values.append(self.value_net(latent_vf).flatten())
                all_log_probs.append(dist.log_prob(chunk_acts))
                all_entropies.append(dist.entropy())

        if not all_values:
            empty = torch.tensor([], device=device)
            return empty, empty, torch.tensor(0.0, device=device)

        return torch.cat(all_values), torch.cat(all_log_probs), torch.cat(all_entropies)

class CustomRecurrentPPO(RecurrentPPO):
    def train(self) -> None:
        super().train()
        if not CONFIG.get('PPO_AUX_TASK', True):
            return
        aux_weight = CONFIG.get('PPO_AUX_LOSS_WEIGHT', 0.3)
        rollout_buffer = self.rollout_buffer
        if len(getattr(rollout_buffer, 'infos', [])) == 0:
            return
        device = self.device
        targets = []
        obs_list = []
        episode_starts_list = []
        lstm_states_list = []
        for i in range(rollout_buffer.buffer_size):
            info_dict = getattr(rollout_buffer, 'infos', [{}])[i] if hasattr(rollout_buffer, 'infos') else {}
            if 'volatility_target' not in info_dict:
                continue
            target = info_dict['volatility_target']
            targets.append(target)
            obs_list.append(obs_as_tensor(rollout_buffer.observations[i], device))
            episode_starts_list.append(torch.tensor([rollout_buffer.episode_starts[i]], device=device, dtype=torch.float32))
            # Reconstruct LSTMStates from buffer (shared)
            lstm_obj = rollout_buffer.lstm_states[i]
            hidden = torch.from_numpy(lstm_obj.pi[0]).to(device)
            cell = torch.from_numpy(lstm_obj.pi[1]).to(device)
            shared_tuple = (hidden, cell)
            lstm_states_list.append(LSTMStates(pi=shared_tuple, vf=shared_tuple))
        if len(obs_list) == 0:
            return
        # Forward pass WITH gradients to allow aux loss to backpropagate
        aux_preds = []
        for obs_t, ep_start, lstm_st in zip(obs_list, episode_starts_list, lstm_states_list):
            _, _, _, _, aux_pred = self.policy(obs_t, lstm_st, ep_start, deterministic=False, return_aux=True)
            aux_preds.append(aux_pred.squeeze(-1))
        aux_preds = torch.cat(aux_preds)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
        aux_loss = F.mse_loss(aux_preds, targets_t)
        self.policy.optimizer.zero_grad(set_to_none=True)
        (aux_weight * aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        logger.debug(f"Auxiliary volatility MSE loss: {aux_loss.item():.6f} (weighted {aux_weight})")

class CustomPPO(PPO):
    def train(self) -> None:
        super().train()
        if not CONFIG.get('PPO_AUX_TASK', True):
            return
        aux_weight = CONFIG.get('PPO_AUX_LOSS_WEIGHT', 0.3)
        rollout_buffer = self.rollout_buffer
        if len(getattr(rollout_buffer, 'infos', [])) == 0:
            return
        device = self.device
        targets = []
        obs_list = []
        for i in range(rollout_buffer.buffer_size):
            info_dict = rollout_buffer.infos[i]
            if 'volatility_target' not in info_dict:
                continue
            target = info_dict['volatility_target']
            targets.append(target)
            obs_list.append(obs_as_tensor(rollout_buffer.observations[i], device))
        if len(obs_list) == 0:
            return
        # Forward pass WITH gradients to allow aux loss to backpropagate
        aux_preds = []
        for obs_t in obs_list:
            _, _, _, aux_pred = self.policy.forward(obs_t, deterministic=False)
            aux_preds.append(aux_pred.squeeze(-1))
        aux_preds = torch.cat(aux_preds)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
        aux_loss = F.mse_loss(aux_preds, targets_t)
        self.policy.optimizer.zero_grad(set_to_none=True)
        (aux_weight * aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        logger.debug(f"Auxiliary volatility MSE loss: {aux_loss.item():.6f} (weighted {aux_weight})")
