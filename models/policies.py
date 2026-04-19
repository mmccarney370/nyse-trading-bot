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
    Minimal pass-through extractor for GTrXL policy.

    NOTE (FIX #17): Reports features_dim=hidden_size but forward() returns raw obs
    (which has obs_dim, potentially != hidden_size). This mismatch is masked because
    GTrXLPolicy overrides extract_features() and never calls this forward() directly.
    For safety, we project obs to features_dim if dimensions don't match.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[-1]
        # FIX #17: Add projection if obs_dim != features_dim to avoid silent shape mismatch
        if obs_dim != features_dim:
            self._proj = nn.Linear(obs_dim, features_dim)
        else:
            self._proj = None
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self._proj is not None:
            return self._proj(observations)
        return observations

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
        """HIGH-2 FIX: Return standard SB3 3-tuple (actions, values, log_prob).
        Aux prediction is only needed during training — store it on self._last_aux_pred
        for CustomPPO.train() to access, or compute separately in train()."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        # Store aux prediction for training access without breaking SB3 interface
        self._last_aux_pred = self.aux_head(latent_pi)
        return actions, values, log_prob

    def forward_with_aux(self, obs, deterministic=False):
        """Explicit aux-returning variant for CustomPPO.train() aux loss computation."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        aux_pred = self.aux_head(latent_pi)
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
        # Apr-19 audit: inference window size — number of recent projected
        # observations (plus current) that _single_step_forward attends over.
        # Training uses 64-step chunks with XL memory; previous inference
        # approximation used only 2 tokens (summary + current), which caused
        # a distribution shift on the value head. Matching inference's
        # attended sequence to the training chunk length closes that gap.
        self.inference_window_size = CONFIG.get('GTRXL_INFERENCE_WINDOW', 32)
        # Rolling buffer of (episode-scoped) projected features used as context
        # during single-step inference. Reset on episode_starts.
        self._inference_window = []

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

        # Parent creates lstm_actor/lstm_critic we never use.
        # We can't delete them (RecurrentPPO._setup_model accesses them after policy init).
        # Instead, freeze them and rebuild the optimizer to exclude their params.
        _lstm_param_ids = set()
        for attr_name in ('lstm_actor', 'lstm_critic'):
            mod = getattr(self, attr_name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False
                    _lstm_param_ids.add(id(p))

        # Rebuild optimizer with only the params we actually use (excludes frozen LSTMs)
        active_params = [p for p in self.parameters() if p.requires_grad and id(p) not in _lstm_param_ids]
        self.optimizer = self.optimizer_class(
            active_params,
            lr=lr_schedule(1),
            **(optimizer_kwargs or {})
        )

        # Aux head for volatility prediction
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Inference step counter for correct positional encoding during rollouts.
        # During training (evaluate_actions), proper pos_offset is computed per chunk.
        # During single-step inference, this counter tracks the absolute position
        # so PE is consistent with training (not always 0,1).
        self._inference_step: int = 0

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

        # L29 FIX: Clamp pos_offset to prevent PE buffer overflow
        max_pe_len = self.sinusoidal_pe.shape[1]
        clamped_start = min(pos_offset, max_pe_len - S) if S > 0 else 0
        clamped_start = max(clamped_start, 0)
        x = x + self.sinusoidal_pe[:, clamped_start:clamped_start + S, :H]

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
        """Extract summary memory vector [B, H] from LSTMStates (various formats).
        Must always return exactly 2D: [batch_size, hidden_size]."""
        if lstm_states is None:
            return torch.zeros(batch_size, self.hidden_size, device=device)
        h = None
        if hasattr(lstm_states, 'pi'):
            h = lstm_states.pi[0]
        elif isinstance(lstm_states, (tuple, list)) and len(lstm_states) > 0:
            h = lstm_states[0]
        if h is None or (hasattr(h, 'nelement') and h.nelement() == 0):
            return torch.zeros(batch_size, self.hidden_size, device=device)
        # Flatten to exactly 2D [B, H] regardless of input shape
        while h.dim() > 2:
            h = h.squeeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        # Ensure hidden dim matches expected size (take last hidden_size elements)
        if h.shape[-1] != self.hidden_size:
            h = h[..., :self.hidden_size]
        # Ensure batch dim matches
        if h.shape[0] != batch_size:
            h = h[-batch_size:]
        return h

    def _make_lstm_states(self, memory: torch.Tensor) -> LSTMStates:
        """Wrap summary memory into LSTMStates namedtuple for sb3 compatibility."""
        hidden_out = memory.unsqueeze(0)
        cell_out = torch.zeros_like(hidden_out)
        shared_tuple = (hidden_out, cell_out)
        return LSTMStates(pi=shared_tuple, vf=shared_tuple)

    def _single_step_forward(self, obs: torch.Tensor, lstm_states, episode_starts: torch.Tensor,
                             increment_step: bool = True) -> torch.Tensor:
        """
        Shared single-step inference: project obs → memory-token attention → output.
        Used by forward(), get_distribution(), predict_values().

        The previous summary vector is prepended as a context token so the transformer
        attends to both current features and compressed history.

        Uses self._inference_step for positional encoding offset so PE positions
        increment across steps within an episode, matching training behavior.

        Args:
            increment_step: Whether to increment _inference_step after this call.
                Only forward() should increment (once per observation). get_distribution()
                and predict_values() are called on the SAME observation during SB3 rollout
                collection, so they must NOT increment to avoid double-counting.

        Apr-19 audit: inference now maintains a rolling window of the last
        `inference_window_size` projected observations and attends over the
        full window (plus the LSTM-derived summary as token 0). This matches
        training's chunked self-attention far more closely than the previous
        2-token approximation, collapsing the train/inference distribution
        shift on the value function. The window is reset on episode_starts.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if episode_starts is not None and episode_starts.dim() == 0:
            episode_starts = episode_starts.unsqueeze(0)

        B = obs.shape[0]
        features = self.obs_projection(obs).unsqueeze(1)  # [B, 1, H]
        memory = self._extract_memory(lstm_states, B, obs.device)

        assert B == 1, (
            f"GTrXLPolicy._inference_step is a scalar — n_envs must be 1, got batch size {B}. "
            "Use RecurrentPPO with n_envs=1 for correct positional encoding."
        )
        if episode_starts is not None and episode_starts.any():
            memory = memory * (1 - episode_starts.float().unsqueeze(-1))
            if episode_starts.any():
                self._inference_step = 0
                self._inference_window = []

        # Dimensional hygiene on memory token
        if memory.dim() == 1:
            memory = memory.unsqueeze(0)
        if memory.dim() == 2:
            mem_token = memory.unsqueeze(1)
        elif memory.dim() == 3:
            mem_token = memory
        else:
            mem_token = memory.reshape(memory.shape[0], -1)[:, :self.hidden_size].unsqueeze(1)
        if features.dim() == 2:
            features = features.unsqueeze(1)

        # Build the rolling inference sequence:
        # [mem_token, ...last K-1 prior obs features, current obs features]
        win = self.inference_window_size
        # Detach prior features to prevent gradient flow through history
        # (we don't want .backward() to trace into past steps during any
        # inadvertent training-mode call on single-step fn).
        history = [h.detach() for h in self._inference_window[-max(0, win - 1):]]
        seq_tokens = [mem_token] + history + [features]
        combined = torch.cat(seq_tokens, dim=1)  # [B, L, H]
        L = combined.shape[1]

        # Add sinusoidal PE to every observation token (not the memory token).
        # Each history token gets its prior PE offset; current token gets offset=_inference_step.
        max_pe = self.sinusoidal_pe.shape[1] - 1
        if L > 1 and max_pe > 0:
            hist_count = len(history)
            current_pos = self._inference_step % (max_pe + 1)
            # Positions: history tokens are earliest..current-1; current token is current_pos
            for i in range(hist_count):
                hist_pos = (current_pos - (hist_count - i)) % (max_pe + 1)
                if hist_pos < 0:
                    hist_pos = 0
                combined[:, 1 + i:2 + i, :] = combined[:, 1 + i:2 + i, :] + \
                    self.sinusoidal_pe[:, hist_pos:hist_pos + 1, :self.hidden_size]
            # Current observation token is at index L-1
            combined[:, L - 1:L, :] = combined[:, L - 1:L, :] + \
                self.sinusoidal_pe[:, current_pos:current_pos + 1, :self.hidden_size]

        causal_mask = self._make_causal_mask(L, 0, combined.device)
        for layer in self.gtrxl_layers:
            combined = layer(combined, attn_mask=causal_mask)
        combined = self.output_norm(combined)

        if increment_step:
            self._inference_step += 1
            # Append CURRENT (post-projection, pre-attention) features to the
            # history window for future steps. Trim to the configured window.
            self._inference_window.append(features.detach())
            if len(self._inference_window) > win:
                self._inference_window = self._inference_window[-win:]

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
        """Single-step inference for backtest/live trading.
        HIGH-1 FIX: increment_step=False — SB3 calls get_distribution() and predict_values()
        on the same observation during rollout collection. Only forward() increments."""
        output = self._single_step_forward(obs, lstm_states, episode_starts, increment_step=False)
        latent_pi, _ = self.mlp_extractor(output)
        distribution = self._get_action_dist_from_latent(latent_pi)
        hidden_out = output.unsqueeze(0)
        cell_out = torch.zeros_like(hidden_out)
        return distribution, (hidden_out, cell_out)

    def predict_values(self, obs: torch.Tensor,
                       lstm_states: Tuple[torch.Tensor, torch.Tensor],
                       episode_starts: torch.Tensor) -> torch.Tensor:
        """Value prediction for GAE computation.
        HIGH-1 FIX: increment_step=False — same observation as get_distribution()."""
        output = self._single_step_forward(obs, lstm_states, episode_starts, increment_step=False)
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
            # HIGH-3 FIX: Return empty tensors for all three — SB3 expects shape [N]
            # not a scalar 0.0 for entropy. Empty [0] tensors are consistent.
            empty = torch.tensor([], device=device)
            return empty, empty, empty

        return torch.cat(all_values), torch.cat(all_log_probs), torch.cat(all_entropies)

class CustomRecurrentPPO(RecurrentPPO):
    """Aux training uses a SEPARATE optimizer for aux_head parameters only,
    so PPO's optimizer momentum/variance estimates are never corrupted."""

    def _setup_model(self) -> None:
        super()._setup_model()
        # Create a separate optimizer for aux head only (prevents PPO gradient corruption)
        if CONFIG.get('PPO_AUX_TASK', True) and hasattr(self.policy, 'aux_head'):
            aux_lr = CONFIG.get('PPO_AUX_LEARNING_RATE', 1e-4)
            self._aux_optimizer = torch.optim.Adam(self.policy.aux_head.parameters(), lr=aux_lr)
            logger.info(f"Aux optimizer created for aux_head (lr={aux_lr})")

    def set_parameters(self, load_path_or_dict, exact_match=True, device="auto"):
        """Override to handle optimizer state mismatch from checkpoints saved before
        LSTM params were excluded from the optimizer. On mismatch, skip optimizer
        state and log a warning — policy weights are still loaded correctly."""
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            from stable_baselines3.common.save_util import load_from_zip_file
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)

        # Always substitute fresh optimizer state — saved checkpoints may have different
        # param counts (e.g., old code deleted LSTM before saving, new code keeps them frozen).
        # Policy weights are architecture-compatible; only the optimizer state differs.
        opt_key = 'policy.optimizer'
        if opt_key in params and hasattr(self.policy, 'optimizer'):
            params[opt_key] = self.policy.optimizer.state_dict()
            logger.info("Substituted fresh optimizer state for checkpoint load (policy weights preserved)")

        super().set_parameters(params, exact_match=exact_match, device=device)

    def train(self) -> None:
        super().train()
        if not CONFIG.get('PPO_AUX_TASK', True):
            return
        aux_weight = CONFIG.get('PPO_AUX_LOSS_WEIGHT', 0.3)
        targets_list = getattr(self, '_aux_vol_targets', [])
        if not targets_list:
            return
        aux_optimizer = getattr(self, '_aux_optimizer', None)
        if aux_optimizer is None:
            return
        rollout_buffer = self.rollout_buffer
        device = self.device
        n_targets = min(len(targets_list), rollout_buffer.buffer_size)
        if n_targets == 0:
            return
        obs_list = []
        episode_starts_list = []
        targets = []
        for i in range(n_targets):
            targets.append(targets_list[i])
            obs_list.append(obs_as_tensor(rollout_buffer.observations[i], device))
            episode_starts_list.append(
                torch.tensor([rollout_buffer.episode_starts[i]], device=device, dtype=torch.float32)
            )
        if not obs_list:
            return
        # Aux head predicts per-bar volatility (not sequential patterns), so fresh zero
        # hidden states are acceptable here. The aux head is a simple MLP on the transformer
        # output — it doesn't need accumulated recurrent state to predict volatility.
        aux_preds = []
        zero_h = torch.zeros(1, 1, self.policy.hidden_size if hasattr(self.policy, 'hidden_size') else 256, device=device)
        zero_lstm = LSTMStates(pi=(zero_h, zero_h.clone()), vf=(zero_h.clone(), zero_h.clone()))
        for obs_t, ep_start in zip(obs_list, episode_starts_list):
            # M48 FIX: Run forward with no_grad to get the hidden state cheaply,
            # then re-apply only aux_head with gradients. Avoids wasteful backward
            # through the full transformer backbone for the aux loss.
            with torch.no_grad():
                _, _, _, new_states = self.policy(obs_t, zero_lstm, ep_start, deterministic=False, return_aux=False)
            hidden = new_states.pi[0][-1].detach()  # detached last-layer hidden
            aux_pred = self.policy.aux_head(hidden)
            aux_preds.append(aux_pred.squeeze(-1))
        aux_preds = torch.cat(aux_preds)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
        aux_loss = F.mse_loss(aux_preds, targets_t)
        # Use separate aux optimizer — does NOT touch PPO's optimizer state
        aux_optimizer.zero_grad(set_to_none=True)
        (aux_weight * aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.aux_head.parameters(), self.max_grad_norm)
        aux_optimizer.step()
        # CRIT-1 FIX: Zero stale gradients on non-aux params accumulated by aux backward().
        # PPO's optimizer.zero_grad() should handle this, but be safe against ordering bugs.
        for name, param in self.policy.named_parameters():
            if param.grad is not None and not name.startswith('aux_head'):
                param.grad = None
        logger.debug(f"Auxiliary volatility MSE loss: {aux_loss.item():.6f} (weighted {aux_weight}, n={n_targets})")
        self._aux_vol_targets = []

class CustomPPO(PPO):
    """Aux training uses a SEPARATE optimizer for aux_head parameters only,
    so PPO's optimizer momentum/variance estimates are never corrupted."""

    def _setup_model(self) -> None:
        super()._setup_model()
        if CONFIG.get('PPO_AUX_TASK', True) and hasattr(self.policy, 'aux_head'):
            aux_lr = CONFIG.get('PPO_AUX_LEARNING_RATE', 1e-4)
            self._aux_optimizer = torch.optim.Adam(self.policy.aux_head.parameters(), lr=aux_lr)
            logger.info(f"Aux optimizer created for aux_head (lr={aux_lr})")

    def train(self) -> None:
        super().train()
        if not CONFIG.get('PPO_AUX_TASK', True):
            return
        aux_weight = CONFIG.get('PPO_AUX_LOSS_WEIGHT', 0.3)
        targets_list = getattr(self, '_aux_vol_targets', [])
        if not targets_list:
            return
        aux_optimizer = getattr(self, '_aux_optimizer', None)
        if aux_optimizer is None:
            return
        rollout_buffer = self.rollout_buffer
        device = self.device
        n_targets = min(len(targets_list), rollout_buffer.buffer_size)
        if n_targets == 0:
            return
        obs_list = []
        targets = []
        for i in range(n_targets):
            targets.append(targets_list[i])
            obs_list.append(obs_as_tensor(rollout_buffer.observations[i], device))
        if not obs_list:
            return
        aux_preds = []
        for obs_t in obs_list:
            # HIGH-2 FIX: Use forward_with_aux() which returns 4 values
            # (forward() now returns standard SB3 3-tuple)
            _, _, _, aux_pred = self.policy.forward_with_aux(obs_t, deterministic=False)
            aux_preds.append(aux_pred.squeeze(-1))
        aux_preds = torch.cat(aux_preds)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
        aux_loss = F.mse_loss(aux_preds, targets_t)
        # Use separate aux optimizer — does NOT touch PPO's optimizer state
        aux_optimizer.zero_grad(set_to_none=True)
        (aux_weight * aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.aux_head.parameters(), self.max_grad_norm)
        aux_optimizer.step()
        # CRIT-1 FIX: Zero stale gradients on non-aux params (same fix as RecurrentCustomPPO)
        for name, param in self.policy.named_parameters():
            if param.grad is not None and not name.startswith('aux_head'):
                param.grad = None
        logger.debug(f"Auxiliary volatility MSE loss: {aux_loss.item():.6f} (weighted {aux_weight}, n={n_targets})")
        self._aux_vol_targets = []
