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

class AuxGTrXLPolicy(RecurrentActorCriticPolicy):
    """
    CUSTOM GTrXL POLICY — FULLY COMPATIBLE WITH sb3_contrib RecurrentPPO
    Uses namedtuple LSTMStates with shared pi/vf tuples
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=CONFIG.get('GTRXL_NUM_HEADS', 8),
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CONFIG.get('GTRXL_NUM_LAYERS', 4))
        self.memory_gate = nn.GRUCell(hidden_size, hidden_size)
        max_seq_len = 512
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        obs_dim = observation_space.shape[0]
        self.obs_projection = nn.Linear(obs_dim, hidden_size) if obs_dim != hidden_size else None
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.hidden_size = hidden_size

    def forward(self, obs: torch.Tensor, lstm_states: LSTMStates, episode_starts: torch.Tensor,
                deterministic: bool = False, return_aux: bool = False):
        batch_size = obs.shape[0]
        # Extract memory from pi (shared)
        if lstm_states is None or lstm_states.pi[0] is None:
            memory = torch.zeros(batch_size, self.hidden_size, device=obs.device)
        else:
            memory = lstm_states.pi[0].squeeze(0) # (1, batch, hidden) -> (batch, hidden)
        if episode_starts.any():
            memory = memory * (1 - episode_starts.float()).unsqueeze(1)
        features = obs
        if self.obs_projection is not None:
            features = self.obs_projection(features)
        if features.dim() == 2:
            features = features.unsqueeze(1)
        features = features + self.pos_encoding[:, :features.shape[1], :]
        transformer_out = self.transformer(features)
        last_out = transformer_out[:, -1, :]
        new_memory = self.memory_gate(last_out, memory)
        latent_pi, latent_vf = self.mlp_extractor(new_memory)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        aux_pred = self.aux_head(last_out)  # volatility target prediction (teacher-forcing)
        # Return shared LSTMStates
        hidden_out = new_memory.unsqueeze(0)
        cell_out = torch.zeros_like(hidden_out)
        shared_tuple = (hidden_out, cell_out)
        new_lstm_states = LSTMStates(pi=shared_tuple, vf=shared_tuple)
        if return_aux:
            return actions, values, log_prob, new_lstm_states, aux_pred
        else:
            return actions, values, log_prob, new_lstm_states

    def get_distribution(self, obs: torch.Tensor, lstm_states: Union[LSTMStates, Tuple[torch.Tensor, torch.Tensor]], episode_starts: torch.Tensor) -> Tuple[Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Custom get_distribution for single-step inference (backtest/live trading).
        Handles both full LSTMStates namedtuple and direct actor tuple (when enable_critic_lstm=False).
        Returns plain shared tuple for compatibility with base class when critic LSTM disabled.
        """
        # Ensure batched
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if episode_starts.dim() == 0:
            episode_starts = episode_starts.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        # Extract memory robustly: from namedtuple.pi[0] or direct tuple[0]
        if lstm_states is None:
            memory = torch.zeros(batch_size, self.hidden_size, device=device)
        elif hasattr(lstm_states, 'pi'):  # Full LSTMStates
            memory = lstm_states.pi[0].squeeze(0) if lstm_states.pi[0] is not None else torch.zeros(batch_size, self.hidden_size, device=device)
        else:  # Direct tuple (actor states)
            memory = lstm_states[0].squeeze(0) if lstm_states[0].nelement() > 0 else torch.zeros(batch_size, self.hidden_size, device=device)

        if episode_starts.any():
            memory = memory * (1 - episode_starts.float()).unsqueeze(1)

        features = obs
        if self.obs_projection is not None:
            features = self.obs_projection(features)
        features = features.unsqueeze(1)  # (batch, 1, hidden)
        features = features + self.pos_encoding[:, :1, :]

        transformer_out = self.transformer(features)
        last_out = transformer_out[:, -1, :]

        new_memory = self.memory_gate(last_out, memory)

        latent_pi, _ = self.mlp_extractor(new_memory)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Return plain shared tuple (compatible with base class when enable_critic_lstm=False)
        hidden_out = new_memory.unsqueeze(0)
        cell_out = torch.zeros_like(hidden_out)
        shared_tuple = (hidden_out, cell_out)

        return distribution, shared_tuple

    def predict_values(self, obs: torch.Tensor, lstm_states: Tuple[torch.Tensor, torch.Tensor], episode_starts: torch.Tensor) -> torch.Tensor:
        """
        Updated to accept the actor state tuple directly when enable_critic_lstm=False.
        Mirrors the forward pass logic for consistent value estimation (runs transformer + memory gate).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if episode_starts.dim() == 0:
            episode_starts = episode_starts.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        # Extract current memory (hidden) from the incoming actor states tuple
        if lstm_states[0] is None or lstm_states[0].nelement() == 0:
            memory = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            memory = lstm_states[0].squeeze(0)  # (1, batch, hidden) -> (batch, hidden)

        # Reset memory on episode start
        if episode_starts.any():
            memory = memory * (1 - episode_starts.float()).unsqueeze(1)

        # Process observation (identical to forward)
        features = obs
        if self.obs_projection is not None:
            features = self.obs_projection(features)

        features = features.unsqueeze(1)  # Add sequence dim for single-step

        features = features + self.pos_encoding[:, :features.shape[1], :]

        transformer_out = self.transformer(features)
        last_out = transformer_out[:, -1, :]

        # Update memory (required for consistent latent_vf, even though we don't return new states)
        new_memory = self.memory_gate(last_out, memory)

        # Value from post-gate latent (matches forward)
        _, latent_vf = self.mlp_extractor(new_memory)
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, lstm_states: LSTMStates, episode_starts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Custom evaluate_actions for training.
        Processes the flattened rollout by splitting into independent episode segments (at episode_starts=True).
        Each segment starts with zero memory. Returns per-timestep flattened values/log_probs/entropy for base class compatibility.
        """
        device = obs.device
        n_steps_total = obs.shape[0]  # Total flattened steps

        # Actions reshaping
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # Find segment boundaries (episode starts)
        episode_starts = episode_starts.bool()
        start_indices = torch.where(episode_starts)[0]
        if start_indices.numel() == 0 or start_indices[0] != 0:
            start_indices = torch.cat([torch.tensor([0], device=device), start_indices])
        end_indices = torch.cat([start_indices[1:], torch.tensor([n_steps_total], device=device)])

        values_list = []
        log_probs_list = []
        entropies_list = []  # Per-timestep for base class masking/mean

        for start, end in zip(start_indices, end_indices):
            segment_length = end - start
            if segment_length <= 0:
                continue

            segment_obs = obs[start:end]
            segment_actions = actions[start:end]

            # Fresh zero memory for each new episode
            memory = torch.zeros(1, self.hidden_size, device=device)

            for i in range(segment_length):
                current_obs = segment_obs[i:i+1]  # (1, obs_dim)

                # Single-step forward
                features = current_obs
                if self.obs_projection is not None:
                    features = self.obs_projection(features)
                features = features.unsqueeze(1)  # (1, 1, hidden)
                features = features + self.pos_encoding[:, :1, :]

                transformer_out = self.transformer(features)
                last_out = transformer_out[:, -1, :]  # (1, hidden)

                new_memory = self.memory_gate(last_out, memory)

                latent_pi, latent_vf = self.mlp_extractor(new_memory)
                dist = self._get_action_dist_from_latent(latent_pi)

                value = self.value_net(latent_vf).flatten()
                log_prob = dist.log_prob(segment_actions[i]).flatten()
                entropy = dist.entropy().flatten()  # Per-step, will be flattened later

                values_list.append(value)
                log_probs_list.append(log_prob)
                entropies_list.append(entropy)

                memory = new_memory

        if not values_list:
            # Fallback for empty rollout (impossible but safe)
            empty = torch.tensor([], device=device)
            return empty, empty, torch.tensor(0.0, device=device)

        values = torch.cat(values_list)
        log_probs = torch.cat(log_probs_list)
        entropy = torch.cat(entropies_list)  # Flattened per-timestep tensor

        return values, log_probs, entropy

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
        aux_preds = []
        targets = []
        for i in range(rollout_buffer.buffer_size):
            info_dict = getattr(rollout_buffer, 'infos', [{}])[i] if hasattr(rollout_buffer, 'infos') else {}
            if 'volatility_target' not in info_dict:
                continue
            target = info_dict['volatility_target']
            targets.append(target)
            obs_tensor = obs_as_tensor(rollout_buffer.observations[i], device)
            episode_starts = torch.tensor([rollout_buffer.episode_starts[i]], device=device, dtype=torch.float32)
            # Reconstruct LSTMStates from buffer (shared)
            lstm_obj = rollout_buffer.lstm_states[i]
            hidden = torch.from_numpy(lstm_obj.pi[0]).to(device)
            cell = torch.from_numpy(lstm_obj.pi[1]).to(device)
            shared_tuple = (hidden, cell)
            lstm_states = LSTMStates(pi=shared_tuple, vf=shared_tuple)
            with torch.no_grad():
                _, _, _, _, aux_pred = self.policy(
                    obs_tensor,
                    lstm_states,
                    episode_starts,
                    deterministic=False,
                    return_aux=True
                )
            aux_preds.append(aux_pred.squeeze(-1).item())
        if len(aux_preds) == 0:
            return
        aux_preds = torch.tensor(aux_preds, device=device)
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
        aux_loss = F.mse_loss(aux_preds, targets)
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
        if len(rollout_buffer.infos) == 0:
            return
        device = self.device
        aux_preds = []
        targets = []
        for i in range(rollout_buffer.buffer_size):
            info_dict = rollout_buffer.infos[i]
            if 'volatility_target' not in info_dict:
                continue
            target = info_dict['volatility_target']
            targets.append(target)
            obs_tensor = obs_as_tensor(rollout_buffer.observations[i], device)
            with torch.no_grad():
                _, _, _, aux_pred = self.policy.forward(obs_tensor, deterministic=False)
            aux_preds.append(aux_pred.squeeze(-1).item())
        if len(aux_preds) == 0:
            return
        aux_preds = torch.tensor(aux_preds, device=device)
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
        aux_loss = F.mse_loss(aux_preds, targets)
        self.policy.optimizer.zero_grad(set_to_none=True)
        (aux_weight * aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        logger.debug(f"Auxiliary volatility MSE loss: {aux_loss.item():.6f} (weighted {aux_weight})")
