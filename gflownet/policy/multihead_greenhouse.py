from typing import List, Optional

import torch
from torch import nn
from omegaconf import OmegaConf

from gflownet.policy.base import Policy


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_layers: int,
    activation: nn.Module,
    dtype,
):
    dims = [input_dim] + [hidden_dim] * max(n_layers, 0) + [output_dim]
    layers = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(din, dout, dtype=dtype))
        if i < len(dims) - 2:
            layers.append(activation.__class__())
    return nn.Sequential(*layers)


class GreenhouseGroupHeadModel(nn.Module):
    def __init__(
        self,
        env,
        float_type,
        trunk_hidden_dim: int = 128,
        trunk_n_layers: int = 2,
        head_hidden_dim: Optional[int] = None,
        head_n_layers: int = 1,
        invalid_logit: float = -1e9,
        shared_trunk: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.state_dim = int(env.policy_input_dim)
        self.output_dim = int(len(env.fixed_policy_output))
        self.depth_dim = int(env.depth_dim)
        self.n_groups = int(env.n_groups)
        self.n_operations = int(env.n_operations)
        self.invalid_logit = float(invalid_logit)

        self.depth_offset = 1
        self.depth_slice = slice(self.depth_offset, self.depth_offset + self.depth_dim)

        self.group_action_ids: List[List[int]] = [
            list(env.slot_action_ids[g]) for g in range(self.n_groups)
        ]

        head_hidden_dim = trunk_hidden_dim if head_hidden_dim is None else int(head_hidden_dim)

        if shared_trunk is None:
            self.trunk = _make_mlp(
                input_dim=self.state_dim,
                output_dim=trunk_hidden_dim,
                hidden_dim=trunk_hidden_dim,
                n_layers=max(int(trunk_n_layers) - 1, 0),
                activation=nn.LeakyReLU(),
                dtype=float_type,
            )
        else:
            self.trunk = shared_trunk

        self.heads = nn.ModuleList(
            [
                _make_mlp(
                    input_dim=trunk_hidden_dim,
                    output_dim=len(action_ids),
                    hidden_dim=head_hidden_dim,
                    n_layers=max(int(head_n_layers), 1),
                    activation=nn.LeakyReLU(),
                    dtype=float_type,
                )
                for action_ids in self.group_action_ids
            ]
        )

    def _infer_depths(self, states: torch.Tensor) -> torch.Tensor:
        depth_feats = states[:, self.depth_slice]
        return torch.argmax(depth_feats, dim=1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.trunk(states)
        logits = torch.full(
            (states.shape[0], self.output_dim),
            self.invalid_logit,
            dtype=x.dtype,
            device=x.device,
        )

        depths = self._infer_depths(states)
        active = depths < self.n_operations
        if not torch.any(active):
            return logits

        active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        active_depths = depths[active_idx]
        active_group_ids = torch.remainder(active_depths, self.n_groups)

        for group_id in range(self.n_groups):
            row_mask = active_group_ids == group_id
            if not torch.any(row_mask):
                continue

            rows = active_idx[row_mask]
            local_logits = self.heads[group_id](x[rows])
            action_ids = self.group_action_ids[group_id]
            action_ids_t = torch.tensor(action_ids, device=logits.device, dtype=torch.long)
            logits[rows[:, None], action_ids_t] = local_logits

        return logits


class MultiheadGreenhousePolicy(Policy):
    """
    Shared trunk + one local head per greenhouse group.

    Important compatibility note:
    this class must tolerate config=None because the repository instantiates the
    backward policy with the same target even when backward: null in the YAML.
    """

    def __init__(self, config, env, device, float_precision, base=None):
        self.env = env
        super().__init__(config, env, device, float_precision, base=base)

    def parse_config(self, config):
        # Let the base class set the standard fields and handle config=None by
        # defaulting to a uniform policy.
        super().parse_config(config)

        if config is None:
            config = OmegaConf.create()
            config.type = "uniform"

        self.head_n_hid = config.get("head_n_hid", self.n_hid)
        self.head_n_layers = config.get("head_n_layers", 1)
        self.invalid_logit = config.get("invalid_logit", -1e9)

    def instantiate(self):
        if self.type == "fixed":
            self.model = self.fixed_distribution
            self.is_model = False
        elif self.type == "uniform":
            self.model = self.uniform_distribution
            self.is_model = False
        elif self.type in {"mlp_multihead", "multihead_greenhouse"}:
            shared_trunk = None
            if (
                self.shared_weights
                and self.base is not None
                and hasattr(self.base, "model")
                and hasattr(self.base.model, "trunk")
            ):
                shared_trunk = self.base.model.trunk

            self.model = GreenhouseGroupHeadModel(
                env=self.env,
                float_type=self.float,
                trunk_hidden_dim=self.n_hid,
                trunk_n_layers=self.n_layers,
                head_hidden_dim=self.head_n_hid,
                head_n_layers=self.head_n_layers,
                invalid_logit=self.invalid_logit,
                shared_trunk=shared_trunk,
            ).to(self.device)
            self.is_model = True
        else:
            raise ValueError(f"Unsupported policy type: {self.type}")
