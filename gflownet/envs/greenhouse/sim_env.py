from typing import List, Union, Optional, Tuple

import numpy as np
import torch
from torch import float32
from torchtyping import TensorType

from .fmu_env import FMUEnv
from gflownet.utils.common import tfloat
from gflownet.envs.greenhouse.constants_unique_actions_vanthoor import (
    GROUPS,
    BASELINE_PARAMETERS,
    GROUP_ORDER,
    PERTURBATION_SCHEME,
    PARAMETER_BOUNDS,
    INITIAL_CONDITIONS,
)


class CropSimEnv(FMUEnv):
    """
    Clean greenhouse environment with:
    - unique action ids per group
    - Vanthoor/baseline start point for perturbations
    - correct self.n_actions update in forward step
    - lightweight policy representation

    Chosen states2policy:
    - step fraction scalar
    - depth one-hot
    - slot-local action history
    - normalized parameter vector

    Rationale:
    - the next group is already a deterministic function of depth
    - so next-group one-hot and valid-action multi-hot are redundant
    - this keeps the policy input minimal while still fully informative
    """

    def __init__(
        self,
        n_cycles=1,
        step_fraction=0.2,
        decay_factor=0.5,
        fmu_path="fmu/FMU/tomato.fmu",
        device="CUDA",
        precomputed=False,
        **kwargs,
    ):
        self.source = [()]
        self.eos = -1

        self.group2id = {g: i for i, g in enumerate(GROUP_ORDER)}
        self.id2group = {i: g for g, i in self.group2id.items()}
        self.pert2id = {
            p: i
            for i, p in enumerate(
                sorted({p for group in PERTURBATION_SCHEME.values() for p in group.keys()})
            )
        }
        self.id2pert = {i: p for p, i in self.pert2id.items()}
        self.action_space = self._build_action_space()
        self.action2idx = {a: i for i, a in enumerate(self.action_space)}

        self.n_params = len(BASELINE_PARAMETERS)
        self.n_groups = len(GROUP_ORDER)
        self.n_cycles = n_cycles
        self.step_fraction = step_fraction
        self.decay_factor = decay_factor
        self.precomputed = precomputed

        self.n_operations = self.n_groups * self.n_cycles
        self.slot_group_ids = [i % self.n_groups for i in range(self.n_operations)]

        self.slot_action_ids = []
        self.slot_action_pos = []
        for group_id in self.slot_group_ids:
            action_ids = [
                self.pert2id[p]
                for p in PERTURBATION_SCHEME[self.id2group[group_id]].keys()
            ]
            self.slot_action_ids.append(action_ids)
            self.slot_action_pos.append({a: j for j, a in enumerate(action_ids)})

        # Lightweight policy input:
        # [step_fraction] + [depth one-hot] + [slot-local history] + [normalized params]
        self.depth_dim = self.n_operations + 1
        self.history_dim = sum(len(aids) + 1 for aids in self.slot_action_ids)
        self.policy_input_dim = 1 + self.depth_dim + self.history_dim + self.n_params

        super().__init__(fmu_path=fmu_path, device=device, **kwargs)

    def _apply_perturbation(self, step_fraction, group_name, perturb_name, values=None):
        if values is None:
            values = {}

        params = GROUPS[group_name]
        for p in params:
            lo, hi = PARAMETER_BOUNDS[p]
            default_val = INITIAL_CONDITIONS.get(p, BASELINE_PARAMETERS[p])
            val = values.get(p, default_val)
            direction = PERTURBATION_SCHEME[group_name][perturb_name][p]
            val = np.clip(val + direction * step_fraction * (hi - lo), lo, hi)
            values[p] = val
        return values

    def _build_parameter_set(self, group_id, perturb_id, step_fraction=None, values=None):
        if step_fraction is None:
            step_fraction = self.step_fraction
        group = self.id2group[group_id]
        perturb = self.id2pert[perturb_id]
        return self._apply_perturbation(step_fraction, group, perturb, values)

    def _build_config(self, states):
        combined_params = {}
        for (cycle, group_id, perturb_id) in states[1:]:
            step_fraction = self._get_step_fraction(cycle)
            combined_params.update(
                self._build_parameter_set(
                    group_id,
                    perturb_id,
                    step_fraction=step_fraction,
                    values=combined_params,
                )
            )
        return combined_params

    def _build_action_space(self):
        return list(self.id2pert.keys())

    def get_action_space(self):
        return self.action_space

    def _total_decisions(self):
        return self.n_groups * self.n_cycles

    def _n_decisions_made(self, state=None):
        state = self._get_state(state)
        return max(0, len(state) - 1)

    def _next_position(self, state=None):
        state = self._get_state(state)
        k = self._n_decisions_made(state)
        cycle = 1 + (k // self.n_groups)
        group_id = k % self.n_groups
        return cycle, group_id

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        # Normalize action to a plain int to avoid tuple/tensor surprises.
        if isinstance(action, (tuple, list)):
            if len(action) != 1:
                raise ValueError(f"Expected scalar or length-1 action, got {action}")
            action = int(action[0])
        elif torch.is_tensor(action):
            action = int(action.item())
        else:
            action = int(action)

        do_step, state, action = self._pre_step(
            action, backward=False, skip_mask_check=skip_mask_check
        )
        if not do_step:
            return state, action, False

        n_done = self._n_decisions_made(state)
        n_total = self._total_decisions()
        if n_done >= n_total:
            return state, action, False

        cycle, group_id = self._next_position(state)
        new_state = state + [(cycle, group_id, action)]
        self.state = new_state
        self.n_actions += 1
        self.done = self._n_decisions_made(new_state) >= n_total
        return new_state, action, True

    def state2action_key(self, state):
        parts = []
        for group_idx in range(1, len(state)):
            _, _, perturb_id = state[group_idx]
            parts.append(self.id2pert[perturb_id])
        return "|".join(parts)

    def states2policy(self, states: Union[List, TensorType["batch", "state_dim"]]):
        out = []

        for state in states:
            state = self._get_state(state)
            decisions = state[1:] if state != [()] else []
            depth = len(decisions)
            vec = [0.0] * self.policy_input_dim

            cursor = 0

            # 1) step fraction
            vec[cursor] = float(self.step_fraction)
            cursor += 1

            # 2) depth one-hot
            vec[cursor + depth] = 1.0
            cursor += self.depth_dim

            # 3) slot-local history
            for slot_idx in range(self.n_operations):
                valid_action_ids = self.slot_action_ids[slot_idx]
                local_pos = self.slot_action_pos[slot_idx]
                block_size = len(valid_action_ids) + 1
                block = [0.0] * block_size

                if slot_idx < depth:
                    chosen_action = decisions[slot_idx][2]
                    if chosen_action in local_pos:
                        block[local_pos[chosen_action]] = 1.0
                    else:
                        block[-1] = 1.0
                else:
                    block[-1] = 1.0

                vec[cursor : cursor + block_size] = block
                cursor += block_size

            # 4) normalized parameters
            param_set = self.build_config(state, normalize=True)
            vec[cursor : cursor + self.n_params] = param_set
            out.append(vec)

        return tfloat(out, float_type=float32, device=self.device)

    def states2proxy(self, states):
        out = []
        for batch in states:
            if self.precomputed:
                out.append(self.state2action_key(batch))
            else:
                out.append(self.build_config(batch))
        return out

    def build_config(self, state, normalize=False):
        config = self._build_config(state)
        parameters = [0.0] * len(BASELINE_PARAMETERS.keys())
        for i, k in enumerate(sorted(BASELINE_PARAMETERS)):
            parameters[i] = config.get(k, INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k]))

        if normalize:
            for i, k in enumerate(sorted(BASELINE_PARAMETERS)):
                lo, hi = PARAMETER_BOUNDS.get(k, (0, 0))
                if lo == hi:
                    parameters[i] = 0.5
                else:
                    parameters[i] = (parameters[i] - lo) / (hi - lo)
        return parameters

    def _get_step_fraction(self, cycle=1):
        return self.step_fraction * self.decay_factor ** (cycle - 1)

    def _get_max_trajectory_length(self):
        return self._total_decisions()

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        state = self._get_state(state)
        if done is None:
            done = self._get_done(done)

        n_done = self._n_decisions_made(state)
        n_total = self._total_decisions()
        if done or n_done >= n_total:
            return [True] * len(self.action_space)

        _, next_group_id = self._next_position(state)
        valid_action_values = {
            self.pert2id[p]
            for p in PERTURBATION_SCHEME[self.id2group[next_group_id]].keys()
        }
        return [a not in valid_action_values for a in self.get_action_space()]

    def get_mask_invalid_actions_backward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a: Optional[List] = None,
    ) -> List:
        if state is None:
            state = self._get_state(state)
        if done is None:
            done = self._get_done(done)

        if state == self.source:
            return [True] * len(self.get_action_space())

        _, _, pert_id = state[-1]
        return [a != pert_id for a in self.get_action_space()]

    def get_parents(self, state=None, done=None, action=None):
        if state is None:
            state = self._get_state(state)
        if done is None:
            done = self._get_done(done)

        if state == [()] or len(state) <= 1:
            return [], []

        parent_state = state[:-1]
        last_pert = state[-1][2]
        return [parent_state], [last_pert]

    def actions2indices(self, actions):
        if torch.is_tensor(actions):
            return actions.long()
        return torch.tensor(actions, dtype=torch.long, device=self.device)
