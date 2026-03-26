
from typing import List, Union, Optional, Tuple
from .fmu_env import FMUEnv
from torchtyping import TensorType
from gflownet.utils.common import tfloat
from torch import float32
from gflownet.envs.greenhouse.constants import GROUPS, BASELINE_PARAMETERS, GROUP_ORDER, PERTURBATION_SCHEME, PARAMETER_BOUNDS, INITIAL_CONDITIONS
import numpy as np
import torch

class CropSimEnv(FMUEnv):

    def __init__(self, n_cycles=1, step_fraction=0.2, decay_factor=0.5, fmu_path = 'fmu/FMU/tomato.fmu',device="CUDA", precomputed=False, **kwargs):
        self.source = [()]
        self.eos = -1
        self.group2id = {g: i for i, g in enumerate(GROUP_ORDER)}
        self.id2group = {i: g for g, i in self.group2id.items()}
        self.pert2id = {p: i for i, p in enumerate(sorted({p for group in PERTURBATION_SCHEME.values() for p in group.keys()}))}
        self.id2pert = {i: p for p, i in self.pert2id.items()}
        self.action_space = self._build_action_space()
        self.action2idx = {a: i for i, a in enumerate(self.action_space)}
        self.n_params = len(BASELINE_PARAMETERS)
        self.n_groups = len(GROUP_ORDER)
        self.n_cycles = n_cycles
        self.step_fraction = step_fraction
        self.decay_factor = decay_factor
        self.precomputed = precomputed

        super().__init__(fmu_path=fmu_path,device=device, **kwargs)


     # --- Perturbation application ---
     # TODO: Change perturbation actions
    def _apply_perturbation(self, step_fraction, group_name, perturb_name, values = None):    
        """
        Takes in a group name and a perturbation name, then returns the subset
        of modified parameters belonging to the group
        
        :param group_name: Description
        :param perturb_name: Description
        """     
        if values is None:
            values = {}
        
        params = GROUPS[group_name]
        for p in params:
            lo, hi = PARAMETER_BOUNDS[p]
            val = values.get(p, (hi+lo)/2) # set default parameter
            try:
                direction = PERTURBATION_SCHEME[group_name][perturb_name][p]
            except Exception as e:
                exit(0)
                print('Key not in perturbation scheme....')
            
            val = np.clip(val + direction * step_fraction * (hi-lo), lo, hi)
            values[p] = val
        
        return values
    
    def _build_parameter_set(self, group_id, perturb_id, step_fraction=None,values = None):
        
        if step_fraction is None:
            step_fraction = self.step_fraction
        
        group = self.id2group[group_id]
        perturb = self.id2pert[perturb_id]
        return self._apply_perturbation(step_fraction, group, perturb, values)
    
    def _build_config(self, states):
        combined_params = {}
        for (cycle, group_id, perturb_id) in states[1:]:
            step_fraction = self._get_step_fraction(cycle)
            combined_params.update(self._build_parameter_set(group_id, perturb_id, step_fraction=step_fraction,values=combined_params))
        return combined_params
    
    def _build_action_space(self):
        """
        Returns all (group, perturbation) action tuples.
        Uses self.allowed_perturbations, a dict:
            { group_name: [pert1, pert2, ...], ... }
        """
        action_space = list(self.id2pert.keys())
        action_space.append(self.eos)
        return action_space
    
    def get_action_space(self):
        return self.action_space
    
    # TODO: Implement, also look at re-implementing step-backwards
    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """

        do_step, state, action = self._pre_step(action, skip_mask_check)

        # print(f"action {action}")
        # if action invalid
        if not do_step:
            return state, action, False        

        if state == [()]:
            new_state = state + [(1, 0, action)]
            self.state = new_state
            return new_state, action, True
        
        last_cycle, last_group_id, _ = state[-1]
        group_id = last_group_id + 1

        if group_id  == self.n_groups:
            last_cycle += 1
            group_id = 0

        new_state = state + [(last_cycle, group_id, action)]
        self.state = new_state

        if len(new_state[1:]) == self.n_groups * self.n_cycles: # if our object is finally constructed
            self.done=True

        # print(f"action {action}")
        return new_state, action, True
    
    def state2action_key(self, state):
        """Convert state's action history to cache key string."""
        modes_per_group = [
            list(PERTURBATION_SCHEME[group].keys())
            for group in GROUP_ORDER
        ]
        parts = []

        for group_idx, mode_list in enumerate(modes_per_group, start=1):
            _, _, perturb_id = state[group_idx]  # however your state encodes this
            
            parts.append(self.id2pert[perturb_id])
        return "|".join(parts)

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        policy vector contains:

        [0]: step number
        [1]: step_fraction @ cycle
        [2..n_operations]: perturbation ids
        [n_operations..n_params]: parameter set
        """
        out = []
        for state in states: # states is a list of batches containing actual states
            n_operations = self.n_groups * self.n_cycles
            vec = [-1.0] * (2 + n_operations + self.n_params) # adding 1 for step index, 1 for step_fraction
            step = 1
            for i, s in enumerate(state, start=1):
                if s == ():
                    continue
                cycle, _, pert_id = s
                vec[i] = pert_id
                step += 1
            vec[1] = self.step_fraction
            vec[0] = step
            param_set = self.build_config(state, normalize=True)
            vec[2+n_operations:] = param_set
            out.append(vec)
        return tfloat(out, float_type=float32, device=self.device)
    
    def states2proxy(self, states):
        """
        Returns a list of states to proxy:
        
        [t, N_fruit, C_fruit, Cum_DLI, Cum_T]
        """
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
            for i,k in enumerate(sorted(BASELINE_PARAMETERS)):
                lo, hi = PARAMETER_BOUNDS.get(k, (0, 0))
                if lo == hi:
                    parameters[i] = 0.5 #no bounds -> parameter stays fixed
                else:
                    parameters[i] = (parameters[i] - lo) / (hi-lo)
        return parameters
    
    # def state2action_key(self, state):
    #     """Convert state's action history to cache key string."""
    #     modes_per_group = [
    #         list(PERTURBATION_SCHEME[group].keys())
    #         for group in GROUP_ORDER
    #     ]
    #     parts = []
    #     for group_idx, mode_list in enumerate(modes_per_group):
    #         mode_idx = state[group_idx]  # however your state encodes this
    #         parts.append(mode_list[mode_idx])
    #     return "|".join(parts)

    def _get_step_fraction(self, cycle=1):
        return self.step_fraction * self.decay_factor ** (cycle-1)
    
    # see base code for doc
    def _get_max_trajectory_length(self):
        return len(GROUP_ORDER)*self.n_cycles +1


    def get_mask_invalid_actions_forward(self,
        state: Optional[List[Tuple[str, dict]]] = None,
        done: Optional[bool] = None,
        ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.
        """

        if state is None:
            state = self._get_state(state)
        
        if done:
            mask = [True] * self.action_space_dim
            return mask
        elif state == [()]: # state is the empty object
            last_group_id = -1
        else:
            _, last_group_id, _ = state[-1]
        next_group_id = (last_group_id + 1) % self.n_groups
        perturbations = [self.pert2id[p] for p in PERTURBATION_SCHEME[self.id2group[next_group_id]].keys()]
        mask = [False if a in perturbations else True for a in self.get_action_space()]

        return mask

    
    def get_mask_invalid_actions_backward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a: Optional[List] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the backward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.

        The base implementation below should be common to all discrete spaces as it
        relies on get_parents, which is environment-specific and must be implemented.
        Continuous environments will probably need to implement its specific version of
        this method.
        """

        if state is None:
            state = self._get_state(state)
    
        if state == self.source: # if we hit the source state
            return [True] * len(self.get_action_space())

        _, _, pert_id = state[-1]
        mask = [False if a == pert_id else True for a in self.get_action_space()]

        return mask
    
    def get_parents(self, state=None, done=None, action=None):
        state = self._get_state(state)
        done = self._get_done(done)

        # Source state → no parents
        if state == [()] or len(state) <= 1:
            return [], []

        # Parent is the state without the last step
        parent_state = state[:-1]

        # Action that led here is the perturbation ID from the last step
        last_pert = state[-1][2]

        return [parent_state], [last_pert]
    
    def actions2indices(self, actions):
        """Convert action ints to indices into the policy output."""
        if torch.is_tensor(actions):
            return actions.long()
        return torch.tensor(actions, dtype=torch.long, device=self.device)