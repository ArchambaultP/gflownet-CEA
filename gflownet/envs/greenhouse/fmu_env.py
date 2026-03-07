from gflownet.envs.base import GFlowNetEnv
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple, Dict
from copy import deepcopy
from torchtyping import TensorType
from gflownet.utils.common import tlong, tfloat
from torch import float32
from itertools import pairwise

@dataclass
class Condition:
    duration: int
    light_intensity: int
    target_temp: int
    target_rh: int = 70

@dataclass
class Profile:
    conditions: List[Condition]

@dataclass
class Crop:
    c_leaf: float
    c_fruit: float
    TS_flower: float
    N_fruit: float
    N_leaf: float

    def list(self):
        return [self.N_fruit, self.c_fruit]

@dataclass
class ActionStep:
    temperature: 1
    light_intensity: 0.1



class FMUEnv(GFlowNetEnv):

    def __init__(self, init_profile: Profile = None, fmu_path = 'fmu/FMU/growth.fmu',device="CUDA", **kwargs):        
        super().__init__(device=device, **kwargs)


    # TODO:
    def get_action_space(self):
        """
        Constructs list with all possible actions

        Action space:
            [-1] = eos
            [0,1] = inc/dec T + 1
            [2,3] = inc/dec light_intensity + 10
            [4] = no-op
        """
        
        pass
    
    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets the state and done of an environment. Environments that cannot be "done"
        at all states (intermediate states are not fully constructed objects) should
        overwrite this method and check for validity.
        """
        self.state = deepcopy(state)
        self.done = done
        return self
    
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
        pass

    def _grow_step(
        self, 
        profile: Profile,
        step : bool = True,
        state: List[float] = None
    ) -> Crop:

        schedule = [] + [profile] * self.growth_step
        formatted_input = self._schedule_to_trace(schedule, ts=3600 * 6)
        env_dict = self._extract_cum_metrics(formatted_input, self.growth_step)

        if state is not None: #keep track of cumulative metrics
            _, _, _, prev_env_dict = state[-1]
            env_dict = {k: env_dict[k] + prev_env_dict[k] for k in env_dict}

        res = self.growth_model.simulate(formatted_input, step=step)
        crop = self._output_to_crop(res)
        
        return crop, env_dict
    


    def _get_new_profile(self, action: int, profile: Profile):
        new_profile, valid  = self._apply_action_to_profile(action, deepcopy(profile))
        return new_profile, valid

    
    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:

        pass
    
    def states2proxy(self, states):
        """
        Returns a list of states to proxy:
        
        [t, N_fruit, C_fruit, Cum_DLI, Cum_T]
        """
        
        pass
    
    
    def get_mask_invalid_actions_forward(self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.
        """

        pass
    
    # TODO, customize mask
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
        pass
    
    @staticmethod
    def _schedule_to_trace(schedule, ts=3600):
        trace = []
        total_duration = 0
        for (day,p) in enumerate(schedule):
            conditions = p.conditions
            avg_temp = 0
            for cond in conditions:
                    avg_temp += cond.target_temp
            avg_temp = avg_temp / len(conditions)

            mod = 0
            for cond in conditions:
                d = {
                    'CO2_Air':600,
                    'PAR': 400 * cond.light_intensity, # should depend on cond.light_intensity. 400 is a magic number to represent possible maximal PPFD
                    'TCan': cond.target_temp,
                    'TCan24': avg_temp,
                    'TSoil24': avg_temp - 2,
                }
                if mod > 0:
                    trace += [(trace[-1][0] + mod, d)]
                    total_duration += ts - mod

                for t in range(int(total_duration / ts), int((cond.duration + total_duration) / ts)):
                    trace += [(t * ts, d)]
                total_duration = int((cond.duration + total_duration) / ts) * ts

                mod = cond.duration % ts
                if mod > 0 and total_duration < 86400: # there is an overlap between two daily temperature conditions
                    trace += [(total_duration, d)]
                    total_duration += mod
        return trace
    
    @staticmethod
    def _extract_cum_metrics(trace, growth_step):
        
        t_final = 86400.0 * growth_step
        dli_umol = 0.0
        T_cumul = 0.0
        for (t1, c1), (t2, _) in pairwise(trace):
            dt = t2-t1
            GDD = c1["TCan"]
            PPFD = c1["PAR"] #mislabeled, really is ppfd
            dli_umol += PPFD * dt
            T_cumul += GDD * dt

        (tf, cf) = trace[-1]
        dt = t_final - tf
        GDD = cf["TCan"]
        PPFD = cf["PAR"] #mislabeled, really is ppfd
        dli_umol += PPFD * dt
        T_cumul += GDD * dt

        dli_umol = dli_umol / 1e6 #divide by 1 million to convert from micromol to mol
        T_cumul = T_cumul / (3600.0 * 24.0) # convert to average daily temp

        return {"DLI":dli_umol, "GDD":T_cumul}
    
    @staticmethod
    def _output_to_crop(sim_result):

        crop = Crop(
            c_leaf=sim_result['C_leaves_cumm'],
            c_fruit=sim_result['C_fruits_cumm'],
            TS_flower=sim_result['TSFlower'],
            N_fruit=sim_result['N_fruits'],
            N_leaf=sim_result['N_leaves']
        )

        return crop