from gflownet.envs.base import GFlowNetEnv
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple
from copy import deepcopy
from models.plant import GrowthController
from torchtyping import TensorType
from gflownet.utils.common import tlong, tfloat
from torch import float32

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
        return [self.N_leaf, self.N_fruit]

@dataclass
class ActionStep:
    temperature: 1
    light_intensity: 10

# @dataclass 
# class EnvState:
#     profile: Profile
#     crop: Crop

class CropEnv(GFlowNetEnv):

    def __init__(self, init_profile: Profile = None, fmu_path = 'FMU/growth.fmu', growth_step=1, growth_period=2,device="CUDA", **kwargs):

        assert(growth_step <= growth_period and growth_period % growth_step == 0)
        self.action_step = ActionStep(1,10)
        self.n_actions = 0
        self.state=None
        self.growth_step = growth_step # time step between states (in days)
        self.growth_period = growth_period # Number of simulated growths. 
        self.growth_model = GrowthController(fmu_path, 
                                             start_time=0, # inital simulation time (in seconds). should not change
                                             stop_time=86400.0 * growth_step, # Final simulation time (in seconds).
                                             step_size=120.0, #numerical solver step size (in seconds)
                                             logger=None)
        self.eos = -1

        if init_profile is None:
            dayCondition = Condition(*[16 * 3600, 0.15, 13, 65])
            nightCondition = Condition(*[8 * 3600, 0, 10, 70])
            self.init_profile = Profile([dayCondition, nightCondition])
            self.current_profile = Profile([dayCondition, nightCondition])
        else:
            self.init_profile = init_profile
            self.current_profile = deepcopy(init_profile)

        # define source
        self.source = self._grow_step(self.init_profile, step=True).list()

        super().__init__(device=device, **kwargs)

        


    # TODO:
    def get_action_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)

        Action space:
            [-1] = eos
            [0,1] = inc/dec T + 1
            [2,3] = inc/dec light_intensity + 10
        """
        
        return [0,1,2,3,-1]
    
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
        do_step, self.state, action = self._pre_step(action, skip_mask_check)
        
        # if action invalid
        if not do_step:
            return self.state, action, False

        print(action)

        # if action is eos (TODO: Implement eos properly)
        if action == self.eos:
            self.done = True
            return self.state, action, True
        
        else: # execute action
            new_profile, valid = self._get_new_profile(action, self.current_profile)

            # breakpoint()

            if valid:
                crop_next = self._grow_step(new_profile)
                # print(crop_next)
                self.state = crop_next.list()
                self.n_actions += 1

        return self.state, action, valid  


    def _grow_step(
        self, 
        profile: Profile,
        step : bool = True,
        state: List[float] = None
    ) -> Crop:

        # TODO: implement state replay        
        if state is None:
            state = self._get_state(None)
        env_state = deepcopy(state)

        schedule = [] + [profile]

        formatted_input = CropEnv._schedule_to_trace(schedule, ts=3600 * 6)
        res = self.growth_model.simulate(formatted_input, step=step)
        crop = CropEnv._output_to_crop(res)
        
        return crop


    def _get_new_profile(self, action: int, profile: Profile):
        new_profile, valid  = self._apply_action_to_profile(action, deepcopy(profile))
        return new_profile, valid

    def set_state(
        self, crop_state: Crop, done: Optional[bool] = False
    ):

        return super().set_state(crop_state.list(), done)

    def _apply_action_to_profile(self, action: int, profile: Profile):

        sign = (-1)**(action % 2 == 1) #evens are positive, odds negative

        check_valid = True
        valid = True
        
        for cond in profile.conditions:
            if action in [0,1]:
                cond.target_temp += self.action_step.temperature * sign
                if check_valid:
                    valid = 8 <= cond.target_temp <= 30

            elif action in [2,3]:
                cond.light_intensity += self.action_step.light_intensity * sign
                if check_valid:
                    valid = 0 <= cond.light_intensity <= 100
            
            check_valid = valid
        
        return profile, valid
    
    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        states = tfloat(states, device=self.device, float_type=float32)
        return states
    
    # TODO, see base code for doc
    def _get_max_trajectory_length(self):
        return 14*7
    
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

        eos = self.n_actions == self.growth_period 
        if eos:
            mask = [True for _ in range(self.action_space_dim)]
            mask[self.action_space.index(self.eos)] = False
            return mask

        mask = [False for _ in range(self.action_space_dim)]
        for cond in self.current_profile.conditions:
            if cond.target_temp >= 30: # limit max temp to 30 deg C
                mask[0] = True 
            elif cond.target_temp <= 8: # limit min temp to 8 deg C
                mask[1] = True
            
            if cond.light_intensity >= 100: # limit max light intensity to 100%
                mask[2] = True
            elif cond.light_intensity <= 0: # limit min light itensity to 0%
                mask[3] = True

        mask[self.action_space.index(self.eos)] = True # eos is always invalid, except at final step

        return mask
    
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

        state = self._get_state(state)
        done = self._get_done(done)
        if parents_a is None:
            _, parents_a = self.get_parents(state, done)
        mask = [True for _ in range(self.action_space_dim)]
        for pa in parents_a:
            # breakpoint()
            mask[self.action_space.index(pa[0])] = False
        return mask
    
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
                    'PAR': 200, # should depend on cond.light_intensity
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
    def _output_to_crop(sim_result):

        crop = Crop(
            c_leaf=sim_result['C_leaves_cumm'],
            c_fruit=sim_result['C_fruits_cumm'],
            TS_flower=sim_result['TSFlower'],
            N_fruit=sim_result['N_fruits'],
            N_leaf=sim_result['N_leaves']
        )

        return crop