from gflownet.envs.base import GFlowNetEnv
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple
from copy import deepcopy
from models.plant import GrowthController
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

class CropEnv(GFlowNetEnv):

    def __init__(self, init_profile: Profile = None, fmu_path = 'FMU/growth.fmu', growth_step=1, growth_period=2,device="CUDA", **kwargs):

        self.action_step = ActionStep(1, 0.1)
        self.n_actions = 0
        self.state=None
        self.growth_step = growth_step # time step between states (in days)
        self.growth_period = growth_period # Number of simulated growth iterations
        self.growth_model = GrowthController(fmu_path, 
                                             start_time=0, # inital simulation time (in seconds). should not change
                                             stop_time=86400.0 * growth_step, # Final simulation time (in seconds).
                                             step_size=120.0, #numerical solver step size (in seconds)
                                             logger=None)
        self.eos = -1
        self.t_index = 0

        if init_profile is None:
            dayCondition = Condition(*[16 * 3600, 0.70, 20, 65])
            nightCondition = Condition(*[8 * 3600, 0, 16, 70])
            self.init_profile = Profile([dayCondition, nightCondition])
            self.current_profile = Profile([dayCondition, nightCondition])
        else:
            self.init_profile = init_profile
            self.current_profile = deepcopy(init_profile)


        # define source
        # States should be (t, last_action (to reverse), crop state)
        crop, env = self._grow_step(self.init_profile, step=True)
        source_state = (self.t_index, -1, crop.list(), env)
        self.source = [source_state]
        # define source
        # self.source = self._grow_step(self.init_profile, step=True).list()

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
        
        return [0,1,2,3,4, self.eos]
    
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

        # if action is eos (TODO: verify eos implementation)
        if action == self.eos:
            self.done = True
            return self.state, action, True
        
        else: # execute action
            new_profile, valid = self._get_new_profile(action, self.current_profile)

            if valid:
                crop_next, env_next = self._grow_step(new_profile, self.state)
                crop_state = crop_next.list()
                self.t_index += 1
                self.a_last = action

                self.state = [(self.t_index, self.a_last, crop_state, env_next)]
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

        schedule = [] + [profile] * self.growth_step
        formatted_input = CropEnv._schedule_to_trace(schedule, ts=3600 * 6)
        env_dict = CropEnv._extract_cum_metrics(formatted_input, self.growth_step)

        if state is not None: #keep track of cumulative metrics
            _, _, _, prev_env_dict = state[-1]
            env_dict = {k: env_dict[k] + prev_env_dict[k] for k in env_dict}

        res = self.growth_model.simulate(formatted_input, step=step)
        crop = CropEnv._output_to_crop(res)
        
        return crop, env_dict


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
        
        #TODO: make checks more robust by using action index (view eos)
        for cond in profile.conditions:
            if action in [0,1]:
                cond.target_temp += self.action_step.temperature * sign
                if check_valid:
                    valid = 8 <= cond.target_temp <= 30

            elif action in [2,3]: # TODO: Probably hard code night time -> cond.light_intensity = 0
                cond.light_intensity += self.action_step.light_intensity * sign
                if check_valid:
                    valid = 0 <= cond.light_intensity <= 1
            
            check_valid = valid
        
        return profile, valid
    
    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:

        # extract the timesteps and crop states
        states = [[t[0] * self.growth_step, *t[2]] for row in states for t in row]
        states = tfloat(states, device=self.device, float_type=float32)
        return states
    
    def states2proxy(self, states):
        """
        Returns a list of states to proxy:
        
        [t, N_fruit, C_fruit, Cum_DLI, Cum_T]
        """
        
        # extract the timesteps and crop states
        states = [[t[0] * self.growth_step, *t[2], *t[3].values()] for row in states for t in row]
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

        #end of the growth period
        if self.n_actions == self.growth_period:
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

        if state is None:
            state = self._get_state(state)


        # getting the last action to retrace the trajectory. This is necessary since
        # trajectories must be unique and acyclical
        last_action = state[0][1]

        # set mask to be valid only from parent state that executed action. 
        mask = [True for _ in range(self.action_space_dim)]
        mask[self.action_space.index(last_action)] = False

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