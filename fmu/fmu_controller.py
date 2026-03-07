from fmpy import read_model_description, extract, simulate_fmu, instantiate_fmu
import numpy as np
from pathlib import Path
from abc import ABC

def fmi_call_logger(message):
    print(f"[FMI call] {message}")

class FMUController(ABC):

    def __init__(self, fmu_path, start_time, stop_time, step_size=1.0, logger=None):
        self.start_time = start_time
        self.stop_time = stop_time
        self.step_size = step_size
        self.fmu_path = Path(fmu_path)
        self.model_description = read_model_description(self.fmu_path)
        unzipdir = extract(self.fmu_path)
        self.load_vars()
        self.fmu = instantiate_fmu(
            unzipdir, 
            self.model_description, 
            logger=logger,
            fmi_call_logger=None)
        self.logger = logger
    


    def load_vars(self):
        vrs = {}
        vr_inp = {}
        vr_out = {}
        vr_param = {}

        for variable in self.model_description.modelVariables:
            vrs[variable.name] = (variable.valueReference, variable.causality)
        for key, (val, t) in vrs.items():
            if t == 'input':
                vr_inp[key] = val
            elif t == 'output':
                vr_out[key] = val
            elif t == 'parameter':
                vr_param[key] = val
        
        self.input_vars = vr_inp
        self.output_vars = vr_out
        self.param_vars = vr_param

    def set_init_cond(self, formatted_res):
        pass

    def get_variables(self):
        return self.input_vars | self.output_vars | self.param_vars

    def format_outputs(self, out):
        vals = {}
        for i, (k,v) in enumerate(self.output_vars.items()):
            vals[k] = (v, out[i])
        
        return vals

    def simulate(self, inputs, **kwargs):
        inp = self.format_input(inputs)
        res = simulate_fmu(self.fmu_path,
                            fmu_instance=self.fmu,
                            start_time=self.start_time,
                            step_size=self.step_size,
                            stop_time=self.stop_time,
                            fmi_type='CoSimulation',
                            output_interval=self.step_size,
                            input=inp,
                            **kwargs,
                            )
        
        res = tuple(res[-1])[1:] # remove time stamp from output
        formatted_res = self.format_outputs(res)
        return formatted_res

    @staticmethod
    def format_input(vals):
        # formats input time series to structured numpy data array
        # input: vals => [(0.0, {inp1:val, inp2:val}) ...]
        # output: ndarray => [(data), dtype]

        dtypes = {}
        dtypes['time'] = np.double
        inputs = []
        for t, inp in vals:
            i = [t]
            for k,v in inp.items():
                if k not in dtypes:
                    dtypes[k] = np.double # Convert types here
                i.append(v)
            inputs.append(tuple(i))
        
        return np.array(inputs, dtype=[(k,v) for k,v in dtypes.items()])


