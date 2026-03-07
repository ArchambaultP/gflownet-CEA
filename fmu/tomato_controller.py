
from gflownet.envs.greenhouse.constants import parse_output_type, BASELINE_PARAMETERS, INITIAL_CONDITIONS
import tempfile
from fmpy import read_model_description, extract
from fmpy.fmi3 import FMU3Slave
from fmu.fmu_controller import FMUController

class TomatoController(FMUController):
    """
    Docstring for TomatoController

    Functions to interact with a Tomato model FMU. This Controller assumes FMU version 3.
    """
    def __init__(self, fmu_path, start_time=0.0, stop_time=86400.0, step_size=120, logger=None, **kwargs):
        super().__init__(fmu_path, start_time, stop_time, step_size, logger, **kwargs)
        fmu, model_dsc, _ = self.instantiate_clean_fmu()
        self.fmu = fmu
        self.model_description = model_dsc

    # def instantiate_clean_fmu(self):

    #     unzipdir = tempfile.mkdtemp()
    #     extract(self.fmu_path, unzipdir)

    #     model_description = read_model_description(unzipdir)

    #     fmu = FMU3Slave(
    #         guid=model_description.guid,
    #         unzipDirectory=unzipdir,
    #         modelIdentifier=model_description.coSimulation.modelIdentifier,
    #         instanceName='instance1'
    #     )

    #     fmu.instantiate()

    #     return fmu, model_description, unzipdir


    def instantiate_clean_fmu(self):
        import sys
        print("[FMU] Creating temp dir...", file=sys.stderr, flush=True)
        unzipdir = tempfile.mkdtemp()
        print(f"[FMU] Extracting to {unzipdir}...", file=sys.stderr, flush=True)
        extract(self.fmu_path, unzipdir)
        print(f"[FMU] Reading model description...", file=sys.stderr, flush=True)
        model_description = read_model_description(unzipdir)
        print(f"[FMU] Creating FMU3Slave...", file=sys.stderr, flush=True)
        fmu = FMU3Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName='instance1'
        )
        print(f"[FMU] Calling instantiate()...", file=sys.stderr, flush=True)
        fmu.instantiate()
        print(f"[FMU] Done", file=sys.stderr, flush=True)
        return fmu, model_description, unzipdir
    
    def set_init_cond(self, parameter_dict, input_dict=None):
        init = self.preprocess_init_cond(parameter_dict)
        self.fmu.enterInitializationMode(startTime=0.0, stopTime=self.stop_time)
        
        for name, (ref, val) in init.items():
            if isinstance(val, list):
                self.fmu.setFloat64([ref], val)
            else:
                self.fmu.setFloat64([ref], [float(val)])
        
        # set inputs during init mode, just like simulate_fmu normally does
        if input_dict is not None:
            self.set_input(input_dict)
        
        self.fmu.exitInitializationMode()
        for name, vr in self.param_vars.items():
            try:
                val = self.fmu.getFloat64([vr])[0]
                # print(name, val)
            except Exception as e:
                val = self.fmu.getFloat64([vr], nValues=50)
                # print(name, val)
    
    def get_sim_params(self, param_names):
        param_dict = {}
        for name in param_names:
            param_dict[name] = self.fmu.getFloat64([self.param_vars[name]])
        return param_dict
    
    def preprocess_init_cond(self, parameter_dict):
        out = {}
        name_mapping = self.get_variables()
        for k, v in parameter_dict.items():
            out[k] = out[k] = (name_mapping[k], v)
        return out
    
    def format_out(self, formatted_res):
        out = {}
        sum_Cleaf = 0
        sum_Cfruit = 0
        sum_Ctruss = 0
        for k, (ref,v) in formatted_res.items():
            if "C_leaves[" in k:
                sum_Cleaf += v
            elif "C_trusses[" in k:
                sum_Ctruss += v
            elif "C_fruits[" in k:
                sum_Cfruit += v
            elif k == "LeafAreaIndex":
                out['LAI'] = v
            else:
                out[k] = v

        out['C_leaves'] = sum_Cleaf
        out['C_fruit'] = sum_Cfruit
        out['C_truss'] = sum_Ctruss

        return out
    
    def set_input(self, input_dict):
            self.fmu.setFloat64(
                list(self.input_vars.values()),
                [input_dict[k] for k in self.input_vars.keys()]
            )

    def get_output(self):
        out = {}
        for k,v in self.output_vars.items():
            type, size = parse_output_type(k)
            if type == "list":
                out[k] = self.fmu.getFloat64([v], nValues=size)
            else:
                out[k] = self.fmu.getFloat64([v])
        
        return out


    
    def simulate(self, inputs, setpoints, init_conds):  
        out = []
        current_time = 0.0
        
        if init_conds is None:
            init_conds = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS}

        self.set_init_cond(init_conds, input_dict={"CO2_Air":400.0, "PAR_gh":500.0, "TCan":20.0, "TCan24":20.0})
        while current_time < self.stop_time:
            if inputs:
                t, inp = inputs[0]
                if current_time <= t:
                    self.set_input(inp)
                    inputs = inputs[1:]

            self.fmu.doStep(currentCommunicationPoint=current_time, 
                            communicationStepSize=self.step_size,
                            noSetFMUStatePriorToCurrentPoint=True)
            current_time += self.step_size

            if setpoints:
                if current_time >= setpoints[0]:
                    setpoints = setpoints[1:]
                    out.append((current_time, self.get_output()))
            else:
                break
        
        self.fmu.reset()

        return out
    
