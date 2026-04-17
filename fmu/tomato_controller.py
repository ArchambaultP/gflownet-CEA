from gflownet.envs.greenhouse.constants_unique_actions_vanthoor import (
    parse_output_type,
    BASELINE_PARAMETERS,
    INITIAL_CONDITIONS,
)
import tempfile
from fmpy import read_model_description, extract
from fmpy.fmi3 import FMU3Slave
from fmu.fmu_controller import FMUController
import shutil
import os


class TomatoController(FMUController):
    """
    Controller for interacting with the tomato FMU (FMU v3).
    """

    def __init__(self, fmu_path, start_time=0.0, stop_time=86400.0, step_size=120, logger=None, **kwargs):
        super().__init__(fmu_path, start_time, stop_time, step_size, logger, **kwargs)
        fmu, model_dsc, unzipdir = self.instantiate_clean_fmu()
        self.fmu = fmu
        self.model_description = model_dsc
        self.unzipdir = unzipdir

    def instantiate_clean_fmu(self):
        unzipdir = tempfile.mkdtemp()
        extract(self.fmu_path, unzipdir)
        model_description = read_model_description(unzipdir)
        fmu = FMU3Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )
        fmu.instantiate()
        return fmu, model_description, unzipdir

    def preprocess_init_cond(self, parameter_dict):
        out = {}
        name_mapping = self.get_variables()
        for k, v in parameter_dict.items():
            out[k] = (name_mapping[k], v)
        return out

    def set_init_cond(self, parameter_dict, input_dict=None):
        init = self.preprocess_init_cond(parameter_dict)

        self.fmu.enterInitializationMode(startTime=0.0, stopTime=self.stop_time)

        for name, (ref, val) in init.items():
            if isinstance(val, list):
                self.fmu.setFloat64([ref], val)
            else:
                self.fmu.setFloat64([ref], [float(val)])

        if input_dict is not None:
            self.set_input(input_dict)

        self.fmu.exitInitializationMode()

        # Optional sanity reads; kept because your original code had them.
        for name, vr in self.param_vars.items():
            try:
                _ = self.fmu.getFloat64([vr])[0]
            except Exception:
                _ = self.fmu.getFloat64([vr], nValues=50)

    def get_sim_params(self, param_names):
        param_dict = {}
        for name in param_names:
            param_dict[name] = self.fmu.getFloat64([self.param_vars[name]])
        return param_dict

    def format_out(self, formatted_res):
        out = {}
        sum_Cleaf = 0.0
        sum_Cfruit = 0.0
        sum_Ctruss = 0.0

        for k, (ref, v) in formatted_res.items():
            if "C_leaves[" in k:
                sum_Cleaf += v
            elif "C_trusses[" in k:
                sum_Ctruss += v
            elif "C_fruits[" in k:
                sum_Cfruit += v
            elif k == "LeafAreaIndex":
                out["LAI"] = v
            else:
                out[k] = v

        out["C_leaves"] = sum_Cleaf
        out["C_fruit"] = sum_Cfruit
        out["C_truss"] = sum_Ctruss

        return out

    def set_input(self, input_dict):
        self.fmu.setFloat64(
            list(self.input_vars.values()),
            [float(input_dict[k]) for k in self.input_vars.keys()],
        )

    def get_output(self):
        out = {}
        for k, v in self.output_vars.items():
            output_type, size = parse_output_type(k)
            if output_type == "list":
                out[k] = self.fmu.getFloat64([v], nValues=size)
            else:
                out[k] = self.fmu.getFloat64([v])
        return out

    def simulate(self, inputs, setpoints, init_conds):
        """
        Simulate the FMU and return outputs sampled at the requested setpoints.

        Parameters
        ----------
        inputs : list[tuple[float, dict]]
            Time-stamped climate/input values, in seconds.
        setpoints : list[float]
            Query times in seconds at which outputs should be recorded.
        init_conds : dict
            Initial parameter/state dictionary.

        Returns
        -------
        list[tuple[float, dict]]
            Pairs of (time_seconds, output_dict).
        """
        inputs = sorted(list(inputs or []), key=lambda x: x[0])
        setpoints = sorted(list(setpoints or []))

        if init_conds is None:
            init_conds = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS}

        default_input = {
            "CO2_Air": 400.0,
            "PAR_gh": 500.0,
            "TCan": 20.0,
            "TCan24": 20.0,
        }

        input_idx = 0
        setpoint_idx = 0
        current_time = float(self.start_time)
        out = []

        # Use the latest input available at t <= start_time as init input.
        init_input = default_input
        while input_idx < len(inputs) and float(inputs[input_idx][0]) <= current_time:
            _, init_input = inputs[input_idx]
            input_idx += 1

        self.set_init_cond(init_conds, input_dict=init_input)

        # Record outputs requested exactly at start_time.
        while setpoint_idx < len(setpoints) and float(setpoints[setpoint_idx]) <= current_time:
            out.append((current_time, self.get_output()))
            setpoint_idx += 1

        # Main simulation loop
        while current_time < self.stop_time:
            # Apply any inputs whose timestamp has been reached.
            while input_idx < len(inputs) and float(inputs[input_idx][0]) <= current_time:
                _, inp = inputs[input_idx]
                self.set_input(inp)
                input_idx += 1

            self.fmu.doStep(
                currentCommunicationPoint=current_time,
                communicationStepSize=self.step_size,
                noSetFMUStatePriorToCurrentPoint=True,
            )
            current_time += self.step_size

            # Record outputs for all setpoints reached by this step.
            while setpoint_idx < len(setpoints) and float(setpoints[setpoint_idx]) <= current_time:
                out.append((float(setpoints[setpoint_idx]), self.get_output()))
                setpoint_idx += 1

            if setpoint_idx >= len(setpoints) and current_time >= self.stop_time:
                break

        return out

    def close(self):
        try:
            self.fmu.terminate()
        except Exception:
            pass
        try:
            self.fmu.freeInstance()
        except Exception:
            pass
        if self.unzipdir and os.path.isdir(self.unzipdir):
            shutil.rmtree(self.unzipdir, ignore_errors=True)