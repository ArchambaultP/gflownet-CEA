"""Parallel FMU execution with hard-kill timeouts."""
import os
import shutil
import pickle
import tempfile
import multiprocessing as mp
import traceback

def _worker(conn, local_fmu, args_file):
    try:
        with open(args_file, 'rb') as f:
            input_trace, setpoints, init_conds, step_size = pickle.load(f)
        conn.send(_run_team_sim(local_fmu, input_trace, setpoints, init_conds, step_size))
    except Exception as e:
        traceback.print_exc()  # prints to stderr so you can see it
        conn.send(e)
    finally:
        conn.close()

def _run_team_sim(fmu_path, input_trace, setpoints, init_conds, step_size):
    from fmu.tomato_controller import TomatoController

    controller = TomatoController(
        fmu_path,
        start_time=0.0,
        stop_time=input_trace[-1][0],
        step_size=step_size,
        logger=None,
    )
    return controller.simulate(input_trace, setpoints, init_conds=init_conds)


def run_parallel(args_by_team, fmu_path, timeout=15,verbose=False):
    """
    Run one FMU simulation per team in parallel.

    Args:
        args_by_team: dict of {team_name: (input_trace, setpoints, init_conds, step_size)}
        fmu_path: path to the .fmu file
        timeout: seconds before hard-killing a worker

    Returns:
        dict of {team_name: sim_output} for teams that succeeded
    """
    tmp_dir = tempfile.mkdtemp()

    # Pre-copy FMU and pre-serialize args before spawning
    team_meta = {}
    for i, (t, args) in enumerate(args_by_team.items()):
        local_fmu = os.path.join(tmp_dir, f"tomato_{i}.fmu")
        shutil.copy2(fmu_path, local_fmu)

        args_file = os.path.join(tmp_dir, f"args_{t}.pkl")
        with open(args_file, 'wb') as f:
            pickle.dump(args, f)

        team_meta[t] = (local_fmu, args_file)

    # Spawn all at once
    procs = {}
    for t, (local_fmu, args_file) in team_meta.items():
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=_worker, args=(child_conn, local_fmu, args_file))
        p.start()
        child_conn.close()
        procs[t] = (p, parent_conn)

    # Collect with hard-kill timeout
    results = {}
    for t, (p, conn) in procs.items():
        p.join(timeout=timeout)
        if p.is_alive():
            p.kill()
            p.join()
            if verbose:
                print(f"Hard-killed FMU for team {t}")
        elif conn.poll():
            obj = conn.recv()
            if not isinstance(obj, Exception):
                results[t] = obj
                if verbose:
                    print(f"Finished FMU for team {t}")
            else:
                print(f"FMU error for team {t}: {obj}")
        else:
            # Process exited but sent nothing — it crashed
            print(f"FMU crashed for team {t}, exit code: {p.exitcode}")
        conn.close()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results