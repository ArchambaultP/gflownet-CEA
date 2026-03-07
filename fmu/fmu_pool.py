"""Parallel FMU execution with hard-kill timeouts."""
import os
import shutil
import pickle
import tempfile
import multiprocessing as mp

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


def _worker(conn, local_fmu, args_file):
    try:
        with open(args_file, 'rb') as f:
            input_trace, setpoints, init_conds, step_size = pickle.load(f)
        conn.send(_run_team_sim(local_fmu, input_trace, setpoints, init_conds, step_size))
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


def run_parallel(args_by_team, fmu_path, timeout=15, verbose=False):
    tmp_dir = tempfile.mkdtemp()

    team_meta = {}
    for i, (t, args) in enumerate(args_by_team.items()):
        local_fmu = os.path.join(tmp_dir, f"tomato_{i}.fmu")
        shutil.copy2(fmu_path, local_fmu)

        args_file = os.path.join(tmp_dir, f"args_{t}.pkl")
        with open(args_file, 'wb') as f:
            pickle.dump(args, f)

        team_meta[t] = (local_fmu, args_file)

    procs = {}
    for t, (local_fmu, args_file) in team_meta.items():
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=_worker, args=(child_conn, local_fmu, args_file))
        p.start()
        child_conn.close()
        procs[t] = (p, parent_conn)
        if verbose:
            print(f"Started FMU for team {t}")

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
                if verbose:
                    print(f"FMU error for team {t}: {obj}")
        conn.close()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results