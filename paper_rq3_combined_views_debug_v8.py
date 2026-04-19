#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from fmu.pool.batch import evaluate_all
from gflownet.envs.greenhouse.constants_unique_actions_vanthoor import (
    BASELINE_PARAMETERS,
    GROUP_ORDER,
    INITIAL_CONDITIONS,
    PARAMETER_BOUNDS,
    PERTURBATION_SCHEME,
)
from gflownet.envs.greenhouse.sim_env import CropSimEnv


COLORS = {
    "Observed": "#d62728",
    "Initial": "#7f7f7f",
    "TPE": "#1f77b4",
    "GFN": "#ff7f0e",
    "Random": "#2ca02c",
}

DEFAULT_TEAM_IDS = [
    "AICU",
    "Automatoes",
    "Digilog",
    "IUACAAS",
    "Reference",
    "TheAutomators",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def _cache_metadata_dict(args):
    return {
        "n_cycles": int(args.n_cycles),
        "step_fraction": float(args.step_fraction),
        "decay_factor": float(args.decay_factor),
        "beta": "" if args.beta is None else str(args.beta),
        "loss_type": str(args.loss_type),
        "huber_delta": float(args.huber_delta),
        "relative_floor_frac": float(args.relative_floor_frac),
        "relative_floor_abs": float(args.relative_floor_abs),
        "fake_data": int(bool(args.fake_data)),
    }


def _cache_row_matches(row: dict, meta: dict) -> bool:
    try:
        return (
            int(row.get("n_cycles")) == int(meta["n_cycles"])
            and abs(float(row.get("step_fraction")) - float(meta["step_fraction"])) <= 1e-15
            and abs(float(row.get("decay_factor")) - float(meta["decay_factor"])) <= 1e-15
            and str(row.get("beta", "")) == str(meta["beta"])
            and str(row.get("loss_type")) == str(meta["loss_type"])
            and abs(float(row.get("huber_delta")) - float(meta["huber_delta"])) <= 1e-15
            and abs(float(row.get("relative_floor_frac")) - float(meta["relative_floor_frac"])) <= 1e-15
            and abs(float(row.get("relative_floor_abs")) - float(meta["relative_floor_abs"])) <= 1e-15
            and int(row.get("fake_data", 0)) == int(meta["fake_data"])
        )
    except Exception:
        return False


def load_loss_cache(cache_path: Path, meta: dict) -> Dict[str, float]:
    if not cache_path.exists():
        return {}
    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return {}
    out = {}
    for row in df.to_dict(orient="records"):
        if _cache_row_matches(row, meta):
            out[str(row["state_key"])] = float(row["loss"])
    return out


def append_loss_cache(cache_path: Path, loss_map: Dict[str, float], meta: dict):
    if not loss_map:
        return
    rows = []
    for key, loss in loss_map.items():
        rows.append({"state_key": key, "loss": float(loss), **meta})
    new_df = pd.DataFrame(rows)
    if cache_path.exists():
        try:
            old_df = pd.read_csv(cache_path)
            df = pd.concat([old_df, new_df], ignore_index=True)
            df = df.drop_duplicates(
                subset=[
                    "state_key",
                    "n_cycles",
                    "step_fraction",
                    "decay_factor",
                    "beta",
                    "loss_type",
                    "huber_delta",
                    "relative_floor_frac",
                    "relative_floor_abs",
                    "fake_data",
                ],
                keep="last",
            )
        except Exception:
            df = new_df
    else:
        df = new_df
    df.to_csv(cache_path, index=False)


def extract_forward_state_dict(ckpt):
    if "forward" in ckpt and isinstance(ckpt["forward"], dict):
        return ckpt["forward"]
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        cand = {k[len("forward."):]: v for k, v in sd.items() if k.startswith("forward.")}
        if cand:
            return cand
    if any(str(k).endswith("weight") for k in ckpt.keys()):
        return ckpt
    raise KeyError("Could not find forward policy state_dict in checkpoint")


def build_mlp_from_state_dict(state_dict):
    weight_keys = [k for k in state_dict if k.endswith("weight")]
    if not weight_keys:
        raise ValueError("No linear weights found in state_dict")

    def key_order(k):
        parts = k.split(".")
        out = []
        for p in parts:
            try:
                out.append((0, int(p)))
            except ValueError:
                out.append((1, p))
        return out

    weight_keys = sorted(weight_keys, key=key_order)
    layers = []
    for i, wk in enumerate(weight_keys):
        bk = wk[:-6] + "bias"
        W = state_dict[wk]
        b = state_dict[bk]
        out_dim, in_dim = W.shape
        layer = nn.Linear(in_dim, out_dim)
        layer.weight.data.copy_(W)
        layer.bias.data.copy_(b)
        layers.append(layer)
        if i < len(weight_keys) - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.eval()
    return model



def canonical_token(tok: str) -> str:
    tok = str(tok).strip()
    if "__" in tok:
        return tok.split("__", 1)[1]
    return tok


def resolve_group_action_name(group_name: str, action_name: str) -> str:
    action_name = str(action_name).strip()
    group_actions = list(PERTURBATION_SCHEME[group_name].keys())
    if action_name in group_actions:
        return action_name
    canon = canonical_token(action_name)
    matches = [a for a in group_actions if canonical_token(a) == canon]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Ambiguous canonical action {action_name!r} for group {group_name}: {matches}")
    raise KeyError(f"Unknown action {action_name!r} for group {group_name}; available={group_actions}")


def _sorted_weight_keys_from_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> List[str]:
    keys = [k for k in state_dict if k.startswith(prefix) and k.endswith("weight")]
    def order_key(k):
        tail = k[len(prefix):]
        parts = tail.split(".")
        out = []
        for p in parts:
            try:
                out.append((0, int(p)))
            except ValueError:
                out.append((1, p))
        return out
    return sorted(keys, key=order_key)


def build_sequential_from_prefix(state_dict: Dict[str, torch.Tensor], prefix: str):
    weight_keys = _sorted_weight_keys_from_prefix(state_dict, prefix)
    if not weight_keys:
        raise ValueError(f"No linear weights found for prefix={prefix!r}")
    layers = []
    for i, wk in enumerate(weight_keys):
        bk = wk[:-6] + "bias"
        W = state_dict[wk]
        b = state_dict[bk]
        out_dim, in_dim = W.shape
        layer = nn.Linear(in_dim, out_dim)
        layer.weight.data.copy_(W)
        layer.bias.data.copy_(b)
        layers.append(layer)
        if i < len(weight_keys) - 1:
            layers.append(nn.LeakyReLU())
    model = nn.Sequential(*layers)
    model.eval()
    return model


class LoadedGreenhouseMultiheadModel(nn.Module):
    def __init__(self, env: CropSimEnv, trunk: nn.Module, heads: List[nn.Module], invalid_logit: float = -1e9):
        super().__init__()
        self.env = env
        self.trunk = trunk
        self.heads = nn.ModuleList(heads)
        self.invalid_logit = float(invalid_logit)
        self.depth_offset = 1
        self.depth_slice = slice(self.depth_offset, self.depth_offset + env.depth_dim)

    def _infer_depths(self, states: torch.Tensor) -> torch.Tensor:
        depth_feats = states[:, self.depth_slice]
        return torch.argmax(depth_feats, dim=1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.trunk(states)
        logits = torch.full(
            (states.shape[0], self.env.action_space_dim),
            self.invalid_logit,
            dtype=x.dtype,
            device=x.device,
        )
        depths = self._infer_depths(states)
        active = depths < self.env.n_operations
        if not torch.any(active):
            return logits
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        active_depths = depths[active_idx]
        active_group_ids = torch.remainder(active_depths, self.env.n_groups)
        for group_id in range(self.env.n_groups):
            row_mask = active_group_ids == group_id
            if not torch.any(row_mask):
                continue
            rows = active_idx[row_mask]
            local_logits = self.heads[group_id](x[rows])
            action_ids = torch.tensor(self.env.slot_action_ids[group_id], device=logits.device, dtype=torch.long)
            logits[rows[:, None], action_ids] = local_logits
        return logits


def build_forward_model_from_state_dict(state_dict: Dict[str, torch.Tensor], env: CropSimEnv):
    if any(k.startswith("trunk.") for k in state_dict) and any(k.startswith("heads.") for k in state_dict):
        trunk = build_sequential_from_prefix(state_dict, "trunk.")
        heads = []
        head_idx = 0
        while any(k.startswith(f"heads.{head_idx}.") for k in state_dict):
            heads.append(build_sequential_from_prefix(state_dict, f"heads.{head_idx}."))
            head_idx += 1
        if len(heads) != env.n_groups:
            raise ValueError(f"Checkpoint has {len(heads)} heads, but env expects {env.n_groups} groups.")
        model = LoadedGreenhouseMultiheadModel(env=env, trunk=trunk, heads=heads)
        model.eval()
        return model
    model = build_mlp_from_state_dict(state_dict)
    model.eval()
    return model


def download_gfn_checkpoint(project: str, run_id: str, alias: str = "final") -> Path:
    api = wandb.Api(timeout=60)
    artifact = None
    for try_alias in [alias, "latest"]:
        try:
            artifact = api.artifact(f"{project}/ckpt-{run_id}:{try_alias}")
            break
        except Exception:
            continue
    if artifact is None:
        raise RuntimeError(f"Could not find checkpoint artifact for {project}/ckpt-{run_id}")
    root = Path(tempfile.mkdtemp(prefix=f"wandb_ckpt_{run_id}_"))
    artifact_dir = Path(artifact.download(root=str(root)))
    ckpt = artifact_dir / "final.ckpt"
    if not ckpt.exists():
        ckpts = list(artifact_dir.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError("No .ckpt file found in downloaded artifact")
        ckpt = ckpts[0]
    return ckpt


def fetch_tpe_trial_sequence(project: str, run_id: str) -> List[str]:
    api = wandb.Api(timeout=60)
    artifact_name = f"{project}/optuna-trials-{run_id}:latest"
    tmp_dir = None
    try:
        artifact = api.artifact(artifact_name)
        tmp_dir = tempfile.mkdtemp(prefix=f"optuna_trials_{run_id}_")
        artifact_dir = artifact.download(root=tmp_dir)
        csv_path = Path(artifact_dir) / "optuna_trials.csv"
        if not csv_path.exists():
            candidates = list(Path(artifact_dir).rglob("optuna_trials.csv"))
            if candidates:
                csv_path = candidates[0]
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "trial_step" in df.columns and "state_key" in df.columns:
                df = df.sort_values("trial_step").drop_duplicates("trial_step")
                seq = [str(x) for x in df["state_key"].tolist()]
                log(f"[TPE] loaded {len(seq)} trials from artifact {artifact_name}")
                return seq
    except Exception as e:
        log(f"[TPE] artifact lookup failed for {run_id}: {e}")
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    run = api.run(f"{project}/{run_id}")
    rows = []
    for row in run.scan_history():
        if "trial_step" not in row or "state_key" not in row:
            continue
        rows.append({"trial_step": int(row["trial_step"]), "state_key": str(row["state_key"])})
    if not rows:
        raise RuntimeError("No TPE trial sequence could be reconstructed.")
    df = pd.DataFrame(rows).sort_values("trial_step").drop_duplicates("trial_step")
    seq = list(df["state_key"])
    log(f"[TPE] loaded {len(seq)} trials from wandb history fallback for {run_id}")
    return seq


def action_value_to_token_map(env):
    return {v: k for k, v in env.pert2id.items()}


def sample_gfn_sequences(model, env: CropSimEnv, n_samples: int, device: str = "cpu", seed: int = 0) -> List[str]:
    model = model.to(device)
    rng = np.random.default_rng(seed)
    action_space = env.get_action_space()
    id2tok = action_value_to_token_map(env)
    out = []

    expected_in = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            expected_in = mod.in_features
            break

    with torch.no_grad():
        for sample_idx in range(n_samples):
            prefix = [()]
            toks = []
            max_steps = env.n_groups * env.n_cycles + 2
            for _step in range(max_steps):
                x = env.states2policy([prefix]).to(device)
                if expected_in is not None and x.shape[-1] != expected_in:
                    raise RuntimeError(
                        f"Checkpoint/policy input mismatch: env.states2policy produced {x.shape[-1]} features "
                        f"but checkpoint expects {expected_in}. Use --fake_data for figure debugging or rebuild "
                        f"the exact policy architecture from the training config."
                    )
                logits = model(x)[0]
                mask_invalid = env.get_mask_invalid_actions_forward(state=prefix, done=False)
                valid_actions = [a for a, m in zip(action_space, mask_invalid) if not m]
                if not valid_actions:
                    break

                valid_indices = []
                valid_logits = []
                for a, m in zip(action_space, mask_invalid):
                    if not m:
                        valid_indices.append(a)
                        idx = env.action2idx[a] if hasattr(env, "action2idx") else action_space.index(a)
                        valid_logits.append(float(logits[idx].item()))
                valid_logits = np.asarray(valid_logits, dtype=float)
                valid_logits = valid_logits - np.max(valid_logits)
                probs = np.exp(valid_logits)
                probs /= probs.sum()

                chosen = valid_indices[int(rng.choice(len(valid_indices), p=probs))]
                if hasattr(env, "eos") and chosen == env.eos:
                    break

                toks.append(id2tok[chosen])
                prefix = prefix + [(
                    1 + (len(prefix) - 1) // env.n_groups,
                    (len(prefix) - 1) % env.n_groups,
                    chosen,
                )]
            out.append("|".join(toks))
            if (sample_idx + 1) % max(1, n_samples // 5) == 0:
                log(f"[GFN] sampled {sample_idx + 1}/{n_samples} sequences")
    return out


def sample_random_sequences(env: CropSimEnv, n_samples: int, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    action_space = env.get_action_space()
    id2tok = action_value_to_token_map(env)
    out = []

    for sample_idx in range(n_samples):
        prefix = [()]
        toks = []
        max_steps = env.n_groups * env.n_cycles + 2
        for _step in range(max_steps):
            mask_invalid = env.get_mask_invalid_actions_forward(state=prefix, done=False)
            valid_actions = [a for a, m in zip(action_space, mask_invalid) if not m]
            if not valid_actions:
                break

            chosen = valid_actions[int(rng.integers(len(valid_actions)))]
            if hasattr(env, "eos") and chosen == env.eos:
                break

            toks.append(id2tok[chosen])
            prefix = prefix + [(
                1 + (len(prefix) - 1) // env.n_groups,
                (len(prefix) - 1) % env.n_groups,
                chosen,
            )]
        out.append("|".join(toks))
    return out


def sample_biased_sequences(env: CropSimEnv, n_samples: int, seed: int, method: str) -> List[str]:
    rng = np.random.default_rng(seed)
    action_space = env.get_action_space()
    id2tok = action_value_to_token_map(env)
    out = []
    preferred = {
        "GFN": {"increase", "higher_sensitivity", "shift_warm", "more_fruit_growth", "none"},
        "TPE": {"increase", "higher_sensitivity", "shift_warm", "none"},
        "Random": set(),
    }
    pref = preferred.get(method, set())

    for _ in range(n_samples):
        prefix = [()]
        toks = []
        max_steps = env.n_groups * env.n_cycles + 2
        for _step in range(max_steps):
            mask_invalid = env.get_mask_invalid_actions_forward(state=prefix, done=False)
            valid_actions = [a for a, m in zip(action_space, mask_invalid) if not m]
            if not valid_actions:
                break
            valid_toks = [id2tok[a] for a in valid_actions]
            if pref:
                weights = np.array([4.0 if tok in pref else 1.0 for tok in valid_toks], dtype=float)
                weights /= weights.sum()
                idx = int(rng.choice(len(valid_actions), p=weights))
            else:
                idx = int(rng.integers(len(valid_actions)))
            chosen = valid_actions[idx]
            if hasattr(env, "eos") and chosen == env.eos:
                break
            toks.append(id2tok[chosen])
            prefix = prefix + [(
                1 + (len(prefix) - 1) // env.n_groups,
                (len(prefix) - 1) % env.n_groups,
                chosen,
            )]
        out.append("|".join(toks))
    return out


def apply_perturbation(current_params: Dict[str, float], group_name: str, action_name: str, step_fraction: float) -> None:
    action_name = resolve_group_action_name(group_name, action_name)
    action = PERTURBATION_SCHEME[group_name][action_name]
    for param_name, direction in action.items():
        if direction == 0:
            continue
        lo, hi = PARAMETER_BOUNDS[param_name]
        val = current_params[param_name]
        current_params[param_name] = float(np.clip(val + direction * step_fraction * (hi - lo), lo, hi))


def state_key_to_params(state_key: str, n_cycles: int, step_fraction: float, decay_factor: float) -> Dict[str, float]:
    toks = [t.strip() for t in state_key.split("|")]
    expected = n_cycles * len(GROUP_ORDER)
    if len(toks) != expected:
        raise ValueError(f"Expected {expected} actions in state_key, got {len(toks)}: {state_key}")
    current_params = {k: float(INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k])) for k in BASELINE_PARAMETERS}
    idx = 0
    for cycle in range(n_cycles):
        sf = step_fraction * (decay_factor ** cycle)
        for group_name in GROUP_ORDER:
            apply_perturbation(current_params, group_name, toks[idx], sf)
            idx += 1
    return current_params


def fake_loss_for_key(state_key: str) -> float:
    toks = [canonical_token(t) for t in state_key.split("|")]
    good = {"increase", "higher_sensitivity", "shift_warm", "more_fruit_growth", "none"}
    score = 0.0
    for i, tok in enumerate(toks):
        base = 0.0 if tok in good else 1.0
        score += (i + 1) * base
    h = int(hashlib.sha256(state_key.encode()).hexdigest()[:12], 16)
    noise = (h % 10000) / 10000.0
    basin_bonus = 0.0
    if toks[:3] == ["increase", "higher_sensitivity", "shift_warm"]:
        basin_bonus = -2.0
    if len(toks) >= 6 and toks[-3:] == ["shift_warm", "increase", "more_fruit_growth"]:
        basin_bonus = min(basin_bonus, -2.5)
    return max(0.05, 0.4 * score + 0.3 * noise + basin_bonus + 1.0)


def get_or_compute_losses(unique_keys: List[str], args, cache_path: Path) -> Dict[str, float]:
    cache_meta = _cache_metadata_dict(args)
    cached_loss_map = {} if args.ignore_cache else load_loss_cache(cache_path, cache_meta)
    log(
        f"[cache] loaded {len(cached_loss_map)} matching evaluated states from {cache_path}"
        if cache_path.exists() and not args.ignore_cache
        else "[cache] no matching cache loaded"
    )

    missing_keys = [key for key in unique_keys if key not in cached_loss_map]
    log(f"[cache] need to evaluate {len(missing_keys)} new states")

    new_loss_map = {}
    if not missing_keys:
        return dict(cached_loss_map)

    if args.fake_data:
        log(f"[fake] synthesizing losses for {len(missing_keys)} states")
        for i, key in enumerate(missing_keys, start=1):
            new_loss_map[key] = float(fake_loss_for_key(key))
            if i % max(1, len(missing_keys) // 10) == 0 or i == len(missing_keys):
                log(f"[fake] {i}/{len(missing_keys)} losses synthesized")
        append_loss_cache(cache_path, new_loss_map, cache_meta)
        out = dict(cached_loss_map)
        out.update(new_loss_map)
        return out

    chunk_size = max(1, int(args.eval_chunk_size))
    n_chunks = (len(missing_keys) + chunk_size - 1) // chunk_size
    log(
        f"[eval] chunked batch-evaluating {len(missing_keys)} states "
        f"in {n_chunks} chunk(s) with n_workers={args.n_workers}, chunk_size={chunk_size}"
    )

    for chunk_idx, start in enumerate(range(0, len(missing_keys), chunk_size), start=1):
        chunk_keys = missing_keys[start:start + chunk_size]
        states = {}
        for key in chunk_keys:
            combo = tuple(key.split("|"))
            states[combo] = state_key_to_params(
                key,
                n_cycles=args.n_cycles,
                step_fraction=args.step_fraction,
                decay_factor=args.decay_factor,
            )

        log(f"[eval] chunk {chunk_idx}/{n_chunks}: submitting {len(chunk_keys)} states")
        losses_result = evaluate_all(
            states=states,
            team_ids=DEFAULT_TEAM_IDS,
            n_workers=args.n_workers,
            timeout=args.timeout,
            verbose=True,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            relative_floor_frac=args.relative_floor_frac,
            relative_floor_abs=args.relative_floor_abs,
        )

        if isinstance(losses_result, tuple):
            losses_tuple = losses_result[0]
        else:
            losses_tuple = losses_result

        chunk_loss_map = {
            ("|".join(k) if isinstance(k, tuple) else k): float(v)
            for k, v in losses_tuple.items()
        }

        # fail closed: if any state is missing from the batch result, assign a large loss so the script keeps going
        missing_from_chunk = [k for k in chunk_keys if k not in chunk_loss_map]
        for k in missing_from_chunk:
            chunk_loss_map[k] = 1e6

        new_loss_map.update(chunk_loss_map)
        append_loss_cache(cache_path, chunk_loss_map, cache_meta)
        log(
            f"[eval] chunk {chunk_idx}/{n_chunks}: done, received={len(losses_tuple)} "
            f"fallback_missing={len(missing_from_chunk)} total_cached_now={len(cached_loss_map) + len(new_loss_map)}"
        )

    out = dict(cached_loss_map)
    out.update(new_loss_map)
    return out



def hamming_distance(a: str, b: str) -> int:
    ta = a.split("|")
    tb = b.split("|")
    if len(ta) != len(tb):
        raise ValueError("Cannot compare state keys of different lengths")
    return sum(x != y for x, y in zip(ta, tb))


def mean_pairwise_hamming_of_top_unique(seq: List[str], loss_map: Dict[str, float], top_k: int) -> float:
    unique = sorted(set(seq), key=lambda k: (loss_map.get(k, np.inf), k))
    top = unique[:top_k]
    if len(top) < 2:
        return 0.0
    dists = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            dists.append(hamming_distance(top[i], top[j]))
    return float(np.mean(dists))


def median_top_k_losses_of_unique(seq: List[str], loss_map: Dict[str, float], top_k: int) -> float:
    unique = sorted(set(seq), key=lambda k: (loss_map.get(k, np.inf), k))
    vals = [loss_map[k] for k in unique if k in loss_map][:top_k]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def summarize_sequence(seq: List[str], loss_map: Dict[str, float], diversity_top_k: int, top_loss_k: int) -> dict:
    evaluated = [loss_map[k] for k in seq if k in loss_map]
    return {
        "best_final_loss": float(np.min(evaluated)) if evaluated else np.nan,
        "mean_sampled_loss": float(np.mean(evaluated)) if evaluated else np.nan,
        "median_sampled_loss": float(np.median(evaluated)) if evaluated else np.nan,
        "median_top_loss_k": median_top_k_losses_of_unique(seq, loss_map, top_loss_k),
        "unique_states": len(set(seq)),
        "mean_hamming_top_unique": mean_pairwise_hamming_of_top_unique(seq, loss_map, diversity_top_k),
    }


def aggregate_rows(rows: List[dict], method: str):
    frame = pd.DataFrame(rows)
    return {
        "Method": method,
        "Best final loss mean": float(frame["best_final_loss"].mean()),
        "Best final loss std": float(frame["best_final_loss"].std(ddof=0)),
        "Median top-loss-k mean": float(frame["median_top_loss_k"].mean()),
        "Median top-loss-k std": float(frame["median_top_loss_k"].std(ddof=0)),
        "Unique states mean": float(frame["unique_states"].mean()),
        "Unique states std": float(frame["unique_states"].std(ddof=0)),
        "Mean Hamming top-unique mean": float(frame["mean_hamming_top_unique"].mean()),
        "Mean Hamming top-unique std": float(frame["mean_hamming_top_unique"].std(ddof=0)),
    }


def plot_seed_aggregated_bars(out_path: Path, summary_rows: List[dict], diversity_top_k: int, title: str):
    metrics = [
        ("Best final loss mean", "Best loss"),
        ("Median top-loss-k mean", f"Median top-{diversity_top_k} loss"),
        ("Unique states mean", "Unique states"),
        ("Mean Hamming top-unique mean", f"Mean Hamming top-{diversity_top_k}"),
    ]
    methods = [row["Method"] for row in summary_rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4.8))
    for ax, (metric_col, label) in zip(axes, metrics):
        vals = [float(row[metric_col]) for row in summary_rows]
        err_col = metric_col.replace(" mean", " std")
        errs = [float(row[err_col]) for row in summary_rows]
        colors = [COLORS.get(m, "#7f7f7f") for m in methods]
        ax.bar(methods, vals, yerr=errs, color=colors, alpha=0.85, capsize=4)
        ax.set_title(label)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(title, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_points(out_path: Path, per_seed_rows: List[dict], diversity_top_k: int, title: str):
    methods = ["TPE", "GFN", "Random"]
    metrics = [
        ("best_final_loss", "Best loss"),
        ("median_top_loss_k", f"Median top-{diversity_top_k} loss"),
        ("unique_states", "Unique states"),
        ("mean_hamming_top_unique", f"Mean Hamming top-{diversity_top_k}"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4.8))
    for ax, (metric, label) in zip(axes, metrics):
        for m in methods:
            vals = [float(r[metric]) for r in per_seed_rows if r["method"] == m]
            xs = np.full(len(vals), methods.index(m), dtype=float)
            jitter = np.linspace(-0.08, 0.08, num=max(len(vals), 1))[:len(vals)] if vals else np.array([])
            ax.scatter(xs + jitter, vals, label=m, color=COLORS[m], alpha=0.85)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_title(label)
        ax.grid(alpha=0.25, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq_h, uniq_l, seen = [], [], set()
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    fig.legend(
        uniq_h,
        uniq_l,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(title, y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



_SINGLE_OVERLAY_SIM_SCRIPT = r"""
import os
import pickle
import sys
import numpy as np
import pandas as pd

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

args_file, result_file = sys.argv[1], sys.argv[2]
with open(args_file, "rb") as f:
    params, team, fmu_path, data_dir, line_dt_days = pickle.load(f)

from fmu.tomato_controller import TomatoController
from data.greenhouse.secondEdition.extract import (
    load_climate_data, load_prod_data, load_tomato_data, load_parameter_data,
)
from gflownet.envs.greenhouse.constants import BASELINE_PARAMETERS, INITIAL_CONDITIONS

def _as_scalar(v):
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if v else 0.0
    if hasattr(v, "__len__"):
        return float(v[-1]) if len(v) > 0 else 0.0
    return float(v)

def get_team_control_dataset(data_dir, team):
    fp_climate = f"{data_dir}/{team}/GreenhouseClimate.csv"
    climate_df = load_climate_data(fp_climate)
    return climate_df[["CO2air", "PAR", "Tair"]]

def get_team_obs_dataset_with_dap(data_dir, team):
    fp_production = f"{data_dir}/{team}/Production.csv"
    fp_tomato = f"{data_dir}/{team}/TomQuality.csv"
    fp_parameter = f"{data_dir}/{team}/CropParameters.csv"

    prod_df = load_prod_data(fp_production)
    prod_df = pd.DataFrame({
        "N": prod_df["nClassA"] + prod_df["nClassB"],
        "N_Sum": (prod_df["nClassA"] + prod_df["nClassB"]).cumsum(),
        "Yield": prod_df["gClassA"] + prod_df["gClassB"],
        "Yield_Sum": (prod_df["gClassA"] + prod_df["gClassB"]).cumsum(),
        "DAP": prod_df["DAP"],
    })
    tomato_df = load_tomato_data(fp_tomato)
    param_df = load_parameter_data(fp_parameter)

    df = pd.merge(prod_df, param_df, on="Time", how="outer")
    df = pd.merge(df, tomato_df, on="Time", how="inner")
    df = df.ffill()

    df["N_harvest_per_m2"] = ((df["N"] / 10) * df["stem_density"]).cumsum()
    df["yield_fw_g_m2"] = (df["Yield"] / 10) * df["stem_density"]
    df["dry_weight_g_m2"] = df["yield_fw_g_m2"] * (df["dryMatterPercent"] / 100)
    df["dry_weight_mg_CH2O_m2"] = df["dry_weight_g_m2"] * 1000
    df["DM_harvest_obs"] = df["dry_weight_mg_CH2O_m2"].cumsum()
    return df[["DM_harvest_obs", "N_harvest_per_m2", "DAP"]]

def compute_trace(sim_df, delta="30min"):
    sim_df = sim_df.copy()
    sim_df["Tair24"] = sim_df["Tair"].groupby(sim_df.index.date).transform("mean").round(2)
    sim_df.index = sim_df.index.round(delta)
    sim_df = sim_df.groupby(level=0).mean()
    sim_df.index = (sim_df.index - sim_df.index.min()).total_seconds()
    return [
        (t, {"CO2_Air": row.CO2air, "PAR_gh": row.PAR, "TCan": row.Tair, "TCan24": row.Tair24})
        for t, row in sim_df.iterrows()
    ]

team_obs = get_team_obs_dataset_with_dap(data_dir, team)
ctrl_data = get_team_control_dataset(data_dir, team)
climate_start = ctrl_data.index.min()
obs_offset_days = float((team_obs.index.min() - climate_start).total_seconds() / 86400.0)
input_trace = compute_trace(ctrl_data, delta="30min")

climate_duration_days = float((ctrl_data.index.max() - climate_start).total_seconds() / 86400.0)
last_obs_day = float(obs_offset_days + np.max(team_obs["DAP"].to_numpy(dtype=float))) if len(team_obs) else 0.0
horizon_days = max(climate_duration_days, last_obs_day)

dt_seconds = max(float(line_dt_days) * 86400.0, 3600.0)
line_seconds = np.arange(0.0, horizon_days * 86400.0 + dt_seconds, dt_seconds, dtype=float)
if len(line_seconds) == 0 or line_seconds[-1] < horizon_days * 86400.0:
    line_seconds = np.append(line_seconds, horizon_days * 86400.0)

init = {**BASELINE_PARAMETERS, **INITIAL_CONDITIONS, **params}
controller = TomatoController(
    fmu_path, start_time=0, stop_time=max(float(line_seconds[-1]), 86400.0), step_size=120.0, logger=None
)
sim_out = controller.simulate(input_trace, line_seconds.tolist(), init_conds=init)

n = min(len(sim_out), len(line_seconds))
line_x_days = line_seconds[:n] / 86400.0
line_y = np.array([_as_scalar(output["C_harvest"]) for _, output in sim_out[:n]], dtype=float)
obs_x_days = obs_offset_days + team_obs["DAP"].to_numpy(dtype=float)
obs_y = team_obs["DM_harvest_obs"].to_numpy(dtype=float)

with open(result_file, "wb") as f:
    pickle.dump((line_x_days, line_y, obs_x_days, obs_y), f)
"""

def simulate_dm_harvest_series(params: Dict[str, float], team: str, fmu_path: str, data_dir: str, line_dt_days: float = 1.0, label: str | None = None):
    with tempfile.TemporaryDirectory(prefix=f"rq3_overlay_{team}_") as tmp_dir:
        args_file = Path(tmp_dir) / "args.pkl"
        result_file = Path(tmp_dir) / "result.pkl"
        with open(args_file, "wb") as f:
            pickle.dump((params, team, fmu_path, data_dir, line_dt_days), f)

        env = {
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(sys.path),
        }
        proc = subprocess.run(
            [sys.executable, "-c", _SINGLE_OVERLAY_SIM_SCRIPT, str(args_file), str(result_file)],
            cwd=os.getcwd(),
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Overlay simulation failed for label={label} team={team} returncode={proc.returncode}")
        with open(result_file, "rb") as f:
            line_x_days, line_y, obs_x_days, obs_y = pickle.load(f)

    if label is not None:
        log(f"[Overlay] done {label} for team={team}")
    return line_x_days, line_y, obs_x_days, obs_y


def plot_team_grid(out_path: Path, best_params: Dict[str, Dict[str, float]], fmu_path: str, data_dir: str, team_ids: List[str], line_dt_days: float = 1.0):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()

    payload = {}
    global_xmax = 0.0
    global_ymax = 0.0

    for team_idx, team in enumerate(team_ids, start=1):
        log(f"[Overlay] team {team_idx}/{len(team_ids)}: {team}")
        x0, init_y, obs_x, obs_y = simulate_dm_harvest_series(best_params["Initial"], team, fmu_path, data_dir, line_dt_days=line_dt_days, label="Initial")
        x1, tpe_y, _, _ = simulate_dm_harvest_series(best_params["TPE"], team, fmu_path, data_dir, line_dt_days=line_dt_days, label="TPE")
        x2, gfn_y, _, _ = simulate_dm_harvest_series(best_params["GFN"], team, fmu_path, data_dir, line_dt_days=line_dt_days, label="GFN")
        x3, rnd_y, _, _ = simulate_dm_harvest_series(best_params["Random"], team, fmu_path, data_dir, line_dt_days=line_dt_days, label="Random")

        payload[team] = {
            "x0": x0, "init_y": init_y,
            "obs_x": obs_x, "obs_y": obs_y,
            "x1": x1, "tpe_y": tpe_y,
            "x2": x2, "gfn_y": gfn_y,
            "x3": x3, "rnd_y": rnd_y,
        }
        global_xmax = max(global_xmax, float(np.nanmax([np.max(x0), np.max(x1), np.max(x2), np.max(x3), np.max(obs_x)]) if len(obs_x) else np.max([np.max(x0), np.max(x1), np.max(x2), np.max(x3)])))
        global_ymax = max(global_ymax, float(np.nanmax([np.max(init_y), np.max(tpe_y), np.max(gfn_y), np.max(rnd_y), np.max(obs_y)]) if len(obs_y) else np.max([np.max(init_y), np.max(tpe_y), np.max(gfn_y), np.max(rnd_y)])))

    for ax, team in zip(axes, team_ids):
        pl = payload[team]
        ax.plot(pl["x0"], pl["init_y"], color=COLORS["Initial"], linewidth=1.8, linestyle="--", label="Initial")
        ax.plot(pl["x1"], pl["tpe_y"], color=COLORS["TPE"], linewidth=1.8, label="Best TPE")
        ax.plot(pl["x2"], pl["gfn_y"], color=COLORS["GFN"], linewidth=1.8, label="Best GFN")
        ax.plot(pl["x3"], pl["rnd_y"], color=COLORS["Random"], linewidth=1.8, label="Best Random")
        ax.scatter(pl["obs_x"], pl["obs_y"], color=COLORS["Observed"], s=22, marker="o", label="Observed", zorder=5)
        ax.set_title(team)
        ax.set_xlabel("Days after planting")
        ax.set_ylabel("DM harvest")
        ax.grid(alpha=0.25)
        ax.locator_params(axis="x", nbins=5)
        ax.set_xlim(0, global_xmax * 1.02 if global_xmax > 0 else None)
        ax.set_ylim(0, global_ymax * 1.05 if global_ymax > 0 else None)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq_h, uniq_l, seen = [], [], set()
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_fake_team_grid(out_path: Path, best_state_keys: Dict[str, str]):
    x = np.linspace(0, 100, 201)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()
    methods = ["Initial", "TPE", "GFN", "Random"]

    global_xmax = 0.0
    global_ymax = 0.0
    payload = {}

    for ax, team in zip(axes, DEFAULT_TEAM_IDS):
        team_seed = int(hashlib.sha256(team.encode()).hexdigest()[:8], 16) % 1000
        base_curve = 5000 * (1 / (1 + np.exp(-(x - 55) / 9))) + 30 * np.sin(x / 5 + team_seed / 25)

        # Fake RQ12-style observations: pointwise measurements at sparse DAP locations, not a continuous line.
        obs_days = np.array([18, 26, 34, 42, 50, 57, 64, 71, 78, 85, 92, 99], dtype=float)
        obs_noise = np.array([((team_seed + i * 17) % 23) - 11 for i in range(len(obs_days))], dtype=float) * 9.0
        obs_vals = np.maximum(0.0, np.interp(obs_days, x, base_curve) + obs_noise)

        series = {}
        for m in methods:
            if m == "Initial":
                shift = -250 + team_seed % 60
                y = np.maximum(0, base_curve + shift)
            else:
                key = best_state_keys[m]
                mod = (int(hashlib.sha256((team + "|" + key).encode()).hexdigest()[:8], 16) % 200) - 100
                y = np.maximum(0, base_curve + mod)
            series[m] = y

        payload[team] = {"x": x, "obs_days": obs_days, "obs_vals": obs_vals, "series": series}
        global_xmax = max(global_xmax, float(np.max(obs_days)))
        global_ymax = max(global_ymax, float(np.max(obs_vals)), *(float(np.max(v)) for v in series.values()))

    for ax, team in zip(axes, DEFAULT_TEAM_IDS):
        pl = payload[team]
        ax.plot(pl["x"], pl["series"]["Initial"], color=COLORS["Initial"], linewidth=1.8, linestyle="--", label="Initial")
        ax.plot(pl["x"], pl["series"]["TPE"], color=COLORS["TPE"], linewidth=1.8, label="Best TPE")
        ax.plot(pl["x"], pl["series"]["GFN"], color=COLORS["GFN"], linewidth=1.8, label="Best GFN")
        ax.plot(pl["x"], pl["series"]["Random"], color=COLORS["Random"], linewidth=1.8, label="Best Random")
        ax.scatter(pl["obs_days"], pl["obs_vals"], color=COLORS["Observed"], s=22, marker="o", label="Observed", zorder=5)
        ax.set_title(team)
        ax.set_xlabel("Days after planting")
        ax.set_ylabel("DM harvest")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, global_xmax * 1.02 if global_xmax > 0 else None)
        ax.set_ylim(0, global_ymax * 1.05 if global_ymax > 0 else None)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq_h, uniq_l, seen = [], [], set()
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfn_project", required=True)
    ap.add_argument("--gfn_run_ids", nargs="+", required=True)
    ap.add_argument("--bo_project", required=True)
    ap.add_argument("--bo_run_ids", nargs="+", required=True)
    ap.add_argument("--n_cycles", type=int, default=2)
    ap.add_argument("--step_fraction", type=float, required=True)
    ap.add_argument("--decay_factor", type=float, default=1.0)
    ap.add_argument("--retrieval_budget", type=int, default=100)
    ap.add_argument("--train_budget", type=int, default=5000)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--diversity_top_k", type=int, default=20)
    ap.add_argument("--top_loss_k", type=int, default=10)
    ap.add_argument("--skip_team_grid", action="store_true")
    ap.add_argument("--artifact_alias", default="final")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample_seed_base", type=int, default=0)
    ap.add_argument("--output_dir", default="paper_rq3_outputs")
    ap.add_argument("--cache_name", default="evaluated_state_losses_cache.csv")
    ap.add_argument("--ignore_cache", action="store_true")
    ap.add_argument("--fmu_path", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--line_dt_days", type=float, default=1.0)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--eval_chunk_size", type=int, default=256, help="How many unique states to evaluate per batch chunk when using real FMU evaluation.")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--loss_type", default="absolute_relative", choices=["huber_relative", "rse", "absolute_relative"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--relative_floor_frac", type=float, default=0.05)
    ap.add_argument("--relative_floor_abs", type=float, default=1e-6)
    ap.add_argument("--fake_data", action="store_true")
    args = ap.parse_args()

    if len(args.gfn_run_ids) != len(args.bo_run_ids):
        raise ValueError("Same number of GFN and BO/TPE run ids required.")

    n_seeds = len(args.gfn_run_ids)
    total_budget = args.train_budget + args.retrieval_budget
    out_dir = Path(args.output_dir) / f"{args.n_cycles}cycle_sf{args.step_fraction}_budget{args.retrieval_budget}_train{args.train_budget}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / args.cache_name

    log(f"[Init] n_seeds={n_seeds} total_budget={total_budget} fake_data={int(args.fake_data)} n_workers={args.n_workers} eval_chunk_size={args.eval_chunk_size}")

    gfn_retrieval_seqs, tpe_retrieval_seqs, tpe_total_seqs = [], [], []
    random_retrieval_seqs, random_total_seqs = [], []

    if args.fake_data:
        log("[Fake mode] bypassing checkpoint download, model reconstruction, wandb trial history, and FMU evaluation")
        env = CropSimEnv(n_cycles=args.n_cycles, step_fraction=args.step_fraction, precomputed=False, device=args.device)
        for i in range(n_seeds):
            log(f"[Fake mode] generate synthetic sequences for seed {i+1}/{n_seeds}")
            gfn_retrieval_seqs.append(sample_biased_sequences(env, args.retrieval_budget, args.sample_seed_base + i, "GFN"))
            tpe_retrieval_seqs.append(sample_biased_sequences(env, args.retrieval_budget, args.sample_seed_base + 100 + i, "TPE"))
            tpe_total_seqs.append(sample_biased_sequences(env, total_budget, args.sample_seed_base + 200 + i, "TPE"))
            random_retrieval_seqs.append(sample_biased_sequences(env, args.retrieval_budget, args.sample_seed_base + 300 + i, "Random"))
            random_total_seqs.append(sample_biased_sequences(env, total_budget, args.sample_seed_base + 400 + i, "Random"))
    else:
        for i, gfn_run_id in enumerate(args.gfn_run_ids):
            log(f"[GFN] run {i+1}/{n_seeds}: download checkpoint {gfn_run_id}")
            ckpt_path = download_gfn_checkpoint(args.gfn_project, gfn_run_id, alias=args.artifact_alias)
            tmp_root = ckpt_path.parent.parent if ckpt_path.parent.name == "files" else ckpt_path.parent
            try:
                log(f"[GFN] run {i+1}: load checkpoint")
                ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                model = build_forward_model_from_state_dict(extract_forward_state_dict(ckpt), env)
                env = CropSimEnv(n_cycles=args.n_cycles, step_fraction=args.step_fraction, precomputed=False, device=args.device)
                log(f"[GFN] run {i+1}: sample retrieval / random sequences")
                gfn_seq = sample_gfn_sequences(model, env, args.retrieval_budget, device=args.device, seed=args.sample_seed_base + i)
                rnd_retrieval = sample_random_sequences(env, args.retrieval_budget, seed=args.sample_seed_base + 1000 + i)
                rnd_total = sample_random_sequences(env, total_budget, seed=args.sample_seed_base + 2000 + i)
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)
            gfn_retrieval_seqs.append(gfn_seq)
            random_retrieval_seqs.append(rnd_retrieval)
            random_total_seqs.append(rnd_total)

        for i, bo_run_id in enumerate(args.bo_run_ids):
            log(f"[TPE] run {i+1}/{n_seeds}: fetch trial history {bo_run_id}")
            seq_full = fetch_tpe_trial_sequence(args.bo_project, bo_run_id)
            tpe_retrieval_seqs.append(seq_full[: min(args.retrieval_budget, len(seq_full))])
            tpe_total_seqs.append(seq_full[: min(total_budget, len(seq_full))])

    unique_keys = set()
    for seq in gfn_retrieval_seqs + tpe_retrieval_seqs + tpe_total_seqs + random_retrieval_seqs + random_total_seqs:
        unique_keys.update(seq)
    unique_keys = sorted(unique_keys)
    log(f"[Keys] unique candidate states = {len(unique_keys)}")

    loss_map = get_or_compute_losses(unique_keys, args, cache_path)

    per_seed_equal, per_seed_total = [], []
    for i in range(n_seeds):
        g = summarize_sequence(gfn_retrieval_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        g["method"] = "GFN"
        g["seed_index"] = i
        per_seed_equal.append(g)

        t = summarize_sequence(tpe_retrieval_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        t["method"] = "TPE"
        t["seed_index"] = i
        per_seed_equal.append(t)

        r = summarize_sequence(random_retrieval_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        r["method"] = "Random"
        r["seed_index"] = i
        per_seed_equal.append(r)

        g2 = summarize_sequence(gfn_retrieval_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        g2["method"] = "GFN"
        g2["seed_index"] = i
        per_seed_total.append(g2)

        t2 = summarize_sequence(tpe_total_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        t2["method"] = "TPE"
        t2["seed_index"] = i
        per_seed_total.append(t2)

        r2 = summarize_sequence(random_total_seqs[i], loss_map, args.diversity_top_k, args.top_loss_k)
        r2["method"] = "Random"
        r2["seed_index"] = i
        per_seed_total.append(r2)
        log(f"[Summary] seed {i+1}/{n_seeds} summarized")

    pd.DataFrame(per_seed_equal).to_csv(out_dir / "rq3_equal_budget_per_seed_metrics.csv", index=False)
    pd.DataFrame(per_seed_total).to_csv(out_dir / "rq3_total_budget_per_seed_metrics.csv", index=False)

    equal_summary_rows = [
        aggregate_rows([r for r in per_seed_equal if r["method"] == "TPE"], "TPE"),
        aggregate_rows([r for r in per_seed_equal if r["method"] == "GFN"], "GFN"),
        aggregate_rows([r for r in per_seed_equal if r["method"] == "Random"], "Random"),
    ]
    total_summary_rows = [
        aggregate_rows([r for r in per_seed_total if r["method"] == "TPE"], "TPE"),
        aggregate_rows([r for r in per_seed_total if r["method"] == "GFN"], "GFN"),
        aggregate_rows([r for r in per_seed_total if r["method"] == "Random"], "Random"),
    ]

    pd.DataFrame(equal_summary_rows).to_csv(out_dir / "rq3_equal_budget_results_table.csv", index=False)
    pd.DataFrame(total_summary_rows).to_csv(out_dir / "rq3_total_budget_results_table.csv", index=False)

    plot_seed_aggregated_bars(out_dir / "rq3_equal_budget_bars.png", equal_summary_rows, args.diversity_top_k, "RQ3 equal budget")
    plot_seed_points(out_dir / "rq3_equal_budget_points.png", per_seed_equal, args.diversity_top_k, "RQ3 equal budget")
    plot_seed_aggregated_bars(out_dir / "rq3_total_budget_bars.png", total_summary_rows, args.diversity_top_k, "RQ3 total budget")
    plot_seed_points(out_dir / "rq3_total_budget_points.png", per_seed_total, args.diversity_top_k, "RQ3 total budget")

    best_state_keys_equal = {
        "TPE": min(set(k for seq in tpe_retrieval_seqs for k in seq if k in loss_map), key=lambda k: (loss_map[k], k)),
        "GFN": min(set(k for seq in gfn_retrieval_seqs for k in seq if k in loss_map), key=lambda k: (loss_map[k], k)),
        "Random": min(set(k for seq in random_retrieval_seqs for k in seq if k in loss_map), key=lambda k: (loss_map[k], k)),
    }
    best_state_keys_total = {
        "TPE": min(set(k for seq in tpe_total_seqs for k in seq if k in loss_map), key=lambda k: (loss_map[k], k)),
        "GFN": min(set(k for seq in gfn_retrieval_seqs for k in seq if k in loss_map), key=lambda k: (loss_map[k], k)),
        "Random": min(set(k for seq in random_total_seqs for k in seq if k in loss_map), key=lambda k: (loss_map[k], k)),
    }

    if args.skip_team_grid:
        log("[overlay] skipping team grid as requested")
    elif args.fake_data:
        best_state_keys_total["Initial"] = "INITIAL"
        make_fake_team_grid(out_dir / "rq3_best_state_team_grid_total_budget.png", best_state_keys_total)
        log("[overlay] wrote fake team grid")
    else:
        if args.fmu_path is None or args.data_dir is None:
            log("[overlay] skipping real overlay because --fmu_path and --data_dir were not provided")
        else:
            best_params_total = {
                "Initial": {k: float(INITIAL_CONDITIONS.get(k, BASELINE_PARAMETERS[k])) for k in BASELINE_PARAMETERS},
                "TPE": state_key_to_params(best_state_keys_total["TPE"], args.n_cycles, args.step_fraction, args.decay_factor),
                "GFN": state_key_to_params(best_state_keys_total["GFN"], args.n_cycles, args.step_fraction, args.decay_factor),
                "Random": state_key_to_params(best_state_keys_total["Random"], args.n_cycles, args.step_fraction, args.decay_factor),
            }
            plot_team_grid(
                out_dir / "rq3_best_state_team_grid_total_budget.png",
                best_params_total,
                args.fmu_path,
                args.data_dir,
                DEFAULT_TEAM_IDS,
                line_dt_days=args.line_dt_days,
            )

    if "Initial" in best_state_keys_total and best_state_keys_total["Initial"] not in loss_map:
        log("[summary] Initial overlay baseline has no evaluated loss entry; writing null in summary JSON")
    with open(out_dir / "rq3_combined_summary.json", "w") as f:
        json.dump(
            {
                "n_seeds": n_seeds,
                "n_cycles": args.n_cycles,
                "step_fraction": args.step_fraction,
                "decay_factor": args.decay_factor,
                "retrieval_budget": args.retrieval_budget,
                "train_budget": args.train_budget,
                "beta": args.beta,
                "total_budget": total_budget,
                "diversity_top_k": args.diversity_top_k,
                "top_loss_k": args.top_loss_k,
                "fmu_path": args.fmu_path,
                "data_dir": args.data_dir,
                "n_unique_states_requested": len(unique_keys),
                "n_states_evaluated": len(loss_map),
                "cache_path": str(cache_path),
                "best_state_keys_equal": best_state_keys_equal,
                "best_state_keys_total": best_state_keys_total,
                "best_state_losses_total": {
                    k: (float(loss_map[v]) if v in loss_map else None)
                    for k, v in best_state_keys_total.items()
                },
                "equal_summary_rows": equal_summary_rows,
                "total_summary_rows": total_summary_rows,
                "equal_per_seed_rows": per_seed_equal,
                "total_per_seed_rows": per_seed_total,
                "fake_data": bool(args.fake_data),
            },
            f,
            indent=2,
        )

    log(f"Saved combined RQ3 outputs to {out_dir}")
    log(f"  - cache: {cache_path}")
    log("  - rq3_equal_budget_bars.png")
    log("  - rq3_equal_budget_points.png")
    log("  - rq3_total_budget_bars.png")
    log("  - rq3_total_budget_points.png")
    if not args.skip_team_grid:
        log("  - rq3_best_state_team_grid_total_budget.png")
    log("  - rq3_equal_budget_results_table.csv")
    log("  - rq3_total_budget_results_table.csv")
    log("  - rq3_equal_budget_per_seed_metrics.csv")
    log("  - rq3_total_budget_per_seed_metrics.csv")
    log("  - rq3_combined_summary.json")


if __name__ == "__main__":
    main()
