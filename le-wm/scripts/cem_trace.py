from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task_cost import task_cost  # noqa: E402

try:
    from aliasing_experiment import _get_env_state, _make_env, _reset_env_to_state, _step_env  # noqa: E402
except Exception:  # noqa: BLE001
    _get_env_state = None
    _make_env = None
    _reset_env_to_state = None
    _step_env = None


EPS = 1e-8


@dataclass
class CEMTraceConfig:
    enabled: bool = False
    trace_dir: Path = Path("results/cem_trace")
    trace_episodes: int = 5
    trace_max_steps: int = 30
    save_full_rollouts: bool = False
    true_replay_candidates: bool = False
    model_name: str = "unknown"
    latent_dim: int = -1
    horizon_blocks: int = -1
    action_block: int = -1
    receding_horizon_raw: int = -1
    env_name: str = "swm/PushT-v1"
    reset_state_tol: float = 1e-3
    true_replay_topk: int = 64
    true_replay_random: int = 64


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _vector_rows(value: Any) -> Optional[np.ndarray]:
    arr = _to_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    if arr.ndim >= 3 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None]
    return arr.reshape(arr.shape[0], -1)


def _safe_task_cost(states: Optional[np.ndarray], goals: Optional[np.ndarray], batch_idx: int) -> float:
    if states is None or goals is None:
        return float("nan")
    if batch_idx >= states.shape[0] or batch_idx >= goals.shape[0]:
        return float("nan")
    try:
        state = np.asarray(states[batch_idx], dtype=np.float64)
        goal = np.asarray(goals[batch_idx], dtype=np.float64)
        return float(task_cost(state, goal))
    except Exception:
        return float("nan")


def _rank_1_based(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + EPS)


def _done_reason_summary(info: Any) -> str:
    if not isinstance(info, dict):
        return str(info)
    keep = {}
    for key, value in info.items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
            keep[str(key)] = _jsonable(value)
    return json.dumps(keep)[:2048]


def _bool_from_cli(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lower = value.lower()
    if lower in {"1", "true", "yes", "y", "on"}:
        return True
    if lower in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def extract_trace_cli_args(argv: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "trace_cem": False,
        "trace_dir": "results/cem_trace",
        "trace_episodes": 5,
        "trace_max_steps": 30,
        "trace_save_full_rollouts": False,
        "trace_true_replay_candidates": False,
    }
    cleaned = [argv[0]]
    idx = 1
    while idx < len(argv):
        item = argv[idx]
        if item == "--trace_cem":
            out["trace_cem"] = True
            idx += 1
        elif item.startswith("--trace_cem="):
            out["trace_cem"] = _bool_from_cli(item.split("=", 1)[1])
            idx += 1
        elif item in {"--trace_dir", "--trace_episodes", "--trace_max_steps"}:
            if idx + 1 >= len(argv):
                raise ValueError(f"{item} requires a value.")
            key = item[2:]
            value = argv[idx + 1]
            out[key] = int(value) if key in {"trace_episodes", "trace_max_steps"} else value
            idx += 2
        elif item.startswith("--trace_dir="):
            out["trace_dir"] = item.split("=", 1)[1]
            idx += 1
        elif item.startswith("--trace_episodes="):
            out["trace_episodes"] = int(item.split("=", 1)[1])
            idx += 1
        elif item.startswith("--trace_max_steps="):
            out["trace_max_steps"] = int(item.split("=", 1)[1])
            idx += 1
        elif item == "--trace_save_full_rollouts":
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
                out["trace_save_full_rollouts"] = _bool_from_cli(argv[idx + 1])
                idx += 2
            else:
                out["trace_save_full_rollouts"] = True
                idx += 1
        elif item.startswith("--trace_save_full_rollouts="):
            out["trace_save_full_rollouts"] = _bool_from_cli(item.split("=", 1)[1])
            idx += 1
        elif item == "--trace_true_replay_candidates":
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
                out["trace_true_replay_candidates"] = _bool_from_cli(argv[idx + 1])
                idx += 2
            else:
                out["trace_true_replay_candidates"] = True
                idx += 1
        elif item.startswith("--trace_true_replay_candidates="):
            out["trace_true_replay_candidates"] = _bool_from_cli(item.split("=", 1)[1])
            idx += 1
        else:
            cleaned.append(item)
            idx += 1
    argv[:] = cleaned
    return out


class CEMTracer:
    def __init__(self, config: CEMTraceConfig):
        self.config = config
        self.trace_dir = Path(config.trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.pool_dir = self.trace_dir / "candidate_pools"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.iter_dir = self.trace_dir / "iteration_costs"
        self.iter_dir.mkdir(parents=True, exist_ok=True)
        self.planning_rows: List[Dict[str, Any]] = []
        self.iter_rows: List[Dict[str, Any]] = []
        self.pool_rows: List[Dict[str, Any]] = []
        self.exec_rows: List[Dict[str, Any]] = []
        self.timeline_rows: List[Dict[str, Any]] = []
        self.angle_rows: List[Dict[str, Any]] = []
        self.true_replay_rows: List[Dict[str, Any]] = []
        self.current_info: Optional[Dict[str, Any]] = None
        self.current_raw_action: Optional[np.ndarray] = None
        self.active_plan: Optional[Dict[str, Any]] = None
        self.raw_call_idx = -1
        self.mpc_step_idx = -1
        self.last_exec_row_by_batch: Dict[int, int] = {}
        self.original_get_cost = None
        self.original_get_action = None
        self.solver = None
        self.policy = None
        self.replay_env = None
        self.solver_return_action = None
        self.solver_return_sequence = None
        self._warned_true_replay = False
        self.external_episode_id: Optional[int] = None

    def start_episode(self, episode_id: int) -> None:
        self.external_episode_id = int(episode_id)
        self.raw_call_idx = -1
        self.mpc_step_idx = -1
        self.last_exec_row_by_batch = {}
        self.active_plan = None

    def _episode_id(self, batch_idx: int) -> int:
        if self.external_episode_id is not None:
            return int(self.external_episode_id)
        return int(batch_idx)

    def mark_episode_result(self, episode_id: int, success: bool, final_task_cost: float) -> None:
        for collection in (self.planning_rows, self.exec_rows, self.timeline_rows, self.pool_rows, self.angle_rows, self.true_replay_rows):
            for row in collection:
                if int(row.get("episode_id", -1)) == int(episode_id):
                    row["episode_final_success"] = int(bool(success))
                    row["episode_final_task_cost"] = float(final_task_cost)
                    if collection is self.exec_rows:
                        try:
                            after = float(row.get("true_task_cost_after_execution", float("nan")))
                        except (TypeError, ValueError):
                            after = float("nan")
                        if not math.isfinite(after):
                            before = float(row.get("true_task_cost_before_execution", float("nan")))
                            row["true_task_cost_after_execution"] = float(final_task_cost)
                            row["true_progress_delta"] = before - float(final_task_cost) if math.isfinite(before) else float("nan")
                            row["progress_improved_after_first_block"] = int(float(final_task_cost) < before) if math.isfinite(before) else ""

    def attach(self, model: Any, policy: Any, cfg: Any = None) -> None:
        self.original_get_cost = model.get_cost
        self.original_get_action = policy.get_action
        self.policy = policy
        self.solver = getattr(policy, "solver", None)
        self._wrap_solver_methods()
        tracer = self

        def traced_get_cost(info_dict, action_candidates):
            return tracer._get_cost_wrapper(info_dict, action_candidates)

        def traced_get_action(info_dict):
            tracer._begin_policy_call(info_dict)
            try:
                action = tracer.original_get_action(info_dict)
            finally:
                pass
            tracer._finish_policy_call(action)
            return action

        model.get_cost = traced_get_cost
        policy.get_action = traced_get_action
        self._write_metadata(cfg, model, policy)
        print(f"[cem_trace] enabled; writing trace to {self.trace_dir}", flush=True)

    def _wrap_solver_methods(self) -> None:
        if self.solver is None:
            return
        for method_name in ("solve", "plan", "sample_action", "get_action", "act"):
            method = getattr(self.solver, method_name, None)
            if method is None or not callable(method):
                continue

            def make_wrapper(fn, name):
                def wrapper(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    self._capture_solver_return(result, name)
                    return result

                return wrapper

            try:
                setattr(self.solver, method_name, make_wrapper(method, method_name))
            except Exception:
                pass

    def _capture_solver_return(self, result: Any, method_name: str) -> None:
        arr = _to_numpy(result)
        if arr is not None and arr.size:
            self.solver_return_action = arr
            if arr.ndim >= 3:
                self.solver_return_sequence = arr
        if isinstance(result, dict):
            for key in ("action_sequence", "actions", "plan", "sequence"):
                if key in result:
                    seq = _to_numpy(result[key])
                    if seq is not None:
                        self.solver_return_sequence = seq
                        break
            for key in ("action", "selected_action", "mean_action"):
                if key in result:
                    action = _to_numpy(result[key])
                    if action is not None:
                        self.solver_return_action = action
                        break

    def _elite_count(self, candidate_count: int) -> Tuple[int, str]:
        for attr in ("elite_k", "num_elites", "n_elites", "_elite_k", "_num_elites", "_n_elites"):
            value = getattr(self.solver, attr, None)
            if value is not None:
                try:
                    return max(1, min(int(value), candidate_count)), f"solver.{attr}"
                except (TypeError, ValueError):
                    pass
        config = getattr(self.solver, "_config", None) or getattr(self.solver, "config", None)
        for attr in ("elite_k", "num_elites", "n_elites"):
            value = getattr(config, attr, None)
            if value is not None:
                try:
                    return max(1, min(int(value), candidate_count)), f"solver.config.{attr}"
                except (TypeError, ValueError):
                    pass
        return max(1, int(math.ceil(0.1 * candidate_count))), "fallback_top_10_percent"

    def _distribution_stats(self) -> Dict[str, float]:
        stats = {
            "distribution_mean_norm": float("nan"),
            "distribution_std_mean": float("nan"),
            "distribution_std_min": float("nan"),
            "distribution_std_max": float("nan"),
        }
        if self.solver is None:
            return stats
        mean_arr = None
        std_arr = None
        for attr in ("mean", "_mean", "mu", "_mu", "action_mean", "_action_mean"):
            mean_arr = _to_numpy(getattr(self.solver, attr, None))
            if mean_arr is not None:
                break
        for attr in ("std", "_std", "sigma", "_sigma", "var", "_var", "action_std", "_action_std"):
            std_arr = _to_numpy(getattr(self.solver, attr, None))
            if std_arr is not None:
                if "var" in attr:
                    std_arr = np.sqrt(np.maximum(std_arr, 0.0))
                break
        if mean_arr is not None and mean_arr.size:
            stats["distribution_mean_norm"] = float(np.linalg.norm(mean_arr.reshape(-1)))
        if std_arr is not None and std_arr.size:
            flat = std_arr.reshape(-1)
            stats["distribution_std_mean"] = float(np.mean(flat))
            stats["distribution_std_min"] = float(np.min(flat))
            stats["distribution_std_max"] = float(np.max(flat))
        return stats

    def _infer_selected_candidate(
        self,
        action_sequence: Optional[np.ndarray],
        batch_idx: int,
        fallback_idx: int,
    ) -> Tuple[int, str, float]:
        if action_sequence is None:
            return fallback_idx, "argmin_final_terminal_cost_no_action_sequence", float("nan")
        candidates = self._actions_for_batch(action_sequence, batch_idx)
        if candidates is None:
            return fallback_idx, "argmin_final_terminal_cost_bad_action_shape", float("nan")
        target = self.solver_return_sequence
        source = "solver_return_sequence"
        if target is None:
            target = self.solver_return_action if self.current_raw_action is None else self.current_raw_action
            source = "executed_action_prefix"
        target = _to_numpy(target)
        if target is None or target.size == 0:
            return fallback_idx, "argmin_final_terminal_cost_no_solver_return", float("nan")
        target = np.asarray(target)
        if target.ndim >= 3 and target.shape[0] > batch_idx:
            target = target[batch_idx]
        elif target.ndim >= 2 and target.shape[0] == 1:
            target = target[0]
        target_flat = target.reshape(-1)
        flat_candidates = candidates.reshape(candidates.shape[0], -1)
        compare_dim = min(flat_candidates.shape[1], target_flat.shape[0])
        if compare_dim <= 0:
            return fallback_idx, "argmin_final_terminal_cost_empty_action_compare", float("nan")
        distances = np.linalg.norm(flat_candidates[:, :compare_dim] - target_flat[:compare_dim][None], axis=1)
        idx = int(np.nanargmin(distances))
        return idx, source, float(distances[idx])

    def _actions_for_batch(self, action_sequence: np.ndarray, batch_idx: int) -> Optional[np.ndarray]:
        actions = np.asarray(action_sequence)
        if actions.ndim == 4:
            if batch_idx >= actions.shape[0]:
                return None
            return actions[batch_idx]
        if actions.ndim == 3:
            return actions
        return None

    def _action_processor(self):
        process = getattr(self.policy, "process", None)
        if isinstance(process, dict):
            return process.get("action")
        return getattr(process, "action", None)

    def _candidate_future_raw_actions(
        self,
        candidate_action: np.ndarray,
        history_size: int,
        raw_action_dim: int,
    ) -> Optional[np.ndarray]:
        action = np.asarray(candidate_action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None]
        processor = self._action_processor()

        def convert(seq: np.ndarray) -> Optional[np.ndarray]:
            if seq.size == 0:
                return None
            flat = seq.reshape(-1, seq.shape[-1])
            if processor is not None:
                mean = np.asarray(getattr(processor, "mean_", np.zeros(raw_action_dim)), dtype=np.float32).reshape(-1)
                scale = np.asarray(getattr(processor, "scale_", np.ones_like(mean)), dtype=np.float32).reshape(-1)
                if flat.shape[-1] != mean.size and flat.shape[-1] % mean.size == 0:
                    repeat = flat.shape[-1] // mean.size
                    mean = np.tile(mean, repeat)
                    scale = np.tile(scale, repeat)
                if flat.shape[-1] == mean.size:
                    flat = flat * scale.reshape(1, -1) + mean.reshape(1, -1)
            if flat.shape[-1] == raw_action_dim:
                return flat.reshape(-1, raw_action_dim)
            if flat.shape[-1] % raw_action_dim == 0:
                return flat.reshape(-1, raw_action_dim)
            return None

        expected = int(self.config.horizon_blocks * self.config.action_block)
        future = convert(action[history_size:])
        whole = convert(action)
        if future is not None and future.shape[0] >= expected:
            return future
        if whole is not None and whole.shape[0] >= expected:
            return whole
        return future if future is not None else whole

    def _true_replay_candidates(
        self,
        batch_idx: int,
        values: np.ndarray,
        ranks: np.ndarray,
        selected_idx: int,
        actions: Optional[np.ndarray],
        history_size: int,
        state_rows: Optional[np.ndarray],
        goal_rows: Optional[np.ndarray],
        elite_threshold: float,
    ) -> None:
        if not self.config.true_replay_candidates:
            return
        if _make_env is None or _reset_env_to_state is None or _step_env is None or _get_env_state is None:
            if not self._warned_true_replay:
                print("[cem_trace] warning: true replay unavailable because aliasing env helpers could not be imported.", flush=True)
                self._warned_true_replay = True
            return
        if actions is None or state_rows is None or goal_rows is None or batch_idx >= state_rows.shape[0] or batch_idx >= goal_rows.shape[0]:
            if not self._warned_true_replay:
                print("[cem_trace] warning: true replay unavailable because action/state/goal tensors are missing.", flush=True)
                self._warned_true_replay = True
            return
        candidate_actions = self._actions_for_batch(actions, batch_idx)
        if candidate_actions is None or candidate_actions.ndim < 2:
            return
        if self.replay_env is None:
            self.replay_env = _make_env(self.config.env_name)
        raw_action_dim = int(np.asarray(getattr(self.replay_env, "action_space").shape).prod())
        top = np.argsort(values)[: min(int(self.config.true_replay_topk), values.shape[0])]
        elite = np.where(values <= elite_threshold)[0]
        rng = np.random.default_rng(0 + int(self.mpc_step_idx) * 1009 + int(batch_idx))
        random_count = min(int(self.config.true_replay_random), values.shape[0])
        random = rng.choice(values.shape[0], size=random_count, replace=False) if random_count > 0 else np.asarray([], dtype=np.int64)
        replay_indices = np.unique(np.concatenate([np.asarray([selected_idx], dtype=np.int64), top, elite, random])).astype(np.int64)
        start_state = np.asarray(state_rows[batch_idx], dtype=np.float64)
        goal_state = np.asarray(goal_rows[batch_idx], dtype=np.float64)
        start_cost = float(task_cost(start_state, goal_state))
        for candidate_idx in replay_indices:
            raw_actions = self._candidate_future_raw_actions(candidate_actions[candidate_idx], history_size, raw_action_dim)
            if raw_actions is None or raw_actions.size == 0:
                continue
            candidate_type = "random"
            if candidate_idx == int(np.nanargmin(values)):
                candidate_type = "argmin"
            elif values[candidate_idx] <= elite_threshold:
                candidate_type = "elite"
            try:
                obs, _info = _reset_env_to_state(self.replay_env, start_state, goal_state, self.config.reset_state_tol)
                done = False
                info = {}
                first_block_cost = float("nan")
                max_raw_steps = int(self.config.horizon_blocks * self.config.action_block)
                replay_actions = raw_actions[:max_raw_steps]
                executed_steps = 0
                for step_idx, action in enumerate(replay_actions):
                    obs, _reward, done, info = _step_env(self.replay_env, action)
                    executed_steps = step_idx + 1
                    if executed_steps == int(self.config.action_block):
                        first_state = np.asarray(_get_env_state(self.replay_env, obs=obs, info=info), dtype=np.float64)[: goal_state.size]
                        first_block_cost = float(task_cost(first_state, goal_state))
                    if done:
                        break
                terminal_state = np.asarray(_get_env_state(self.replay_env, obs=obs, info=info), dtype=np.float64)[: goal_state.size]
                terminal_cost = float(task_cost(terminal_state, goal_state))
                self.true_replay_rows.append(
                    {
                        "episode_id": int(self._episode_id(batch_idx)),
                        "mpc_step_idx": int(self.mpc_step_idx),
                        "raw_env_step": int(self._raw_env_step()),
                        "candidate_idx": int(candidate_idx),
                        "candidate_type": candidate_type,
                        "predicted_terminal_cost_c25": float(values[candidate_idx]),
                        "rank_by_predicted_cost": int(ranks[candidate_idx]),
                        "selected_flag": int(candidate_idx == selected_idx),
                        "elite_flag": int(values[candidate_idx] <= elite_threshold),
                        "true_terminal_task_cost": terminal_cost,
                        "true_terminal_progress": float(start_cost - terminal_cost),
                        "true_progress_delta": float(start_cost - terminal_cost),
                        "first_block_progress_delta": float(start_cost - first_block_cost) if math.isfinite(first_block_cost) else float("nan"),
                        "true_start_task_cost": start_cost,
                        "executed_raw_steps": int(executed_steps),
                        "expected_raw_steps": max_raw_steps,
                        "done": int(bool(done)),
                        "done_reason": _done_reason_summary(info),
                        "terminal_state": json.dumps(_jsonable(terminal_state)),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                if not self._warned_true_replay:
                    print(f"[cem_trace] warning: exact true candidate replay failed: {exc}", flush=True)
                    self._warned_true_replay = True
                return
        self._true_replay_solver_return(batch_idx, values, ranks, selected_idx, history_size, start_state, goal_state, start_cost, raw_action_dim)

    def _select_solver_return_for_batch(self, batch_idx: int) -> Optional[np.ndarray]:
        seq = self.solver_return_sequence
        if seq is None:
            seq = self.solver_return_action
        arr = _to_numpy(seq)
        if arr is None or arr.size == 0:
            return None
        arr = np.asarray(arr)
        if arr.ndim >= 3 and arr.shape[0] > batch_idx:
            return arr[batch_idx]
        if arr.ndim >= 2 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def _true_replay_solver_return(
        self,
        batch_idx: int,
        values: np.ndarray,
        ranks: np.ndarray,
        selected_idx: int,
        history_size: int,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        start_cost: float,
        raw_action_dim: int,
    ) -> None:
        seq = self._select_solver_return_for_batch(batch_idx)
        if seq is None:
            return
        raw_actions = self._candidate_future_raw_actions(seq, 0, raw_action_dim)
        if raw_actions is None or raw_actions.size == 0:
            return
        try:
            obs, _info = _reset_env_to_state(self.replay_env, start_state, goal_state, self.config.reset_state_tol)
            done = False
            info = {}
            max_raw_steps = int(self.config.horizon_blocks * self.config.action_block)
            first_block_cost = float("nan")
            executed_steps = 0
            for step_idx, action in enumerate(raw_actions[:max_raw_steps]):
                obs, _reward, done, info = _step_env(self.replay_env, action)
                executed_steps = step_idx + 1
                if executed_steps == int(self.config.action_block):
                    first_state = np.asarray(_get_env_state(self.replay_env, obs=obs, info=info), dtype=np.float64)[: goal_state.size]
                    first_block_cost = float(task_cost(first_state, goal_state))
                if done:
                    break
            terminal_state = np.asarray(_get_env_state(self.replay_env, obs=obs, info=info), dtype=np.float64)[: goal_state.size]
            terminal_cost = float(task_cost(terminal_state, goal_state))
            predicted_cost = float(values[selected_idx]) if 0 <= selected_idx < values.shape[0] else float("nan")
            self.true_replay_rows.append(
                {
                    "episode_id": int(self._episode_id(batch_idx)),
                    "mpc_step_idx": int(self.mpc_step_idx),
                    "raw_env_step": int(self._raw_env_step()),
                    "candidate_idx": -1,
                    "candidate_type": "solver_return",
                    "predicted_terminal_cost_c25": predicted_cost,
                    "predicted_cost_source": "matched_sampled_candidate_if_match_distance_small_else_proxy",
                    "rank_by_predicted_cost": int(ranks[selected_idx]) if 0 <= selected_idx < ranks.shape[0] else -1,
                    "selected_flag": 1,
                    "elite_flag": int(values[selected_idx] <= np.sort(values)[max(0, min(len(values) - 1, self._elite_count(len(values))[0] - 1))]) if 0 <= selected_idx < values.shape[0] else 0,
                    "true_terminal_task_cost": terminal_cost,
                    "true_terminal_progress": float(start_cost - terminal_cost),
                    "true_start_task_cost": start_cost,
                    "true_progress_delta": float(start_cost - terminal_cost),
                    "first_block_progress_delta": float(start_cost - first_block_cost) if math.isfinite(first_block_cost) else float("nan"),
                    "executed_raw_steps": int(executed_steps),
                    "expected_raw_steps": max_raw_steps,
                    "done": int(bool(done)),
                    "done_reason": _done_reason_summary(info),
                    "terminal_state": json.dumps(_jsonable(terminal_state)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            if not self._warned_true_replay:
                print(f"[cem_trace] warning: solver_return true replay failed: {exc}", flush=True)
                self._warned_true_replay = True

    def _write_metadata(self, cfg: Any, model: Any, policy: Any) -> None:
        payload = {
            "trace_config": _jsonable(self.config.__dict__),
            "discovered_paths": {
                "eval_entry_point": "eval.py:run",
                "policy_wrapper": f"{policy.__class__.__module__}.{policy.__class__.__name__}",
                "cem_solver": getattr(getattr(policy, "solver", None), "__class__", type(None)).__module__
                + "."
                + getattr(getattr(policy, "solver", None), "__class__", type(None)).__name__,
                "plan_config": "stable_worldmodel.PlanConfig",
                "model_rollout": f"{model.__class__.__module__}.{model.__class__.__name__}.rollout",
                "model_get_cost": f"{model.__class__.__module__}.{model.__class__.__name__}.get_cost",
                "model_criterion": f"{model.__class__.__module__}.{model.__class__.__name__}.criterion",
            },
            "known_limitations": [
                "The hook is read-only and records model.get_cost calls made by the actual CEM solver.",
                "Candidate true replay is best-effort: it resets a fresh PushT env from the state in the policy info and replays candidate future actions if action de-normalization is inferable.",
                "selected_candidate_idx is matched to the solver return/action prefix when possible, otherwise it falls back to argmin final terminal cost.",
                "For vectorized eval, episode_id is the vectorized batch slot.",
            ],
        }
        if cfg is not None:
            try:
                from omegaconf import OmegaConf

                payload["eval_cfg"] = OmegaConf.to_container(cfg, resolve=True)
            except Exception:
                payload["eval_cfg"] = str(cfg)
        with (self.trace_dir / "trace_metadata.json").open("w") as file:
            json.dump(_jsonable(payload), file, indent=2)

    def _raw_env_step(self) -> int:
        return max(0, int(self.raw_call_idx))

    def _begin_policy_call(self, info_dict: Dict[str, Any]) -> None:
        self.raw_call_idx += 1
        self.current_info = info_dict
        self.current_raw_action = None
        self.active_plan = None
        self.solver_return_action = None
        self.solver_return_sequence = None

    def _start_plan_if_needed(self, cost: np.ndarray) -> None:
        if self.active_plan is not None:
            return
        self.mpc_step_idx += 1
        traced_batch = min(int(cost.shape[0]), int(self.config.trace_episodes))
        state_rows = _vector_rows((self.current_info or {}).get("state"))
        goal_rows = _vector_rows((self.current_info or {}).get("goal_state"))
        current_costs = [_safe_task_cost(state_rows, goal_rows, i) for i in range(traced_batch)]
        for batch_idx in range(traced_batch):
            previous_idx = self.last_exec_row_by_batch.get(batch_idx)
            if previous_idx is not None and previous_idx < len(self.exec_rows):
                before = float(self.exec_rows[previous_idx].get("true_task_cost_before_execution", float("nan")))
                after = current_costs[batch_idx]
                self.exec_rows[previous_idx]["true_task_cost_after_execution"] = after
                self.exec_rows[previous_idx]["true_progress_delta"] = before - after if math.isfinite(before) and math.isfinite(after) else float("nan")
                self.exec_rows[previous_idx]["progress_improved_after_first_block"] = (
                    int(after < before) if math.isfinite(before) and math.isfinite(after) else ""
                )

        self.active_plan = {
            "cem_iter": -1,
            "traced_batch": traced_batch,
            "current_costs": current_costs,
            "final": None,
        }

    def _get_cost_wrapper(self, info_dict: Dict[str, Any], action_candidates: torch.Tensor):
        cost_tensor = self.original_get_cost(info_dict, action_candidates)
        cost = _to_numpy(cost_tensor)
        if cost is None:
            return cost_tensor
        cost = np.asarray(cost, dtype=np.float64)
        if cost.ndim == 1:
            cost = cost[None]
        self._start_plan_if_needed(cost)
        if self.active_plan is None:
            return cost_tensor
        if self.mpc_step_idx >= int(self.config.trace_max_steps):
            return cost_tensor
        self.active_plan["cem_iter"] += 1
        cem_iter = int(self.active_plan["cem_iter"])
        traced_batch = int(self.active_plan["traced_batch"])
        actions = _to_numpy(action_candidates)
        pred_emb = _to_numpy(info_dict.get("predicted_emb"))
        goal_emb = _to_numpy(info_dict.get("goal_emb"))
        self._record_iteration(cost, cem_iter, traced_batch)
        self.active_plan["final"] = {
            "cost": cost,
            "actions": actions,
            "predicted_emb": pred_emb,
            "goal_emb": goal_emb,
            "cem_iter": cem_iter,
        }
        return cost_tensor

    def _record_iteration(self, cost: np.ndarray, cem_iter: int, traced_batch: int) -> None:
        for batch_idx in range(traced_batch):
            values = cost[batch_idx]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            elite_count, elite_count_source = self._elite_count(int(finite.size))
            elite = np.sort(finite)[:elite_count]
            dist_stats = self._distribution_stats()
            iter_path = self.iter_dir / f"cem_costs_ep{self._episode_id(batch_idx)}_step{self.mpc_step_idx}_iter{cem_iter}.npz"
            np.savez_compressed(iter_path, terminal_cost_c25=values.astype(np.float32))
            self.iter_rows.append(
                {
                    "episode_id": int(self._episode_id(batch_idx)),
                    "mpc_step_idx": int(self.mpc_step_idx),
                    "raw_env_step": int(self._raw_env_step()),
                    "cem_iter": int(cem_iter),
                    "candidate_count": int(values.shape[0]),
                    "elite_count": int(elite_count),
                    "elite_count_source": elite_count_source,
                    "best_cost": float(np.min(finite)),
                    "mean_cost": float(np.mean(finite)),
                    "median_cost": float(np.median(finite)),
                    "std_cost": float(np.std(finite)),
                    "min_cost": float(np.min(finite)),
                    "max_cost": float(np.max(finite)),
                    "elite_mean_cost": float(np.mean(elite)),
                    "elite_max_cost": float(np.max(elite)),
                    "selected_best_idx_this_iter": int(np.nanargmin(values)),
                    "distribution_mean_norm_before": dist_stats["distribution_mean_norm"],
                    "distribution_std_mean_before": dist_stats["distribution_std_mean"],
                    "distribution_std_min_before": dist_stats["distribution_std_min"],
                    "distribution_std_max_before": dist_stats["distribution_std_max"],
                    "distribution_mean_norm_after": dist_stats["distribution_mean_norm"],
                    "distribution_std_mean_after": dist_stats["distribution_std_mean"],
                    "iteration_cost_npz": str(iter_path),
                }
            )

    def _finish_policy_call(self, action: Any) -> None:
        self.current_raw_action = _to_numpy(action)
        if self.active_plan is None:
            return
        if self.mpc_step_idx >= int(self.config.trace_max_steps):
            return
        final = self.active_plan.get("final")
        if final is None:
            return
        self._save_final_candidate_pools(final)
        self._record_executed_action(action, final)

    def _save_final_candidate_pools(self, final: Dict[str, Any]) -> None:
        cost = np.asarray(final["cost"], dtype=np.float64)
        actions = final.get("actions")
        pred_emb = final.get("predicted_emb")
        goal_emb = final.get("goal_emb")
        traced_batch = min(int(cost.shape[0]), int(self.config.trace_episodes))
        state_rows = _vector_rows((self.current_info or {}).get("state"))
        goal_rows = _vector_rows((self.current_info or {}).get("goal_state"))
        for batch_idx in range(traced_batch):
            values = cost[batch_idx]
            ranks = _rank_1_based(values)
            best_idx = int(np.nanargmin(values))
            selected_idx, selection_rule, selection_distance = self._infer_selected_candidate(actions, batch_idx, best_idx)
            terminal = None
            z0 = None
            zgoal = None
            rollout = None
            history_size = 1
            if pred_emb is not None:
                rollout = pred_emb[batch_idx]
                pixels = (self.current_info or {}).get("pixels")
                pixels_np = _to_numpy(pixels)
                if pixels_np is not None and pixels_np.ndim >= 3:
                    history_size = int(pixels_np.shape[2]) if pixels_np.ndim >= 6 else 1
                terminal = rollout[:, -1, :]
                z0 = rollout[:, max(0, min(history_size - 1, rollout.shape[1] - 1)), :]
            if goal_emb is not None:
                goal_arr = goal_emb[batch_idx]
                zgoal = goal_arr[-1] if goal_arr.ndim == 2 else goal_arr.reshape(-1)
            pool_path = self.pool_dir / f"candidate_pool_ep{self._episode_id(batch_idx)}_step{self.mpc_step_idx}.npz"
            save_payload = {
                "terminal_cost_c25": values.astype(np.float32),
                "rank_by_terminal_cost": ranks.astype(np.int64),
                "best_idx_argmin_cost": np.asarray(best_idx, dtype=np.int64),
                "selected_candidate_idx": np.asarray(selected_idx, dtype=np.int64),
            }
            if actions is not None:
                action_batch = self._actions_for_batch(actions, batch_idx)
                if action_batch is not None:
                    save_payload["action_sequence"] = np.asarray(action_batch).astype(np.float32)
            solver_return_sequence = self._select_solver_return_for_batch(batch_idx)
            if solver_return_sequence is not None:
                save_payload["solver_return_sequence"] = np.asarray(solver_return_sequence).astype(np.float32)
            if self.current_raw_action is not None:
                current_action = _to_numpy(self.current_raw_action)
                if current_action is not None:
                    if current_action.ndim >= 2 and current_action.shape[0] > batch_idx:
                        save_payload["executed_first_action"] = np.asarray(current_action[batch_idx]).astype(np.float32)
                    else:
                        save_payload["executed_first_action"] = np.asarray(current_action).astype(np.float32)
            if state_rows is not None and batch_idx < state_rows.shape[0]:
                save_payload["current_state"] = np.asarray(state_rows[batch_idx]).astype(np.float32)
            if goal_rows is not None and batch_idx < goal_rows.shape[0]:
                save_payload["goal_state"] = np.asarray(goal_rows[batch_idx]).astype(np.float32)
            if terminal is not None:
                save_payload["predicted_terminal_latent"] = terminal.astype(np.float32)
            if self.config.save_full_rollouts and rollout is not None:
                save_payload["predicted_rollout_latents"] = rollout.astype(np.float32)
            if z0 is not None:
                save_payload["start_latent"] = z0.astype(np.float32)
            if zgoal is not None:
                save_payload["goal_latent"] = zgoal.astype(np.float32)
            np.savez_compressed(pool_path, **save_payload)

            finite_values = values[np.isfinite(values)]
            elite_count, elite_count_source = self._elite_count(int(finite_values.size)) if finite_values.size else (1, "empty_cost_fallback")
            elite_threshold = float(np.sort(finite_values)[elite_count - 1]) if finite_values.size else float("nan")
            selected_cost = float(values[selected_idx])
            current_cost = _safe_task_cost(state_rows, goal_rows, batch_idx)
            self._true_replay_candidates(batch_idx, values, ranks, selected_idx, actions, history_size, state_rows, goal_rows, elite_threshold)
            self.planning_rows.append(
                {
                    "episode_id": int(self._episode_id(batch_idx)),
                    "mpc_step_idx": int(self.mpc_step_idx),
                    "raw_env_step": int(self._raw_env_step()),
                    "model_name": self.config.model_name,
                    "latent_dim": int(self.config.latent_dim),
                    "horizon_blocks": int(self.config.horizon_blocks),
                    "action_block": int(self.config.action_block),
                    "planning_horizon_raw": int(self.config.horizon_blocks * self.config.action_block),
                    "receding_horizon_raw": int(self.config.receding_horizon_raw),
                    "num_candidates": int(values.shape[0]),
                    "num_elites": int(elite_count),
                    "elite_count_source": elite_count_source,
                    "num_cem_iterations": int(final["cem_iter"] + 1),
                    "selected_candidate_idx": int(selected_idx),
                    "selected_candidate_source": selection_rule,
                    "selected_candidate_match_distance": selection_distance,
                    "solver_return_sequence_saved": int(solver_return_sequence is not None),
                    "selected_rank_by_cost": int(ranks[selected_idx]),
                    "selected_predicted_cost": selected_cost,
                    "best_predicted_cost": float(values[best_idx]),
                    "final_elite_mean_cost": float(np.mean(np.sort(finite_values)[:elite_count])) if finite_values.size else float("nan"),
                    "current_task_cost": current_cost,
                    "current_task_progress": float("nan"),
                    "success_flag_so_far": "",
                    "goal_index_or_offset": "",
                }
            )
            self.timeline_rows.append(
                {
                    "episode_id": int(self._episode_id(batch_idx)),
                    "mpc_step_idx": int(self.mpc_step_idx),
                    "raw_env_step": int(self._raw_env_step()),
                    "task_cost": current_cost,
                    "task_progress": float("nan"),
                    "selected_predicted_cost": selected_cost,
                    "selected_true_progress_delta": float("nan"),
                    "success_flag": "",
                    "failure_flag": "",
                }
            )
            for candidate_idx, candidate_cost in enumerate(values):
                is_elite = bool(candidate_cost <= elite_threshold)
                is_selected = candidate_idx == selected_idx
                self.pool_rows.append(
                    {
                        "episode_id": int(self._episode_id(batch_idx)),
                        "mpc_step_idx": int(self.mpc_step_idx),
                        "candidate_idx": int(candidate_idx),
                        "terminal_cost_c25": float(candidate_cost),
                        "rank_by_terminal_cost": int(ranks[candidate_idx]),
                        "is_elite_final": int(is_elite),
                        "is_selected_final": int(is_selected),
                        "selected_candidate_source": selection_rule if is_selected else "",
                        "solver_return_sequence_saved": int(solver_return_sequence is not None),
                        "candidate_pool_npz": str(pool_path),
                        "predicted_terminal_latent_norm": float(np.linalg.norm(terminal[candidate_idx])) if terminal is not None else float("nan"),
                        "start_latent_norm": float(np.linalg.norm(z0[candidate_idx])) if z0 is not None else float("nan"),
                        "goal_latent_norm": float(np.linalg.norm(zgoal)) if zgoal is not None else float("nan"),
                    }
                )
            if terminal is not None and z0 is not None and zgoal is not None:
                self._record_angle_scores(batch_idx, values, ranks, selected_idx, terminal, z0, zgoal)

    def _record_angle_scores(
        self,
        batch_idx: int,
        values: np.ndarray,
        ranks: np.ndarray,
        best_idx: int,
        terminal: np.ndarray,
        z0: np.ndarray,
        zgoal: np.ndarray,
    ) -> None:
        q = zgoal[None, :] - z0
        d = terminal - z0
        cos = _cosine(d, q)
        norm = np.sum(d * d, axis=-1)
        align = -2.0 * np.sum(q * d, axis=-1)
        q_norm = np.sum(q * q, axis=-1)
        decomposed_cost = q_norm + norm + align
        for candidate_idx in range(values.shape[0]):
            self.angle_rows.append(
                {
                    "episode_id": int(self._episode_id(batch_idx)),
                    "mpc_step_idx": int(self.mpc_step_idx),
                    "candidate_idx": int(candidate_idx),
                    "terminal_cost_c25": float(values[candidate_idx]),
                    "rank_by_terminal_cost": int(ranks[candidate_idx]),
                    "is_selected_final": int(candidate_idx == best_idx),
                    "cos_to_goal_direction": float(cos[candidate_idx]),
                    "transition_norm_sq": float(norm[candidate_idx]),
                    "alignment_term": float(align[candidate_idx]),
                    "goal_displacement_norm_sq": float(q_norm[candidate_idx]),
                    "decomposed_terminal_cost": float(decomposed_cost[candidate_idx]),
                    "decomposition_abs_error": float(abs(decomposed_cost[candidate_idx] - values[candidate_idx])),
                }
            )

    def _record_executed_action(self, action: Any, final: Dict[str, Any]) -> None:
        action_np = _to_numpy(action)
        if action_np is None:
            return
        if action_np.ndim == 1:
            action_np = action_np[None]
        cost = np.asarray(final["cost"], dtype=np.float64)
        traced_batch = min(int(cost.shape[0]), int(self.config.trace_episodes), int(action_np.shape[0]))
        state_rows = _vector_rows((self.current_info or {}).get("state"))
        goal_rows = _vector_rows((self.current_info or {}).get("goal_state"))
        for batch_idx in range(traced_batch):
            values = cost[batch_idx]
            best_idx = int(np.nanargmin(values))
            actions = final.get("actions")
            selected_idx, selection_rule, selection_distance = self._infer_selected_candidate(actions, batch_idx, best_idx)
            row = {
                "episode_id": int(self._episode_id(batch_idx)),
                "mpc_step_idx": int(self.mpc_step_idx),
                "raw_env_step": int(self._raw_env_step()),
                "selected_candidate_idx": int(selected_idx),
                "argmin_candidate_idx": int(best_idx),
                "selection_rule": selection_rule,
                "selected_candidate_match_distance": selection_distance,
                "selected_terminal_cost_c25": float(values[selected_idx]),
                "selected_rank_by_cost": int(_rank_1_based(values)[selected_idx]),
                "executed_action": json.dumps(_jsonable(action_np[batch_idx])),
                "true_task_cost_before_execution": _safe_task_cost(state_rows, goal_rows, batch_idx),
                "true_task_cost_after_execution": float("nan"),
                "true_progress_delta": float("nan"),
                "progress_improved_after_first_block": "",
            }
            self.last_exec_row_by_batch[batch_idx] = len(self.exec_rows)
            self.exec_rows.append(row)

    def close(self, metrics: Any = None) -> None:
        self._write_audit_aliases()
        _write_csv(self.trace_dir / "planning_step_metadata.csv", self.planning_rows)
        _write_csv(self.trace_dir / "cem_iteration_summary.csv", self.iter_rows)
        _write_csv(self.trace_dir / "candidate_pool_summary.csv", self.pool_rows)
        _write_csv(self.trace_dir / "executed_step_trace.csv", self.exec_rows)
        _write_csv(self.trace_dir / "episode_timeline.csv", self.timeline_rows)
        _write_csv(self.trace_dir / "candidate_angle_scores.csv", self.angle_rows)
        _write_csv(self.trace_dir / "true_replay_candidates.csv", self.true_replay_rows)
        with (self.trace_dir / "trace_run_summary.json").open("w") as file:
            json.dump(
                _jsonable(
                    {
                        "metrics": metrics,
                        "num_planning_rows": len(self.planning_rows),
                        "num_cem_iteration_rows": len(self.iter_rows),
                        "num_candidate_rows": len(self.pool_rows),
                        "num_angle_rows": len(self.angle_rows),
                        "num_true_replay_rows": len(self.true_replay_rows),
                        "expected_min_planning_rows_if_all_active": int(
                            max(0, min(self.config.trace_episodes, self.config.trace_episodes))
                            * max(0, min(self.config.trace_max_steps, self.mpc_step_idx + 1))
                        ),
                    }
                ),
                file,
                indent=2,
            )
        print(f"[cem_trace] wrote trace outputs to {self.trace_dir}", flush=True)

    def _write_audit_aliases(self) -> None:
        exec_by_step = {
            (int(row.get("episode_id", -1)), int(row.get("mpc_step_idx", -1))): row
            for row in self.exec_rows
        }
        full_rows = []
        for row in self.planning_rows:
            key = (int(row.get("episode_id", -1)), int(row.get("mpc_step_idx", -1)))
            exec_row = exec_by_step.get(key, {})
            full_rows.append(
                {
                    "episode_id": row.get("episode_id", ""),
                    "mpc_step_idx": row.get("mpc_step_idx", ""),
                    "raw_env_step": row.get("raw_env_step", ""),
                    "current_task_cost": row.get("current_task_cost", ""),
                    "current_progress": row.get("current_task_progress", ""),
                    "best_predicted_cost": row.get("best_predicted_cost", ""),
                    "elite_mean_predicted_cost": row.get("final_elite_mean_cost", ""),
                    "solver_return_predicted_cost": row.get("selected_predicted_cost", ""),
                    "executed_first_block_true_progress": exec_row.get("true_progress_delta", ""),
                    "task_cost_after_first_block": exec_row.get("true_task_cost_after_execution", ""),
                    "episode_final_success": row.get("episode_final_success", ""),
                }
            )
        _write_csv(self.trace_dir / "full_episode_trace.csv", full_rows)
        _write_csv(self.trace_dir / "candidate_true_replay_audit.csv", self.true_replay_rows)
        _write_csv(self.trace_dir / "solver_return_replay.csv", [row for row in self.true_replay_rows if row.get("candidate_type") == "solver_return"])
        cem_rows = []
        grouped_iters: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for row in self.iter_rows:
            grouped_iters.setdefault((int(row.get("episode_id", -1)), int(row.get("mpc_step_idx", -1))), []).append(row)
            cem_rows.append(
                {
                    "episode_id": row.get("episode_id", ""),
                    "mpc_step_idx": row.get("mpc_step_idx", ""),
                    "raw_env_step": row.get("raw_env_step", ""),
                    "iteration": row.get("cem_iter", ""),
                    "best_predicted_cost": row.get("best_cost", ""),
                    "elite_mean_predicted_cost": row.get("elite_mean_cost", ""),
                    "cost_mean": row.get("mean_cost", ""),
                    "cost_std": row.get("std_cost", ""),
                    "cost_min": row.get("min_cost", ""),
                    "cost_max": row.get("max_cost", ""),
                    "distribution_mean_norm": row.get("distribution_mean_norm_after", ""),
                    "distribution_std_mean": row.get("distribution_std_mean_after", ""),
                }
            )
        for row in cem_rows:
            group = grouped_iters.get((int(row.get("episode_id", -1)), int(row.get("mpc_step_idx", -1))), [])
            ordered = sorted(group, key=lambda item: int(item.get("cem_iter", 0)))
            best = [float(item.get("best_cost", "nan")) for item in ordered]
            row["best_predicted_cost_decreases_over_iterations"] = int(best[-1] <= best[0]) if len(best) >= 2 and all(math.isfinite(v) for v in best) else ""
        _write_csv(self.trace_dir / "cem_iteration_audit.csv", cem_rows)
        _write_csv(self.trace_dir / "planning_call_failure_types.csv", self._failure_type_rows(exec_by_step))
        self._write_failure_summary()

    def _failure_type_rows(self, exec_by_step: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for row in self.true_replay_rows:
            grouped.setdefault((int(row.get("episode_id", -1)), int(row.get("mpc_step_idx", -1))), []).append(row)
        threshold = 0.01
        for key, group in sorted(grouped.items()):
            solver = [row for row in group if row.get("candidate_type") == "solver_return"]
            solver_row = solver[0] if solver else {}
            solver_progress = float(solver_row.get("true_terminal_progress", float("nan")))
            solver_pred = float(solver_row.get("predicted_terminal_cost_c25", float("nan")))
            progress = np.asarray([float(row.get("true_terminal_progress", float("nan"))) for row in group], dtype=np.float64)
            pred = np.asarray([float(row.get("predicted_terminal_cost_c25", float("nan"))) for row in group], dtype=np.float64)
            candidate_types = [str(row.get("candidate_type", "")) for row in group]
            support_failure = not bool(np.any(progress > threshold))
            better = progress > solver_progress + threshold if math.isfinite(solver_progress) else np.zeros_like(progress, dtype=bool)
            ranking_failure = bool(np.any(better & (pred > solver_pred))) if math.isfinite(solver_pred) else False
            low_pred = solver_pred <= np.nanpercentile(pred, 10) if np.any(np.isfinite(pred)) and math.isfinite(solver_pred) else False
            imagination_failure = bool(low_pred and (not math.isfinite(solver_progress) or solver_progress <= threshold))
            sampled_good = any((typ in {"argmin", "elite"} and prog > threshold) for typ, prog in zip(candidate_types, progress))
            mean_sequence_failure = bool(sampled_good and (not math.isfinite(solver_progress) or solver_progress <= threshold))
            exec_row = exec_by_step.get(key, {})
            first_delta = float(exec_row.get("true_progress_delta", float("nan"))) if exec_row else float("nan")
            rows.append(
                {
                    "episode_id": key[0],
                    "mpc_step_idx": key[1],
                    "support_failure": int(support_failure),
                    "ranking_failure": int(ranking_failure),
                    "imagination_failure": int(imagination_failure),
                    "mean_sequence_failure": int(mean_sequence_failure),
                    "first_block_failure": int(first_delta < 0) if math.isfinite(first_delta) else "",
                    "replan_compounding": "",
                    "solver_return_true_progress": solver_progress,
                    "solver_return_predicted_cost": solver_pred,
                    "best_replayed_true_progress": float(np.nanmax(progress)) if progress.size else float("nan"),
                    "best_replayed_predicted_cost": float(pred[int(np.nanargmax(progress))]) if progress.size and np.any(np.isfinite(progress)) else float("nan"),
                }
            )
        return rows

    def _write_failure_summary(self) -> None:
        path = self.trace_dir / "planning_call_failure_types.csv"
        rows = []
        if path.exists() and path.stat().st_size:
            with path.open() as file:
                rows = list(csv.DictReader(file))
        lines = ["# State8 CEM Failure Episode Summary", ""]
        lines.append(f"Planning calls with true replay: {len(rows)}")
        for key in ("support_failure", "ranking_failure", "imagination_failure", "mean_sequence_failure", "first_block_failure"):
            vals = [int(row[key]) for row in rows if str(row.get(key, "")).strip() in {"0", "1"}]
            lines.append(f"- {key}: {sum(vals)}/{len(vals)}")
        lines.append("")
        lines.append("Notes: solver_return predicted cost is exact only if the solver return sequence can be matched or rescored; otherwise the trace records the source/proxy in CSV.")
        (self.trace_dir / "failure_episode_summary.md").write_text("\n".join(lines) + "\n")
