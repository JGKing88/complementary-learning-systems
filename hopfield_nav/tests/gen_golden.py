"""Generate the golden fixtures in ``tests/golden/``.

    python -m hopfield_nav.tests.gen_golden          # write
    python -m hopfield_nav.tests.gen_golden --check  # compare, write nothing

These files pin the *current* behavior of code that the 2026-08 refactor
rewrites: the three policy-input assembly sites, the Hopfield signal
computation, and per-trial evaluator outcomes. ``test_golden.py`` asserts that
the live code still reproduces them bit-for-bit.

Regeneration is deliberate. If a golden file changes, either the refactor
altered behavior (investigate) or the change is intended (regenerate, and say
so in the commit message). Never regenerate to make a red test go green
without reading the diff -- that is the one thing these files exist to prevent.

What is covered
---------------
observations  The assembled ``rnn_input`` from all three sites, for a sweep of
              AgentConfig channel combinations. The failure mode this catches
              is the silent one: right shape, wrong channel order.
signals       ``q``, the memory mask, and the Gram-Schmidt basis out of
              ``_hopfield_signal_at`` / ``_compute_multistep_q``, per-env and
              shared-Hopfield.
evaluators    Per-trial success *bits* (not aggregate floats) from the three
              headline evaluators. Bits survive phase 4's rebatching; the
              aggregate floats would not, because batching changes RNG order.
long_horizon  The full returned dict of the four evaluators that have no
              per-trial recorder -- realistic, repeat, union_coverage and
              sequential_episodes. Added after an audit found that no test
              executed any of them, while phases 5a and 5c had rewritten code
              inside three.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from hopfield_nav.world.env import GridEnv, make_env
from hopfield import Hopfield
from hopfield_nav.tests.fixtures import (
    RecordingAgent, StubVectorHash, make_collector, make_stub_cfg,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


# ---------------------------------------------------------------------------
# The observation-channel sweep
# ---------------------------------------------------------------------------
#
# Each entry is a name plus make_stub_cfg kwargs. The first four correspond to
# the configurations that actually appear in live runs; the rest isolate one
# channel at a time so a golden diff points at a specific channel rather than
# "something changed".
OBS_CONFIGS: list[tuple[str, dict]] = [
    ("live_discrete_signal_only", dict(
        movement_mode="discrete", input_hopfield_signal=True)),
    ("live_continuous_signal_only", dict(
        movement_mode="continuous", input_hopfield_signal=True)),
    ("live_continuous_full", dict(
        movement_mode="continuous", input_hopfield_signal=True,
        input_encoded_state=True, input_prev_action=True,
        input_prev_reward=True, input_sensory=True)),
    ("live_continuous_goal_in_memory", dict(
        movement_mode="continuous", input_hopfield_signal=True,
        input_goal_in_memory=True)),
    ("encoded_state_only", dict(
        movement_mode="discrete", input_hopfield_signal=False,
        input_encoded_state=True)),
    ("sensory_only", dict(
        movement_mode="discrete", input_hopfield_signal=False,
        input_sensory=True)),
    ("hopfield_raw", dict(
        movement_mode="continuous", input_hopfield_signal=True,
        input_hopfield_raw=True)),
    ("multistep_1", dict(
        movement_mode="continuous", input_hopfield_signal=True,
        input_hopfield_multistep=[1])),
    ("multistep_1_2", dict(
        movement_mode="continuous", input_hopfield_signal=True,
        input_hopfield_multistep=[1, 2])),
    ("prev_action_and_reward", dict(
        movement_mode="discrete", input_hopfield_signal=True,
        input_prev_action=True, input_prev_reward=True)),
    ("all_channels_discrete", dict(
        movement_mode="discrete", input_hopfield_signal=True,
        input_encoded_state=True, input_prev_action=True,
        input_prev_reward=True, input_sensory=True,
        input_goal_in_memory=True)),
]

EMBED_DIM = 8


def _build_hopfields(vh: StubVectorHash, n: int, *, populated: bool,
                     seed: int = 0) -> list[Hopfield]:
    """n independent Hopfields, optionally pre-loaded with fixed patterns."""
    rng = np.random.RandomState(seed)
    hops = []
    for _ in range(n):
        h = Hopfield(EMBED_DIM, beta=1.0, device="cpu")
        if populated:
            for _ in range(3):
                gx, gy = rng.randint(0, vh.Npos), rng.randint(0, vh.Npos)
                h.input_memory(torch.from_numpy(vh.encoded_Phi[gx, gy]).float())
        hops.append(h)
    return hops


def gen_observations() -> dict[str, np.ndarray]:
    """Assembled policy inputs from the rollout main loop and the bootstrap.

    ``RolloutBatch.obs`` is the main loop's copy; the RecordingAgent also sees
    the bootstrap call, which is the last entry it records.
    """
    out: dict[str, np.ndarray] = {}
    for name, kwargs in OBS_CONFIGS:
        for populated in (False, True):
            tag = f"{name}__{'populated' if populated else 'empty'}"
            cfg = make_stub_cfg(batch_envs=3, steps_per_rollout=4, **kwargs)
            collector, agent, vh = make_collector(cfg, EMBED_DIM, seed=0)
            rec = RecordingAgent(agent)
            env = make_env(cfg.env, cfg.agent.movement_mode, seed=1234)
            hops = _build_hopfields(vh, cfg.batch_envs, populated=populated)

            torch.manual_seed(0)
            np.random.seed(0)
            batch = collector.collect_rollout(env, rec, hops, update_idx=1)

            # (B, T, D): the main-loop site, one row per step.
            out[f"obs__{tag}"] = batch.obs.cpu().numpy()
            # (n_calls, B, D): every site, including the bootstrap (last call).
            out[f"calls__{tag}"] = rec.recorded
            out[f"bootstrap_obs__{tag}"] = rec.recorded[-1]
            out[f"bootstrap_value__{tag}"] = batch.bootstrap_value.cpu().numpy()
    return out


def gen_eval_observations() -> dict[str, np.ndarray]:
    """The third assembly site: ``eval.agent_step``, B=1.

    Driven over a fixed action sequence from a fixed start so the trajectory --
    and therefore every observation on it -- is reproducible.
    """
    from hopfield_nav.evaluation.metrics import agent_step

    out: dict[str, np.ndarray] = {}
    device = torch.device("cpu")
    for name, kwargs in OBS_CONFIGS:
        for populated in (False, True):
            tag = f"{name}__{'populated' if populated else 'empty'}"
            cfg = make_stub_cfg(batch_envs=1, steps_per_rollout=4, **kwargs)
            _, agent, vh = make_collector(cfg, EMBED_DIM, seed=0)
            rec = RecordingAgent(agent)
            env = make_env(cfg.env, cfg.agent.movement_mode, seed=1234)
            env.reset()
            env.set_position((1, 1))
            hop = _build_hopfields(vh, 1, populated=populated)[0]

            torch.manual_seed(0)
            np.random.seed(0)
            h_rnn = None
            prev_reward = None
            prev_action = None
            for _ in range(4):
                res = agent_step(
                    rec, env, (0, 0), vh, hop, h_rnn, cfg, device,
                    deterministic=True, goal_local=env.goal_location,
                    goal_in_memory=populated,
                    prev_reward=prev_reward, prev_action=prev_action,
                )
                h_rnn = res["h_rnn"]
                prev_reward = res["next_prev_reward"]
                prev_action = res["next_prev_action"]
            out[f"evalobs__{tag}"] = rec.recorded
    return out


def gen_signals() -> dict[str, np.ndarray]:
    """q, memory mask and Gram-Schmidt basis straight out of the collector."""
    out: dict[str, np.ndarray] = {}
    for movement_mode in ("discrete", "continuous"):
        for shared in (False, True):
            for populated in (False, True):
                tag = (f"{movement_mode}__"
                       f"{'shared' if shared else 'perenv'}__"
                       f"{'populated' if populated else 'empty'}")
                cfg = make_stub_cfg(movement_mode=movement_mode, batch_envs=3)
                collector, _, vh = make_collector(cfg, EMBED_DIM, seed=0)
                B = cfg.batch_envs
                hops = _build_hopfields(vh, B, populated=populated)
                hop_arg = hops[0] if shared else hops

                positions = np.array([[1, 1], [2, 3], [4, 0]], dtype=np.int32)
                emb_np = vh.get_encoded_state(positions, (0, 0))
                emb = torch.from_numpy(emb_np).float()
                signal_dim = 4 if cfg.agent.hopfield_mode == "discrete" else 2

                sig, q, mask, W = collector._hopfield_signal_at(
                    emb_np, emb, positions, (0, 0), hop_arg, shared,
                    signal_dim, cached_W=None, recompute_mask=None,
                )
                out[f"sig__{tag}"] = sig.cpu().numpy()
                out[f"q__{tag}"] = np.asarray(q)
                out[f"mask__{tag}"] = np.asarray(
                    mask.cpu().numpy() if torch.is_tensor(mask) else mask)
                # The Gram-Schmidt basis is None when no Hopfield holds a
                # memory (nothing to project). Record that as a flag rather
                # than an object array, so it is part of the pinned contract.
                out[f"W_is_none__{tag}"] = np.array(int(W is None))
                if W is not None:
                    out[f"W__{tag}"] = np.asarray(W)

                msq = collector._compute_multistep_q(
                    emb_np, emb, hop_arg, shared, W, [1, 2],
                )
                for s, v in sorted(msq.items()):
                    out[f"msq{s}__{tag}"] = np.asarray(v)
    return out


def gen_evaluators() -> dict[str, np.ndarray]:
    """Per-trial outcome records from the three headline evaluators.

    Per-trial records, not aggregate floats: phase 4 rebatches these evaluators
    onto VecEnv, which changes RNG consumption order. Per-trial outcomes under a
    deterministic policy survive that; the aggregates do not. The aggregates are
    stored too, but as a diagnostic -- test_golden.py asserts on the records.
    """
    from hopfield_nav.evaluation import metrics as ev

    out: dict[str, np.ndarray] = {}
    cfg = make_stub_cfg(movement_mode="discrete", batch_envs=4)
    _, agent, vh = make_collector(cfg, EMBED_DIM, seed=0)
    device = torch.device("cpu")
    val_envs = [make_env(cfg.env, "discrete", seed=100 + i) for i in range(2)]
    # The stub has no register_envs(); place the two envs at distinct offsets.
    vh.env_offsets = [(0, 0), (8, 8)]

    for fn_name in ("evaluate_navigation", "evaluate_goal_discovery",
                    "evaluate_exploration"):
        fn = getattr(ev, fn_name)
        torch.manual_seed(0)
        np.random.seed(0)
        records: list[tuple] = []
        res = fn(agent, val_envs, vh, [0, 1], cfg, device,
                 num_trials=4, max_steps=20, n_distractors_list=[0, 2],
                 per_trial=records)
        # (n_trials, n_fields) integer records -- exactly comparable.
        arr = np.array(records, dtype=np.int64)
        out[f"{fn_name}__per_trial"] = arr
        if fn_name == "evaluate_goal_discovery" and arr[:, 6].sum() == 0:
            # Column 6 is n_arrivals. On this small stub world an untrained
            # policy reaches the goal rarely, so this fixture pins the at-goal
            # branch only weakly -- but zero would mean it pins nothing at all,
            # which is how the long-horizon fixture was vacuous on its first
            # attempt. The real coverage of the teleport is
            # test_goal_contract.py::test_goal_discovery_arrivals_are_never_consecutive.
            raise AssertionError(
                "goal_discovery fixture is vacuous: no trial reached the goal, "
                "so the at-goal branch never ran. Widen the goal radius or "
                "raise max_steps.")

        keys, vals = [], []
        for n_dist in sorted(res.keys()):
            for metric in sorted(res[n_dist].keys()):
                keys.append(f"{n_dist}/{metric}")
                vals.append(float(res[n_dist][metric]))
        out[f"{fn_name}__agg_keys"] = np.array(keys)
        out[f"{fn_name}__agg_vals"] = np.array(vals, dtype=np.float64)
    return out


def _flatten(d, prefix: str = "") -> dict[str, str]:
    """Nested result dict -> flat {dotted.key: repr(value)}.

    These evaluators return dicts of dicts with mixed value types (floats,
    ints, lists, tuple keys), so repr of a flattened view is what is both
    storable in an .npz and readable in a diff.
    """
    flat: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        else:
            flat[key] = repr(v)
    return flat


def gen_long_horizon_evaluators() -> dict[str, np.ndarray]:
    """The four evaluators the per-trial goldens do not reach.

    evaluate_realistic, evaluate_repeat, evaluate_union_coverage and
    evaluate_sequential_episodes have no per-trial recorder and were, until
    this fixture existed, executed by no test at all -- while phases 5a and 5c
    rewrote the at-goal handling inside two of them and the whole protocol
    inside a third. Their full returned dict is pinned instead.
    """
    from hopfield_nav.evaluation import metrics as ev

    def _world(n=2):
        # A small grid and a wide goal radius, so an untrained policy actually
        # reaches the goal within the step budget. Without that, the at-goal
        # branch these evaluators hand-rolled -- the code phase 5a rewrote --
        # never executes and the fixture pins nothing. The radius also puts the
        # off-cell store path in scope, which is the other thing that changed.
        cfg = make_stub_cfg(movement_mode="discrete")
        cfg.env.size = 4
        cfg.env.goal_radius = 1.5
        _c, agent, vh = make_collector(cfg, EMBED_DIM, seed=0)
        vh.env_offsets = [(0, 0), (8, 0), (0, 8)][:n]
        envs = [make_env(cfg.env, "discrete", seed=100 + i) for i in range(n)]
        return cfg, agent, vh, envs

    results: dict[str, dict] = {}

    for lock in (False, True):
        cfg, agent, vh, envs = _world()
        torch.manual_seed(0)
        np.random.seed(0)
        results[f"realistic_lock{lock}"] = ev.evaluate_realistic(
            agent, envs, vh, [0, 1], cfg, torch.device("cpu"),
            steps_per_env=120, seed=13, lock_store_after_goal=lock)

    cfg, agent, vh, envs = _world()
    torch.manual_seed(0)
    np.random.seed(0)
    results["repeat"] = ev.evaluate_repeat(
        agent, envs, vh, [0, 1], cfg, torch.device("cpu"),
        n_trials=3, steps_per_env=80, seed=21)

    cfg, agent, vh, envs = _world()
    torch.manual_seed(0)
    np.random.seed(0)
    results["union_coverage"] = ev.evaluate_union_coverage(
        agent, envs, vh, [0, 1], cfg, torch.device("cpu"),
        num_trials=4, max_steps=20)

    # Guard against a silently vacuous fixture. If the agent never reaches the
    # goal, none of the at-goal handling runs and this golden pins nothing --
    # which is exactly the state this file was in when first written.
    for tag in ("realistic_lockFalse", "realistic_lockTrue"):
        reaches = sum(v["n_reaches"] for v in results[tag]["primary"].values())
        if reaches == 0:
            raise AssertionError(
                f"{tag} fixture is vacuous: the agent never reached the goal, "
                f"so the at-goal branch was never exercised. Widen the goal "
                f"radius, shrink the grid, or raise steps_per_env.")
    if results["repeat"]["summary"]["mean_reaches"] <= 0:
        raise AssertionError("repeat fixture is vacuous: no goal reaches")

    for oracle in (False, True):
        cfg, agent, vh, envs = _world(3)
        torch.manual_seed(0)
        np.random.seed(0)
        results[f"sequential_oracle{oracle}"] = ev.evaluate_sequential_episodes(
            agent, envs, vh, [0, 1, 2], cfg, torch.device("cpu"),
            iters_per_block=6, max_steps=12, seed=7,
            oracle_store_at_goal=oracle)

    out: dict[str, np.ndarray] = {}
    for name, res in results.items():
        flat = _flatten(res)
        keys = sorted(flat)
        out[f"{name}__keys"] = np.array(keys)
        out[f"{name}__vals"] = np.array([flat[k] for k in keys])
    return out


GENERATORS = {
    "observations": gen_observations,
    "eval_observations": gen_eval_observations,
    "signals": gen_signals,
    "evaluators": gen_evaluators,
    "long_horizon_evaluators": gen_long_horizon_evaluators,
}


def compare(path: Path, fresh: dict[str, np.ndarray]) -> list[str]:
    """Human-readable differences between a stored golden and fresh output.

    Empty list means bit-identical. Shared by ``--check`` and ``test_golden.py``
    so there is exactly one definition of "matches the golden".

    NaN equals NaN here: several evaluator aggregates are NaN by design when no
    trial succeeded, and that is a stable outcome, not a difference.
    """
    if not path.exists():
        return [f"{path.name}: MISSING (run `python -m hopfield_nav.tests.gen_golden`)"]
    saved = np.load(path, allow_pickle=False)
    problems = []
    for k in sorted(set(saved.files) - set(fresh)):
        problems.append(f"{path.name}[{k}]: in golden, not regenerated")
    for k in sorted(set(fresh) - set(saved.files)):
        problems.append(f"{path.name}[{k}]: newly generated, not in golden")
    for k in sorted(set(saved.files) & set(fresh)):
        a, b = saved[k], np.asarray(fresh[k])
        if a.shape != b.shape:
            problems.append(f"{path.name}[{k}]: shape {a.shape} -> {b.shape}")
        elif a.dtype.kind in "US":
            if not np.array_equal(a, b):
                problems.append(f"{path.name}[{k}]: string values differ")
        elif a.dtype.kind == "f":
            if not np.array_equal(a, b, equal_nan=True):
                d = np.abs(a - b)
                problems.append(
                    f"{path.name}[{k}]: values differ, max|Δ|={np.nanmax(d):.3e}")
        elif not np.array_equal(a, b):
            d = np.abs(a.astype(np.int64) - b.astype(np.int64))
            problems.append(f"{path.name}[{k}]: values differ, max|Δ|={d.max()}")
    return problems


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.npz"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="Compare against the stored goldens; write nothing. "
                        "Exits non-zero on any difference.")
    p.add_argument("--only", nargs="*", choices=sorted(GENERATORS),
                   help="Regenerate only these files.")
    args = p.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(GENERATORS)
    problems: list[str] = []
    for name in names:
        data = GENERATORS[name]()
        path = golden_path(name)
        if args.check:
            problems.extend(compare(path, data))
            print(f"checked {path.name}  ({len(data)} arrays)")
        else:
            np.savez_compressed(path, **data)
            print(f"wrote {path.name}  ({len(data)} arrays, "
                  f"{path.stat().st_size / 1024:.0f} KiB)")

    if problems:
        print("\nDIFFERENCES:")
        for line in problems:
            print("  " + line)
        return 1
    if args.check:
        print("\nall goldens match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
