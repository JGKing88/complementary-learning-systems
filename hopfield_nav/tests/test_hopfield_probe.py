"""Tests for ``analysis.hopfield_probe``.

Lives here rather than under the package because ``pyproject.toml`` points
pytest at ``hopfield_nav/tests`` only, and a test the gate does not run is not
a gate. ``tests/`` is exempt from the layering rules, and
``test_phase_decoding.py`` already reaches into ``analysis`` the same way.

What is actually pinned, in rough order of how much would break silently:

1. **Lazy encoding equals the real field.** ``grid_codes`` bit-for-bit against
   ``gen_gbook_2d`` + ``smooth_gbook``, and ``Field`` against a real
   ``VectorHash`` with ``encoded_Phi`` materialised at a tiny ``Npos``.
   Everything downstream inherits this, so it is the first test.
2. **The angle convention.** ``q`` is ``(East, North) = (dx, dy)``. A
   transpose mirrors every angle while leaving every aggregate plausible, so
   this is pinned against hand-built due-East / due-North cases.
3. **The radius convention.** First failure, not last success.
4. **``retr_dist`` is NaN outside the test env**, so a cross-room retrieval can
   never be averaged into a real-space distance.
5. **The goal is absorbing in Test D**, without which reach-rate collapses for
   a reason that is about modelling, not about the encoder.
"""
from __future__ import annotations

import re
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import torch

from analysis.hopfield_probe.attractor import (
    classify_outcomes, first_failure_radius,
)
from analysis.hopfield_probe.encode import Field, grid_codes
from analysis.hopfield_probe.flow import (
    discrete_flow, discrete_successor, terminal_structure,
)
from analysis.hopfield_probe.harness import (
    OUTCOMES, ProbeConfig, local_cells, sample_worlds,
)
from analysis.hopfield_probe.qfield import bearing, project_q, q_error
from analysis.hopfield_probe.stats import BinnedStat, Map2D, wrap_to_pi
from encoder_training.config import EncoderModelConfig
from encoder_training.models import create_encoder
from hopfield_nav.config import VectorHashConfig
from hopfield_nav.utils import smooth_gbook
from hopfield_nav.world.scaffold import VectorHash


def _tiny(lambdas=(2, 3), fwhm=0.25, out_dim=8):
    torch.manual_seed(0)
    npos = int(np.prod(lambdas))
    vh = VectorHash(VectorHashConfig(lambdas=list(lambdas), Np=16, Npos=npos,
                                     static_vectorhash=True))
    vh.build_scaffold()
    mcfg = EncoderModelConfig(lambdas=list(lambdas), out_dim=out_dim,
                              hidden_dim=8, num_hidden_layers=1, gain=2.0)
    enc = create_encoder(mcfg, "cpu")
    enc.eval()
    enc.requires_grad_(False)
    vh.precompute_encoded_phi(enc, fwhm, device="cpu")
    field = Field(enc, list(lambdas), fwhm, 2.0, npos)
    return vh, field, npos


# ---------------------------------------------------------------------------
# 1. Lazy encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lambdas,fwhm", [((2, 3), 0.25), ((3, 4), 0.5),
                                          ((2, 3), 0.0)])
def test_grid_codes_are_bit_identical_to_the_real_gbook(lambdas, fwhm):
    """The whole package reads positions through this. Not "close" -- equal.

    The float expressions in ``grid_codes`` are written in ``smooth_gbook``'s
    exact form for this reason; an algebraically equal rearrangement would
    round differently in the last bit and this test would be the only thing
    that noticed.
    """
    vh, _field, npos = _tiny(lambdas, fwhm)
    sgb = (smooth_gbook(vh.gbook, vh.lambdas, fwhm) if fwhm > 0
           else vh.gbook.copy())
    ref = sgb.reshape(vh.Ng, npos * npos).T.astype(np.float32)

    xs, ys = np.meshgrid(np.arange(npos), np.arange(npos), indexing="ij")
    got = grid_codes(list(lambdas), xs.ravel(), ys.ravel(), fwhm)
    assert np.array_equal(ref, got)


def test_field_matches_precomputed_encoded_phi():
    vh, field, npos = _tiny()
    xs, ys = np.meshgrid(np.arange(npos), np.arange(npos), indexing="ij")
    got = field.encode(xs.ravel(), ys.ravel())
    ref = vh.encoded_Phi[xs.ravel(), ys.ravel()]
    # float32 epsilon, not bit-equality: the encoder runs in batches and a
    # matmul is not required to associate the same way at every batch size.
    assert np.allclose(ref, got, atol=1e-6)


def test_local_basis_matches_gram_schmidt_projection():
    vh, field, npos = _tiny()
    off = (1, 1)
    loc = np.array([[i, j] for i in range(npos - 2) for j in range(npos - 2)],
                   dtype=np.int32)
    assert np.allclose(vh.gram_schmidt_projection(loc, off),
                       field.local_basis(loc, off), atol=1e-5)
    assert np.allclose(vh.get_encoded_state(loc, off),
                       field.encoded_state(loc, off), atol=1e-6)


def test_swapped_basis_is_orthonormal_and_still_east_north():
    _vh, field, npos = _tiny()
    loc = np.array([[1, 1], [2, 2]], dtype=np.int32)
    W = field.local_basis(loc, (1, 1), swap_gram_schmidt=True)
    assert np.allclose((W * W).sum(-1), 1.0, atol=1e-5)
    assert np.allclose((W[:, 0] * W[:, 1]).sum(-1), 0.0, atol=1e-5)
    # Row 0 must still be the East row: under the swap East is the vector kept
    # exactly, so it is parallel to the raw East displacement.
    gx = np.clip(loc[:, 0] + 1, 1, npos - 2)
    gy = np.clip(loc[:, 1] + 1, 1, npos - 2)
    d_rgt = field.encode(gx + 1, gy) - field.encode(gx, gy)
    d_rgt /= np.linalg.norm(d_rgt, axis=1, keepdims=True)
    assert np.allclose(np.abs((W[:, 0] * d_rgt).sum(-1)), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. The angle convention
# ---------------------------------------------------------------------------

def test_q_is_east_north_and_a_due_east_goal_reads_zero_error():
    """A transpose here mirrors every angle and every aggregate still looks
    plausible. Hand-built, so it cannot."""
    q = np.array([[1.0, 0.0]])                       # points due East
    theta = bearing(np.array([3.0]), np.array([0.0]))  # goal 3 cells East
    assert np.isclose(q_error(q, theta)[0], 0.0, atol=1e-12)

    q_n = np.array([[0.0, 1.0]])                     # due North
    theta_n = bearing(np.array([0.0]), np.array([5.0]))
    assert np.isclose(q_error(q_n, theta_n)[0], 0.0, atol=1e-12)

    # East q against a North goal is a quarter turn, signed negative.
    assert np.isclose(q_error(q, theta_n)[0], -np.pi / 2, atol=1e-12)


def test_project_q_recovers_a_planted_displacement():
    basis = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])   # East, North
    cur = np.array([[0.0, 0.0, 0.0]])
    rec = np.array([[2.0, -3.0, 9.0]])       # 9 is off-plane, must vanish
    assert np.allclose(project_q(basis, cur, rec), [[2.0, -3.0]])


def test_wrap_to_pi_is_half_open_on_the_left():
    """``[-pi, pi)``: exactly opposite reads as -pi, never +pi.

    Only the antipodal case can tell the two apart and every aggregate here
    takes abs, but a sector or sign-of-error plot would notice, so the
    convention is pinned rather than left to the reader.
    """
    assert np.isclose(wrap_to_pi(np.pi), -np.pi)
    assert np.isclose(wrap_to_pi(-np.pi), -np.pi)
    assert np.isclose(wrap_to_pi(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_to_pi(0.3), 0.3)
    assert np.isclose(wrap_to_pi(2 * np.pi + 0.3), 0.3)


# ---------------------------------------------------------------------------
# 3. Radius convention
# ---------------------------------------------------------------------------

def test_radius_stops_at_the_first_failure_not_the_last_success():
    """The condition is not monotone in r, and a radius only means anything if
    the guarantee nests all the way in."""
    dist = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    hit = np.array([True, True, False, True, True])
    # Passes at 0 and 1, fails at 2, would "pass" again at 4 by fraction --
    # the answer is 1.
    assert first_failure_radius(hit, dist, 1.0) == 1.0


def test_radius_is_minus_one_when_the_goal_cell_itself_misses():
    dist = np.array([0.0, 1.0])
    assert first_failure_radius(np.array([False, True]), dist, 1.0) == -1.0


def test_radius_95_tolerates_a_single_miss():
    dist = np.arange(100.0)
    hit = np.ones(100, dtype=bool)
    hit[50] = False
    assert first_failure_radius(hit, dist, 1.0) == 49.0
    assert first_failure_radius(hit, dist, 0.95) > 49.0


# ---------------------------------------------------------------------------
# 4. Outcome classification
# ---------------------------------------------------------------------------

def test_retr_dist_is_nan_outside_the_test_env():
    """Real-space distance across two rooms is not a quantity. Averaging one
    in would manufacture a small, reassuring number."""
    ret_env = np.array([0, 0, 1, -1])
    ret_x = np.array([2, 5, 2, 0])
    ret_y = np.array([2, 2, 2, 0])
    out, dist = classify_outcomes(ret_env, ret_x, ret_y, 0, (2, 2))

    assert out[0] == OUTCOMES.index("exact")
    assert out[1] == OUTCOMES.index("far_same_env")
    assert out[2] == OUTCOMES.index("other_env")
    assert out[3] == OUTCOMES.index("alias")
    assert dist[0] == 0.0
    assert np.isnan(dist[2]) and np.isnan(dist[3])


def test_near_is_split_from_far_same_env():
    ret_env = np.zeros(3, dtype=int)
    out, _ = classify_outcomes(ret_env, np.array([3, 4, 9]),
                               np.array([2, 2, 2]), 0, (2, 2))
    assert out[0] == OUTCOMES.index("near")        # 1 cell away
    assert out[1] == OUTCOMES.index("near")        # 2 cells, the boundary
    assert out[2] == OUTCOMES.index("far_same_env")


# ---------------------------------------------------------------------------
# 5. Flow
# ---------------------------------------------------------------------------

def test_terminal_structure_finds_sinks_and_cycles():
    #  0 -> 1 -> 2 -> 1  (cycle {1,2});  3 -> 3 (sink)
    nxt = np.array([1, 2, 1, 3])
    terminal, cycles = terminal_structure(nxt)
    assert len(cycles) == 2
    assert terminal[0] == terminal[1] == terminal[2]
    assert terminal[3] != terminal[0]
    sizes = sorted(len(c) for c in cycles)
    assert sizes == [1, 2]


def test_goal_is_absorbing_so_a_good_field_reaches_it():
    """Without the absorbing goal, the goal cell's own near-zero q classifies
    to some cardinal and steps the agent straight back off it -- nothing is
    ever terminal at the goal and reach_rate collapses regardless of the
    field's quality."""
    size = 5
    goal = (2, 2)
    cells = local_cells(size)
    delta = (np.array(goal) - cells).astype(float)
    n = np.linalg.norm(delta, axis=1, keepdims=True)
    q = np.divide(delta, n, out=np.zeros_like(delta), where=n > 0)

    res = discrete_flow(q, size, goal)
    assert res["reach_rate"] == 1.0
    assert res["n_success"] == size * size
    assert res["sinks"] == []
    assert res["mean_steps"] > 0


def test_a_field_pointing_the_wrong_way_reaches_nothing_and_reports_a_sink():
    size = 5
    goal = (2, 2)
    cells = local_cells(size)
    delta = (cells - np.array(goal)).astype(float)   # away from the goal
    n = np.linalg.norm(delta, axis=1, keepdims=True)
    q = np.divide(delta, n, out=np.zeros_like(delta), where=n > 0)

    res = discrete_flow(q, size, goal)
    assert res["reach_rate"] < 0.1
    assert res["sinks"], "a field flowing outward must terminate somewhere"


def test_zero_q_is_a_sink_not_a_spurious_cardinal():
    size = 3
    q = np.zeros((size * size, 2))
    nxt = discrete_successor(q, size)
    assert np.array_equal(nxt, np.arange(size * size))


# ---------------------------------------------------------------------------
# 6. Accumulators
# ---------------------------------------------------------------------------

def test_binned_stat_mean_is_exact_and_percentiles_track_the_histogram():
    """Means are exact running sums; percentiles come from a value histogram.

    That approximation is the only one in the package, so its accuracy is
    pinned at the sample size the probes actually produce -- hundreds per bin,
    where a 1-degree histogram resolves the median to a bin width. With two
    samples any histogram percentile has to pick one of them, which says
    nothing about the estimator.
    """
    rng = np.random.RandomState(0)
    lo = rng.uniform(0.0, 60.0, 4000)
    hi = rng.uniform(100.0, 160.0, 4000)
    b = BinnedStat(np.array([0.0, 1.0, 2.0]), "angle_deg")
    b.add(np.full(lo.size, 0.5), lo)
    b.add(np.full(hi.size, 1.5), hi)
    j = b.to_json()

    assert j["n"] == [4000, 4000]
    assert np.isclose(j["mean"][0], lo.mean())        # exact, not binned
    assert np.isclose(j["mean"][1], hi.mean())
    assert abs(j["p50"][0] - np.median(lo)) <= 1.5    # one bin width
    assert abs(j["p50"][1] - np.median(hi)) <= 1.5
    assert abs(j["p25"][0] - np.percentile(lo, 25)) <= 1.5
    assert abs(j["p90"][1] - np.percentile(hi, 90)) <= 1.5


def test_binned_stat_ignores_non_finite_values():
    b = BinnedStat(np.array([0.0, 1.0]), "angle_deg")
    b.add(np.array([0.5, 0.5, 0.5]), np.array([10.0, np.nan, np.inf]))
    assert b.to_json()["n"] == [1]


def test_map2d_pools_a_running_mean():
    m = Map2D((2, 2))
    m.add(np.array([0, 0]), np.array([0, 0]), np.array([1.0, 3.0]))
    j = m.to_json()
    assert j["mean"][0][0] == 2.0
    assert j["n"][0][0] == 2
    assert j["mean"][1][1] is None


# ---------------------------------------------------------------------------
# 7. Config and worlds
# ---------------------------------------------------------------------------

def test_world_is_pinned_at_max_k_so_placement_does_not_move_with_load():
    """Sec 2.3: raising K must raise the memory load, not repack the scaffold.

    The first K envs of a world are identical at every K, so the K axis is load
    alone.
    """
    cfg = ProbeConfig(n_worlds=1, n_envs_per_world=8, k_values=(2, 4, 8),
                      env_size=4, Npos=36, seed=3)
    cfg.validate()
    w = sample_worlds(cfg)[0]
    assert len(w.specs) == 8
    again = sample_worlds(cfg)[0]
    assert [s.to_json() for s in w.specs] == [s.to_json() for s in again.specs]


def test_scored_env_population_is_capped_so_k_is_a_load_axis():
    """Scoring all K envs would make the K axis change *which* envs are
    measured, not just how many memories compete.

    The populations would be nested and growing, so a K-to-K comparison would
    not be like-for-like -- which is enough, at these envs' spread, to make
    exact_hit rise from K=5 to K=10, backwards and entirely an artifact. Above
    ``n_score_envs`` the measured set is identical at every K.
    """
    from analysis.hopfield_probe.harness import scored_envs

    cfg = ProbeConfig(n_envs_per_world=50, n_score_envs=5,
                      k_values=(1, 3, 5, 10, 50), env_size=4, Npos=36)
    assert scored_envs(cfg, 1) == [0]
    assert scored_envs(cfg, 3) == [0, 1, 2]
    # Constant from n_score_envs upward -- the same envs, more memories.
    for k in (5, 10, 50):
        assert scored_envs(cfg, k) == [0, 1, 2, 3, 4]

    # The other modes define a single test env by construction.
    for mode in ("goal+distractors", "same_env_goals"):
        other = ProbeConfig(memory_mode=mode, n_envs_per_world=50,
                            k_values=(5,), env_size=4, Npos=36)
        assert scored_envs(other, 5) == [0]


def test_config_rejects_k_larger_than_the_world():
    cfg = ProbeConfig(n_envs_per_world=4, k_values=(2, 8), env_size=4,
                      Npos=36)
    with pytest.raises(ValueError, match="pins the world size"):
        cfg.validate()


def test_worlds_are_deterministic_in_the_seed():
    a = sample_worlds(ProbeConfig(n_worlds=2, n_envs_per_world=3, env_size=4,
                                  Npos=36, k_values=(3,), seed=11))
    b = sample_worlds(ProbeConfig(n_worlds=2, n_envs_per_world=3, env_size=4,
                                  Npos=36, k_values=(3,), seed=11))
    c = sample_worlds(ProbeConfig(n_worlds=2, n_envs_per_world=3, env_size=4,
                                  Npos=36, k_values=(3,), seed=12))
    assert [s.to_json() for s in a[0].specs] == [s.to_json() for s in b[0].specs]
    assert [s.to_json() for s in a[0].specs] != [s.to_json() for s in c[0].specs]


# ---------------------------------------------------------------------------
# 8. Encoder loading
# ---------------------------------------------------------------------------

def test_missing_fwhm_is_an_error_not_a_default(tmp_path):
    """``validate_config`` accepts a fwhm_ratio and checks only lambdas, so a
    silent default is how an encoder gets evaluated at a smoothing width it was
    never fitted to."""
    from analysis.hopfield_probe.harness import load_probe_encoder

    mcfg = EncoderModelConfig(lambdas=[2, 3], out_dim=8, hidden_dim=8,
                              num_hidden_layers=1, gain=2.0)
    enc = create_encoder(mcfg, "cpu")
    path = tmp_path / "no_fwhm.pt"
    torch.save({"model_config": mcfg.__dict__,
                "model_state_dict": enc.state_dict(), "gain": 2.0}, path)

    with pytest.raises(ValueError, match="fwhm_ratio"):
        load_probe_encoder(str(path))

    _e, _c, gain, fwhm, header = load_probe_encoder(str(path),
                                                    fwhm_override=0.3)
    assert fwhm == 0.3 and gain == 2.0
    assert header["fwhm_was_overridden"] is True


def test_fwhm_is_read_from_train_config(tmp_path):
    from analysis.hopfield_probe.harness import load_probe_encoder

    mcfg = EncoderModelConfig(lambdas=[2, 3], out_dim=8, hidden_dim=8,
                              num_hidden_layers=1, gain=2.0)
    enc = create_encoder(mcfg, "cpu")
    path = tmp_path / "with_fwhm.pt"
    torch.save({"model_config": mcfg.__dict__,
                "train_config": {"fwhm_ratio": 0.4},
                "model_state_dict": enc.state_dict(), "gain": 7.0}, path)

    _e, _c, gain, fwhm, header = load_probe_encoder(str(path))
    assert fwhm == 0.4
    assert gain == 7.0            # top-level "gain" wins, as in production
    assert header["fwhm_was_overridden"] is False


# ---------------------------------------------------------------------------
# 9. End to end
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_suite_runs_and_the_report_builds(tmp_path):
    from analysis.hopfield_probe.attractor import run_test_a
    from analysis.hopfield_probe.controls import run_controls
    from analysis.hopfield_probe.flow import run_test_d
    from analysis.hopfield_probe.harness import write_json
    from analysis.hopfield_probe.qfield import run_tests_bc
    from analysis.hopfield_probe.report.build import build

    _vh, field, npos = _tiny(lambdas=(4, 5), fwhm=0.25, out_dim=16)
    cfg = ProbeConfig(
        Npos=npos, env_size=4, n_worlds=2, n_envs_per_world=3,
        k_values=(1, 3), steps=(1, 2), n_alias=40, n_cont_samples=400,
        n_cont_annulus=100, cos_chunk=64,
    )
    cfg.validate()
    worlds = sample_worlds(cfg)

    payload = {
        "header": {"label": "tiny", "gain": 2.0, "fwhm_ratio": 0.25,
                   "out_dim": 16, "n_params": 1, "lambdas": [4, 5],
                   "unique_radius": None},
        "config": cfg.to_json(),
        "test_a": run_test_a(field, worlds, cfg),
        "test_bc": run_tests_bc(field, worlds, cfg),
        "test_d": run_test_d(field, worlds, cfg),
        "controls": run_controls(field, worlds, cfg),
    }
    write_json(tmp_path / "tiny.json", payload)
    out = build(tmp_path)

    for name in ("index.html", "test_a.html", "test_b.html", "test_c.html",
                 "test_d.html", "controls.html"):
        text = (out / name).read_text()
        assert text.startswith("<!doctype html>")
        assert "<svg" in text
        # A NaN or None in an SVG *attribute* silently drops the mark. Checked
        # on attribute values only -- the prose legitimately says "NaN" when
        # explaining why retr_dist is one.
        bad = re.findall(r'=\"[^\"]*\b(?:NaN|nan|None|undefined)\b[^\"]*\"',
                         text)
        assert not bad, f"{name}: {bad[:3]}"
        # Every SVG must parse, or a browser silently drops the rest of it.
        svgs = re.findall(r"<svg\b.*?</svg>", text, re.S)
        assert svgs, f"{name} has no parseable <svg>"
        for svg in svgs:
            ET.fromstring(svg)

    # The one-file variants: a whole document, and a body-only fragment for a
    # host that supplies its own skeleton. Both must stay self-contained -- a
    # fragment that dropped its <style> would render unthemed.
    whole = (out / "report.html").read_text()
    frag = (out / "report.fragment.html").read_text()
    assert whole.startswith("<!doctype html>")
    assert not frag.lstrip().startswith("<!doctype")
    for f in (whole, frag):
        assert "<style>" in f and "<script>" in f
        assert 'id="tip"' in f
        assert "<link" not in f and 'src="http' not in f   # no external asset
    # Tabs become in-page anchors, and every anchor has a section to land on.
    nav = re.findall(r'<nav class="tabs">(.*?)</nav>', frag, re.S)
    assert nav, "single page has no tab nav"
    targets = re.findall(r'href="#([^"]+)"', nav[0])
    assert targets
    for t in targets:
        assert f'id="{t}"' in frag, f"tab #{t} has no section"


@pytest.mark.slow
def test_every_page_carries_every_encoder(tmp_path):
    """A comparison run must not render one encoder everywhere but the index.

    The test pages were built from ``results[0]`` alone, so a four-encoder run
    showed only the first outside the overview -- which is what the viz spec's
    encoder filter exists to prevent, and is invisible unless something counts
    the blocks.
    """
    from analysis.hopfield_probe.harness import write_json
    from analysis.hopfield_probe.report.build import build

    _vh, field, npos = _tiny(lambdas=(4, 5), fwhm=0.25, out_dim=16)
    cfg = ProbeConfig(Npos=npos, env_size=4, n_worlds=1, n_envs_per_world=3,
                      k_values=(1, 3), steps=(1, 2), n_alias=20,
                      n_cont_samples=200, n_cont_annulus=50, cos_chunk=64)
    cfg.validate()
    worlds = sample_worlds(cfg)

    from analysis.hopfield_probe.attractor import run_test_a
    from analysis.hopfield_probe.qfield import run_tests_bc
    payload = {"config": cfg.to_json(),
               "test_a": run_test_a(field, worlds, cfg),
               "test_bc": run_tests_bc(field, worlds, cfg)}
    labels = ["alpha", "beta-2", "gamma"]
    for lab in labels:
        write_json(tmp_path / f"{lab}.json",
                   {**payload,
                    "header": {"label": lab, "gain": 2.0, "fwhm_ratio": 0.25,
                               "out_dim": 16, "n_params": 1, "lambdas": [4, 5],
                               "unique_radius": None}})
    out = build(tmp_path)

    for name in ("test_a.html", "test_b.html", "test_c.html"):
        text = (out / name).read_text()
        for lab in labels:
            assert f'<div data-encoder="{lab}"' in text, f"{name}: no {lab}"
            assert f'data-encoder="{lab}"' in text
        opts = re.findall(r'data-key="encoder">(.*?)</select>', text, re.S)
        assert opts, f"{name}: no encoder selector"
        for lab in labels:
            assert f'value="{lab}"' in opts[0]

    # The one-file variant: one selector, and a header per encoder so the
    # provenance bar tracks the selection.
    frag = (out / "report.fragment.html").read_text()
    assert len(re.findall(r'data-key="encoder"', frag)) == 1
    for lab in labels:
        assert f'<span class="enc-hdr" data-encoder="{lab}"' in frag
