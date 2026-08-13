"""The layering rules, enforced by walking every module's AST.

Phase 6 gave the tree a direction: `world` knows nothing about anything,
`analysis` may know about everything, and each layer between may only look
downward. That direction is only worth having if something checks it, because
the way it decays is invisible -- one convenient import, and the next person
finds a cycle they cannot break without moving three files.

Five rules:

1. **No upward imports at module scope.** A module may import from its own
   layer or below, never above. This is what stopped `rollout.collector`
   importing `RolloutBatch` from `ppo`, and `train_rnn` importing a matplotlib
   figure helper at module scope. A *function-scoped* upward import is a
   weaker thing -- it creates no import-time dependency and cannot form an
   import cycle -- so it is allowed, but only from `DEFERRED_UPWARD_IMPORTS`,
   so the set stays small and visible instead of growing quietly.
2. **`encoder_training` never imports `hopfield_nav`.** The two packages are
   siblings, and the encoder side has to stay usable without the navigation
   stack -- `encoder_training/sweep.py` runs on nodes that never build a
   `GridEnv`. The one experiment that violated this moved to
   `analysis/scaffold_experiments/`.
3. **No cross-module private imports.** `from x import _y` says the author of
   `x` did not mean `_y` to be depended on. Either it is public or it is not.
4. **Figure code lives in `analysis/`.** A module-scope `import matplotlib`
   outside `analysis/` means a figure generator is sitting in a library
   package. Phase 6 initially left three there — `visualize_trajectories`,
   `viz_sensory` and `encoder_training/plot_sweep` — because they were sorted
   by "has a `__main__` guard" rather than by what they produce. This rule is
   the reason that cannot recur.
5. **Nothing imports a CLI.** The layer-7 modules are programs. When
   `analysis/continual/baseline.py` imported `build_envs` out of `train_rnn`,
   the figure pipeline depended on a training entry point while `train_rnn`
   deferred an import back into `analysis.continual` — a mutual dependency that
   rules 1–3 all permit, because 8 → 7 is downward and the 7 → 8 edge was
   declared. Shared setup goes in `training/`, which is what
   `training/rnn_setup.py` and `training/world_setup.py` are for.

6. **Only `cls_paths` names an output directory.** Building a path out of the
   literal `"checkpoint"`, `"encoders"`, `"histories"`, … is how all five
   trainers came to ignore `CLS_RUNS`: the string is *relative*, so it resolved
   against the CWD and then through a repo symlink to the default root, whatever
   the environment said. The failure was invisible for months because from the
   repo root the answer is right. This rule is scoped to path *construction* --
   `os.path.join`, `Path(...)`, `/` and `+` -- so a reader comparing
   `parts[0] == "encoders"`, or an argparse `help=` string quoting a default,
   is untouched.

The layer numbers are the whole specification; a new package that is not in the
table fails rule 1 loudly rather than being silently exempt.

`tests/` is exempt from all five: a characterization test that pins
`env._at_goal_l2` is doing its job, and a test importing across layers is not a
dependency anything ships.

Every rule is mutation-verified -- an upward import in `world/env.py`, an
`encoder_training -> hopfield_nav` import, a private import from a lower layer,
a module-scope matplotlib import in `train.py`, an `analysis -> train_rnn`
import, and `os.path.join("checkpoint", sub)` restored in `train_navigate`
each fail exactly the rule they should, and nothing else.
"""
from __future__ import annotations

import ast
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Longest prefix wins. Lower number = deeper in the stack = fewer dependencies.
LAYERS: dict[str, int] = {
    # Leaves: no intra-repo dependencies at all, so everything may use them.
    "hopfield_nav.config": 0,
    "hopfield_nav.utils": 0,
    "hopfield_nav.encoder_io": 0,     # the one deliberate edge to encoder_training
    "encoder_training": 0,
    "gridcode": 0,                     # the live remnant of cls/vectorhash, phase 7
    "hopfield": 0,                     # the memory model, shared by both stacks
    "cls_paths": 0,
    "run_manifest": 0,                 # run.json: written by the CLIs, read by analysis

    "hopfield_nav.world": 1,           # env, vec_env, scaffold, memory, episode
    "hopfield_nav.policy": 2,          # agent, agent_rnn, channels
    "hopfield_nav.rollout": 3,         # collector, rnn, signal, oracles, distractors, types
    "hopfield_nav.updates": 4,         # ppo, bc, bc_rnn
    "hopfield_nav.evaluation": 5,      # metrics, rnn, protocols, batched, checkpoint_io
    "hopfield_nav.training": 6,        # world_setup, rnn_sequential

    # The CLIs. They wire the layers together; rule 5 keeps them unimported.
    "hopfield_nav.train": 7,
    "hopfield_nav.train_phased": 7,
    "hopfield_nav.train_navigate": 7,
    "hopfield_nav.train_store": 7,
    "hopfield_nav.train_rnn": 7,
    "hopfield_nav.eval_all": 7,
    # Offline probes of the inputs a run depends on. Entry points like the
    # trainers -- they wire world + rollout + evaluation together and are only
    # ever run with `python -m`, never imported -- so they take the CLI layer
    # and rule 5 keeps them that way.
    "hopfield_nav.diagnostics": 7,

    "analysis": 8,                     # figure + experiment pipelines
}

CLI_LAYER = 7

# `hopfield_nav` itself is only the namespace package; a bare `import
# hopfield_nav` carries no dependency.
NAMESPACE_ONLY = {"hopfield_nav"}

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "docs", "tests"}

# Privates that are imported across modules and should not be. Empty by intent:
# every entry here is a thing phase 6 was supposed to promote or inline. Add a
# row only with a reason, and prefer fixing the import.
PRIVATE_IMPORT_ALLOWLIST: set[tuple[str, str, str]] = set()

# Packages exempt from rule 4 (figure code lives in analysis/). Empty since
# phase 7: the only entry was `cls`, whose eight module-scope matplotlib imports
# went with the package. `encoder_training/nav_eval` inherited one of them and
# now imports lazily inside the single branch that plots.
FIGURE_RULE_EXEMPT: tuple[str, ...] = ()

# (importer, target) pairs allowed to import upward *inside a function body*.
# Each one is a deliberate inversion: the lower layer hands off to the higher
# one at a leaf of the call graph, and paying for it at import time would drag
# the whole upper layer into every process that touches the lower one.
DEFERRED_UPWARD_IMPORTS: set[tuple[str, str]] = {
    # train_rnn writes its two forgetting plots at the end of a sequential run.
    # At module scope this would make matplotlib -- and the entire figure
    # stack -- an import-time dependency of every training job, including the
    # headless sbatch ones that never draw anything.
    ("hopfield_nav.train_rnn", "analysis.continual.plotting"),
}


# Directory names owned by cls_paths. Each is the first component of a path
# under CLS_RUNS, and each existed as an in-repo directory before the 2026-08
# storage migration left it behind as a symlink -- which is exactly why a
# hardcoded literal still resolves and the mistake is silent.
RESERVED_OUTPUT_DIRS = frozenset({
    "checkpoint", "checkpoints", "checkpoint_rnn", "agent_ckpts",
    "agent_ckpts_legacy", "encoders", "histories", "scaffold_cache",
    "results", "figures", "sweeps", "logs",
})

# Modules allowed to build a path from one of those names. `cls_paths` *is* the
# definition of the layout, so it is the only one. Unlike rules 1-5 this rule
# covers every source file in the tree, `scripts/` included: a maintenance
# script that hardcodes `checkpoint/` is wrong in exactly the same way.
OUTPUT_DIR_LITERAL_EXEMPT = frozenset({"cls_paths"})

# Calls whose string arguments are path components.
_PATH_BUILDERS = ("join", "Path", "PosixPath", "makedirs", "joinpath")


def _layer(module: str) -> int | None:
    """Layer of a dotted module, by longest matching prefix."""
    if module in NAMESPACE_ONLY:
        return None
    best, best_len = None, -1
    for prefix, layer in LAYERS.items():
        if (module == prefix or module.startswith(prefix + ".")) and len(prefix) > best_len:
            best, best_len = layer, len(prefix)
    return best


def _iter_modules():
    """(dotted module, path, AST) for every source file the rules cover."""
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = os.path.relpath(dirpath, REPO_ROOT)
        parts = [] if rel == "." else rel.split(os.sep)
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            name = fn[:-3]
            mod = ".".join(parts + ([] if name == "__init__" else [name])) or name
            if _layer(mod) is None and mod not in NAMESPACE_ONLY:
                continue
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError:  # pragma: no cover
                continue
            yield mod, os.path.relpath(path, REPO_ROOT), tree


def _iter_all_source():
    """(dotted module, path, AST) for every source file, layered or not.

    `_iter_modules` yields only modules the layer table classifies, which
    deliberately excludes `scripts/`. Rule 6 is not a layering rule and applies
    to the whole tree, so it needs its own walk.
    """
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = os.path.relpath(dirpath, REPO_ROOT)
        parts = [] if rel == "." else rel.split(os.sep)
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            name = fn[:-3]
            mod = ".".join(parts + ([] if name == "__init__" else [name])) or name
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            yield mod, os.path.relpath(path, REPO_ROOT), tree


def _imports(mod: str, tree: ast.AST, *, module_scope_only: bool = False):
    """(target module, imported names, lineno) for every import statement.

    ``module_scope_only`` restricts to the module body, i.e. the imports that
    execute the moment anything imports this module.
    """
    pkg = mod.rsplit(".", 1)[0] if "." in mod else ""
    nodes = tree.body if module_scope_only else ast.walk(tree)
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            yield target, [a.name for a in node.names], node.lineno
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, [], node.lineno


def test_no_upward_imports_at_module_scope():
    """Every module-scope import goes to the importer's own layer or below."""
    violations = []
    for mod, path, tree in _iter_modules():
        here = _layer(mod)
        if here is None:
            continue
        for target, _names, lineno in _imports(mod, tree, module_scope_only=True):
            there = _layer(target)
            if there is None:
                continue                      # third-party, stdlib, namespace
            if there > here:
                violations.append(
                    f"{path}:{lineno}  {mod} (layer {here}) imports "
                    f"{target} (layer {there})"
                )
    assert not violations, (
        "imports that go up the stack:\n  " + "\n  ".join(sorted(violations))
    )


def test_deferred_upward_imports_are_declared():
    """Function-scoped upward imports must be on the list, and the list live."""
    found: set[tuple[str, str]] = set()
    violations = []
    for mod, path, tree in _iter_modules():
        here = _layer(mod)
        if here is None:
            continue
        module_scope = {
            (t, ln) for t, _n, ln in _imports(mod, tree, module_scope_only=True)
        }
        for target, _names, lineno in _imports(mod, tree):
            if (target, lineno) in module_scope:
                continue                      # covered by the strict rule above
            there = _layer(target)
            if there is None or there <= here:
                continue
            found.add((mod, target))
            if (mod, target) not in DEFERRED_UPWARD_IMPORTS:
                violations.append(
                    f"{path}:{lineno}  {mod} (layer {here}) defers an import of "
                    f"{target} (layer {there}) without declaring it"
                )
    assert not violations, (
        "undeclared deferred upward imports:\n  " + "\n  ".join(sorted(violations))
    )
    stale = DEFERRED_UPWARD_IMPORTS - found
    assert not stale, (
        f"DEFERRED_UPWARD_IMPORTS lists edges that no longer exist: {sorted(stale)}"
    )


def test_encoder_training_never_imports_hopfield_nav():
    """The encoder side stays usable without the navigation stack."""
    violations = []
    for mod, path, tree in _iter_modules():
        if not (mod == "encoder_training" or mod.startswith("encoder_training.")):
            continue
        for target, _names, lineno in _imports(mod, tree):
            if target == "hopfield_nav" or target.startswith("hopfield_nav."):
                violations.append(f"{path}:{lineno}  imports {target}")
    assert not violations, (
        "encoder_training reaches into hopfield_nav:\n  " + "\n  ".join(violations)
    )


def test_figure_code_lives_in_analysis():
    """A module-scope matplotlib import outside `analysis/` is misplaced.

    Lazy imports inside a plotting function are fine and are what the library
    does: `eval_all` draws five optional figures that way, and `train_rnn`
    defers its forgetting plot. What this catches is a *figure generator* filed
    under a library package because it happened to have a `__main__` guard —
    which is how `visualize_trajectories`, `viz_sensory` and
    `encoder_training/plot_sweep` were left behind by the phase-6 move.
    """
    violations = []
    for mod, path, tree in _iter_modules():
        if mod.startswith("analysis"):
            continue
        if any(mod == e or mod.startswith(e + ".") for e in FIGURE_RULE_EXEMPT):
            continue
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] == "matplotlib" for n in names):
                violations.append(f"{path}:{node.lineno}  imports matplotlib at module scope")
    assert not violations, (
        "figure generators outside analysis/ (move them, or make the import "
        "lazy if the module is a library):\n  " + "\n  ".join(sorted(violations))
    )


def test_nothing_imports_a_cli():
    """Layer-7 modules are programs; importing one couples you to a CLI.

    Rules 1-3 do not catch this: a layer-8 module importing layer 7 is a
    *downward* import and therefore legal. But it is how `analysis.continual`
    and `hopfield_nav.train_rnn` ended up mutually dependent. Tests are exempt
    (`SKIP_DIRS`), so `test_protocols.py` may still reach for
    `train_rnn.train_sequential`.
    """
    clis = {m for m, layer in LAYERS.items() if layer == CLI_LAYER}
    violations = []
    for mod, path, tree in _iter_modules():
        if mod in clis:
            continue
        for target, _names, lineno in _imports(mod, tree):
            if target in clis or any(target.startswith(c + ".") for c in clis):
                violations.append(f"{path}:{lineno}  {mod} imports the CLI {target}")
    assert not violations, (
        "modules importing a CLI (move the shared part into training/):\n  "
        + "\n  ".join(sorted(violations))
    )


def test_no_cross_module_private_imports():
    """A leading underscore means "not part of my interface"."""
    violations = []
    for mod, path, tree in _iter_modules():
        for target, names, lineno in _imports(mod, tree):
            if _layer(target) is None and target not in NAMESPACE_ONLY:
                continue
            if target == mod:
                continue
            for name in names:
                if not name.startswith("_") or name.startswith("__"):
                    continue
                if (mod, target, name) in PRIVATE_IMPORT_ALLOWLIST:
                    continue
                violations.append(f"{path}:{lineno}  {mod} imports {name} from {target}")
    assert not violations, (
        "private names imported across modules:\n  " + "\n  ".join(sorted(violations))
    )


def _reserved_literals_in_paths(tree: ast.AST):
    """(literal, lineno) for reserved dir names used to *build* a path.

    Three shapes, which between them cover every way the five trainers and the
    scripts have ever spelled it:

        os.path.join("checkpoint", sub)      -> argument of a path builder
        Path("encoders") / run                  (same; `Path` is a builder)
        "checkpoint_rnn/" + sub              -> operand of + or /

    A bare literal that never reaches a path is not a violation: rule 6 is
    about constructing locations, not about mentioning them. That is what keeps
    `parts[0] == "encoders"` in `analysis/trajectories.py`'s encoder-path
    fallback legal, and it has to stay legal -- ~300 checkpoints store their
    encoder as a repo-relative `encoders/...` string and that resolver is how
    they are still found.
    """
    def reserved(node):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            return None
        first = node.value.replace("\\", "/").lstrip("/").split("/")[0]
        return first if first in RESERVED_OUTPUT_DIRS else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else "")
            if name not in _PATH_BUILDERS:
                continue
            for arg in node.args:
                lit = reserved(arg)
                if lit:
                    yield lit, arg.lineno
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
            for side in (node.left, node.right):
                lit = reserved(side)
                if lit:
                    yield lit, side.lineno


def test_no_hardcoded_output_dirs():
    """Only `cls_paths` may turn an output-directory name into a path.

    The bug this exists for: every trainer resolved its default `save_dir` as
    `os.path.join("checkpoint", sub)`. Relative, so it resolved against the CWD
    and then through a repo symlink to the *default* runs root -- `CLS_RUNS`
    was ignored for the one artifact class there are 8.5 GB of, a job started
    outside the repo root silently wrote a stray `./checkpoint/`, and the test
    suite's sandboxed `CLS_RUNS` leaked 36 junk run directories into the real
    tree because `train_phase_b_only` had no `--save_dir` to escape with.
    """
    violations = []
    for mod, path, tree in _iter_all_source():
        if mod in OUTPUT_DIR_LITERAL_EXEMPT:
            continue
        for literal, lineno in _reserved_literals_in_paths(tree):
            violations.append(
                f"{path}:{lineno}  builds a path from {literal!r} "
                f"(use cls_paths.run_dir / the *_dir accessors)"
            )
    assert not violations, (
        "hardcoded output directories:\n  " + "\n  ".join(sorted(violations))
    )


def _trainer_modules() -> list[str]:
    """Every `hopfield_nav/train*.py`, discovered rather than listed.

    A hardcoded list goes stale in exactly the direction that hurts: a *sixth*
    trainer is the one that could reintroduce the bug, and naming five would
    leave the sixth silently exempt from the rule written to catch it.
    """
    root = os.path.join(REPO_ROOT, "hopfield_nav")
    return sorted(
        f[:-3] for f in os.listdir(root)
        if f.startswith("train") and f.endswith(".py") and f != "__init__.py"
    )


def test_every_trainer_accepts_save_dir():
    """Every trainer takes `--save_dir`, so every one can be sandboxed.

    `train_store` (then `train_phase_b_only`) was the one that did not, which is the whole reason
    the pytest leak was possible: the smoke fixture sets `CLS_RUNS` to a
    tmpdir, and with the default path ignoring it there was no second way out.
    """
    trainers = _trainer_modules()
    # Vacuity guard: a discovery that found nothing would pass silently, which
    # is the failure mode of replacing a list with a glob.
    assert len(trainers) >= 5, (
        f"expected at least the five known trainers, discovered {trainers}"
    )

    missing = []
    for module in trainers:
        src = open(os.path.join(REPO_ROOT, "hopfield_nav", f"{module}.py"),
                   encoding="utf-8").read()
        if '"--save_dir"' not in src:
            missing.append(module)
    assert not missing, f"trainers with no --save_dir flag: {missing}"


@pytest.mark.parametrize("package", sorted(LAYERS))
def test_every_declared_layer_exists(package):
    """A stale row in LAYERS would silently exempt a package from rule 1."""
    top = package.split(".")[0]
    candidates = [
        os.path.join(REPO_ROOT, package.replace(".", os.sep)),
        os.path.join(REPO_ROOT, package.replace(".", os.sep) + ".py"),
        os.path.join(REPO_ROOT, top),
        os.path.join(REPO_ROOT, top + ".py"),
    ]
    assert any(os.path.exists(c) for c in candidates), (
        f"LAYERS declares {package}, which does not exist"
    )


def test_every_source_package_is_in_the_table():
    """A new top-level package must pick a layer rather than be unclassified."""
    unclassified = []
    for entry in sorted(os.listdir(REPO_ROOT)):
        path = os.path.join(REPO_ROOT, entry)
        if entry.startswith(".") or entry in SKIP_DIRS:
            continue
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "__init__.py")):
            if _layer(entry) is None and entry not in NAMESPACE_ONLY:
                unclassified.append(entry)
    assert not unclassified, (
        f"packages with no layer in LAYERS: {unclassified}"
    )


def test_hopfield_nav_subpackages_are_all_classified():
    """Same, one level down: a new hopfield_nav/<pkg>/ must declare a layer."""
    root = os.path.join(REPO_ROOT, "hopfield_nav")
    unclassified = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if entry in SKIP_DIRS or entry.startswith("."):
            continue
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "__init__.py")):
            if _layer(f"hopfield_nav.{entry}") is None:
                unclassified.append(entry)
        elif entry.endswith(".py") and entry != "__init__.py":
            if _layer(f"hopfield_nav.{entry[:-3]}") is None:
                unclassified.append(entry)
    assert not unclassified, (
        f"hopfield_nav members with no layer in LAYERS: {unclassified}"
    )
