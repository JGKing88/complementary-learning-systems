"""The layering rules, enforced by walking every module's AST.

Phase 6 gave the tree a direction: `world` knows nothing about anything,
`analysis` may know about everything, and each layer between may only look
downward. That direction is only worth having if something checks it, because
the way it decays is invisible -- one convenient import, and the next person
finds a cycle they cannot break without moving three files.

Three rules:

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

The layer numbers are the whole specification; a new package that is not in the
table fails rule 1 loudly rather than being silently exempt.

`tests/` is exempt from all three: a characterization test that pins
`env._at_goal_l2` is doing its job, and a test importing across layers is not a
dependency anything ships.

Each rule is mutation-verified: an upward import in `world/env.py`, an
`encoder_training -> hopfield_nav` import, and a private import from a lower
layer each fail exactly one of the three.
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
    "cls": 0,                          # legacy, retired in phase 7
    "cls_paths": 0,

    "hopfield_nav.world": 1,           # env, vec_env, scaffold, memory, episode
    "hopfield_nav.policy": 2,          # agent, agent_rnn, channels
    "hopfield_nav.rollout": 3,         # collector, rnn, signal, oracles, distractors, types
    "hopfield_nav.updates": 4,         # ppo, bc, bc_rnn
    "hopfield_nav.evaluation": 5,      # metrics, rnn, protocols, batched, checkpoint_io
    "hopfield_nav.training": 6,        # world_setup, rnn_sequential

    # The CLIs. They wire the layers together and nothing may import them.
    "hopfield_nav.train": 7,
    "hopfield_nav.train_phased": 7,
    "hopfield_nav.train_phase_a_only": 7,
    "hopfield_nav.train_phase_b_only": 7,
    "hopfield_nav.train_rnn": 7,
    "hopfield_nav.eval_all": 7,
    "hopfield_nav.visualize_trajectories": 7,
    "hopfield_nav.viz_sensory": 7,

    "analysis": 8,                     # figure + experiment pipelines
}

# `hopfield_nav` itself is only the namespace package; a bare `import
# hopfield_nav` carries no dependency.
NAMESPACE_ONLY = {"hopfield_nav"}

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "docs", "tests"}

# Privates that are imported across modules and should not be. Empty by intent:
# every entry here is a thing phase 6 was supposed to promote or inline. Add a
# row only with a reason, and prefer fixing the import.
PRIVATE_IMPORT_ALLOWLIST: set[tuple[str, str, str]] = set()

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
