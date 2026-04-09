# Development Guidelines

## Code Quality

- **No bandaids.** If something needs fixing, fix it properly. Don't patch over a symptom — find the root cause and address it structurally.
- **Single source of truth.** If a value is derived from another (e.g., snapped position from float position), compute it in one place, store it as state, and read it everywhere else. Never recompute ad-hoc in multiple locations.
- **Clean interfaces.** Pass data explicitly through function arguments. Don't use hidden state, cache attributes, or fragile lookups when a parameter would do.
- **Consistent abstractions.** If two code paths do the same thing (e.g., discrete vs continuous stepping), they should share the common logic and only diverge where they actually differ.
- **Think before implementing.** Before writing code, ask: where does this state live? Who owns it? Is there already a place for this? Will this create a second source of truth?

## Architecture

- `encoder_training/` — self-contained encoder training package. Produces checkpoints.
- `hopfield_nav/` — Hopfield navigation system. Loads encoder checkpoints.
- Both import scaffold utilities from `cls.vectorhash`.
- Env handles grid state only. Neural processing (recall, encode, Hopfield, projection) lives in the rollout collector.
- Hopfield is standalone — not owned by VectorHash. VectorHash owns spatial geometry (scaffold, encoded_Phi, Gram-Schmidt).

## Common Pitfalls

- `env_offsets` indices: when `register_envs(all_envs)` is called with train+val combined, val envs start at index `len(train_envs)`, not 0. Always use the global index.
- `nonlin` double-thresholding: the recall chain skips the second threshold. Don't re-add it.
- PyTorch `Categorical` has a `.mean` property — don't use `hasattr(dist, 'mean')` to detect continuous distributions. Use `isinstance(dist, Normal)`.
