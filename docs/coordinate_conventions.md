# Coordinate Conventions

This document describes how spatial coordinates flow through the grid codebook,
encoder, Hopfield network, and navigation evaluation.

## Grid codebook (`gen_gbook_2d`)

`gbook` has shape `(Ng, Npos, Npos)`.

- `gbook[:, a, b]` is the grid-cell activity vector at position `(a, b)`.
- Internally, `phi1 = xs % period` varies with dim 0, `phi2 = ys % period`
  varies with dim 1, and `meshgrid(xs, ys, indexing='ij')` is used.
- **Convention: dim 0 = x, dim 1 = y.**

`smooth_gbook` applies per-module Gaussian bumps and preserves axis ordering.

## Encoded grid (`encoded_Phi`)

Both `VectorHash.precompute_encoded_phi` and `nav_eval.encode_full_grid`
reshape `(Ng, Npos, Npos)` to flat, pass through the encoder, and reshape back
to `(Npos, Npos, embed_dim)`. The reshape is a simple flatten/unflatten with no
transpose, so axis ordering is preserved.

**`encoded_Phi[a, b]` = encoded vector at position `(a, b)` where dim 0 = x,
dim 1 = y.**

## RHC path

`RHCEncoder.encode_positions(xs, ys)` takes explicit x and y coordinate arrays.
When building the full grid, `np.mgrid[0:Npos, 0:Npos]` produces `(2, Npos, Npos)`
where `[0]` varies along dim 0 and `[1]` varies along dim 1. After flattening
and encoding, the reshape back to `(Npos, Npos, embed_dim)` preserves the same
convention.

## Environment (`WMEnv`)

- Positions are `(x, y)` tuples. `_codebook[x, y, heading]`.
- `_simulate_move` adds `(dx, dy)` to `(x, y)`.
- `goal_location` returns `(x, y)`.

## Heading (`world/env.py`)

Heading is a single angle **ψ, radians, clockwise from North**, so forward is
`(sin ψ, cos ψ)`. This is the same convention the foveal ray angles use, which
is what makes rotation cheap: a ray at cone offset θ points along `ψ + θ`, so
facing a direction is *adding ψ to every ray angle* — no rotation matrix.

- ψ = 0 is North. That is where the cone was hard-wired before headings existed,
  so `egocentric_heading=False` (every view read at ψ=0) reproduces the old
  behavior exactly.
- ψ follows the **realized** displacement, never the requested action — a step
  clipped by a wall leaves the agent facing where it was. Discrete movement
  takes ψ from `CARDINAL_RADIANS` (exactly `k·π/2`); continuous movement uses
  `atan2(dx, dy)`, arguments in that order.
- `CARDINAL_RADIANS[k]` is the heading of `CARDINAL_ACTIONS[k]`: an agent that
  steps action `k` ends up facing heading `k`. `cardinal_index(ψ)` inverts it,
  returning `-1` for angles between the cardinals.

`_codebook[x, y, h]` holds the four **cardinal** views. It is not what the agent
observes — a live observation is ray-cast at the continuous ψ — but it is the
canonical per-cell artifact for the scaffold's sbook and the generator's
env-identity check, and it is the gather path whenever ψ is exactly cardinal.

Movement stays **allocentric**: actions are world-frame, and sensory input is
the only thing heading affects.

## VectorHash coordinate mapping

`env_offset = (C_X, C_Y)`. Local position `(lx, ly)` maps to global grid
position `(gx, gy) = (lx + C_X, ly + C_Y)`.

Neighbor definitions in `compute_hopfield_direction_batch`:

| Direction | Neighbor lookup              | Array axis |
|-----------|------------------------------|------------|
| North     | `encoded_Phi[gx, gy + 1]`   | +dim 1     |
| East      | `encoded_Phi[gx + 1, gy]`   | +dim 0     |

Gram-Schmidt builds projection matrix `W` with shape `(2, embed_dim)`:
- `W[0]` = East basis (orthogonalized right vector)
- `W[1]` = North basis (normalized forward vector)

After projection `q = W @ displacement`:
- `q[0]` = East component (+dim 0 = +x)
- `q[1]` = North component (+dim 1 = +y)

## Navigation eval (`nav_eval.py`)

Uses the same convention as VectorHash. Positions are stored as `[gx, gy]`
arrays (dim 0 = x, dim 1 = y).

`_compute_projection_matrix(encoded_Phi, gx, gy)`:
- North: `encoded_Phi[gx, gy + 1]` (+dim 1)
- East: `encoded_Phi[gx + 1, gy]` (+dim 0)
- Returns `W[0]` = East, `W[1]` = North (same as VectorHash).

`_continuous_step` returns `(dgx, dgy)`:
- `dgx = q[0]` (East component, +dim 0)
- `dgy = q[1]` (North component, +dim 1)

`simulate_trajectory` stores `position = [gx, gy]` and increments by
`[dgx, dgy]`.

`run_navigation_eval` receives placements as `(gx0, gy0)` tuples. Goals are
sampled at `(gx0 + rand, gy0 + rand)` and looked up as
`encoded_Phi[goal_gx, goal_gy]`.

## Hopfield network

The Hopfield network is purely associative and dimension-agnostic. It stores
and recalls flat vectors of shape `(embed_dim,)`. Spatial structure is handled
entirely by the projection matrix `W` that maps Hopfield displacement back to
2D grid movement.

## Summary

Everywhere in the pipeline:

| Array dimension | Spatial axis | Cardinal direction |
|-----------------|-------------|--------------------|
| dim 0           | x           | East (+), West (-) |
| dim 1           | y           | North (+), South (-)|

All coordinate tuples are `(x, y)` / `(gx, gy)`, and all position arrays are
`[dim0, dim1]` = `[x, y]`.
