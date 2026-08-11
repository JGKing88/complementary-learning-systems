# The sensory code

What the agent actually sees, what is settled about it, and which plausible
claims about it turned out to be wrong.

## The mechanism

`_wall_code` is `(4, size * wall_resolution)` of ±1 — a barcode painted on the
four walls. At `wall_resolution=1` (the default) that is one segment per grid
cell, so **32 numbers for an 8×8 room is the entire texture of the world**.

The agent has a 120° foveal cone of `observation_size` rays, centred on its
heading ψ. Each ray is traced until it hits a wall and returns the ±1 value of
the segment it landed on. The observation is that vector, ordered left to right
across the cone — one ±1 per ray, width `observation_size`.

Heading is a continuous angle, radians clockwise from North, set to whatever
direction the agent last actually moved. Sensory input is the only thing it
affects: actions stay world-frame and the policy gets no heading channel.

All four *cardinal* views of every cell are precomputed into
`_codebook[x, y, h]`, so discrete rollouts are a table gather; continuous
headings are ray-cast live. Both paths agree by construction.

## What is settled

**Position is not identifiable from one view at `wall_resolution=1`.** Segment
boundaries sit exactly on cell boundaries, so every ray landing anywhere inside
a cell reads the same value and the wall carries nothing about where *within* a
cell the agent stands. ~6–10% of cells then share a bit-identical observation
with another cell. That is information-theoretic: there is no difference for any
readout to find. It concentrates against the wall the agent faces, where a 120°
cone spans only ~3.5·d cells of wall — about 3.5 segments, against the ~6 bits
needed to name one of 64 cells. `wall_resolution ≥ 4` closes it, and the
requirement is near-flat in room size, because the cone sees a fixed extent of
wall however big the room. → `positional_identifiability.py`

**Nearby views are a warp of each other, not a perturbation.** An observation is
the wall barcode resampled onto the ray-angle axis; translating shifts that
sampling and approaching dilates it. Two cells one apart have zero-lag cosine
~0.02 and best-lag cosine ~0.85 — the same content, displaced. The lag is a
readout of displacement, linear in `dx` and falling as `1/D`, so
cross-correlating two views recovers the displacement without localising either.
→ `warp_structure.py`

## What was wrong

Recorded so the dead ends are not re-run. Each of these was measured, believed,
and then contradicted by a later measurement in this same series.

**"Capacity tracks the sensory code's effective dimensionality."** Soft rank
(participation ratio of the singular values) explains saturation *within* one
wall code — more rays resample the same segments and buy nothing — but it does
not survive comparison *across* wall codes. Two codes matched on soft rank and
on interpolability differed 4.7× in twin rate. What tracks recall is aliasing,
i.e. whether distant cells can look alike, not how many directions carry
variance.

**"The code must be spatially smooth."** Argued from a cosine-nearest-neighbour
decode, which measures whether a *local similarity* reader can use the code —
not whether the structure is there. It is there: see the warp result above.
Smoothness governs sample efficiency for a local reader, not possibility. A
smoothing knob was prototyped on the strength of this and **deliberately not
shipped**, because on the metric that assumes no reader (exact twins) it was
strictly harmful.

**"A 1/f multi-scale wall would beat a single correlation length."** The
reasoning was that coarse bands would serve interpolation while fine detail
served disambiguation. Measured: at matched stripe width, Gaussian smoothing
beat pink noise on *both* axes. The power-law tail correlates far-apart wall
regions, which is exactly the distant similarity that causes aliasing, while its
fine structure sits too low in amplitude to survive the `sign()`.

**"Many-to-one association is intrinsically hard."** Used to explain why four
cardinal views cannot share one place code. Wrong: with random patterns, k
patterns sharing a target track the k=1 curve almost exactly, so the sharing
costs only its own pattern count. The real cause was capacity — 4 views × 256
cells against an effective dimensionality of ~10–26.

## Open

- `observation_size=12` (the `train_navigate` default) is below what the code
  needs: ~27% twins against ~9% at 60, same wall. 60 costs almost nothing.
- Extra rays still buy displacement *precision* even where they buy no
  identifiability — `dx=1` is ~15 lags at 240 rays and ~4 at 60.
- At `wall_resolution=8` a pure-shift correlation matches much more weakly
  (~0.38 vs ~0.85). The true warp is shift *plus* dilation and a roll cannot
  express the dilation; whether a shift+scale estimator recovers it is untested.
- The policy reads the ray vector through one linear layer into a GRU, which
  cannot express a cross-correlation between successive observations. Finer
  walls give more to work with; exploiting the warp structure would need
  architecture over the ray axis.
- `encoded_state` is `encoded_Phi[true positions]` — ground-truth position handed
  to the policy. While that channel is on, nothing forces the sensory→position
  inversion to happen at all.
