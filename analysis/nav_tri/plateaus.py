"""Two plateaus, reported separately: ACCURACY and BEELINE.

Jack's target is the beeline -- reached as quickly and as stably as possible --
with accuracy as the precondition rather than the goal. These are different
updates: on p17_gain success pinned at 1.000 by ~u200 while path quality kept
improving to ~u400.

BEELINE METRIC. `mean_steps` alone is not it, because step count tracks the
speed cap (section 9.9): the same path walked slower costs more steps. The
speed-invariant quantity is the DISTANCE WALKED,

    path = mean_steps * mean_speed          [cells]

whose floor is the mean straight-line start-goal distance. That floor is ~10.5
cells, inferred two independent ways from section 9.9's own table:
    p10_pol_v1   10.95 steps * 1.00 speed / 1.043 directness = 10.50
    p12_lo       11.79 steps * 0.94 speed / 1.081 directness = 10.25
and confirmed by p17_gain's best observed path, 11.2 * 0.95 = 10.64. So
`directness = path / 10.5`, and 1.00 is a perfect beeline.

THE SURVIVORSHIP TRAP. `mean_steps` and `mean_speed` are computed over
SUCCESSFUL episodes only. At low success only the nearest goals are reached, so
path looks spuriously good -- p19_nc at u100 shows 9.3 steps at success 0.073.
Directness is therefore only quoted where success >= MIN_SUCC, and suppressed
elsewhere rather than reported with a caveat nobody will read.
"""
import re
import sys

LOG = '/orcd/pool/003/jackking/cls_runs/logs/nav_p2_%s.out'
STRAIGHT = 10.5      # mean straight-line start-goal distance, cells
MIN_SUCC = 0.90      # below this, path is survivorship and is not quoted
ACC_THR = 0.95       # "accurate"
BEE_THR = 1.10       # "beeline": within 10% of the straight line

ARMS = [('p19_kcap', '21656252', 'none'),
        ('p19_kcur', '21659098', '0->10 over 100'),
        ('p19_nc', '21651001', 'kappa 148 ctrl')]
if len(sys.argv) > 1:                       # reference runs, e.g. 21623992
    ARMS = [(a, a, '') for a in sys.argv[1:]]

NAV = re.compile(
    r"navigate_u(\d+)\] nav=\{0: \{'success_rate': ([0-9.e+-]+), "
    r"'mean_speed': ([0-9.e+-]+), 'mean_steps': ([0-9.e+-]+).*?"
    r"10: \{'success_rate': ([0-9.e+-]+), "
    r"'mean_speed': ([0-9.e+-]+), 'mean_steps': ([0-9.e+-]+)")


def series(job):
    try:
        txt = open(LOG % job).read()
    except FileNotFoundError:
        return []
    out = []
    for m in NAV.finditer(txt):
        u, s0, _v0, _t0, s10, v10, t10 = m.groups()
        path = float(t10) * float(v10)
        out.append(dict(u=int(u), s0=float(s0), s10=float(s10),
                        steps=float(t10), spd=float(v10), path=path,
                        direct=path / STRAIGHT))
    return out


def plateau(rows, key, thr, lower_is_better, gate=None):
    """First update reaching thr, and the WORST value at or after it -- the
    threshold-free pair section 9.9 settled on."""
    ok = [r for r in rows
          if (r[key] <= thr if lower_is_better else r[key] >= thr)
          and (gate is None or r['s10'] >= gate)]
    if not ok:
        return None, None
    first = ok[0]['u']
    tail = [r[key] for r in rows
            if r['u'] >= first and (gate is None or r['s10'] >= gate)]
    worst = max(tail) if lower_is_better else min(tail)
    return first, worst


hdr = (f"{'arm':10s} {'curric':16s} {'n':>3s} {'lastu':>5s} | "
       f"{'ACC first':>9s} {'worst':>6s} | {'BEE first':>9s} {'worst':>6s} | "
       f"{'now succ':>8s} {'now direct':>10s}")
print(hdr)
print('-' * len(hdr))
for arm, job, cur in ARMS:
    rows = series(job)
    if not rows:
        print(f'{arm:10s} {cur:16s}  -- no evals yet --')
        continue
    a_u, a_w = plateau(rows, 's10', ACC_THR, False)
    b_u, b_w = plateau(rows, 'direct', BEE_THR, True, gate=MIN_SUCC)
    last = rows[-1]
    nd = f'{last["direct"]:.3f}' if last['s10'] >= MIN_SUCC else '(low succ)'
    print(f'{arm:10s} {cur:16s} {len(rows):3d} {last["u"]:5d} | '
          f'{str("u" + str(a_u)) if a_u else "--":>9s} '
          f'{("%.3f" % a_w) if a_w is not None else "--":>6s} | '
          f'{str("u" + str(b_u)) if b_u else "--":>9s} '
          f'{("%.3f" % b_w) if b_w is not None else "--":>6s} | '
          f'{last["s10"]:8.3f} {nd:>10s}')

print(f'\nACC  = succ@10 >= {ACC_THR};  BEE = directness <= {BEE_THR} '
      f'(path/{STRAIGHT} cells), only counted where succ@10 >= {MIN_SUCC}.')
print('"worst" is the worst value at or after the first crossing = stability.\n')

for arm, job, _ in ARMS:
    rows = series(job)
    if not rows:
        continue
    print(f'--- {arm} ---')
    print(f"{'u':>6s} {'succ@10':>8s} {'steps':>6s} {'speed':>6s} "
          f"{'path':>6s} {'direct':>7s}")
    for r in rows:
        d = f'{r["direct"]:7.3f}' if r['s10'] >= MIN_SUCC else '      -'
        print(f'{r["u"]:6d} {r["s10"]:8.3f} {r["steps"]:6.1f} '
              f'{r["spd"]:6.2f} {r["path"]:6.1f} {d}')
    print()
