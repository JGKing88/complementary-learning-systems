"""Is the readout good, and is the policy following it?

Three cosines per §9.7: q_accuracy = cos(q, goal-pos) says whether the READOUT
points at the goal; follow_q = cos(a, q) says whether the POLICY follows it;
align_true = cos(a, goal-pos) is the baseline follow_q must be read against.
"""
import json

import numpy as np

F = ('/orcd/pool/003/jackking/cls_runs/results/exploit_diag/'
     'diag_p19_nc_u225.json')
d = json.load(open(F))

for g in d['groups']:
    rows = g['rows']
    n = len(rows)
    ok = [r for r in rows if r['success']]
    bad = [r for r in rows if not r['success']]

    def mean(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None
             and not isinstance(r[k], str) and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float('nan')

    print(f"=== n_distractors={g['n_dist']}   {len(ok)}/{n} success "
          f"({len(ok) / max(n, 1):.3f}) ===")
    for lab, rs in (('SUCCESS', ok), ('FAIL', bad)):
        if not rs:
            continue
        print(f'  {lab:8s} n={len(rs):3d}  '
              f'q_acc={mean(rs, "q_acc"):+.3f}  '
              f'follow_q={mean(rs, "follow_q"):+.3f}  '
              f'align_true={mean(rs, "align_true"):+.3f}  '
              f'd_min={mean(rs, "d_min"):.2f}')
    if bad:
        from collections import Counter
        print(f'  symptoms: {dict(Counter(r["symptom"] for r in bad))}')
        print(f'  motions : {dict(Counter(r["motion"] for r in bad))}')
    print()
