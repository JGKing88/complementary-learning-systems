"""Does the encoder actually hand the policy a bigger readout?

§17.7 inferred this from `dir_norm` at u1, which is a POLICY statistic. This
measures the thing itself: the field JSONs store `field` = the raw, UNnormalised
q at every (2x-downsampled) cell, so ||q|| is directly available for each
encoder without running anything.
"""
import json

import numpy as np

D = '/orcd/pool/003/jackking/cls_runs/results/exploit_diag'
TAGS = [('field_p10v1', 'P2 fixed  gain   5  beta   5', 0.128),
        ('field_p17_final', 'P2 fixed  gain 300  beta   5', 0.126),
        ('field_p18_final', 'knee      gain 300  beta 300', 0.351),
        ('field_p19_w52_full', 'w52       gain 100  beta 100', 0.336)]

print(f'{"encoder / gain / beta":32s} {"mean|q|":>9s} {"median":>9s} '
      f'{"p90":>9s} {"max":>9s}   {"dir@u1":>7s}')
print('-' * 84)
base = None
rows = []
for tag, lab, diru1 in TAGS:
    envs = json.load(open(f'{D}/{tag}.json'))['envs']
    mags = []
    for e in envs:
        for c in e['cells']:
            f = np.asarray(c['field'], dtype=np.float64)   # (h, w, 2)
            mags.append(np.linalg.norm(f, axis=-1).ravel())
    m = np.concatenate(mags)
    rows.append((lab, m.mean(), diru1))
    if base is None:
        base = m.mean()
    print(f'{lab:32s} {m.mean():9.4f} {np.median(m):9.4f} '
          f'{np.percentile(m, 90):9.4f} {m.max():9.4f}   {diru1:7.3f}')

print()
print('ratios against the P2-fixed gain-5 row:')
for lab, mu, diru1 in rows:
    print(f'  {lab:32s} |q| x{mu / base:5.2f}   dir@u1 x{diru1 / 0.128:5.2f}')
