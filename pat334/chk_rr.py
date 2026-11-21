import sys, time, torch
sys.path.insert(0, 'pat334'); import rr as M
orig = torch.load
def small(f, *a, **k):
    x = orig(f, *a, **k)
    if isinstance(x, dict) and 'n_inst' in x: x = dict(x); x['n_inst'] = 6
    return x
M.torch.load = small
p = M.FC.load_backbone(M.ROOT / M.CK['PaTH'], M.dev)
for name in ('val', 'test'):
    t0 = time.time(); A = M.run_set(name, p, (0, 1)); dt = (time.time() - t0) / 6
    E = A['E'][128]; print(name, 'sec/inst', round(dt, 2), 'nkeys', len(E), 'queries', len(A['order']))
    for H in M.HS:
        E = A['E'][H]
        for m in 'cj':
            a0, b0 = E[f'{m}|cos|0.0|u5'][0], E[f'{m}|eu|0.0|u5'][0]; assert torch.equal(a0, b0)
            print(H, m, 'f %.3f' % E['f'][0].median(), ' '.join('lam%s %.3f' % (l, E[f'{m}|cos|{l}|u5'][0].median()) for l in M.LAMS))
