import sys, torch
sys.path.insert(0, 'pat334'); import mh_hid as M
orig = torch.load
def small(f, *a, **k):
    x = orig(f, *a, **k)
    if isinstance(x, dict) and 'n_inst' in x: x = dict(x); x['n_inst'] = 5
    return x
M.torch.load = small
p = M.FC.load_backbone(M.ROOT / M.CK['PaTH'], M.dev)
A = M.run_set('val', p, (0, 1)); B = [M.run_set('val', p, (i, 2)) for i in range(2)]
oa = A['order']; ob = torch.cat([b['order'] for b in B]); ia = oa.argsort(); ib = ob.argsort()
assert torch.equal(oa[ia], ob[ib])
mx = 0.0
for H in M.HS:
    assert A['E'][H].keys() == B[0]['E'][H].keys()
    for k in A['E'][H]:
        for j in range(2):
            x = A['E'][H][k][j][ia]; y = torch.cat([b['E'][H][k][j] for b in B])[ib]; mx = max(mx, float((x - y).abs().max()))
print('nkeys', len(A['E'][128]), 'queries', len(oa), 'max diff', mx)
