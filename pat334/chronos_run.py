"""PAT-334 D: Chronos-T5-base samples for the test queries of one setting (last 512 raw context steps, 20 samples, 512 steps,
seed 0). Runs in .venv_chronos. usage: chronos_run.py DATA_DIR TAG i n   |   chronos_run.py merge TAG n"""
import sys, torch
from pathlib import Path
O = Path(__file__).resolve().parent / 'out'
if sys.argv[1] == 'merge':
    tag, n = sys.argv[2], int(sys.argv[3]); P = [torch.load(O / f'chronos_raw_{tag}_{i}of{n}.pt') for i in range(n)]
    nq = sum(len(p['q']) for p in P); out = torch.empty(nq, 20, 512, dtype=torch.float16)
    for p in P: out[p['q']] = p['s']
    torch.save({'samples': out}, O / f'chronos_{tag}_test.pt'); print('merged', out.shape); sys.exit()
from chronos import ChronosPipeline
dd, tag, i, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = torch.load(dd / 'test.pt'); S, rows, t = d['S'], d['rows'], d['t']; nq = len(rows)
q = torch.arange(i, nq, n)
pipe = ChronosPipeline.from_pretrained('amazon/chronos-t5-base', device_map='cuda', torch_dtype=torch.bfloat16)
torch.manual_seed(0); out = []
for b in range(0, len(q), 32):
    qq = q[b:b + 32]; ctx = torch.stack([S[rows[j], int(t[j]) - 512:int(t[j])] for j in qq.tolist()]).float()
    out.append(pipe.predict(ctx, prediction_length=512, num_samples=20, limit_prediction_length=False).to(torch.float16))
    if b % 3200 == 0: print(b, len(q), flush=True)
torch.save({'q': q, 's': torch.cat(out)}, O / f'chronos_raw_{tag}_{i}of{n}.pt'); print('done', flush=True)
