"""PAT-334 plugin step 2: other foundation models at the test origins and the Gbt backtest origins (runs in .venv_chronos).
Every model sees the last 512 raw steps before its origin (as Panda), predicts 512 steps; point forecast = median.
  bolt    : amazon/chronos-bolt-base, univariate per channel, median quantile
  dynamix : DurstewitzLab dynamix-3d-alrnn-v1.0; 3-channel instances jointly (native N = 3), others per channel with
            delay_embedding (the repo's time-series mode); standardize=True
Origins: b = 't' -> t; b in 0..3 -> t - 512 - 128 b, valid if the context is inside the allowed history (ctx: >= t - CTX).
usage: P321_OUT=... P321_CTX=... P333_HIST=... python pat334/bbx.py MODEL B   -> out/gbt_bb_{MODEL}_{TAG}_{B}.pt"""
import os, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / 'pat334' / 'out'
CTX = int(os.environ.get('P321_CTX', 512)); MODE = os.environ.get('P333_HIST', 'full'); TAG = f'ctx{CTX}_{MODE}'
dev = torch.device('cuda'); L = 512; HP = 512


def load(name):
    if name == 'bolt':
        from chronos import BaseChronosPipeline
        p = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-base', device_map='cuda', torch_dtype=torch.bfloat16)
        def run(ctx):                                                                     # [n, L] -> [n, HP]
            out = []
            for i in range(0, len(ctx), 1024):
                q, _ = p.predict_quantiles(context=ctx[i:i + 1024], prediction_length=HP, quantile_levels=[0.5], limit_prediction_length=False)
                out.append(q[..., 0].float().cpu())
            return torch.cat(out)
        return run, False
    from dynamix.utilities.utilities import load_hf_model
    from dynamix.model.forecaster import DynaMixForecaster
    m = load_hf_model('dynamix-3d-alrnn-v1.0').to(dev).eval(); fc = DynaMixForecaster(m)
    def run(ctx, joint):                                                                  # ctx [n, C, L] -> [n, C, HP]
        out = []
        for i in range(0, len(ctx), 256):
            c = ctx[i:i + 256].to(dev).float().permute(2, 0, 1)                           # (L, S, C)
            r = fc.forecast(context=c, horizon=HP, preprocessing_method='pos_embedding' if joint else 'delay_embedding', standardize=True)
            out.append(r.permute(1, 2, 0).float().cpu())
        return torch.cat(out)
    return run, True


@torch.no_grad()
def main(model, b):
    d = torch.load(Path(os.environ['P321_OUT']) / 'test.pt'); S, rows, t, ids = d['S'], d['rows'], d['t'], d['ids']
    if b == 't': tp, ok = t.clone(), torch.ones(len(t), dtype=torch.bool)
    else:
        tp = t - 512 - 128 * int(b); ok = (tp - L) >= ((t - CTX) if MODE == 'ctx' else 0); tp = torch.where(ok, tp, t)
    run, inst = load(model); t0 = time.time(); f = torch.zeros(len(rows), HP)
    if not inst:
        ctx = S[rows[:, None], tp[:, None] - L + torch.arange(L)]
        f = run(ctx)
    else:
        qi = [torch.nonzero(ids == i)[:, 0] for i in range(d['n_inst'])]; byc = {}
        for i, q in enumerate(qi): byc.setdefault(len(q), []).append(i)
        for C, insts in byc.items():
            Q = torch.stack([qi[i] for i in insts])                                       # [n, C]
            ctx = S[rows[Q][..., None], tp[Q][..., None] - L + torch.arange(L)]           # [n, C, L]
            if C == 3: pr = run(ctx, True)
            else: pr = run(ctx.reshape(-1, 1, L), False).reshape(len(insts), C, HP)
            f[Q.reshape(-1)] = pr.reshape(-1, HP)
    f = torch.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6)
    torch.save({'f': f.half() if f.abs().max() < 6e4 else f, 'ok': ok, 'tp': tp}, O / f'gbt_bb_{model}_{TAG}_{b}.pt')
    print(f'{model} {TAG} b{b}: valid {float(ok.float().mean()):.3f}, {time.time() - t0:.0f}s, finite {float(torch.isfinite(f).float().mean()):.4f}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
