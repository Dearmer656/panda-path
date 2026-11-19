import glob, os, collections, pyarrow as pa, pyarrow.ipc as ipc, numpy as np
for d in ['data/final_skew40/train','data/final_skew40/test_zeroshot','data/final_base40/train','data/final_base40/test_zeroshot']:
    systems=sorted(os.listdir(d)); files=glob.glob(f'{d}/*/*.arrow')
    dims=collections.Counter(); ntraj=0; lens=collections.Counter(); ddr=collections.Counter()
    for i,f in enumerate(files):
        try: t=ipc.open_file(f).read_all()
        except Exception: t=ipc.open_stream(f).read_all()
        if i==0: print(' schema', t.schema)
        col=t.column('target').to_pylist() if 'target' in t.column_names else None
        for row in col:
            a=np.asarray(row); dims[a.shape[0]]+=1; lens[a.shape[-1]]+=1; ntraj+=1
    print(d,'systems',len(systems),'files',len(files),'traj',ntraj,'dims',dict(sorted(dims.items())),'len',dict(lens))
