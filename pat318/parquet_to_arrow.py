"""PAT-318: rebuild the official directory layout (<split>/<system>/<idx>_T-4096.arrow) from the HF skew40 parquet files.
Usage: python pat318/parquet_to_arrow.py data/skew40/train.parquet data/final_skew40/train
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from panda.utils.data_utils import convert_to_arrow

src, dst = sys.argv[1], Path(sys.argv[2])
pf = pq.ParquetFile(src)
n = 0
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg).to_pydict()
    for shape, target, d, f in zip(t['target._np_shape'], t['target'], t['_source_directory'], t['_source_filename']):
        x = np.asarray(target, dtype=np.float64).reshape(shape)
        (dst / d).mkdir(parents=True, exist_ok=True)
        convert_to_arrow(dst / d / f, x)
        n += 1
print(src, '->', dst, n, 'files', len(list(dst.iterdir())), 'systems')
