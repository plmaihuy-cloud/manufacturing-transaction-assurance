from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'simulation_v2'
"""E9: reference exact-reachability enumeration (explicit b^n path expansion)."""
import time, itertools, numpy as np, pandas as pd
def enumerate_paths(b, n):
    states = {0}
    frontier = [0]
    for _ in range(n):           # breadth-first expansion of every path (paths kept, not merged)
        nxt = []
        for s in frontier:
            for k in range(b):
                nxt.append(s * b + k)
        frontier = nxt
    return len(frontier)
rows = []
for b, nmax in [(2, 16), (3, 11), (4, 9)]:
    for n in range(1, nmax + 1):
        ts = []
        for rep in range(15):
            t0 = time.perf_counter(); m = enumerate_paths(b, n); ts.append((time.perf_counter() - t0) * 1000)
        rows.append(dict(b=b, n=n, paths=m, median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95))))
df = pd.DataFrame(rows); df.to_csv(str(DATA / 'E9_reachability.csv'), index=False)
print(df.groupby('b').tail(1).to_string(index=False))
