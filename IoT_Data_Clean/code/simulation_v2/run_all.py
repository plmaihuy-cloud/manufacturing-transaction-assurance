from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'simulation_v2'
"""Run every experiment of ANALYSIS_PLAN.md and write run-level CSVs to results/."""
import itertools, json, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from raf_sim import Config, Sim

OUT = str(DATA)
POL6 = ['naive', 'static', 'fixed', 'adaptive', 'matched', 'oracle']


def one(kw):
    tag = kw.pop('_tag')
    rec = kw.pop('_rec', False)
    samp = kw.pop('_samp', 0.0)
    c = Config(**kw, record_events=rec, sample_every=samp)
    s = Sim(c); r = s.run(); r['experiment'] = tag
    extra = {}
    if rec: extra['events'] = s.events
    if samp: extra['samples'] = s.samples
    return r, extra


def jobs():
    J = []
    seeds = range(1, 21)
    # E1a
    for p in POL6:
        for s in seeds: J.append(dict(_tag='E1a_central', policy=p, seed=s))
    # E1b
    for c, m, p in itertools.product([1, 2, 4, 8], [2, 4, 8, 16], ['fixed', 'adaptive']):
        for s in seeds: J.append(dict(_tag='E1b_capacity', policy=p, query_servers=c, query_service_mean=m, seed=s))
    # E1c
    for tau, c in itertools.product(np.round(np.arange(0.25, 3.51, 0.25), 2), [1, 2, 4, 8]):
        for s in seeds: J.append(dict(_tag='E1c_threshold', policy='fixed', threshold=float(tau), query_servers=c, seed=s))
    # E1d
    for sc, p in itertools.product([0.5, 1, 2, 4], POL6):
        for s in seeds: J.append(dict(_tag='E1d_delay', policy=p, delay_scale=sc, seed=s))
    # E1e
    for dr, b, a in itertools.product([0.0, 0.015, 0.05, 0.10], [2, 3, 5], [6, 8, 10]):
        for s in seeds: J.append(dict(_tag='E1e_stress', policy='adaptive', packet_drop=dr, buffer_size=b, arrival_mean=a, seed=s))
    # E1f
    for h, act, p in itertools.product([225, 450, 900], [0.1, 0.35, 1.0, 2.0], ['fixed', 'adaptive']):
        for s in seeds: J.append(dict(_tag='E1f_hold_actuation', policy=p, hold_mtbf=h, actuation_mean=act, seed=s))
    # E2
    for p in ['fixed', 'adaptive', 'matched']:
        for s in range(1001, 1201): J.append(dict(_tag='E2_fresh200', policy=p, seed=s))
    # E3
    for p in ['fixed', 'adaptive', 'matched']:
        for s in range(2001, 2301): J.append(dict(_tag='E3_R2', policy=p, seed=s))
    # E4
    for c, m in itertools.product([1, 2, 4, 8], [2, 4, 8, 16]):
        for s in range(3001, 3101): J.append(dict(_tag='E4_capacity100', policy='adaptive', query_servers=c, query_service_mean=m, seed=s))
    # E5
    for tau, c in itertools.product(np.round(np.arange(0.05, 3.501, 0.05), 2), [1, 2, 4, 8]):
        for s in seeds: J.append(dict(_tag='E5_fine_sweep', policy='fixed', threshold=float(tau), query_servers=c, seed=s))
    # E6
    for s in seeds: J.append(dict(_tag='E6_oracle_events', policy='oracle', seed=s, _rec=True))
    # E7
    for c, m in itertools.product([1, 2, 4], [2, 4, 8, 16]):
        for s in range(5001, 5041): J.append(dict(_tag='E7_calibration', policy='adaptive', query_servers=c, query_service_mean=m, seed=s, _samp=2.0))
    # E8
    for (cond, c, m), dl, sv, p in itertools.product([('central', 2, 8), ('bottleneck', 1, 16)], ['lognormal', 'gamma'],
                                                    ['exponential', 'weibull'], ['fixed', 'adaptive', 'matched']):
        for s in range(4001, 4101): J.append(dict(_tag='E8_distribution_' + cond, policy=p, query_servers=c, query_service_mean=m,
                                                   delay_family=dl, service_family=sv, seed=s))
    return J


if __name__ == '__main__':
    J = jobs()
    print('runs', len(J), flush=True)
    t0 = time.time()
    rows, events, samples = [], [], []
    with Pool(2) as pool:
        for k, (r, extra) in enumerate(pool.imap(one, J, chunksize=20)):
            rows.append(r)
            for e in extra.get('events', []): events.append(e)
            for smp in extra.get('samples', []):
                samples.append(dict(regime_c=r['query_servers'], regime_s=r['query_service_mean'], seed=r['seed'],
                                    t=smp[0], u=smp[1], q=smp[2]))
            if k % 2000 == 0: print(k, round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    for tag, g in df.groupby('experiment'):
        g.to_csv(f'{OUT}/{tag}.csv', index=False)
    pd.DataFrame(events).to_csv(f'{OUT}/E6_oracle_event_detail.csv', index=False)
    pd.DataFrame(samples).to_csv(f'{OUT}/E7_calibration_samples.csv', index=False)
    print('done', len(df), round(time.time() - t0), flush=True)
