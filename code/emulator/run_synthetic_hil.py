import numpy as np, pandas as pd
from pathlib import Path
rng=np.random.default_rng(20261004)
classes=[('delayed_heartbeat',20,0.95,0.020,0.040,42,0.40,0.010),('late_hold',20,0.98,0.030,0.060,48,0.38,0.015),('reservation_race',20,0.97,0.040,0.050,54,0.36,0.020)]
rows=[]
for cls,n,pcover,pfs,pfv,med,sigma,pmis in classes:
    for r in range(1,n+1):
        oracle=bool(rng.random()<0.55)  # admissible absent injected unsafe state
        covered=bool(rng.random()<pcover)
        # RAF decision conditioned on oracle; error rates pre-specified above
        if oracle:
            raf_accept=not (rng.random()<pfv)
        else:
            raf_accept=bool(rng.random()<pfs)
        latency=float(rng.lognormal(np.log(med),sigma))
        mismatch=bool(rng.random()<pmis)
        # transaction timestamps in ms, all relative to episode start
        qstart=float(rng.uniform(5,15)); qend=qstart+float(rng.lognormal(np.log(10),0.28))
        cert=qend+float(rng.uniform(1.0,4.0)); commit=cert+latency
        act=commit+float(rng.lognormal(np.log(18),0.25))
        rows.append(dict(episode_id=f'{cls}_{r:02d}',fault_class=cls,replicate=r,
                         latent_state_covered=covered,raf_accept=raf_accept,oracle_admissible=oracle,
                         false_safe=raf_accept and not oracle,false_veto=(not raf_accept) and oracle,
                         commit_latency_ms=latency,actuation_mismatch=mismatch,
                         query_start_ms=qstart,query_end_ms=qend,certificate_time_ms=cert,
                         commit_time_ms=commit,actuation_time_ms=act,
                         evidence_origin='synthetic_HIL_stress_test',random_seed=20261004))
df=pd.DataFrame(rows)
out=Path(__file__).resolve().parents[2] / 'data' / 'emulator'; out.mkdir(parents=True, exist_ok=True)
df.to_csv(out/'hil_60_synthetic_executed.csv',index=False)
summary=[]
for name,g in [('overall',df),*list(df.groupby('fault_class'))]:
    unsafe=(~g.oracle_admissible).sum(); safe=g.oracle_admissible.sum()
    summary.append(dict(group=name,n=len(g),coverage=g.latent_state_covered.mean(),false_safe_count=g.false_safe.sum(),
                        false_safe_rate=(g.false_safe.sum()/unsafe if unsafe else np.nan),false_veto_count=g.false_veto.sum(),
                        false_veto_rate=(g.false_veto.sum()/safe if safe else np.nan),mismatch_count=g.actuation_mismatch.sum(),
                        mismatch_rate=g.actuation_mismatch.mean(),latency_median_ms=g.commit_latency_ms.median(),
                        latency_p95_ms=g.commit_latency_ms.quantile(.95),latency_p99_ms=g.commit_latency_ms.quantile(.99)))
pd.DataFrame(summary).to_csv(out/'hil_60_synthetic_summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
