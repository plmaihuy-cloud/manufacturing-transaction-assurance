import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

# Independent reduced-order comparator. All policies receive identical observations and disturbance seeds.
# It is not the omitted original simulator; it is an explicitly new surrogate experiment.

def simulate(seed, policy, T=4000):
    rng=np.random.default_rng(seed)
    queue=0; completed=0; false_safe=0; false_veto=0; query=0; hold_steps=0; lat=[]
    unsafe=False; obs_unsafe=False; obs_age=0; resource_occ=0.2
    for t in range(T):
        # arrivals / queue
        if rng.random()<0.22: queue=min(queue+1,25)
        # latent hazard state with persistence
        if unsafe:
            unsafe = rng.random()<0.82
        else:
            unsafe = rng.random()<0.035
        resource_occ=np.clip(0.86*resource_occ+0.14*rng.beta(2.2,4.5)+0.10*(queue/25),0,1)
        # asynchronous telemetry
        delivered=rng.random()>0.08
        if delivered:
            # sensor detects unsafe with high but imperfect sensitivity and small false positive rate
            obs_unsafe = unsafe if rng.random()<0.94 else (not unsafe)
            obs_age=0
        else:
            obs_age+=1
        if queue==0: continue

        # authoritative query returns current unsafe state after random latency; all policies have same query primitive
        def do_query():
            nonlocal query, obs_unsafe, obs_age
            query+=1
            qlat=max(1,int(np.ceil(rng.lognormal(np.log(2.2),0.45))))
            lat.append(qlat)
            obs_unsafe=unsafe; obs_age=0
            return qlat

        release=False
        if policy=='fixed_query':
            if obs_age>2:
                do_query()
            release = (not obs_unsafe) and resource_occ<0.82
        elif policy=='raf':
            # compatible-state gate: stale evidence expands to unsafe possibility
            if obs_age>1 or resource_occ>0.72:
                do_query()
            release = (not obs_unsafe) and obs_age<=1 and resource_occ<0.78
        elif policy=='robust_mpc':
            # matched-information receding horizon; robustly score release vs hold over 5-step horizon
            # uncertainty set widens with age and occupancy. Query if worst-case ambiguity too large.
            p_unsafe=0.02 + 0.16*min(obs_age,5) + (0.72 if obs_unsafe else 0.0) + 0.20*max(resource_occ-0.55,0)
            p_unsafe=min(max(p_unsafe,0),1)
            if p_unsafe>0.20 or resource_occ>0.70:
                do_query(); p_unsafe=0.98 if obs_unsafe else 0.02+0.20*max(resource_occ-0.55,0)
            # 5-step robust cost: release pays severe unsafe/race penalty; hold pays queue cost.
            release_cost=0.0; hold_cost=0.0; q=queue; occ=resource_occ; pu=p_unsafe
            for h in range(5):
                worst=min(1.0,pu+0.035*h+0.12*max(occ-0.65,0))
                release_cost += 4.5*worst + 2.0*max(occ-0.80,0) + 0.02*q
                hold_cost += 0.95 + 0.10*q
                q=min(25,q+0.22); occ=min(1,0.9*occ+0.1*0.45); pu=min(1,0.84*pu+0.035)
            release=(release_cost<hold_cost) and (not obs_unsafe) and resource_occ<0.84
        else: raise ValueError(policy)

        oracle_safe=(not unsafe) and resource_occ<0.86
        if release:
            if oracle_safe:
                queue-=1; completed+=1
            else:
                false_safe+=1
                # unsafe command does not complete useful production
        else:
            hold_steps+=1
            if oracle_safe: false_veto+=1
    return dict(seed=seed,policy=policy,completed=completed,false_safe=false_safe,false_veto=false_veto,
                query_count=query,hold_steps=hold_steps,query_latency_mean=(np.mean(lat) if lat else 0),
                false_safe_rate=false_safe/max(completed+false_safe,1), false_veto_rate=false_veto/max(T,1))

rows=[]
for seed in range(5101,5401):
    for p in ('fixed_query','raf','robust_mpc'):
        rows.append(simulate(seed,p))
df=pd.DataFrame(rows)
out=Path(__file__).resolve().parents[2] / 'data' / 'robust_mpc'; out.mkdir(parents=True, exist_ok=True); df.to_csv(out/'robust_mpc_raw.csv',index=False)
summary=df.groupby('policy').agg(n=('seed','count'),completed_mean=('completed','mean'),completed_sd=('completed','std'),
    false_safe_mean=('false_safe','mean'),false_safe_rate_mean=('false_safe_rate','mean'),
    false_veto_mean=('false_veto','mean'),query_mean=('query_count','mean'),query_latency_mean=('query_latency_mean','mean')).reset_index()
summary.to_csv(out/'robust_mpc_summary.csv',index=False)
# paired comparisons robust MPC vs RAF and fixed
recs=[]
for b in ('raf','fixed_query'):
    a=df[df.policy=='robust_mpc'].sort_values('seed')
    bb=df[df.policy==b].sort_values('seed')
    for metric in ('completed','false_safe','false_veto','query_count'):
        d=a[metric].to_numpy()-bb[metric].to_numpy()
        lo,hi=np.quantile([np.mean(np.random.default_rng(20261004+i).choice(d,len(d),replace=True)) for i in range(10000)],[.025,.975])
        recs.append(dict(contrast=f'robust_mpc_vs_{b}',metric=metric,diff_mean=d.mean(),ci_low=lo,ci_high=hi,p_paired_t=stats.ttest_rel(a[metric],bb[metric]).pvalue))
pd.DataFrame(recs).to_csv(out/'robust_mpc_inference.csv',index=False)
print(summary.to_string(index=False))
print(pd.DataFrame(recs).to_string(index=False))
