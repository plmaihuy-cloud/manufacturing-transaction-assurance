import importlib.util, csv, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

exp_path='/mnt/data/IoT_Thesis_Q1_Final_Package/code/experiment_original_v7.py'
spec=importlib.util.spec_from_file_location('expv7',exp_path)
exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

class MatchedSim(exp.Sim):
    def __init__(self,c,base=2.0,occ_penalty=0.8,wait_bonus=0.0):
        super().__init__(c); self.base=base; self.occ_penalty=occ_penalty; self.wait_bonus=wait_bonus
    def attempt(self,t,st):
        if self.pending[st] or not self.buf[st]: return
        j=self.buf[st][0]; o=self.obs[st]; age=max(0,t-o['src'])
        if not self.safe_obs(st):
            self.abstain+=1; self.push(t+1,'retry',(st,)); return
        down_occ=float(o.get('down_occ',0.0))
        qage=max(0.0,t-self.jobs[j]['ready'])/max(1e-9,self.c.q_limit)
        factor=max(0.35,min(1.50,1.0-self.occ_penalty*down_occ+self.wait_bonus*min(qage,1.0)))
        tau=self.base*factor
        if age<=tau: self.certify(t,st,j,age)
        else: self.pending[st]=True; self.query_request(t,st,j)

def run_policy(kind,seed):
    c=exp.Config(policy='adaptive' if kind in ('adaptive','matched') else 'fixed_query',delay_scale=1,threshold=1.5,query_servers=2,query_service_mean=8,seed=seed)
    r=MatchedSim(c).run() if kind=='matched' else exp.run(c)
    r['eval_policy']=kind; r['seed_eval']=seed
    r['hold_rate']=r['viol_hold']/r['dispatches'] if r['dispatches'] else 0.0
    return r

seeds=range(2001,2301)
rows=[]
for s in seeds:
    for kind in ['fixed_query','adaptive','matched']:
        rows.append(run_policy(kind,s))

outdir=Path('/mnt/data/q1_v4_assets'); outdir.mkdir(exist_ok=True)
df=pd.DataFrame(rows); df.to_csv(outdir/'r2_primary_replication.csv',index=False)

rng=np.random.default_rng(20261003)
def paired_arrays(a,b,metric):
    aa=df[df.eval_policy==a].sort_values('seed_eval')[metric].to_numpy(float)
    bb=df[df.eval_policy==b].sort_values('seed_eval')[metric].to_numpy(float)
    return aa,bb,aa-bb

def boot_ci(d,B=20000):
    n=len(d); idx=rng.integers(0,n,(B,n)); means=d[idx].mean(axis=1)
    return np.quantile(means,[.025,.975])

def tost_paired(d,margin):
    n=len(d); m=d.mean(); se=stats.sem(d)
    if se==0: return (0.0 if abs(m)<margin else 1.0), np.nan, np.nan
    # H01: mean <= -margin, H02: mean >= +margin. Reject both for equivalence.
    t1=(m+margin)/se; p1=1-stats.t.cdf(t1,n-1)
    t2=(m-margin)/se; p2=stats.t.cdf(t2,n-1)
    return max(p1,p2),p1,p2

def cohen_dz(d):
    sd=d.std(ddof=1); return d.mean()/sd if sd>0 else np.nan

records=[]
raw_tests=[]
for a,b in [('matched','fixed_query'),('matched','adaptive')]:
    for metric in ['completed','hold_rate']:
        aa,bb,d=paired_arrays(a,b,metric); lo,hi=boot_ci(d)
        t_p=stats.ttest_rel(aa,bb).pvalue
        try: w_p=stats.wilcoxon(d).pvalue
        except: w_p=np.nan
        rec={'contrast':f'{a}_vs_{b}','metric':metric,'a_mean':aa.mean(),'b_mean':bb.mean(),'paired_diff':d.mean(),'ci_low':lo,'ci_high':hi,'paired_t_p':t_p,'wilcoxon_p':w_p,'cohen_dz':cohen_dz(d)}
        if metric=='hold_rate':
            tp,p1,p2=tost_paired(d,0.001); rec.update({'tost_margin':0.001,'tost_p':tp,'tost_p_lower':p1,'tost_p_upper':p2})
        else:
            rec.update({'tost_margin':np.nan,'tost_p':np.nan,'tost_p_lower':np.nan,'tost_p_upper':np.nan})
        records.append(rec); raw_tests.append((len(records)-1,t_p))

# Holm adjustment across the 4 planned t-tests
m=len(raw_tests); sorted_tests=sorted(raw_tests,key=lambda x:x[1]); prev=0
adj=[None]*m
for rank,(idx,p) in enumerate(sorted_tests):
    val=min(1.0,(m-rank)*p)
    val=max(prev,val); prev=val; adj[idx]=val
for i,r in enumerate(records): r['holm_adjusted_p']=adj[i]

res=pd.DataFrame(records); res.to_csv(outdir/'r2_primary_inference.csv',index=False)

# descriptive summary
summary=[]
for pol,g in df.groupby('eval_policy'):
    summary.append({'policy':pol,'n':len(g),'completed_mean':g.completed.mean(),'completed_sd':g.completed.std(ddof=1),'hold_rate_mean':g.hold_rate.mean(),'aggregate_conflict_mean':g.conditional_violation.mean(),'query_rate_mean':g.query_rate.mean()})
pd.DataFrame(summary).to_csv(outdir/'r2_policy_summary.csv',index=False)

# Conservative MDE for completion pairwise at alpha=.0125 two-sided, 80% power, using observed paired SD from R2 only as a precision descriptor (not prospective power).
for a,b in [('matched','fixed_query'),('matched','adaptive')]:
    aa,bb,d=paired_arrays(a,b,'completed'); sd=d.std(ddof=1); n=len(d)
    zcrit=stats.norm.ppf(1-0.0125/2); zpow=stats.norm.ppf(.80)
    mde=(zcrit+zpow)*sd/math.sqrt(n)
    print(a,b,'diff',d.mean(),'sd',sd,'MDE_approx',mde)

print(res.to_string(index=False))
print(pd.DataFrame(summary).to_string(index=False))
