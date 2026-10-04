import importlib.util, math, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import gamma as gammafn

exp_path='/mnt/data/IoT_Thesis_Q1_Final_Package/code/experiment_original_v7.py'
spec=importlib.util.spec_from_file_location('expv7',exp_path); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

class VariantSim(exp.Sim):
    def __init__(self,c,delay_family='lognormal',query_family='exponential',matched=False,base=2.0,occ_penalty=0.8,wait_bonus=0.0):
        super().__init__(c); self.delay_family=delay_family; self.query_family=query_family; self.matched=matched; self.base=base; self.occ_penalty=occ_penalty; self.wait_bonus=wait_bonus
    def delay(self):
        if self.delay_family=='lognormal':
            d=self.r.lognormvariate(math.log(max(.02,self.c.delay_scale)),self.c.delay_sigma)
        elif self.delay_family=='gamma':
            sigma=self.c.delay_sigma
            mean=max(.02,self.c.delay_scale)*math.exp(0.5*sigma*sigma)
            cv=math.sqrt(math.exp(sigma*sigma)-1.0)
            shape=1.0/(cv*cv); scale=mean/shape
            d=self.r.gammavariate(shape,scale)
        else: raise ValueError(self.delay_family)
        if self.r.random()<self.c.burst_prob: d*=self.r.uniform(3,9)
        return d
    def query_request(self,t,st,j):
        k=min(range(len(self.qfree)), key=lambda x:self.qfree[x]); start=max(t,self.qfree[k]); wait=start-t
        if self.query_family=='exponential':
            svc=self.r.expovariate(1/self.c.query_service_mean)
        elif self.query_family=='weibull':
            shape=1.5
            scale=self.c.query_service_mean/gammafn(1+1/shape)
            svc=self.r.weibullvariate(scale,shape)
        else: raise ValueError(self.query_family)
        self.query_service_sum+=svc; done=start+svc; self.qfree[k]=done
        self.query_wait.append(wait); self.query_sojourn.append(wait+svc); self.queries+=1
        self.push(done,'query_done',(st,j,t))
    def attempt(self,t,st):
        if not self.matched: return super().attempt(t,st)
        if self.pending[st] or not self.buf[st]: return
        j=self.buf[st][0]; o=self.obs[st]; age=max(0,t-o['src'])
        if not self.safe_obs(st):
            self.abstain+=1; self.push(t+1,'retry',(st,)); return
        down_occ=float(o.get('down_occ',0.0)); qage=max(0.0,t-self.jobs[j]['ready'])/max(1e-9,self.c.q_limit)
        factor=max(0.35,min(1.50,1.0-self.occ_penalty*down_occ+self.wait_bonus*min(qage,1.0)))
        tau=self.base*factor
        if age<=tau: self.certify(t,st,j,age)
        else: self.pending[st]=True; self.query_request(t,st,j)

def run_one(policy,seed,delay_family,query_family,condition):
    if condition=='central': cservers,qmean=2,8.0
    elif condition=='bottleneck': cservers,qmean=1,16.0
    else: raise ValueError(condition)
    c=exp.Config(policy='adaptive' if policy in ('adaptive','matched') else 'fixed_query',delay_scale=1,threshold=1.5,query_servers=cservers,query_service_mean=qmean,seed=seed)
    sim=VariantSim(c,delay_family,query_family,matched=(policy=='matched'))
    r=sim.run(); r.update({'eval_policy':policy,'delay_family':delay_family,'query_family':query_family,'condition':condition,'seed_eval':seed})
    r['hold_rate']=r['viol_hold']/r['dispatches'] if r['dispatches'] else 0
    return r

rows=[]
for condition in ['central','bottleneck']:
  for dfam in ['lognormal','gamma']:
    for qfam in ['exponential','weibull']:
      for seed in range(4001,4101):
        for pol in ['fixed_query','adaptive','matched']:
          rows.append(run_one(pol,seed,dfam,qfam,condition))
outdir=Path('/mnt/data/q1_v4_assets'); outdir.mkdir(exist_ok=True)
df=pd.DataFrame(rows); df.to_csv(outdir/'distribution_sensitivity_raw.csv',index=False)
summary=df.groupby(['condition','delay_family','query_family','eval_policy']).agg(n=('seed_eval','size'),completed_mean=('completed','mean'),completed_sd=('completed','std'),conflict_mean=('conditional_violation','mean'),hold_rate_mean=('hold_rate','mean'),query_wait_p95_mean=('query_wait_p95','mean'),query_rate_mean=('query_rate','mean')).reset_index()
summary.to_csv(outdir/'distribution_sensitivity_summary.csv',index=False)
print(summary.to_string(index=False))
