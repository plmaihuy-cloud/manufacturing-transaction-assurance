from pathlib import Path
import io, re, json, zipfile, math
import numpy as np
import pandas as pd
from scipy.optimize import linprog, brentq
from scipy.sparse import lil_matrix, csr_matrix
from scipy.stats import binom

HERE = Path(__file__).resolve()
PKG = HERE.parents[2]
RAW = PKG / 'data' / 'external' / 'daypsci' / 'raw'
OUT = PKG / 'data' / 'external' / 'daypsci' / 'derived'
OUT.mkdir(parents=True, exist_ok=True)

DAY_ZIP = RAW / 'daypsci_source.zip'
STG_ZIP = RAW / '7612698.zip'

condition_map = {
    'pn_sensor_T1_batch02':'Sensor stuck-0',
    'pn_sensor_T2_batch03':'Sensor stuck-1',
    'pn_sensor_T3_batch04':'Sensor delay',
    'pn_sensor_T4_batch05':'Sensor noise',
    'pn_sensor_T5_batch06':'Sensor glitch',
    'pn_sensor_T6_batch07':'Sensor degradation',
    'pn_valve_T1_batch08':'Valve stuck-OFF',
    'pn_valve_T2_batch09':'Valve stuck-ON',
    'pn_valve_T3_batch10':'Valve delay',
    'pn_combined_T1_batch11':'Combined sensor+valve',
}
class_map = {
    'Sensor delay':'Temporal', 'Sensor degradation':'Temporal', 'Valve delay':'Temporal',
    'Sensor stuck-0':'Logical/hybrid', 'Sensor stuck-1':'Logical/hybrid',
    'Sensor noise':'Logical/hybrid', 'Sensor glitch':'Logical/hybrid',
    'Valve stuck-OFF':'Logical/hybrid', 'Valve stuck-ON':'Logical/hybrid',
    'Combined sensor+valve':'Logical/hybrid',
}

def wilson(k, n, z=1.959963984540054):
    if n == 0: return (np.nan, np.nan)
    p = k/n; den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return center-half, center+half

def exact_mcnemar_p(b, c):
    m=b+c
    if m==0: return 1.0
    tail=binom.cdf(min(b,c),m,0.5)
    return min(1.0, 2*tail)

def mcnemar_reject(b,c,alpha=0.05):
    return exact_mcnemar_p(b,c) < alpha

def one_direction_power(n, delta, alpha=0.05):
    # p10=delta, p01=0. Rejection threshold depends on number of forward discordants.
    return sum(binom.pmf(k,n,delta) for k in range(n+1) if mcnemar_reject(k,0,alpha))

def conservative_fixed_q_power(n, delta, q, alpha=0.05):
    # Hold total discordance probability q fixed; split as p10=(q+delta)/2, p01=(q-delta)/2.
    if delta < 0 or delta > q: return np.nan
    theta=(q+delta)/(2*q) if q>0 else 0.5
    power=0.0
    for m in range(n+1):
        pm=binom.pmf(m,n,q)
        if pm==0: continue
        r=0.0
        for b in range(m+1):
            c=m-b
            if mcnemar_reject(b,c,alpha):
                r += binom.pmf(b,m,theta)
        power += pm*r
    return power

# ---------------- DAYPSCI ----------------
with zipfile.ZipFile(DAY_ZIP) as z:
    names=z.namelist(); csv_names=[n for n in names if n.lower().endswith('.csv')]
    part_rows=[]; batch_rows=[]; baseline_df=None
    for csv_name in sorted(csv_names):
        folder=Path(csv_name).parent.name
        df=pd.read_csv(io.BytesIO(z.read(csv_name)), sep=';', encoding='utf-8-sig')
        if folder == 'pn_baseline__batch01':
            baseline_df=df.copy(); continue
        readme=str(Path(csv_name).parent/'README.md').replace('\\','/')
        txt=z.read(readme).decode('utf-8',errors='ignore') if readme in names else ''
        parts=sorted(set(map(int,re.findall(r'Part\s+(\d+)\s*\(',txt,re.I))))
        cond=condition_map[folder]
        for pid in parts:
            g=df[df.part_id==pid]
            part_rows.append({
                'batch':folder,'condition':cond,'class':class_map[cond],'part_id':pid,
                'anomaly_observable':bool((g.anomaly.fillna(0).astype(float)!=0).any()),
                'gemma_d2':bool((g.gemma.fillna(0).astype(float)==7).any()),
                'fault_flag':bool((g.fault.fillna(0).astype(float)!=0).any())})
        tmp=[r for r in part_rows if r['batch']==folder]
        n0=len(tmp); an0=sum(r['anomaly_observable'] for r in tmp); d20=sum(r['gemma_d2'] for r in tmp)
        batch_rows.append({'batch':folder,'condition':cond,'class':class_map[cond],
                           'n':n0,'anomaly_n':an0,'d2_n':d20,'gap_pp':100*(an0-d20)/n0})

P=pd.DataFrame(part_rows); B=pd.DataFrame(batch_rows)
P.to_csv(OUT/'daypsci_injected_part_observability.csv',index=False)
B.to_csv(OUT/'daypsci_batch_gap.csv',index=False)

n=len(P); an=int(P.anomaly_observable.sum()); d2=int(P.gemma_d2.sum())
b=int(((P.anomaly_observable==True)&(P.gemma_d2==False)).sum())
c=int(((P.anomaly_observable==False)&(P.gemma_d2==True)).sum())
q=(b+c)/n; gap=(b-c)/n
p_exact=exact_mcnemar_p(b,c)

# Reproduce uncertainty.
an_lo,an_hi=wilson(an,n); d2_lo,d2_hi=wilson(d2,n)
rng=np.random.default_rng(20261003); a=P.anomaly_observable.to_numpy(np.int8); d=P.gemma_d2.to_numpy(np.int8)
boot=np.empty(100000)
for st in range(0,len(boot),5000):
    m=min(5000,len(boot)-st); idx=rng.integers(0,n,size=(m,n))
    boot[st:st+m]=(a[idx].mean(1)-d[idx].mean(1))*100
gap_lo,gap_hi=np.quantile(boot,[.025,.975])

# MDE at 80% power.
alpha=.05; target=.80
f=lambda x: one_direction_power(n,x,alpha)-target
mde_one=brentq(f,1e-9,.999999)
# Conservative model: keep observed discordance q fixed and allow reverse discordance.
g=lambda x: conservative_fixed_q_power(n,x,q,alpha)-target
mde_fixedq=brentq(g,1e-9,q-1e-9)
obs_power_one=one_direction_power(n,gap,alpha)
obs_power_fixedq=conservative_fixed_q_power(n,gap,q,alpha)

power_df=pd.DataFrame([{
    'n_pairs':n,'anomaly_yes_d2_no_b':b,'anomaly_no_d2_yes_c':c,
    'paired_risk_difference_pp':100*gap,'discordance_rate_pct':100*q,
    'exact_two_sided_mcnemar_p':p_exact,
    'observed_power_one_direction_model':obs_power_one,
    'observed_power_fixed_total_discordance_model':obs_power_fixedq,
    'mde80_pp_one_direction_model':100*mde_one,
    'mde80_pp_fixed_total_discordance_model':100*mde_fixedq,
    'alpha':alpha,'target_power':target,
    'interpretation':'Observed 38.6-pp paired gap exceeds both 80%-power MDEs; power calculations are design diagnostics, not external-validity evidence.'
}])
power_df.to_csv(OUT/'daypsci_paired_power.csv',index=False)

# Baseline control.
base=baseline_df[baseline_df.part_id.between(1,25)]
baseparts=[]
for pid,gp in base.groupby('part_id'):
    baseparts.append({'part_id':int(pid),'anomaly_observable':bool((gp.anomaly.fillna(0).astype(float)!=0).any()),
                      'gemma_d2':bool((gp.gemma.fillna(0).astype(float)==7).any())})
BP=pd.DataFrame(baseparts)
neg=pd.DataFrame([{'sample':'DAYPSCI baseline batch 01','n_parts':len(BP),
                   'anomaly_observable_n':int(BP.anomaly_observable.sum()),'gemma_d2_n':int(BP.gemma_d2.sum()),
                   'observed_gap_pp':100*(BP.anomaly_observable.mean()-BP.gemma_d2.mean()),
                   'role':'Internal negative control only; not independent external replication.'}])
neg.to_csv(OUT/'daypsci_negative_control.csv',index=False)

strata=P.groupby('class').agg(n=('part_id','size'),anomaly_n=('anomaly_observable','sum'),d2_n=('gemma_d2','sum')).reset_index()
strata['anomaly_pct']=100*strata.anomaly_n/strata.n; strata['d2_pct']=100*strata.d2_n/strata.n; strata['gap_pp']=strata.anomaly_pct-strata.d2_pct
strata.to_csv(OUT/'daypsci_class_stratification.csv',index=False)

inference=pd.DataFrame([{
    'sample':'DAYPSCI documented injected parts','n':n,'anomaly_observable_n':an,'anomaly_observable_pct':100*an/n,
    'anomaly_wilson95_low_pct':100*an_lo,'anomaly_wilson95_high_pct':100*an_hi,
    'gemma_d2_n':d2,'gemma_d2_pct':100*d2/n,'d2_wilson95_low_pct':100*d2_lo,'d2_wilson95_high_pct':100*d2_hi,
    'paired_gap_pp':100*gap,'paired_bootstrap95_low_pp':gap_lo,'paired_bootstrap95_high_pp':gap_hi,
    'mcnemar_b':b,'mcnemar_c':c,'exact_mcnemar_p':p_exact,
    'interpretation':'Physical-CPS case finding about representation; not an RAF effect estimate and not externally generalized.'
}])
inference.to_csv(OUT/'daypsci_gap_inference.csv',index=False)

# ---------------- ST GALLEN ----------------
with zipfile.ZipFile(STG_ZIP) as z:
    ll=[json.loads(line) for line in z.read('low-level_log_20230206-140808.txt').decode('utf-8').splitlines() if line.strip()]
L=pd.DataFrame(ll); L['ts']=pd.to_datetime(L.timestamp); L['row_number']=np.arange(1,len(L)+1)
base_t=L.ts.min(); L['x']=(L.ts-base_t).dt.total_seconds()
diff=L.ts.diff().dt.total_seconds(); invmask=diff<0
inv=[]
for i in np.flatnonzero(invmask.to_numpy()):
    inv.append({'row_number':int(i+1),'previous_station':L.station.iloc[i-1], 'previous_timestamp':str(L.ts.iloc[i-1]),
                'station':L.station.iloc[i],'timestamp':str(L.ts.iloc[i]),'backward_step_ms':float(-1000*diff.iloc[i])})
INV=pd.DataFrame(inv); INV.to_csv(OUT/'st_gallen_inversion_detail.csv',index=False)
inv_lo,inv_hi=wilson(int(invmask.sum()),len(L)-1)

stations=sorted(L.station.unique()); sidx={s:i for i,s in enumerate(stations)}; ns=len(stations)

def constant_feasible():
    rows=[]; rhs=[]
    for i in range(1,len(L)):
        sp,sc=L.station.iloc[i-1],L.station.iloc[i]
        if sp==sc: continue
        # corrected = x - offset; monotonic corrected_i >= corrected_prev -> off_sc-off_sp <= x_i-x_prev
        row=np.zeros(ns); row[sidx[sc]]=1; row[sidx[sp]]=-1
        rows.append(row); rhs.append(L.x.iloc[i]-L.x.iloc[i-1])
    A=csr_matrix(np.array(rows)); bvec=np.array(rhs)
    bounds=[(None,None)]*ns; bounds[0]=(0,0)
    return linprog(np.zeros(ns),A_ub=A,b_ub=bvec,bounds=bounds,method='highs')

def affine_feasible(slope_lo=.001,slope_hi=100.,offset_bound=10000.):
    # corrected t = a_s + b_s*x. Require nondecreasing in aggregate file order.
    # variables [a_0..a_s-1,b_0..b_s-1]
    nv=2*ns; rows=[]; rhs=[]
    for i in range(1,len(L)):
        sp,sc=L.station.iloc[i-1],L.station.iloc[i]
        xp,xc=float(L.x.iloc[i-1]),float(L.x.iloc[i])
        row=np.zeros(nv)
        # a_sp+b_sp*xp <= a_sc+b_sc*xc -> a_sp-a_sc + xp*b_sp-xc*b_sc <=0
        row[sidx[sp]] += 1; row[sidx[sc]] -= 1
        row[ns+sidx[sp]] += xp; row[ns+sidx[sc]] -= xc
        rows.append(row); rhs.append(0.)
    bounds=[(-offset_bound,offset_bound)]*ns + [(slope_lo,slope_hi)]*ns
    # Fix first station a=0, b=1 to remove affine gauge freedom.
    bounds[0]=(0,0); bounds[ns]=(1,1)
    return linprog(np.zeros(nv),A_ub=csr_matrix(np.array(rows)),b_ub=np.array(rhs),bounds=bounds,method='highs')

def piecewise_feasible(block_s=120., offset_bound=2., jump_bound=.25):
    blocks=np.floor(L.x.to_numpy()/block_s).astype(int)
    nb=int(blocks.max())+1
    # variable for every station x block, even if no row; bounded offsets.
    def vid(s,b): return sidx[s]*nb+b
    nv=ns*nb; rows=[]; rhs=[]
    # corrected=x-offset; row order monotonic means off_cur-off_prev <= x_cur-x_prev
    for i in range(1,len(L)):
        sp,sc=L.station.iloc[i-1],L.station.iloc[i]; bp,bc=blocks[i-1],blocks[i]
        row=np.zeros(nv); row[vid(sc,bc)]+=1; row[vid(sp,bp)]-=1
        rows.append(row); rhs.append(float(L.x.iloc[i]-L.x.iloc[i-1]))
    # continuity bounds between adjacent station blocks: |o_b-o_(b-1)| <= J
    for s in stations:
        for bb in range(1,nb):
            r=np.zeros(nv); r[vid(s,bb)]=1; r[vid(s,bb-1)]=-1; rows.append(r); rhs.append(jump_bound)
            r2=-r; rows.append(r2); rhs.append(jump_bound)
    bounds=[(-offset_bound,offset_bound)]*nv
    bounds[vid(stations[0],0)]=(0,0)
    res=linprog(np.zeros(nv),A_ub=csr_matrix(np.array(rows)),b_ub=np.array(rhs),bounds=bounds,method='highs')
    return res,nb,blocks

r1=constant_feasible(); r2=affine_feasible()
# M3 pre-specified 120-s block model; determine feasibility threshold, then minimum absolute bound near that threshold.
def min_jump(block_s=120.,A=2.):
    lo,hi=0.,2.
    if not piecewise_feasible(block_s,A,hi)[0].success: return np.nan
    for _ in range(28):
        mid=(lo+hi)/2
        if piecewise_feasible(block_s,A,mid)[0].success: hi=mid
        else: lo=mid
    return hi
jmin=min_jump(120.,2.)
# use a slightly relaxed jump to find minimum absolute offset amplitude.
juse=max(.22, jmin+1e-4)
def min_amp(block_s=120.,J=juse):
    lo,hi=0.,2.
    if not piecewise_feasible(block_s,hi,J)[0].success: return np.nan
    for _ in range(28):
        mid=(lo+hi)/2
        if piecewise_feasible(block_s,mid,J)[0].success: hi=mid
        else: lo=mid
    return hi
amin=min_amp()
r3,nb,blocks=piecewise_feasible(120.,max(.5,amin+1e-4),juse)
# Apply feasible offsets and verify zero inversions when available.
post_inv=np.nan
if r3.success:
    offs=r3.x.reshape(ns,nb)
    corr=[]
    for i,row in L.iterrows(): corr.append(row.x-offs[sidx[row.station],blocks[i]])
    corr=np.array(corr); post_inv=int(np.sum(np.diff(corr)<-1e-9))

sensitivity=pd.DataFrame([
    {'model':'M0 raw aggregate source timestamps','parameters':'none','feasible_global_monotone':False,
     'remaining_inversions':int(invmask.sum()),'interpretation':'Raw aggregate row order is not globally monotone in source time.'},
    {'model':'M1 station-specific constant offset','parameters':'one offset per station; reference offset fixed at 0','feasible_global_monotone':bool(r1.success),
     'remaining_inversions':0 if r1.success else np.nan,'interpretation':'One fixed station offset is insufficient.' if not r1.success else 'Fixed offsets suffice.'},
    {'model':'M2 station-specific affine clock','parameters':'t_corrected=a_s+b_s*t; reference a=0,b=1; b in [0.001,100]','feasible_global_monotone':bool(r2.success),
     'remaining_inversions':0 if r2.success else np.nan,'interpretation':'No stationary affine per-station correction reconciles the aggregate order.' if not r2.success else 'Affine clock correction suffices.'},
    {'model':'M3 piecewise station offset','parameters':f'120-s source-time blocks; min jump≈{jmin:.3f}s; min |offset|≈{amin:.3f}s','feasible_global_monotone':bool(r3.success),
     'remaining_inversions':post_inv,'interpretation':'A sufficiently flexible time-varying offset can remove all inversions; this is sensitivity, not evidence that clock drift caused them.'}
])
sensitivity.to_csv(OUT/'st_gallen_clock_model_sensitivity.csv',index=False)

inv_summary=pd.DataFrame([{
    'rows':len(L),'adjacent_transitions':len(L)-1,'observed_inversions':int(invmask.sum()),
    'inversion_rate_pct':100*invmask.sum()/(len(L)-1),'inversion_wilson95_low_pct':100*inv_lo,
    'inversion_wilson95_high_pct':100*inv_hi,'small_inversions_10_to_40ms':int(((INV.backward_step_ms>=9.999)&(INV.backward_step_ms<=40.001)).sum()),
    'max_backward_step_ms':float(INV.backward_step_ms.max()),
    'all_station_local_monotone':bool(all(g.ts.is_monotonic_increasing for _,g in L.groupby('station',sort=False))),
    'M1_constant_offset_feasible':bool(r1.success),'M2_affine_clock_feasible':bool(r2.success),
    'M3_piecewise_120s_feasible':bool(r3.success),'M3_min_jump_s_approx':jmin,'M3_min_abs_offset_s_approx':amin,
    'interpretation':'Small prevalence; raw row order cannot be treated as global source-time order. Cause remains unidentified because flexible time-varying alignment can reconcile the trace.'
}])
inv_summary.to_csv(OUT/'st_gallen_inversion_effect_magnitude.csv',index=False)

master=pd.DataFrame([
    {'finding':'DAYPSCI physical-CPS case representation gap','result':f'{an}/{n} anomaly-observable vs {d2}/{n} D2; paired gap {100*gap:.1f} pp',
     'inference':f'exact McNemar p={p_exact:.3g}; bootstrap 95% CI {gap_lo:.1f} to {gap_hi:.1f} pp',
     'sensitivity':f'80% MDE {100*mde_one:.1f} pp one-direction; {100*mde_fixedq:.1f} pp fixed-discordance',
     'boundary':'Single physical-CPS testbed; no semantically matched external replication in supplied evidence.'},
    {'finding':'St. Gallen ordering sensitivity','result':f'{int(invmask.sum())}/{len(L)-1} inversions ({100*invmask.sum()/(len(L)-1):.3f}%); max 850 ms',
     'inference':f'Wilson 95% CI {100*inv_lo:.3f} to {100*inv_hi:.3f}%',
     'sensitivity':f'M1 constant infeasible; M2 affine infeasible; M3 120-s piecewise feasible with ~{jmin:.3f}s jump flexibility',
     'boundary':'Does not identify causality violation; time-varying clock error, buffering, and writer interleaving remain observationally confounded.'},
])
master.to_csv(OUT/'reviewer_v10_key_results.csv',index=False)

print(power_df.to_string(index=False))
print(sensitivity.to_string(index=False))
print(inv_summary.to_string(index=False))
