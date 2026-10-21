from pathlib import Path
import re, math, json
import pandas as pd
import numpy as np
from scipy.stats import binomtest

ROOT=Path('/mnt/data/fullwork/daypsci/DAYPSCI_v1')
OUT=Path('/mnt/data/fullwork/replay_out'); OUT.mkdir(parents=True,exist_ok=True)
condition_map={
'pn_sensor_T1_batch02':'Sensor stuck-0','pn_sensor_T2_batch03':'Sensor stuck-1','pn_sensor_T3_batch04':'Sensor delay',
'pn_sensor_T4_batch05':'Sensor noise','pn_sensor_T5_batch06':'Sensor glitch','pn_sensor_T6_batch07':'Sensor degradation',
'pn_valve_T1_batch08':'Valve stuck-OFF','pn_valve_T2_batch09':'Valve stuck-ON','pn_valve_T3_batch10':'Valve delay',
'pn_combined_T1_batch11':'Combined sensor+valve'}
class_map={'Sensor delay':'Temporal','Sensor degradation':'Temporal','Valve delay':'Temporal',
'Sensor stuck-0':'Logical/hybrid','Sensor stuck-1':'Logical/hybrid','Sensor noise':'Logical/hybrid','Sensor glitch':'Logical/hybrid',
'Valve stuck-OFF':'Logical/hybrid','Valve stuck-ON':'Logical/hybrid','Combined sensor+valve':'Logical/hybrid'}

def wilson(k,n,z=1.959963984540054):
    if n==0:return (np.nan,np.nan)
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

rows=[]
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():continue
    csvs=list(folder.glob('*.csv')); readme=folder/'README.md'
    if not csvs:continue
    df=pd.read_csv(csvs[0],sep=';')
    txt=readme.read_text(encoding='utf-8',errors='ignore') if readme.exists() else ''
    injected=set(map(int,re.findall(r'Part\s+(\d+)\s*\(',txt,re.I)))
    # Physical batches are documented as 25 parts; part_id=0 is setup/pre-cycle state and is excluded.
    for pid in range(1,26):
        g=df[df.part_id==pid].copy()
        if g.empty: continue
        anomaly=bool((pd.to_numeric(g.anomaly,errors='coerce').fillna(0)!=0).any())
        fault=bool((pd.to_numeric(g.fault,errors='coerce').fillna(0)!=0).any())
        d2=bool((pd.to_numeric(g.gemma,errors='coerce').fillna(-1)==7).any())
        # Independent experimental-plan oracle: a documented controlled injection makes a nominal downstream release inadmissible
        # for this replay assay. It is not a factory safety oracle.
        oracle_admissible = pid not in injected
        # Comparator: controller hard-fault state only.
        d2_accept = not d2
        # Evidence-aware replay gate: preserve any source-visible abnormality (anomaly or hard fault) rather than D2 alone.
        evidence_accept = not (anomaly or fault)
        rows.append({
            'batch':folder.name,
            'condition':'Baseline' if folder.name=='pn_baseline__batch01' else condition_map.get(folder.name,folder.name),
            'class':'Baseline' if folder.name=='pn_baseline__batch01' else class_map.get(condition_map.get(folder.name,''),''),
            'part_id':pid,'documented_injection':pid in injected,'oracle_admissible':oracle_admissible,
            'anomaly_observable':anomaly,'fault_flag':fault,'gemma_d2':d2,
            'd2_only_accept':d2_accept,'evidence_gate_accept':evidence_accept,
            'event_count':len(g),
            'first_ts_plc':str(g.ts_plc.iloc[0]),'last_ts_plc':str(g.ts_plc.iloc[-1]),
            'max_delta_time_ms':float(pd.to_numeric(g.delta_time_ms,errors='coerce').max()),
        })
R=pd.DataFrame(rows)
R['d2_false_safe']=(~R.oracle_admissible)&R.d2_only_accept
R['d2_false_veto']=R.oracle_admissible&(~R.d2_only_accept)
R['evidence_false_safe']=(~R.oracle_admissible)&R.evidence_gate_accept
R['evidence_false_veto']=R.oracle_admissible&(~R.evidence_gate_accept)
R.to_csv(OUT/'daypsci_release_replay_parts.csv',index=False)

def summary(name,accept_col):
    n=len(R); unsafe=int((~R.oracle_admissible).sum()); safe=int(R.oracle_admissible.sum())
    fs=int(((~R.oracle_admissible)&R[accept_col]).sum()); fv=int((R.oracle_admissible&(~R[accept_col])).sum())
    detect=unsafe-fs; admit=safe-fv
    fslo,fshi=wilson(fs,unsafe); fvlo,fvhi=wilson(fv,safe)
    dlo,dhi=wilson(detect,unsafe); slo,shi=wilson(admit,safe)
    return {'gate':name,'n_parts':n,'oracle_inadmissible_n':unsafe,'oracle_admissible_n':safe,
            'false_safe_n':fs,'false_safe_pct':100*fs/unsafe,'false_safe_wilson95_low_pct':100*fslo,'false_safe_wilson95_high_pct':100*fshi,
            'false_veto_n':fv,'false_veto_pct':100*fv/safe,'false_veto_wilson95_low_pct':100*fvlo,'false_veto_wilson95_high_pct':100*fvhi,
            'injected_part_detection_n':detect,'injected_part_detection_pct':100*detect/unsafe,
            'safe_part_admission_n':admit,'safe_part_admission_pct':100*admit/safe}
S=pd.DataFrame([summary('D2-only comparator','d2_only_accept'),summary('Evidence-aware replay gate','evidence_gate_accept')])
S.to_csv(OUT/'daypsci_release_replay_summary.csv',index=False)

# Paired discordance of correctness across all parts.
R['d2_correct']=R.d2_only_accept==R.oracle_admissible
R['evidence_correct']=R.evidence_gate_accept==R.oracle_admissible
b=int((~R.d2_correct & R.evidence_correct).sum())
c=int((R.d2_correct & ~R.evidence_correct).sum())
p=binomtest(min(b,c),n=b+c,p=0.5,alternative='two-sided').pvalue if b+c else 1.0
paired=pd.DataFrame([{'n_pairs':len(R),'d2_wrong_evidence_right_b':b,'d2_right_evidence_wrong_c':c,
                      'net_accuracy_gain_pp':100*(R.evidence_correct.mean()-R.d2_correct.mean()),
                      'd2_accuracy_pct':100*R.d2_correct.mean(),'evidence_accuracy_pct':100*R.evidence_correct.mean(),
                      'exact_mcnemar_p':p}])
paired.to_csv(OUT/'daypsci_release_replay_paired.csv',index=False)

# Breakdown by source-defined fault class and condition.
def group_metrics(g):
    unsafe=(~g.oracle_admissible); safe=g.oracle_admissible
    return pd.Series({
      'n':len(g),'injected_n':int(unsafe.sum()),'safe_n':int(safe.sum()),
      'd2_false_safe_n':int((unsafe & g.d2_only_accept).sum()),
      'evidence_false_safe_n':int((unsafe & g.evidence_gate_accept).sum()),
      'd2_false_veto_n':int((safe & ~g.d2_only_accept).sum()),
      'evidence_false_veto_n':int((safe & ~g.evidence_gate_accept).sum()),
      'anomaly_observable_in_injected_n':int((unsafe & g.anomaly_observable).sum()),
      'd2_in_injected_n':int((unsafe & g.gemma_d2).sum())})
R.groupby('class',dropna=False).apply(group_metrics,include_groups=False).reset_index().to_csv(OUT/'daypsci_release_replay_by_class.csv',index=False)
R.groupby(['condition','class'],dropna=False).apply(group_metrics,include_groups=False).reset_index().to_csv(OUT/'daypsci_release_replay_by_condition.csv',index=False)

# Key outcomes JSON for deterministic manuscript injection.
key={'n_parts':len(R),'injected':int((~R.oracle_admissible).sum()),'safe':int(R.oracle_admissible.sum()),
     'd2_false_safe':int(R.d2_false_safe.sum()),'evidence_false_safe':int(R.evidence_false_safe.sum()),
     'd2_false_veto':int(R.d2_false_veto.sum()),'evidence_false_veto':int(R.evidence_false_veto.sum()),
     'd2_accuracy':float(R.d2_correct.mean()),'evidence_accuracy':float(R.evidence_correct.mean()),
     'mcnemar_b':b,'mcnemar_c':c,'mcnemar_p':float(p)}
(OUT/'daypsci_release_replay_key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
print(R[['batch','part_id','documented_injection','anomaly_observable','gemma_d2','d2_only_accept','evidence_gate_accept','oracle_admissible']].to_string(index=False))
print('\nSUMMARY\n',S.to_string(index=False))
print('\nPAIRED\n',paired.to_string(index=False))
print('\nCLASS\n',pd.read_csv(OUT/'daypsci_release_replay_by_class.csv').to_string(index=False))
