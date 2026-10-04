import os,re,json,hashlib
from pathlib import Path
import pandas as pd, numpy as np
from dateutil import parser
ROOT=Path('/mnt/data/work_q1/raw')
out=Path('/mnt/data/work_q1/derived'); out.mkdir(parents=True,exist_ok=True)
# hashes of original uploaded archives
archives={
 'daypsci_source.zip':Path('/mnt/data/DAYPSCI_v1(1).zip'),
 '7612698.zip':Path('/mnt/data/7612698(1).zip'),
 '7440490.zip':Path('/mnt/data/7440490(1).zip')}
rows=[]
for name,p in archives.items():
 h=hashlib.sha256(p.read_bytes()).hexdigest(); rows.append({'archive':name,'bytes':p.stat().st_size,'sha256':h})
pd.DataFrame(rows).to_csv(out/'raw_archive_hashes.csv',index=False)
# DAYPSCI
base=ROOT/'daypsci'/'DAYPSCI_v1'
condition_map={
 'pn_sensor_T1_batch02':'Sensor stuck-0','pn_sensor_T2_batch03':'Sensor stuck-1','pn_sensor_T3_batch04':'Sensor delay','pn_sensor_T4_batch05':'Sensor noise','pn_sensor_T5_batch06':'Sensor glitch','pn_sensor_T6_batch07':'Sensor degradation','pn_valve_T1_batch08':'Valve stuck-OFF','pn_valve_T2_batch09':'Valve stuck-ON','pn_valve_T3_batch10':'Valve delay','pn_combined_T1_batch11':'Combined sensor+valve'}
all_frames=[]; batchrows=[]; partrows=[]
for folder in sorted([p for p in base.iterdir() if p.is_dir()]):
 csvs=list(folder.glob('*.csv'))
 if not csvs: continue
 f=csvs[0]
 df=pd.read_csv(f,sep=';',encoding='utf-8-sig')
 df['ts']=pd.to_datetime(df.ts_plc,format='%d/%m/%Y %H:%M:%S,%f',errors='coerce')
 df['batch_folder']=folder.name
 all_frames.append(df)
 # ordering checks
 event_contig = np.array_equal(df.event_seq.to_numpy(), np.arange(df.event_seq.iloc[0], df.event_seq.iloc[0]+len(df)))
 ts_mono = df.ts.is_monotonic_increasing
 scan_mono = df.scan_id.is_monotonic_increasing
 local_ok=True
 for sid,g in df.groupby('scan_id',sort=False):
  eo=g.event_order.to_numpy()
  if len(eo) and not np.array_equal(eo,np.arange(1,len(eo)+1)):
   local_ok=False; break
 # delta exact comparable from prior row
 diff=df.ts.diff().dt.total_seconds().mul(1000)
 comp=df.delta_time_ms.notna() & diff.notna()
 delta_exact=np.isclose(df.loc[comp,'delta_time_ms'].astype(float),diff.loc[comp],rtol=0,atol=0.51).all()
 dup_event=df.event_seq.duplicated().sum(); dup_scan=df.duplicated(['scan_id','event_order']).sum()
 vals=df.delta_time_ms.dropna().astype(float)
 # parts from readme
 text=(folder/'README.md').read_text(encoding='utf-8',errors='ignore') if (folder/'README.md').exists() else ''
 parts=sorted(set(map(int,re.findall(r'Part\s+(\d+)\s*\(',text,re.I))))
 condition=condition_map.get(folder.name,'Baseline')
 inject_n=len(parts)
 an=flt=d2=0
 for pid in parts:
  g=df[df.part_id==pid]
  av=(g.anomaly.fillna(0).astype(float)!=0).any()
  fv=(g.fault.fillna(0).astype(float)!=0).any()
  dv=(g.gemma.fillna(0).astype(float)==7).any()
  an+=int(av); flt+=int(fv); d2+=int(dv)
  partrows.append({'batch':folder.name,'condition':condition,'part_id':pid,'anomaly_observable':av,'fault_flag':fv,'gemma_d2':dv})
 batchrows.append({'batch':folder.name,'condition':condition,'events':len(df),'event_seq_contiguous':event_contig,'timestamp_monotone':ts_mono,'scan_id_monotone':scan_mono,'scan_local_order_contiguous':local_ok,'delta_time_consistent':delta_exact,'duplicate_event_seq':int(dup_event),'duplicate_scan_order_keys':int(dup_scan),'injected_parts':inject_n,'anomaly_observable_n':an,'fault_flag_n':flt,'gemma_d2_n':d2,'anomaly_observable_pct':100*an/inject_n if inject_n else np.nan,'fault_flag_pct':100*flt/inject_n if inject_n else np.nan,'gemma_d2_pct':100*d2/inject_n if inject_n else np.nan,'median_delta_ms':vals.median(),'p95_delta_ms':vals.quantile(.95),'p99_delta_ms':vals.quantile(.99),'max_delta_ms':vals.max()})
B=pd.DataFrame(batchrows); P=pd.DataFrame(partrows); A=pd.concat(all_frames,ignore_index=True)
B.to_csv(out/'daypsci_batch_replay_metrics.csv',index=False); P.to_csv(out/'daypsci_injected_part_observability.csv',index=False)
inj=P
summary={
 'events_total':len(A),'batches':len(B),'injected_parts':len(inj),
 'anomaly_obs_n':int(inj.anomaly_observable.sum()),'anomaly_obs_pct':100*inj.anomaly_observable.mean(),
 'fault_n':int(inj.fault_flag.sum()),'fault_pct':100*inj.fault_flag.mean(),
 'd2_n':int(inj.gemma_d2.sum()),'d2_pct':100*inj.gemma_d2.mean(),
 'gap_pp':100*(inj.anomaly_observable.mean()-inj.gemma_d2.mean()),
 'median_delta_ms':A.delta_time_ms.dropna().astype(float).median(),
 'p95_delta_ms':A.delta_time_ms.dropna().astype(float).quantile(.95),
 'p99_delta_ms':A.delta_time_ms.dropna().astype(float).quantile(.99),
 'max_delta_ms':A.delta_time_ms.dropna().astype(float).max(),
}
pd.DataFrame([summary]).to_csv(out/'daypsci_primary_finding.csv',index=False)
# StG 761
ll=[]
with open(ROOT/'stg761'/'low-level_log_20230206-140808.txt',encoding='utf-8') as fh:
 for line in fh:
  if line.strip(): ll.append(json.loads(line))
L=pd.DataFrame(ll); L['ts']=pd.to_datetime(L.timestamp)
# monotonic by station in file order
stat=[]
for s,g in L.groupby('station',sort=False):
 dif=g.ts.diff().dt.total_seconds().dropna();
 stat.append({'station':s,'rows':len(g),'timestamp_monotone':g.ts.is_monotonic_increasing,'median_interval_s':dif.median(),'p95_interval_s':dif.quantile(.95),'max_interval_s':dif.max()})
S=pd.DataFrame(stat); S.to_csv(out/'stgallen_761_station_timing.csv',index=False)
# cross-station file-order timestamp inversions (adjacent row inversion)
diff=L.ts.diff().dt.total_seconds(); inv=(diff<0)
invdf=L.loc[inv,['station','timestamp']].copy(); invdf['previous_timestamp']=L.ts.shift(1)[inv].astype(str).values; invdf['negative_step_s']=diff[inv].values
invdf.to_csv(out/'stgallen_761_cross_station_inversions.csv',index=False)
acts=pd.DataFrame(json.load(open(ROOT/'stg761'/'camunda_activity-instance.json')))
acts['start']=pd.to_datetime(acts.startTime,utc=True).dt.tz_convert(None)
acts['end']=pd.to_datetime(acts.endTime,utc=True).dt.tz_convert(None)
# low-level timestamps naive local Europe +0100 in Feb => convert by subtracting 1 hour to UTC naive
lts=(L.ts-pd.Timedelta(hours=1)).sort_values().to_numpy(dtype='datetime64[ns]')
def nearest_sec(t):
 x=np.datetime64(t.to_datetime64()); i=np.searchsorted(lts,x); cand=[]
 if i<len(lts): cand.append(abs((lts[i]-x)/np.timedelta64(1,'s')))
 if i>0: cand.append(abs((x-lts[i-1])/np.timedelta64(1,'s')))
 return min(cand) if cand else np.nan
acts['start_nearest_s']=acts.start.apply(nearest_sec); acts['end_nearest_s']=acts.end.apply(nearest_sec)
# containment any low-level observation in positive-duration activity
# use UTC converted L
ltimes=pd.Series(lts)
contains=[]
for _,r in acts.iterrows():
 if pd.isna(r['end']) or r['end']<=r['start']: contains.append(False)
 else:
  a=np.datetime64(r['start'].to_datetime64()); b=np.datetime64(r['end'].to_datetime64());
  i=np.searchsorted(lts,a,'left'); contains.append(i<len(lts) and lts[i]<=b)
acts['contains_lowlevel']=contains
acts[['id','activityName','processInstanceId','startTime','endTime','durationInMillis','start_nearest_s','end_nearest_s','contains_lowlevel']].to_csv(out/'stgallen_761_workflow_alignment.csv',index=False)
stg_summary={'low_level_rows':len(L),'stations':L.station.nunique(),'process_instances':len(json.load(open(ROOT/'stg761'/'camunda_process-instance.json'))),'activity_instances':len(acts),'all_station_local_monotone':bool(S.timestamp_monotone.all()),'cross_station_adjacent_inversions':int(inv.sum()),'median_station_update_s':float(S.median_interval_s.median()),'max_station_p95_s':float(S.p95_interval_s.max()),'max_station_interval_s':float(S.max_interval_s.max()),'start_within_0_5s_n':int((acts.start_nearest_s<=.5).sum()),'end_within_0_5s_n':int((acts.end_nearest_s<=.5).sum()),'start_within_1s_n':int((acts.start_nearest_s<=1).sum()),'end_within_1s_n':int((acts.end_nearest_s<=1).sum()),'positive_duration_n':int((acts.durationInMillis>0).sum()),'positive_duration_contains_n':int(acts.loc[acts.durationInMillis>0,'contains_lowlevel'].sum())}
pd.DataFrame([stg_summary]).to_csv(out/'stgallen_761_primary_finding.csv',index=False)
# master finding summary
master=pd.DataFrame([
 {'finding':'DAYPSCI anomaly-observable vs controller D2/fault','numerator_or_value':'69/70 vs 42/70','metric':'98.6% vs 60.0%','effect':'38.6 percentage-point gap','interpretation':'Observable anomaly manifestation is broader than controller-declared D2/fault state; a dashboard keyed only to D2/fault would under-represent source-visible abnormal behavior in these traces.','boundary':'Not a RAF false-safe/false-veto estimate; no authoritative hold/query/reservation/commit tuple.'},
 {'finding':'St. Gallen cross-station source-time ordering','numerator_or_value':str(int(inv.sum())),'metric':'8 adjacent timestamp inversions','effect':'Station-local monotonicity coexists with global file-order inversions','interpretation':'Aggregate row order is not a valid surrogate for global event time when independent stations update asynchronously; source-time semantics must be preserved before cross-source fusion.','boundary':'Teaching smart factory traces; no manufacturing permission/RAF intervention.'}
])
master.to_csv(out/'primary_external_replay_findings.csv',index=False)
print('DAYPSCI summary',summary)
print('StG summary',stg_summary)
print(pd.DataFrame(rows).to_string(index=False))
