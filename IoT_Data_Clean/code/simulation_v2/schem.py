from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'simulation_v2'
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif']})
O=json.load(open(str(DATA / 'summary.json'))); NF=O['factorial_runs']; TOTAL=15160
def boxes(ax, items, y, w, h, fs=10):
    n=len(items); gap=(1-n*w)/(n+1)
    for k,(t,b) in enumerate(items):
        x=gap+k*(w+gap)
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',fc='white',ec='black',lw=1))
        ax.text(x+w/2,y+h*0.72,t,ha='center',va='center',fontsize=fs,weight='bold')
        ax.text(x+w/2,y+h*0.33,b,ha='center',va='center',fontsize=fs-2)
        if k<n-1: ax.annotate('',xy=(x+w+gap,y+h/2),xytext=(x+w,y+h/2),arrowprops=dict(arrowstyle='->'))
# Figure 11
fig,ax=plt.subplots(figsize=(10,3.6)); ax.axis('off')
ax.text(.5,.93,'Simulation experiment: one frozen simulator (v2), multiple controlled stress families',ha='center',fontsize=13,weight='bold')
boxes(ax,[('Plant','14 serial stations\nfinite buffers, blocking\nminor stoppages, holds'),('Telemetry','lognormal delay\n5.5% bursts, packet drop\nheartbeat, versioned updates'),
          ('Information service','c = 1, 2, 4, 8\nservice mean = 2, 4, 8, 16'),('Policies','Naive, Static, Oracle\nFixed + Query sweep\nAdaptive-A, Matched-State-B'),
          ('Outcomes','completion\nconflict provenance\nquery wait, degradation')],.38,.16,.38)
ax.text(.5,.22,f'{NF:,}-run factorial  •  20 seeds per design point  •  {TOTAL:,} runs including replications',ha='center',fontsize=11,weight='bold')
ax.text(.5,.1,'Primary denominator: execution conflicts / executed releases. Hold-involved permission escape is reported separately from mechanical/resource conflict.',ha='center',fontsize=9)
fig.savefig(str(DATA / 'figures') + '/image11.png',dpi=200,bbox_inches='tight'); plt.close(fig)
# Figure 23
c1=[r for r in O['capacity100'] if r['c']==1]
fig,ax=plt.subplots(figsize=(10,4.6)); ax.axis('off')
ax.text(.5,.96,'Completed evidence synthesis',ha='center',fontsize=14,weight='bold')
ax.text(.5,.9,'State uncertainty, information capacity and transaction timing are separate evidence dimensions',ha='center',fontsize=10)
boxes(ax,[('Asynchronous\ntelemetry','source time ≠ receipt time'),('Compatible-state\nrepresentation','preserve anomaly and\nfault evidence'),('Runtime\nadmissibility','ADMIT / QUERY / HOLD'),
          ('Finite query\nservice','waiting consumes the\ndecision window'),('Commit /\nactuation','state may change after\ncertification')],.6,.17,.22,fs=8)
cols=[('External physical-CPS evidence','DAYPSCI: 275-part post-cycle replay\nCoverage: 89.82% → 99.64%\nReplay false-safe: 40.00% → 1.43%\nSt. Gallen: timing/order sensitivity'),
      ('Controlled computational evidence',f'14-station model (simulator v2); {TOTAL:,} runs\nOne server, service 2 → 16: {c1[0]["completed"]:.1f} → {c1[3]["completed"]:.1f} jobs\nProvenance separates mechanical conflict\nfrom hold-involved permission escape'),
      ('Explicit evidence boundary','Replay is outside the live command path\nNo physical RAF commit-latency estimate\nNo live reservation-race result\nFactory effectiveness is not claimed')]
for k,(t,b) in enumerate(cols):
    x=.03+k*.325
    ax.add_patch(FancyBboxPatch((x,.1),.3,.36,boxstyle='round,pad=0.01',fc='white',ec='black'))
    ax.text(x+.02,.4,t,fontsize=10,weight='bold'); ax.text(x+.02,.33,b,fontsize=8.5,va='top')
fig.savefig(str(DATA / 'figures') + '/image23.png',dpi=200,bbox_inches='tight'); plt.close(fig)
