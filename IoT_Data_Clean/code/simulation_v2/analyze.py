from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'simulation_v2'
"""Analyses, tables and figures for simulator v2 results."""
import json, math
import numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = str(DATA) + '/'; F = str(DATA / 'figures') + '/'
import os; os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': 10, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 200})
OUT = {}
rng = np.random.default_rng(20261019)
NAME = {'naive': 'Naive', 'static': 'Static', 'fixed': 'Fixed + Query', 'adaptive': 'Adaptive-A',
        'matched': 'Matched-State-B', 'oracle': 'Oracle'}


def load(t): return pd.read_csv(R + t + '.csv')


def boot_mean(x, B=20000):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (B, len(x))); m = x[idx].mean(1)
    return np.quantile(m, [.025, .975])


def paired(df, a, b, metric):
    A = df[df.policy == a].sort_values('seed')[metric].to_numpy(float)
    Bv = df[df.policy == b].sort_values('seed')[metric].to_numpy(float)
    d = A - Bv
    lo, hi = boot_mean(d)
    p = stats.ttest_rel(A, Bv).pvalue if d.std() > 0 else 1.0
    try: w = stats.wilcoxon(A, Bv).pvalue
    except Exception: w = float('nan')
    dz = d.mean() / d.std(ddof=1) if d.std() > 0 else 0.0
    return dict(a=a, b=b, metric=metric, a_mean=A.mean(), b_mean=Bv.mean(), diff=d.mean(), lo=lo, hi=hi, p=p,
                wilcoxon=w, dz=dz, sd=d.std(ddof=1), n=len(d))


def tost(d, margin):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n)
    p1 = 1 - stats.t.cdf((m + margin) / se, n - 1); p2 = stats.t.cdf((m - margin) / se, n - 1)
    return max(p1, p2)


def summ(df, by):
    g = df.groupby(by)
    s = g.agg(n=('seed', 'size'), completed=('completed', 'mean'), completed_sd=('completed', 'std'),
              conflict=('aggregate_conflict', 'mean'), hold=('hold_rate', 'mean'), qrate=('query_rate', 'mean'),
              p95=('query_wait_p95', 'mean'), load=('offered_load', 'mean'), disp=('dispatches', 'sum'),
              failed=('failed', 'sum'), mech=('mech_only', 'sum'), hold_only=('hold_only', 'sum'),
              compound=('compound', 'sum'), hold_inv=('hold_involved', 'sum'), race=('race', 'sum'),
              degr=('degradation', 'mean'))
    return s.reset_index()


# ---------------- E1 ----------------
e1a = load('E1a_central'); s1a = summ(e1a, 'policy'); OUT['E1a'] = s1a.to_dict('records')
e1b = load('E1b_capacity'); s1b = summ(e1b, ['policy', 'query_servers', 'query_service_mean']); OUT['E1b'] = s1b.to_dict('records')
e1c = load('E1c_threshold'); s1c = summ(e1c, ['threshold', 'query_servers']); OUT['E1c'] = s1c.to_dict('records')
e1d = load('E1d_delay'); s1d = summ(e1d, ['policy', 'delay_scale']); OUT['E1d'] = s1d.to_dict('records')
e1e = load('E1e_stress'); s1e = summ(e1e, ['packet_drop', 'buffer_size', 'arrival_mean']); OUT['E1e'] = s1e.to_dict('records')
e1f = load('E1f_hold_actuation'); s1f = summ(e1f, ['policy', 'hold_mtbf', 'actuation_mean']); OUT['E1f'] = s1f.to_dict('records')
OUT['factorial_runs'] = int(sum(len(x) for x in [e1a, e1b, e1c, e1d, e1e, e1f]))

# ---------------- E2 / E3 ----------------
e2 = load('E2_fresh200'); OUT['E2_summary'] = summ(e2, 'policy').to_dict('records')
OUT['E2_contrasts'] = [paired(e2, 'adaptive', 'fixed', 'completed'), paired(e2, 'matched', 'adaptive', 'completed'),
                       paired(e2, 'adaptive', 'fixed', 'aggregate_conflict'), paired(e2, 'adaptive', 'fixed', 'hold_rate')]
e3 = load('E3_R2'); s3 = summ(e3, 'policy'); OUT['R2_summary'] = s3.to_dict('records')
tests = [paired(e3, 'matched', 'fixed', 'completed'), paired(e3, 'matched', 'adaptive', 'completed'),
         paired(e3, 'matched', 'fixed', 'hold_rate'), paired(e3, 'matched', 'adaptive', 'hold_rate')]
ps = np.array([t['p'] for t in tests]); order = np.argsort(ps); holm = np.empty(4); run = 0
for r, k in enumerate(order):
    run = max(run, min(1, (4 - r) * ps[k])); holm[k] = run
for t, h in zip(tests, holm): t['holm'] = h
for t in tests[2:]:
    d = (e3[e3.policy == t['a']].sort_values('seed').hold_rate.to_numpy() - e3[e3.policy == t['b']].sort_values('seed').hold_rate.to_numpy())
    t['tost_p'] = tost(d, 0.001)
OUT['R2_tests'] = tests
# descriptive aggregate conflict
OUT['R2_conflict'] = [paired(e3, 'matched', 'fixed', 'aggregate_conflict'), paired(e3, 'matched', 'adaptive', 'aggregate_conflict'),
                      paired(e3, 'adaptive', 'fixed', 'aggregate_conflict')]
# power
z = lambda a: stats.norm.ppf(1 - a / 2); zb = stats.norm.ppf(.8)
pw = []
for t in tests[:2]:
    dz = abs(t['dz'])
    pw.append(dict(contrast=f"{t['a']} vs {t['b']}", dz=t['dz'], diff=t['diff'], sd=t['sd'],
                   n05=math.ceil(((z(.05) + zb) / dz) ** 2) if dz > 0 else None,
                   n0125=math.ceil(((z(.0125) + zb) / dz) ** 2) if dz > 0 else None,
                   mde_dz=(z(.0125) + zb) / math.sqrt(300), mde_jobs=(z(.0125) + zb) / math.sqrt(300) * t['sd']))
OUT['R2_power'] = pw

# ---------------- provenance (central, E1a) ----------------
OUT['provenance'] = s1a[['policy', 'disp', 'mech', 'hold_only', 'compound', 'hold_inv', 'race', 'conflict', 'hold']].to_dict('records')
ev = pd.read_csv(R + 'E6_oracle_event_detail.csv')
if len(ev):
    ev['lag'] = ev.t_exec - ev.t_cert
    OUT['oracle_events'] = dict(n=len(ev), hold_false_at_cert=int((~ev.hold_cert).sum()), seeds=int(ev.seed.nunique()),
                                lag_mean=ev.lag.mean(), lag_median=ev.lag.median(), lag_max=ev.lag.max(),
                                busy_exec=int(ev.busy_exec.sum()), down_exec=int(ev.down_full_exec.sum()))
else:
    OUT['oracle_events'] = dict(n=0)

# ---------------- E4 capacity ----------------
e4 = load('E4_capacity100')
cap = []
for (c, m), g in e4.groupby(['query_servers', 'query_service_mean']):
    cap.append(dict(c=c, s=m, completed=g.completed.mean(), c_lo=boot_mean(g.completed)[0], c_hi=boot_mean(g.completed)[1],
                    conflict=g.aggregate_conflict.mean(), p95=g.query_wait_p95.mean(), p_lo=boot_mean(g.query_wait_p95)[0],
                    p_hi=boot_mean(g.query_wait_p95)[1], load=g.offered_load.mean(), hold=g.hold_rate.mean()))
OUT['capacity100'] = cap
g4 = e4[(e4.query_service_mean == 8) & (e4.query_servers.isin([4, 8]))]
a4 = g4[g4.query_servers == 4].sort_values('seed').completed.to_numpy(); a8 = g4[g4.query_servers == 8].sort_values('seed').completed.to_numpy()
OUT['cap_4v8_p'] = stats.ttest_rel(a4, a8).pvalue

# ---------------- E5 fine sweep vs adaptive (E1b, service 8) ----------------
e5 = load('E5_fine_sweep'); s5 = summ(e5, ['threshold', 'query_servers'])
dom = []
for c in [1, 2, 4, 8]:
    ad = s1b[(s1b.policy == 'adaptive') & (s1b.query_servers == c) & (s1b.query_service_mean == 8)].iloc[0]
    g = s5[s5.query_servers == c]
    cand = g[(g.completed >= ad.completed) & (g.conflict <= ad.conflict)]
    if len(cand):
        best = cand.sort_values('completed', ascending=False).iloc[0]
        dom.append(dict(c=c, ad_comp=ad.completed, ad_conf=ad.conflict, dominated=True, tau=best.threshold, f_comp=best.completed, f_conf=best.conflict))
    else:
        g2 = g[g.completed >= ad.completed]
        dom.append(dict(c=c, ad_comp=ad.completed, ad_conf=ad.conflict, dominated=False,
                        min_conf=g2.conflict.min() if len(g2) else None, max_comp=g.completed.max()))
OUT['dominance'] = dom

# ---------------- E7 calibration ----------------
sm = pd.read_csv(R + 'E7_calibration_samples.csv')
e7 = load('E7_calibration')
cal = []
for (c, m), g in sm.groupby(['regime_c', 'regime_s']):
    X1, X2, Y1, Y2 = [], [], [], []
    for sd, h in g.groupby('seed'):
        h = h.sort_values('t'); h = h[h.t >= 120]
        u = h.u.to_numpy(); q = h.q.to_numpy(); dt = 2.0
        du = np.diff(u) / dt; dq = np.diff(q) / dt
        X1.append(np.c_[np.ones(len(du)), -u[:-1], q[:-1]]); Y1.append(du)
        X2.append(np.c_[u[:-1], -q[:-1]]); Y2.append(dq)
    X1 = np.vstack(X1); Y1 = np.concatenate(Y1); X2 = np.vstack(X2); Y2 = np.concatenate(Y2)
    c1, _ = optimize.nnls(X1, Y1); c2, _ = optimize.nnls(X2, Y2)
    d_, a_, b_ = c1; k_, s_ = c2
    r2 = lambda X, Y, co: 1 - ((Y - X @ co) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    G = b_ * k_ / (a_ * s_) if a_ > 0 and s_ > 0 else float('nan')
    rr = e7[(e7.query_servers == c) & (e7.query_service_mean == m)]
    cal.append(dict(c=int(c), s=int(m), load=rr.offered_load.mean(), completed=rr.completed.mean(), p95=rr.query_wait_p95.mean(),
                    G=G, r2u=r2(X1, Y1, c1), r2q=r2(X2, Y2, c2), n=len(Y1)))
OUT['calibration'] = cal

# ---------------- E8 distribution ----------------
dd = []
for tag in ['central', 'bottleneck']:
    e8 = load('E8_distribution_' + tag)
    for (dl, sv, p), g in e8.groupby(['delay_family', 'service_family', 'policy']):
        dd.append(dict(cond=tag, delay=dl, service=sv, policy=p, completed=g.completed.mean(), p95=g.query_wait_p95.mean(),
                       conflict=g.aggregate_conflict.mean(), hold=g.hold_rate.mean()))
OUT['distribution'] = dd
e9 = pd.read_csv(R + 'E9_reachability.csv'); OUT['reach'] = e9.groupby('b').tail(1).to_dict('records')

json.dump(OUT, open(str(DATA / 'summary.json'), 'w'), indent=1, default=float)

# ================= FIGURES =================
col = {'fixed': '#1f77b4', 'adaptive': '#ff7f0e', 'matched': '#2ca02c', 'naive': '#7f7f7f', 'static': '#9467bd', 'oracle': '#d62728'}
# Fig 12 R2 operating points
fig, ax = plt.subplots(figsize=(6.2, 4))
for r in OUT['R2_summary']:
    ax.scatter(r['conflict'] * 100, r['completed'], s=60, color=col[r['policy']], label=NAME[r['policy']])
    ax.annotate(f"q={r['qrate']:.2f}", (r['conflict'] * 100, r['completed']), xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Aggregate execution conflict (% of executed releases)'); ax.set_ylabel('Completed jobs (mean)')
ax.set_title('R2 operating points (300 paired seeds per policy)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(F + 'image12.png'); plt.close(fig)
# Fig 13 R2 completed with CI
fig, ax = plt.subplots(figsize=(6, 3.6))
for k, p in enumerate(['fixed', 'adaptive', 'matched']):
    x = e3[e3.policy == p].completed; lo, hi = boot_mean(x)
    ax.errorbar(k, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], fmt='o', color=col[p], capsize=6, ms=8)
ax.set_xticks(range(3)); ax.set_xticklabels([NAME[p] for p in ['fixed', 'adaptive', 'matched']])
ax.set_ylabel('Completed jobs (mean, 95% bootstrap CI)'); ax.set_title('R2 completed jobs')
fig.tight_layout(); fig.savefig(F + 'image13.png'); plt.close(fig)
# Fig 14 provenance
fig, ax = plt.subplots(figsize=(6.4, 3.8))
pol = ['naive', 'static', 'fixed', 'adaptive', 'matched', 'oracle']
sp = s1a.set_index('policy').loc[pol]
m = sp.mech / sp.disp * 100; h = sp.hold_only / sp.disp * 100; cpd = sp.compound / sp.disp * 100
ax.bar(range(6), m, label='Mechanical/resource only', color='#9ecae1')
ax.bar(range(6), h, bottom=m, label='Hold only', color='#de2d26')
ax.bar(range(6), cpd, bottom=m + h, label='Compound', color='#756bb1')
ax.set_xticks(range(6)); ax.set_xticklabels([NAME[p] for p in pol], fontsize=8)
ax.set_ylabel('% of executed releases'); ax.set_title('Conflict provenance at the central condition (20 seeds)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(F + 'image14.png'); plt.close(fig)
# Fig 15/16 capacity
cp = pd.DataFrame(cap)
for fn, yk, lo, hi, yl in [('image15.png', 'completed', 'c_lo', 'c_hi', 'Completed jobs (mean)'), ('image16.png', 'p95', 'p_lo', 'p_hi', 'Mean run-level P95 query wait')]:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for c, g in cp.groupby('c'):
        ax.errorbar(g.s, g[yk], yerr=[g[yk] - g[lo], g[hi] - g[yk]], marker='o', capsize=3, label=f'c = {c}')
    ax.set_xscale('log', base=2); ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels(['2', '4', '8', '16'])
    ax.set_xlabel('Mean query service time'); ax.set_ylabel(yl); ax.legend(fontsize=8)
    ax.set_title('Capacity surface, 100 fresh seeds per cell (Adaptive-A)')
    fig.tight_layout(); fig.savefig(F + fn); plt.close(fig)
# Fig 17 equal-capacity sets
fig, axs = plt.subplots(2, 2, figsize=(7, 5.6))
for ax, c in zip(axs.flat, [1, 2, 4, 8]):
    g = s5[s5.query_servers == c].sort_values('threshold')
    ax.plot(g.conflict * 100, g.completed, '.-', color=col['fixed'], ms=3, lw=.8, label='Fixed + Query, 0.05 grid')
    ad = s1b[(s1b.policy == 'adaptive') & (s1b.query_servers == c) & (s1b.query_service_mean == 8)].iloc[0]
    ax.scatter(ad.conflict * 100, ad.completed, marker='s', color=col['adaptive'], s=50, zorder=3, label='Adaptive-A')
    ax.set_title(f'c = {c}', fontsize=9); ax.set_xlabel('Aggregate conflict (%)', fontsize=8); ax.set_ylabel('Completed jobs', fontsize=8)
axs[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(F + 'image17.png'); plt.close(fig)
# Fig 18 degradation sensitivity (diagnostic): mean degradation per run for q0 grid x beta (adaptive central)
qs = [c for c in e1a.columns if c.startswith('degr_unit_q0_')]
ad = e1a[e1a.policy == 'adaptive']
betas = [0.5, 1, 2, 2.8, 4, 6]
M = np.array([[ad[q].mean() * b for q in qs] for b in betas])
fig, ax = plt.subplots(figsize=(6.4, 3.6))
im = ax.imshow(M, aspect='auto', cmap='Greys')
ax.set_xticks(range(len(qs))); ax.set_xticklabels([q.split('_')[-1] for q in qs], fontsize=7)
ax.set_yticks(range(len(betas))); ax.set_yticklabels(betas)
ax.set_xlabel('q0 (x100)'); ax.set_ylabel('beta'); fig.colorbar(im, label='Mean degradation score per run')
ax.set_title('Queue-time degradation sensitivity (Adaptive-A, central)'); ax.grid(False)
fig.tight_layout(); fig.savefig(F + 'image18.png'); plt.close(fig)
# Fig 19 robustness map
pv = s1e.copy(); pv['row'] = pv.apply(lambda r: f"drop {r.packet_drop:.3f} | buffer {int(r.buffer_size)}", axis=1)
rows = list(dict.fromkeys(pv.sort_values(['packet_drop', 'buffer_size']).row))
Mx = np.array([[pv[(pv.row == r) & (pv.arrival_mean == a)].completed.iloc[0] for a in [6, 8, 10]] for r in rows])
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(Mx, aspect='auto', cmap='Greys')
for i in range(Mx.shape[0]):
    for j in range(3): ax.text(j, i, f'{Mx[i, j]:.1f}', ha='center', va='center', color='white' if Mx[i, j] > Mx.mean() else 'black', fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8); ax.set_xticks(range(3)); ax.set_xticklabels([6, 8, 10])
ax.set_xlabel('Mean interarrival time'); fig.colorbar(im, label='Mean completed jobs'); ax.grid(False)
ax.set_title('Adaptive-A under packet-loss, buffer and arrival stress')
fig.tight_layout(); fig.savefig(F + 'image19.png'); plt.close(fig)
# Fig 20 actuation x hold
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
for h, g in s1f[s1f.policy == 'adaptive'].groupby('hold_mtbf'):
    axs[0].plot(g.actuation_mean, g.conflict * 100, 'o-', label=f'hold MTBF = {h}')
    axs[1].plot(g.actuation_mean, g.completed, 'o-', label=f'hold MTBF = {h}')
axs[0].set_xlabel('Mean actuation delay'); axs[0].set_ylabel('Aggregate conflict (%)')
axs[1].set_xlabel('Mean actuation delay'); axs[1].set_ylabel('Completed jobs'); axs[0].legend(fontsize=7)
fig.suptitle('Actuation-delay and hold-frequency stress (Adaptive-A)', fontsize=10)
fig.tight_layout(); fig.savefig(F + 'image20.png'); plt.close(fig)
# Fig 21 power
fig, ax = plt.subplots(figsize=(6, 3.6))
n = np.arange(50, 3001, 10)
for t in tests[:2]:
    dz = abs(t['dz'])
    pwr = stats.norm.cdf(dz * np.sqrt(n) - z(.0125))
    ax.plot(n, pwr, label=f"{NAME[t['a']]} vs {NAME[t['b']]} (d_z = {dz:.3f})")
ax.axhline(.8, ls='--', c='k', lw=.8); ax.axvline(300, ls=':', c='k', lw=.8)
ax.set_xlabel('Paired seeds'); ax.set_ylabel('Power (alpha = 0.0125)'); ax.legend(fontsize=8); ax.set_title('R2 design resolution')
fig.tight_layout(); fig.savefig(F + 'image21.png'); plt.close(fig)
# Fig 8/9 calibration
cl = pd.DataFrame(cal)
fig, ax = plt.subplots(figsize=(6, 3.6))
for c, g in cl.groupby('c'):
    ax.plot(g.load, g.G, 'o-', label=f'c = {c}')
    for _, r in g.iterrows(): ax.annotate(str(r.s), (r.load, r.G), xytext=(3, 3), textcoords='offset points', fontsize=7)
ax.axhline(1, ls='--', c='k', lw=.8); ax.set_xlabel('Mean offered query load'); ax.set_ylabel('Fitted G = bk/(as)')
ax.set_title('Reduced-order calibration (labels: mean service time)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(F + 'image8.png'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3.6))
for c, g in cl.groupby('c'):
    ax.plot(g.load, g.completed, 'o-', label=f'c = {c}')
    for _, r in g.iterrows(): ax.annotate(f'G={r.G:.2f}', (r.load, r.completed), xytext=(3, 3), textcoords='offset points', fontsize=6)
ax.set_xlabel('Mean offered query load'); ax.set_ylabel('Completed jobs'); ax.legend(fontsize=8)
ax.set_title('Completion versus offered query load')
fig.tight_layout(); fig.savefig(F + 'image9.png'); plt.close(fig)
# Fig 24 reachability
fig, ax = plt.subplots(figsize=(6, 3.6))
for b, g in e9.groupby('b'): ax.plot(g.n, g.median_ms, 'o-', label=f'b = {b}')
ax.set_yscale('log'); ax.set_xlabel('Unresolved events n'); ax.set_ylabel('Median time (ms)'); ax.legend(fontsize=8)
ax.set_title('Reference exact-reachability enumeration')
fig.tight_layout(); fig.savefig(F + 'image24.png'); plt.close(fig)
# Fig 25/26 distribution
dfd = pd.DataFrame(dd); dfd['combo'] = dfd.delay + '/' + dfd.service
for fn, cond, key, yl in [('image25.png', None, 'completed', 'Completed jobs'), ('image26.png', 'bottleneck', 'p95', 'Mean P95 query wait')]:
    conds = ['central', 'bottleneck'] if cond is None else [cond]
    fig, axs = plt.subplots(1, len(conds), figsize=(7.2 if cond is None else 5.2, 3.6), squeeze=False)
    for ax, cd in zip(axs[0], conds):
        g = dfd[dfd.cond == cd]; combos = list(dict.fromkeys(g.combo)); w = .25
        for k, p in enumerate(['fixed', 'adaptive', 'matched']):
            v = [g[(g.combo == cb) & (g.policy == p)][key].iloc[0] for cb in combos]
            ax.bar(np.arange(len(combos)) + (k - 1) * w, v, w, label=NAME[p], color=col[p])
        ax.set_xticks(range(len(combos))); ax.set_xticklabels(combos, fontsize=7, rotation=15); ax.set_title(cd, fontsize=9); ax.set_ylabel(yl)
    axs[0][0].legend(fontsize=7); fig.tight_layout(); fig.savefig(F + fn); plt.close(fig)
print('analysis done')
