# Frozen analysis plan — RAF simulator v2 (written before any confirmatory run)

Date: 2026-10-19. Simulator: raf_sim.py (SHA-256 recorded in plan_hashes.txt).

## Experiments
E1 Factorial, 20 seeds per design point (seeds 1-20):
  a) central policies: naive, static, fixed (tau=1.5), adaptive, matched, oracle (c=2, query service mean 8)
  b) capacity surface: c in {1,2,4,8} x query service mean in {2,4,8,16} x {fixed, adaptive}
  c) threshold sweep: tau in {0.25,...,3.50} step 0.25 x c in {1,2,4,8} (fixed, service mean 8)
  d) delay scale in {0.5,1,2,4} x all six policies
  e) packet drop {0, 0.015, 0.05, 0.10} x buffer {2,3,5} x arrival mean {6,8,10} (adaptive)
  f) hold MTBF {225,450,900} x actuation mean {0.1,0.35,1,2} x {fixed, adaptive}
E2 Fresh central replication: 200 seeds (1001-1200), fixed / adaptive / matched.
E3 R2 primary replication: 300 seeds (2001-2300), fixed / adaptive / matched.
E4 Capacity replication: 100 seeds per cell (3001-3100), adaptive, 4 x 4 surface.
E5 Equal-capacity fine sweep: fixed tau in {0.05,...,3.50} step 0.05 x c in {1,2,4,8}, 20 seeds (1-20), service mean 8.
E6 Oracle event diagnostics: oracle, 20 seeds (1-20), event-level hold records.
E7 Reduced-order calibration: adaptive, 40 seeds (5001-5040) x c in {1,2,4} x service mean {2,4,8,16}; samples every 2 units; burn-in 120.
E8 Distribution sensitivity: {central (c=2, s=8), bottleneck (c=1, s=16)} x delay {lognormal, gamma} x query service {exponential, weibull} x {fixed, adaptive, matched}, 100 seeds (4001-4100).

## R2 confirmatory analysis (E3)
Co-primary endpoints: completed jobs; hold-involved permission-escape rate (per executed release).
Planned contrasts: matched vs fixed; matched vs adaptive (paired by seed).
Tests: paired t-test; Holm-Bonferroni across the four tests (2 contrasts x 2 endpoints), familywise alpha 0.05.
Effect sizes: paired mean difference with 20,000-resample bootstrap 95% CI (rng seed 20261019); Cohen's d_z.
Equivalence (sensitivity only): paired TOST for hold-involved rate, margin +/-0.1 percentage points.
Aggregate execution conflict: descriptive only.
No endpoint, contrast, margin or exclusion will be changed after outcomes are generated.
