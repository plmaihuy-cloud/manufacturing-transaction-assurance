# IoT Research Package: Runtime Release Certification under Asynchronous Telemetry

Code, data and provenance records supporting the dissertation "Runtime Release Certification for Manufacturing under Asynchronous Telemetry" (Mai Ngoc Huy).

## Contents
- `code/simulation_v2/` - Simulator version 2 (`raf_sim.py`), the frozen analysis plan (`ANALYSIS_PLAN.md`) with SHA-256 hashes recorded before any run (`plan_hashes.txt`), the experiment runner (`run_all.py`), analysis and figures (`analyze.py`, `schem.py`) and the reachability benchmark.
- `data/simulation_v2/` - Run-level outputs of all 15,160 simulation runs, oracle event details, calibration samples, `summary.json` and the regenerated figures.
- `code/external/`, `data/external/` - DAYPSCI and St. Gallen raw archives, derived tables and replay scripts; HAI episode registry and simulated transaction overlay.
- `code/emulator/`, `data/emulator/` - 60-episode closed-loop transaction stress test on a plant emulator.
- `code/robust_mpc/`, `data/robust_mpc/` - Matched-information reduced-order Robust MPC comparator.
- `data/provenance/`, `MANIFEST_SHA256.csv`, `DATA_INVENTORY.csv` - hashes and claim boundaries.

## Reproducing the simulation chapter
```
cd code/simulation_v2
python run_all.py          # about 18 minutes on two CPU cores
python reachability_benchmark.py
python analyze.py          # summary.json, tables and figures
python schem.py            # schematic figures 11 and 23
```
Requires Python 3.10+, numpy, scipy, pandas, matplotlib.

## Simulator history
The source of an earlier development simulator was lost before submission, so its outputs are not used. Simulator version 2 was re-implemented from the specification in the dissertation; all simulation results in the dissertation come from version 2.

## Evidence boundaries
- DAYPSCI: archival HIL-originated trace evidence and post-cycle release replay; RAF did not command the PLC.
- HAI: real HIL episode provenance plus a separately labeled simulated RAF transaction overlay. The 24 attack episodes A401-A424 correspond to HAI 22.04 test4; the six benign schedules come from the HAI 21.03 documentation.
- Emulator: synthetic closed-loop transaction stress test, not PLC hardware validation.
- Robust MPC: synthetic reduced-order comparator.
- Simulation: controlled 14-station experiments and replications.

The external replay scripts contain absolute input paths from the original working environment; edit them before rerunning.
