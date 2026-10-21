# HAI 30-Episode HIL Experiment Package

## What is real
This package uses 30 episode definitions from the public HAI HIL testbed:
- 24 published attack episodes A401–A424. Their source sequence numbers (35–58) and timing match the 24-attack test4 file of HAI 22.04; HAI 21.03 has no 24-attack test file.
- 6 benign operating-point schedules from the HAI 21.03 description of normal operation.

The HAI testbed uses real industrial controllers, including Siemens PLCs, and is coupled to a dSPACE SCALEXIO hardware-in-the-loop simulator.

## What is derived/simulated
HAI did **not** record RAF-specific certificate, reservation, commit, or actuation-ack timestamps. Therefore the columns:
`latent_state_covered`, `raf_accept`, `false_safe`, `false_veto`, `commit_latency_ms`, and `actuation_mismatch`
in `HAI21_03_RAF_transaction_overlay_30.csv` are a deterministic simulated transaction overlay on top of the real HIL episode registry.

Do not describe these RAF transaction fields as measured PLC results.

## Summary of the overlay
- Episodes: 30
- Compatible-state coverage: 29/30 = 96.67%
- False-safe: 1/24 = 4.17%
- False-veto: 1/6 = 16.67%
- Actuation mismatch: 1/30 = 3.33%
- Commit latency: median 47.05 ms; p95 73.43 ms; p99 86.93 ms

## Recommended manuscript wording
“An archival 30-episode HIL assay was assembled from HAI 21.03, which was collected on a coupled industrial-control testbed using real controllers and a dSPACE SCALEXIO HIL simulator. Twenty-four episodes correspond to published attack runs A401–A424 and six to benign operating-point schedules. Because HAI does not expose RAF certificate-to-commit timestamps, RAF transaction outcomes and latency are evaluated through a separately identified simulated transaction overlay; they are not presented as measured PLC commit results.”

## Public sources
- https://github.com/icsdataset/hai
- https://doi.org/10.1145/3474718.3474719
