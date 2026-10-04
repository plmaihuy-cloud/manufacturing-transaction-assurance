# IoT Research Package - Clean Release

This package contains only the current dissertation/journal deliverables and the data/code needed to support the results retained in the manuscript. Historical manuscript versions, reviewer-response history, obsolete intermediate tables, duplicate derived files, old rendered PDFs, layout/build scripts, and unexecuted live-HIL scaffolding were removed.

## Naming
All package-level filenames use stable descriptive names. Version suffixes such as `_v9`, `_v10`, `_v12`, `_v15`, `_v20`, and `_v21` were removed. Raw-source identity is preserved in `data/provenance/source_archive_hashes.csv` and in this package manifest.

## Evidence boundaries
- DAYPSCI: archival HIL-originated trace evidence and post-cycle release replay; RAF did not command the PLC.
- HAI: real HIL episode provenance plus a separately labeled simulated RAF transaction overlay.
- Emulator: synthetic closed-loop transaction stress test, not PLC hardware validation.
- Robust MPC: synthetic reduced-order comparator.
- Main simulation: controlled 14-station experiments and replications.

## Current documents
- `documents/thesis_dissertation.docx`
- `documents/jms_submission_14p.docx`

## Notes
The archive intentionally excludes old revision history. Use `DATA_INVENTORY.csv` for provenance and claim boundaries and `MANIFEST_SHA256.csv` for integrity checks.
