# Paste-ready local-machine handoff prompt

Use the prompt below only if this work must be resumed on another local
machine.

```text
Continue the seizure sensor-selection project in:
26-the-optimizers-analysis/final_project

The technical rescue is complete. Do not restart the analysis or restore
legacy files without first reading:
- final_project/README.md
- final_project/docs/METHODS.md
- final_project/docs/PROJECT_RESCUE_AUDIT.md

Completed work:
1. Inspected every legacy script, notebook, figure, result, and data input.
2. Replaced incompatible Gen1/K/P/G implementations with one shared
   per-sensor logistic ensemble.
3. Integrated five-second artifact rejection and training-only imputation.
4. Removed invalid low-gamma input at 64 Hz; the model uses 21 valid features
   per channel.
5. Corrected the full montage from a claimed 31 to 29 EEG channels common to
   all 14 patients.
6. Repaired PN00 raw-window leakage: 15 controls regrouped and 564 duplicate
   landmark rows removed.
7. Made P test splits recording-disjoint with exactly two test seizures each;
   PN10 S07/S08 were excluded because they share the S09 test recording.
8. Made G train on the other four eligible patients only and made P/G use
   identical target test rows.
9. Rebuilt all 14 feature caches directly from EDF data through
   build_feature_cache.py: 14,100 original landmark rows, 21 features/channel.
10. Reran the complete analysis through run_analysis.py.
11. Replaced the README, methods, data guide, and project rescue audit.
12. Deleted stale generated outputs and legacy code; preserved the team's
    pre-existing edited LOPO notebook in final_project/archive/.

Final verified results:
- Observed K = 16 of 29.
- K16 worst-patient AUPRC loss from full = 0.0167206.
- K15 loss = 0.0427454 and fails the 0.03 rule.
- Connected conservative plateau begins at K=25.
- K16 mean AUPRC = 0.243391; full29 = 0.237775.
- Both are below positive-rate reference in 6 of 14 patients.
- Mean P cross-entropy = 0.670329.
- Mean G cross-entropy = 0.552244.
- P lower loss in 2/5 patients; G lower in 3/5.
- Exact paired sign-flip p = 0.6875.
- Strict core across all ten P/G sets is empty.
- Conclusion: reduced sensors preserve a weak model, but accurate prediction
  and a personalization advantage were not established.

Validation already completed:
- python build_feature_cache.py --force --quiet
  -> 14 patient caches, 14,100 landmark rows
- python run_analysis.py --n-jobs -1
  -> completed; K=16; all final tables/figures regenerated
- python -m pytest -q tests
  -> 15 passed
- All final CSV/JSON files parsed.
- P/G test-row identity, probability bounds, recording isolation, K rule, and
  code/manifest SHA-256 provenance were asserted.
- git diff --cached --check passed.

Repository state:
- Branch: project-hail-mary
- Base commit recorded by the run: 6c688691ffdacc8327af32a3b5d04f740881e0fe
- The complete rescue changeset is staged.
- No commit or remote push has been made.
- Raw EDF files, split arrays, and generated feature caches are intentionally
  ignored by Git and remain on the original machine.

Required work remaining:
1. Run `git status --short` and confirm there are no unexpected unstaged files.
2. Review the staged summary with `git diff --cached --stat`.
3. If authorized, commit the staged rescue, for example:
   `git commit -m "Rebuild reduced-sensor seizure analysis"`
4. Push only if the user explicitly wants a remote push and the correct remote
   branch is known.

No further model tuning is required for the requested project. Any change to
features, regularization, split rules, K, or selected channels requires a full
cache/analysis rerun and updated README numbers. Do not claim clinical seizure
prediction accuracy from these results.
```
