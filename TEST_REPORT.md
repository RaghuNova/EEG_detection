# NEUROSCAN software test report

## Checks run

- Python syntax compilation passed for the changed backend modules.
- Frontend production build passed. Vite reports a bundle-size advisory for charting dependencies.
- Focused storage-history test passed.
- Real EDF workflow smoke checks completed at single-recording and small-batch scale. ML, CNN-BiLSTM, and Attention inference returned successful statuses, and the compact result format omitted raw signal arrays. Patient identifiers, filenames, and case-level outputs are intentionally excluded from this report.
- API health and compact-result endpoints returned successfully. Legacy large-result requests are guarded with HTTP 409.

## Existing full-batch status

A prior local full-folder run is present in local storage. It was not repeated after the workflow fix. Its patient-specific details remain local and are not included here.

## Existing test-suite limitation

Some older tests fail collection because imports refer to `backend.dl` and `backend.mlout`, while the current repository keeps `dl/` and `mlout.py` at its root. This is an existing test-layout mismatch; patient recordings were not used in the focused unit test.

No patient recordings, annotations, generated reports, identifiers, or model predictions are included in this report.
