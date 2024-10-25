# Changelog
All notable changes will be documented in this file.

## 2024-10-04

Thanks to @8188zq and @chtmp223 for pointing out some issues in the current repo: some results are not fully reproducible due to random seeding problems.
This affects the results for ICL, Re-reranking, and the RAG tasks, where the demo sample for each question may differ.
We have updated the code to fix this issue, and will update the results on the paper and the spreadsheet soon to reflect the changes.
Specifically, we make sure the seeding is consistent across runs and independent of system settings.

Other minor changes:
 - Clean up `data.py` and remove unused code.
 - Update argument descriptions.
 - Log exceptions