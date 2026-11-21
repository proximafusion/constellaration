# Metric reference fixtures

These three cases check equilibrium, Boozer spectrum, and metric regressions.
Values were pinned in PR #148 using VMEC++ 0.5.4 and BOOZ_XFORM 0.0.9.
The first two boundaries come from `tests/geometry/test_data`; the finite-beta
case uses the seed in `tests/mhd/finite_beta_metrics_test.py`. Each case stores
its inputs, solved `wout.nc`, Boozer spectra, and expected `metrics.json`.

The `.nc` and `.npz` files use Git LFS. If they are missing or appear as
pointer files, install Git LFS and run `git lfs pull` before running the tests.
