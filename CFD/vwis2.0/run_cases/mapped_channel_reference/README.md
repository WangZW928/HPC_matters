# C8-G1 mapped channel figures

These figures render the checked-in `16/32/64` C8-G1 JSON and profile CSV
outputs. They are a general report/web export, not a claim of compliance with a
specific journal.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/avwis-mpl-cache \
python3 run_cases/mapped_channel_reference/plot_results.py --force
```

`profiles` shows the analytic Poiseuille curve and all numerical cell-center
observations. Lines between numerical points are visual guides; no values are
interpolated for analysis. Its second panel uses a logarithmic error axis; all
plotted absolute errors are strictly positive.

`convergence` shows the raw three-grid velocity, residual, flow-rate, and wall
shear metrics on logarithmic axes. Dashed/dotted slope guides are normalized at
`N_y=16` and are not fitted models. Raw data and the exact transformations are
listed in `raw/figure_manifest.json`.

Alt text for `profiles`: Two-panel plot of plane Poiseuille streamwise velocity
against wall-normal position. Numerical profiles for 16, 32, and 64 wall-normal
cells overlap the analytic parabola; the logarithmic error panel shows decreasing
pointwise error as the mesh is refined.

Alt text for `convergence`: Two-panel log-log convergence plot. Velocity L2,
steady residual, and flow-rate errors follow a second-order guide across 16, 32,
and 64 cells. Wall-shear error decreases more slowly, between first and roughly
1.3 order.
