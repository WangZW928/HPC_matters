# VWiS AMReX skeleton

This is an independent, single-level data-layout skeleton. It does not replace
`vwis2.0/` and intentionally implements no physical solver.

Prerequisite: an AMReX installation exposing an `AMReXConfig.cmake` package.

```bash
cmake -S amrex_port -B build/amrex_port -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX
cmake --build build/amrex_port
./build/amrex_port/vwis_amrex_skeleton
```

Useful runtime parameters are `vwis.n_cell`, `vwis.max_grid_size`,
`vwis.nghost`, `vwis.dt`, and `vwis.is_periodic`. The current code allocates
cell-centred `P`, `Phi`, `Nvert`, and directional face-centred `Ucont`, fills
periodic/inter-box ghost cells, initializes zeros, and runs one no-op step.

Not implemented: physical BCs, RHS, LES, SNES-style momentum solve, Poisson or
MAC projection, IBM/EB, FSI, AMR tagging, checkpoint/plotfile output, and any
curvilinear metric operator. This skeleton has not been compiled locally unless
an AMReX package is detected during the validation described in the planning
document.
