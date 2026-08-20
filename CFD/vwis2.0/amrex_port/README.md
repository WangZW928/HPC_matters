# VWiS AMReX P0/P1 foundation

`amrex_port/` is independent of the original `vwis2.0/` PETSc/HYPRE solver.
It is a single-level Cartesian data/runtime framework, not a CFD solver.
`advance_one_step()` only validates a positive `dt` and exchanges same-level
halos; it implements no RHS, pressure solve/projection, LES, IBM/EB, FSI,
curvilinear metric, AMR, physical boundary fill, plotfile payload, or checkpoint payload.

## Version and configuration contract

The requested baseline is AMReX `25.02`, CMake 3.20+, C++17, CPU baseline;
the record is [amrex_version.lock](amrex_version.lock). It is not proof that
this package exists on a host. CMake gives a direct diagnostic if
`AMReXConfig.cmake` is unavailable and rejects a different exported AMReX
version unless the build is explicitly labelled exploratory.

```bash
cmake -S amrex_port -B build/amrex_port \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX
cmake --build build/amrex_port
ctest --test-dir build/amrex_port --output-on-failure
./build/amrex_port/vwis_amrex_skeleton amrex_port/inputs/p1_smoke.in
mpiexec -n 2 ./build/amrex_port/vwis_amrex_skeleton amrex_port/inputs/p1_multibox.in
```

The final MPI command is a test recipe, not a claimed result. For a different
AMReX during porting, add `-DVWIS_AMREX_ALLOW_VERSION_MISMATCH=ON` and record
the result as unvalidated. Without AMReX, run:

```bash
bash amrex_port/tests/static_contract_check.sh
```

## P1 data and lifecycle contract

`Geometry`, `BoxArray`, and `DistributionMapping` define one Cartesian level.
`P`, `Phi`, `Nvert`, `Ucat`, and `Ucat_old` are cell-centred. `Ucont`,
`Ucont_old`, and `Ucont_older` are three separate one-component face-centred
arrays: x/y/z faces respectively. Every field uses `vwis.nghost` grow cells
(the supplied inputs set two). `Ucat` components are x/y/z Cartesian values;
each `Ucont[dir]` component is the normal contravariant flux/velocity in that
face direction. Historical normalization and physical unit conversion are not
yet frozen, so metadata says so rather than inventing SI units.

`n`, `n-1`, and `n-2` allocation preserves the old `Ucont/Ucont_o/Ucont_rm1`
shape for a later integrator but P1 never rotates or updates them. `Phi` is a
workspace, not a pressure time layer. `Nvert` is a legacy IBM classification
placeholder, not an AMReX EB volume fraction. Pressure datum is unset: only P4
can choose null-space removal or a physical reference after case-specific BC review.

`FillBoundary(periodicity)` performs inter-Box and MPI halo exchange and fills
periodic images. It never supplies non-periodic physical ghosts. Registered
`BCRec` values are `ext_dir`, assigning future physical ghosts to a BC functor;
no functor exists in P1. Write only valid cells, then call
`fill_ghost_cells()` before a stencil reads ghosts. Long-lived temporaries must
be class-owned; do not allocate an owning `MultiFab` inside `MFIter`.

`write_metadata_manifest` writes rank-0 JSON schema only when
`vwis.metadata_file` is set. `payload_written=false` makes clear it is not a
plotfile, checkpoint, or restart result.
