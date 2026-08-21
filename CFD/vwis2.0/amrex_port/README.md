# VWiS AMReX P0--P2 Cartesian contract

`amrex_port/` is independent of the original `vwis2.0/` PETSc/HYPRE solver.
It is a single-level Cartesian data/runtime framework, not a CFD solver.
`advance_one_step()` only validates a positive `dt` and exchanges same-level
halos; it implements no RHS, pressure solve/projection, LES, IBM/EB, FSI,
curvilinear metric, AMR, physical boundary fill, plotfile payload, or checkpoint payload.

## Version and configuration contract

The local baseline is AMReX `26.04` at the git SHA recorded in
[amrex_version.lock](amrex_version.lock), CMake 3.20+, and C++17. The lock is
not proof that a compatible package has been built on a host: the test-result
record must name the install prefix and enabled backend. CMake gives a direct diagnostic if
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
the result as exploratory. Without AMReX, run:

```bash
bash amrex_port/tests/static_contract_check.sh
```

## P0--P2 data and lifecycle contract

`Geometry`, `BoxArray`, and `DistributionMapping` define one Cartesian level.
`P`, `Phi`, `Nvert`, `Ucat`, and `Ucat_old` are cell-centred. `Ucont`,
`Ucont_old`, and `Ucont_older` are three separate one-component face-centred
arrays: x/y/z faces respectively. Every field uses `vwis.nghost` grow cells
(the supplied inputs set two). `Ucat` components are x/y/z Cartesian values;
each `Ucont[dir]` component is the integrated normal volume flux
$U_d=u_dA_d$, matching the legacy area-cofactor/`Aj` divergence semantics.
The nondimensional-to-SI conversion is not frozen; this does not make `Ucont`
an ambiguous velocity/flux field.

`n`, `n-1`, and `n-2` allocation preserves the old `Ucont/Ucont_o/Ucont_rm1`
shape for a later integrator but P2 never rotates or updates them. `Phi` is a
workspace, not a pressure time layer. `Nvert` is a legacy IBM classification
placeholder, not an AMReX EB volume fraction. Pressure datum is unset: only P4
can choose null-space removal or a physical reference after case-specific BC review.

`FillBoundary(periodicity)` performs inter-Box and MPI halo exchange and fills
periodic images. It never supplies non-periodic physical ghosts. Registered
`BCRec` values are `ext_dir`, assigning future physical ghosts to a BC functor;
no functor exists through P2. Write only valid cells, then call
`fill_ghost_cells()` before a stencil reads ghosts; it first `OverrideSync`s
the owner value across overlapping face valid regions. Long-lived temporaries must
be class-owned; do not allocate an owning `MultiFab` inside `MFIter`.

`write_metadata_manifest` writes rank-0 JSON schema only when
`vwis.metadata_file` is set. `payload_written=false` makes clear it is not a
plotfile, checkpoint, or restart result.

`p1_contract.in` invokes only the base runtime check: it checks the field
schema and zero initialization, fills a global-index field over a multi-box
periodic domain and verifies every allocated ghost value (including MPI-owned
box boundaries when MPI is enabled), then verifies that a non-periodic
`FillBoundary` leaves out-of-domain ghosts at a sentinel. It is a framework
contract test, not CFD validation.

## P2-003/004/005 Cartesian contract

`sync_ucont_from_ucat()` linearly averages the two adjacent cell velocities and
multiplies the normal component by the Cartesian face area. At a non-periodic
domain face it uses zero-order extrapolation from the adjacent valid cell until
P3 supplies a physical BC; this is an algebraic closure, not a boundary
condition. `sync_ucat_from_ucont()` divides the two bounding fluxes by face area
and averages the resulting normal velocities. Both paths call
`OverrideSync(periodicity)` before `FillBoundary`. Shared face reductions use
`OwnerMask`/`sum_unique`; iterating every face FAB valid box directly would
double count interfaces.

The manifest declares `dx`, cell volume, Cartesian face areas and an unallocated
`legacy_Aj_equivalent=1/cell_volume` for unit-index computational coordinates;
no curvilinear metric field or curved-grid input is allocated. `p2_contract.in`
checks periodic multi-box/MPI transforms, unique face count/flux sum, face
ghosts, and a linear net-flux/cell-volume divergence stencil.
`p2_boundary_face.in` checks non-periodic face ghosts and the explicit
boundary-face extrapolation across multiple boxes. These are contract and
manufactured algebra tests, not physical BC, projection, conservation, or CFD
validation.
