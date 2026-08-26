# VWiS AMReX P0--P4 Cartesian projection contract

`amrex_port/` is independent of the original `vwis2.0/` PETSc/HYPRE solver.
It is a single-level Cartesian data/runtime framework with a narrow pressure
projection, not a complete CFD solver. `project_cartesian()` forms a pressure
RHS from integrated face flux, solves a Cartesian cell-centred Poisson problem
with MLMG, corrects face flux, and synchronizes `Ucat`. `advance_one_step()` is
still a no-op: there is no momentum RHS, time integration, LES, IBM/EB, FSI,
curvilinear metric/operator, AMR, plotfile payload, or checkpoint payload.

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

## P0--P4 data and lifecycle contract

The implementation is organized by responsibility: `CartesianBoundaryConfig.*`
owns input parsing and legacy-code mapping, `VwisAmrExBoundary.cpp` owns
physical ghosts and boundary-face fluxes, and `VwisAmrExContracts.cpp` owns
runtime checks, reductions, diagnostics, and schema output.  The solver header
and `VwisAmrExSolver.cpp` retain the data owner, lifecycle, halo exchange, and
Cartesian face/cell transforms.  Boundary kernels copy scalar configuration
into fixed-size GPU value arrays before launch; host strings and containers are
never captured by device lambdas.

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
placeholder, not an AMReX EB volume fraction. P4 gives correction pressure an
explicit datum policy below; deferred legacy cases still need case review.

`fill_ghost_cells()` performs `OverrideSync` plus inter-Box/MPI/periodic
`FillBoundary`; it never supplies non-periodic physical ghosts.
`fill_physical_ghost_cells()` is a separate P3 operation and refuses to run if
the halo epoch is stale. `apply_boundary_pipeline()` fixes the order as halo
first, physical cell ghosts/boundary faces second, diagnostics last. Any valid
write invalidates both freshness epochs. Long-lived temporaries must be
class-owned; do not allocate an owning `MultiFab` inside `MFIter`.

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

## P3 Cartesian physical boundaries

Set `vwisbcs.enabled=1` and explicitly name `vwisbcs.lo/hi` for every
non-periodic side. Supported names are `noslip`, `slip`, `symmetry`, `inflow`,
and `outflow`. Alternatively `legacy_codes` accepts only the evidenced general
old codes 1/3/4/5. Periodicity remains a Geometry property, as in the old code;
old case-specific codes are rejected with an error.

The minimal inlet supports `uniform` and `linear_plane`; MPI-global profile
normalization makes the integrated incoming `Ucont` equal
`inlet_target_flux`. The outlet pressure value is imposed in cell ghosts, and
`constrain_outlet_flux=1` makes its integrated outgoing `Ucont` match the inlet.
Walls/symmetry set normal boundary flux to zero. P3 diagnostics report freshness
epochs, every physical boundary-face flux, their global sum, and the global
net-flux/cell-volume divergence identity with explicit MPI reductions.

The P3 tests remain boundary/data-flow tests. P4 consumes the same boundary
classification only when explicitly requested by a P4 input; P3 itself still
does not advance physical state.

## P4 Cartesian pressure projection

`Ucont[d]` remains integrated normal volume flux, never face velocity. Cell
divergence is net integrated face flux divided once by cell volume. With
positive `projection_time_coefficient=alpha`, P4 uses

```text
rhs = (alpha/dt) div(Ucont*)
L(Phi) = rhs
Ucont[d] <- Ucont[d] - (dt/alpha) A[d] grad_d(Phi)
```

AMReX `MLPoisson`/MLMG exposes its signed operator face flux through
`getFluxes`; the adapter applies that sign and multiplies by Cartesian face
area before updating `Ucont`. No AMReX API receives integrated flux as ordinary
velocity. `P` is incremented by `Phi`, then `Ucat` is reconstructed from the
corrected integrated face flux.

Periodic sides use periodic pressure BC. Inflow, wall, slip, and symmetry use
homogeneous Neumann correction pressure, so their prescribed normal flux is
unchanged. A fixed-pressure outflow uses homogeneous Dirichlet `Phi` and is the
only physical boundary whose normal flux may be corrected. P3's optional
pre-projection outlet total-flux constraint is deliberately not reimposed after
projection. Supported physical combinations are exactly one inflow plus one
outflow, or a closed no-penetration domain; other combinations are rejected.

Fully periodic and closed all-Neumann problems are singular. P4 validates that
the assembled RHS mean is zero within roundoff, disables MLMG's automatic
solvability repair, rejects incompatible data, and removes only the converged
`Phi` mean to choose a gauge. It never silently subtracts the RHS mean. A
Dirichlet outflow supplies the datum for a nonsingular solve.

The P4 CTests cover periodic and closed singular systems (including an
incompatible-RHS rejection probe and constant-pressure/no-velocity-change
check) plus the supported P3 inflow/outflow path. These are manufactured
Cartesian projection tests, not full CFD or time-integration validation.

The legacy curvilinear Poisson matrix contains non-orthogonal cross terms and a
19-point stencil. `MLPoisson` is not a replacement for it. The curved-metric
operator choice remains deferred until metric fields, cross-term discretization,
boundary treatment, and a validation case exist.

## P3 host verification on 2026-08-24

The current WSL2 host reused locked AMReX 26.04 install caches but configured
the port in new build directories. Evidence is deliberately split by stage:

- CPU configure/compile/link and CTest passed (7/7) in
  `/tmp/vwis-p3-host-20260824-cpu-make`.
- MPI configure/compile/link passed in `/tmp/vwis-p3-host-20260824-mpi` against
  the OpenMPI 4.1.6 AMReX package. Both `mpiexec --oversubscribe -n 2` and a
  loopback-constrained retry were blocked before `MPI_Init`: the execution
  environment denied `socket()` and PMIx could not start its listener. This is
  not a P3 multi-rank runtime pass and no application code was entered.
- CUDA 12.6.85 configure and nvcc compilation passed for the cached
  `sm_75` AMReX package. The verbose build explicitly produced
  `cmake_device_link.o` and linked the executable. `nvidia-smi` reported that
  GPU access was blocked, no NVIDIA/WSL device node was visible, and both P3
  CTest and direct runtime stopped during AMReX CUDA initialization with CUDA
  error 35. This is compile/device-link evidence only, not a CUDA runtime pass.

The exact commands and error boundaries are recorded in
`_Docs/AMReX_P3_边界与诊断设计及测试_20260824.md`. A host session that permits
OpenMPI sockets is still required for the 2-rank result, and a host with a
visible NVIDIA device and compatible driver is required for the final CUDA
runtime result.

## P4 host verification on 2026-08-26

The P4 design, exact commands, solver norms, and MPI evidence boundary are
recorded in `_Docs/AMReX_P4_Cartesian压力投影设计及测试_20260826.md`. CPU clean
configure/build and CTest pass against locked AMReX 26.04. MPI is attempted
separately and is not called a runtime pass if PMIx/socket restrictions prevent
entry into the application.
