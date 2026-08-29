# VWiS AMReX P0--P8-003 Cartesian contract

`amrex_port/` is independent of the original `vwis2.0/` PETSc/HYPRE solver.
It is a single-level Cartesian data/runtime framework with a narrow pressure
projection and conservative advective/viscous RHS, not a complete CFD solver. `project_cartesian()` forms a pressure
RHS from integrated face flux, solves a Cartesian cell-centred Poisson problem
with MLMG, corrects face flux, and synchronizes `Ucat`.
`compute_cartesian_advection_rhs()` and `compute_cartesian_viscous_rhs()` form
the P5-001 convective and P5-002 constant-coefficient viscous parts of the
momentum RHS. P5-004 adds a provisional explicit Euler predictor followed by
the P4 projection, with explicit `n/n-1/n-2` state rotation. This is not the
legacy SNES residual solve and does not implement semi-implicit or BDF2
advancement. P8-001/P8-002 now add a versioned single-level CPU
checkpoint/restart payload using AMReX `VisMF`, including all three fluid time
layers and strict Header validation. There is still no LES, IBM/EB, FSI,
curvilinear metric/operator, AMR, plotfile/HDF5 output, MPI/GPU restart, or full
CFD case validation. P8-003 adds CPU/single-rank uniform-grid point probes,
plane summaries/CSV extraction, reusable flow statistics, and equal-spacing
post-step time averages in the physical channel report. These CSV/JSON files
are diagnostics and are explicitly not AMReX plotfiles.

The Cartesian boundary vocabulary also includes `moving_wall`, with
`vwisbcs.moving_wall_velocity` defaulting to `1 0 0`. Its normal component
must be zero. Cell ghosts use mirror Dirichlet data
`Ughost=2*Uwall-Uinterior`, tangential face-flux ghosts use the same wall
state after face-area scaling, and the physical normal face flux remains zero.
Existing `noslip` behavior is unchanged. Checkpoint schema version 2 records
and strictly checks the moving-wall vector in addition to the prior P8 state;
the reader still accepts schema-v1 checkpoints that predate this field.

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
# Repeatable P0-004/P0-005 manufactured Cartesian benchmark; JSON is emitted by the runner.
ctest --test-dir build/amrex_port -R vwis_amrex_cartesian_benchmark --output-on-failure
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
`vwis.metadata_file` is set. The P8 checkpoint path is separate: it writes a
versioned `Header` plus AMReX `VisMF` payloads. It is intentionally limited to
one CPU rank and must not be confused with a plotfile or MPI restart.

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

## P5-001 Cartesian advective RHS

For each Cartesian face direction `d` and transported velocity component `m`,
the face momentum flux and cell RHS are

```text
F[d,m] = Ucont[d] * 0.5 * (Ucat[L,m] + Ucat[R,m])
RHS_adv[m] = -sum_d(F[d,m]_hi - F[d,m]_lo) / cell_volume
```

This is the Cartesian form comparable to the legacy `RHSSolver.C`
`-second_order` branch, whose stored inviscid face flux has the minus sign
inside the flux before taking face differences. `Ucont` remains integrated
volume flux; it is not divided by `dx` and the conservative divergence divides
once by cell volume. Shared faces retain the existing `OverrideSync` ownership
rule, and the kernel reads `Ucat` only after periodic/inter-Box halos and, when
configured, physical ghosts have been filled.

The P5 CTests use a fully periodic multi-Box manufactured field
`U=(0.75,sin(2*pi*x),0)` at 16 and 32 x cells, plus a constant physical
inflow/constrained-outflow multi-Box case. They check the exact discrete
second-order stencil, continuous manufactured error, zero RHS for constant
advected components, and zero RHS for constant boundary through-flow. This is
not a curvilinear equivalent, time advance, LES, or IBM path.

## P5-002 Cartesian viscous RHS

For constant kinematic viscosity `nu`, the cell-centred Cartesian operator is

```text
RHS_visc[m] = nu * (Dxx(Ucat[m]) + Dyy(Ucat[m]) + Dzz(Ucat[m]))
```

The implementation uses the second-order centred three-point stencil and reads
`Ucat` only after the existing periodic/inter-Box halo and physical no-slip
ghost pipeline. Non-periodic directions require explicit `vwisbcs`; variable
viscosity, curvilinear metrics, IBM/EB and time integration remain out of scope.

The P5-002 CPU tests cover the exact discrete periodic manufactured eigenvalue
on 16 and 32 cells, global momentum conservation, negative kinetic-energy rate,
and no-slip boundary flux balance on a multi-Box domain. With locked AMReX
26.04, the clean CPU build and all 16 CTests pass; this is not MPI/CUDA runtime
or full CFD time-integration validation.

## P5-004 explicit Cartesian time baseline

The selected executable baseline is

```text
U* = U^n + dt * (RHS_adv(U^n) + RHS_visc(U^n))
U^(n+1) = project(U*, projection_time_coefficient=1)
```

Before overwriting the current state, both `Ucat` and every `Ucont[d]` rotate
from `(n,n-1)` to `(n-1,n-2)`. The solver then records `time`, `step`, and a
history depth capped at three. The explicit guard requires both the conservative
advective CFL estimate `sum_d(dt*max(abs(Ucat_d))/dx_d)` and diffusion number
`2*nu*dt*sum_d(1/dx_d^2)` to be at most one. These limits catch clearly invalid
steps; they are not a nonlinear stability proof for centred advection.

The periodic manufactured regression uses the divergence-free shear
`U=(0,sin(2*pi*x),0)`. Its advective RHS and pressure correction are zero, so
the full predictor/project path reduces to an exactly known semi-discrete
diffusion eigenmode. Runs with `dt=1e-3` and `dt/2` to `t=8e-3` give a temporal
error ratio near two, while also checking momentum drift, post-projection
divergence, and exact history rotation. A second expected-failure test exercises
the diffusion-number rejection.

On 2026-08-28 the locked 26.04 CPU clean build and complete suite passed 18/18;
the measured coarse/fine temporal error ratio was `2.002575751`. MPI build/link
passed, but 2-rank execution was blocked by the host PMIx socket restriction.
See `_Docs/AMReX_P5-004_时间推进设计及测试_20260828.md` for the time-step
baseline and `_Docs/AMReX_P8-001_P8-002_checkpoint_restart_20260829.md` for
checkpoint/restart commands, evidence, limitations, and the distinction from
CFD validation.

The legacy `Integrator::SolveFunction` remains materially different: its
reachable `timeCoeff=1` branch is a nonlinear SNES residual with current and old
RHS weighted by one half, while its BDF2-shaped branch is unreachable because
`UData::getTimeCoeff()` returns `1.0`. P5-004 does not erase that distinction;
the semi-implicit/BDF2 solver decision remains P5-005 work. P8 now provides a
separate restart payload for the validated single-level CPU path; it does not
make the explicit integrator equivalent to legacy SNES or extend restart to
MPI, GPU, metric, IBM, FSI, or AMR state.

## P8-001/P8-002 checkpoint and restart

The checkpoint directory contains a strict text `Header` and AMReX `VisMF`
payloads. The Header records the schema/version, locked AMReX version and SHA,
single-rank CPU scope, dimension/precision, domain/dx, boundary configuration,
ghost width, component/layout contract, time/step/history depth, and the fixed
field list. The payload stores `Ucat`, `Ucat_old`, `Ucat_older`, all three
`Ucont` history layers, `P`, `Phi`, and `Nvert`.

Restart validates the Header before reading any field. It rejects an invalid
magic/version, unsupported rank/backend, dimension or precision mismatch,
field/layout mismatch, wrong geometry or BC configuration, missing payload, or
unexpected extra field. A real regression runs an uninterrupted trajectory and
a checkpoint-at-K trajectory to N steps, then compares every persistent field,
time, step, and history depth. The current CPU result is bitwise identical.

On 2026-08-29, the locked CPU build in `build/amrex_port_p8` passed the static
contract check, clean rebuild, both P8 tests, all 20 CTests, and `git diff
--check`. MPI/GPU runtime and plotfile/HDF5 output remain outside this
increment. Full evidence is recorded in the P8 report.

## P8-003 uniform Cartesian sampling and statistics

`VwisAmrExDiagnostics.cpp` provides a cell-containing physical-coordinate
point probe, cell-plane statistics, deterministic plane CSV extraction, and an
instantaneous diagnostic reduction. The diagnostic schema reports integrated
and maximum absolute divergence, net physical-boundary mass flux, outlet flow,
three-component momentum, kinetic energy, and pressure mean/minimum/maximum.
Plane statistics report the mean velocity and pressure and the integrated
normal flow on the plane's low face. Point values and plane rows are
cell-centred; there is no interpolation in this increment.

The exact `p8_sampling_statistics.in` contract uses a periodic `4x3x2`,
multi-Box constant velocity field and indexed pressure field. It verifies the
probe cell/value, a six-cell x section and CSV row count, zero divergence,
section flow, momentum, kinetic energy, and pressure statistics within a
roundoff-scaled tolerance. This is a manufactured contract test, not CFD
validation. The physical-channel report reuses the APIs for every post-step
record and adds arithmetic means over its 40 equally spaced samples.

P8-003 remains deliberately single-level, uniform Cartesian, CPU, and one MPI
rank. It has no AMR coverage arbitration, MPI gather/reduction contract, GPU
runtime, interpolated probes, IBM force/moment or IBM-neighbour statistics.
There is no plotfile writer: JSON and CSV outputs must not be presented as
plotfile-compatible. The 2026-08-29 commands and results are recorded in
`_Docs/AMReX_P8-003_采样统计诊断_20260829.md`.

## Lid-driven cavity engineering demonstration

`../run_cases/lid_driven_cavity/` contains a reproducible `32x32x1`, periodic-z
square cavity at `Re=100`. It writes solver-native CSV/JSON diagnostics and
Matplotlib PNG/SVG figures; it does not claim AMReX plotfile output or
reference validation. The focused `vwis_amrex_lid_driven_cavity_sanity` CTest
checks deterministic schemas, row counts, finite output, and the solver's hard
post-projection divergence guard. The case record documents the explicit Euler
limitation and an AMReX 26.04 thin-domain BoxArray restriction observed during
the run.

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

## Physical Cartesian channel benchmark

`inputs/physical_channel.in` adds a physical, single-level Cartesian plane
channel without changing the legacy `vwis2.0/` source. The domain is
`[0,1]^3` with `32x16x4` cells and `dx=(0.03125,0.0625,0.25)`. The fluid starts
at rest; `xlo` is a uniform inlet with target volume flux 1, `xhi` is a fixed
pressure (`p=0`) outflow, `ylo/yhi` are no-slip walls, and `zlo/zhi` are
periodic. With `nu=0.1`, `dt=2e-4`, 40 steps reach `t=0.008` and
`Re=U_in H/nu=10` for `U_in=1`, `H=1`. No body force or manufactured source is
used. This uses existing P3/P4 BC and projection paths; no benchmark-specific
boundary capability was added.

Run it with `ctest --test-dir build/amrex_port --output-on-failure -R
vwis_amrex_physical_channel`. The runner writes `physical_channel.json`, with
one record per step containing timing, post-projection divergence, net mass
flux, momentum, kinetic energy, outlet flow, section means, centerline
velocity, pressure drop, and pressure statistics. It also writes arithmetic
time averages over the equally spaced post-step records. The report status is deliberately
`physical run / not yet validated` and `reference_available=false`: no legacy
or literature reference data is available, so a passing CTest means only that
the configured physical run and report contract completed.
