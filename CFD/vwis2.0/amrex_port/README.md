# AVWiS (AMReX–VWiS) Cartesian contract and P5-003 C5.2 mapped boundaries

## Naming convention

AVWiS means **AMReX–VWiS** and is the product name of this AMReX port. VWiS
continues to mean the original solver/algorithm. The AMReX implementation,
its executable (`avwis`), and its core C++ types use the `AVWiS` spelling.
The former `vwis.*` runtime namespace remains accepted as a compatibility
alias; new inputs and documentation use `avwis.*`. The boundary namespace
follows the same rule: `avwisbcs.*` is canonical and `vwisbcs.*` is legacy.

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
general curvilinear operator, AMR, plotfile/HDF5 output, MPI/GPU restart, or full
CFD case validation. P5-003 C1 makes each solver own an immutable identity
`MetricData` on its exact `BoxArray/DistributionMapping` and uses a narrow
adapter as the cell-volume source for the existing integrated-flux divergence.
P5-003 C2.1 adds a production analytic orthogonal metric provider and geometry
contract. C2.2 adds explicit metric-aware gradient/divergence, the diagonal
orthogonal `Ucat<->Ucont` transform, and a periodic orthogonal MLMG projection
path. The analytic path is deliberate opt-in; Cartesian identity remains the
default. C5.1 adds stationary no-slip walls and periodic pairs; C5.2 extends
that same analytic separable-orthogonal path with mapped inlet/fixed-pressure
outlet, tangential moving wall, slip, and symmetry modes. Mapped advection,
viscosity, time advancement, checkpoint/restart, general non-orthogonal
surfaces, and non-orthogonal pressure terms remain rejected.
P8-003 adds CPU/single-rank uniform-grid point probes,
plane summaries/CSV extraction, reusable flow statistics, and equal-spacing
post-step time averages in the physical channel report. These CSV/JSON files
are diagnostics and are explicitly not AMReX plotfiles.

The Cartesian boundary vocabulary also includes `moving_wall`, with
`avwisbcs.moving_wall_velocity` defaulting to `1 0 0`. Its normal component
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
./build/amrex_port/avwis amrex_port/inputs/p1_smoke.in
mpiexec -n 2 ./build/amrex_port/avwis amrex_port/inputs/p1_multibox.in
# Repeatable P0-004/P0-005 manufactured Cartesian benchmark; JSON is emitted by the runner.
ctest --test-dir build/amrex_port -R avwis_cartesian_benchmark --output-on-failure
```

The final MPI command is a test recipe, not a claimed result. For a different
AMReX during porting, add `-DAVWIS_ALLOW_VERSION_MISMATCH=ON` and record
the result as exploratory. Without AMReX, run:

```bash
bash amrex_port/tests/static_contract_check.sh
```

## Runtime modes

Every input selects exactly one operation with the required `avwis.mode` key.
The accepted values are:

- `solve`: initialize or restart, advance `avwis.run_steps`, and optionally
  write `avwis.checkpoint_file` and `avwis.metadata_file`.
- `contract_base`, `contract_p2`, `contract_p3`, and `contract_p4`: run the
  corresponding framework, transform, boundary, or projection contract.
  `contract_base_p2` preserves the established combined base-plus-P2 contract
  exercised by `inputs/p2_contract.in` as one explicit operation.
- `contract_p5_advection`, `contract_p5_viscous`, and `contract_p5_time`: run
  the focused Cartesian P5 contracts.
- `contract_p5_orthogonal`: run the focused periodic analytic-orthogonal C2.2
  solver projection contract.
- `contract_p5_mapped_boundary`: run the focused C5.1 analytic-orthogonal wall,
  periodic translation, ghost, ownership, validation, and projection contract.
- `contract_p5_mapped_boundary_c52`: run the C5.2 inlet/outlet,
  moving/slip/symmetry, mixed periodic, ownership, ghost-order, and matching
  pressure-projection contracts.
- `contract_p8_restart` and `contract_p8_sampling`: run the restart or uniform
  sampling/statistics contract.
- `benchmark_cartesian` and `benchmark_physical_channel`: run the manufactured
  Cartesian or physical plane-channel benchmark.
- `case_lid_cavity`: run the lid-driven-cavity engineering case.

For example:

```text
avwis.mode = solve
avwis.n_cell = 16 16 16
avwis.is_periodic = 1 1 1
avwis.run_steps = 8
avwis.dt = 1.0e-3
avwis.viscosity = 0.1
```

The former independent `vwis.run_*` boolean selectors are not retained as a
compatibility interface. Inputs using the compatibility namespace must still
choose one `vwis.mode`; a missing or unknown mode is rejected with an explicit
diagnostic. This makes simultaneous operations impossible by construction.

## Source ownership

`src/` owns the reusable production application core: solver state and
lifecycle, Cartesian operators and boundary handling, checkpoint/restart,
runtime diagnostic APIs, and the `CoordinateMapping`/`MetricData` geometry
module plus its explicit identity/orthogonal mapped operator layer.
`benchmarks/` owns the manufactured Cartesian and
physical channel runners, while `cases/` owns the lid-driven cavity runner.
`tests/contracts/` owns the P1--P8 contract-runner facade, its test-only access
bridge, and every manufactured/regression contract implementation. Contract checks may call
production operators through the composed bridge, but the checks themselves are
test code and do not belong to production sources. `tests/curvilinear/` owns
the independent metric assertions and mapping fixtures. Other `tests/` files contain
CTest wrappers, and `inputs/` retains the public runtime configurations.

Case and benchmark drivers are free runner functions accepting
`AVWiSSolver&`; they are not part of the solver's public API. The P1--P5
checks are methods of `AVWiSContractRunner`, which is composed with an existing
`AVWiSSolver&`. A contract test is not a solver, so inheritance is rejected:
it would express a false is-a relationship, complicate protected/private
access, and encourage construction of a second solver merely to run checks.
The composed runner instead uses the narrow `AVWiSContractTestAccess` friend
bridge. Raw `MultiFab` state remains private, and reusable operations such as
initialization, stepping, diagnostics, and field transforms stay on the solver.
Runtime dispatch remains in the small `src/RunMode.*` helper so established
`avwis.mode=contract_*` inputs continue to work. CMake names the boundary
explicitly: `AVWIS_PRODUCTION_SOURCES` contains only reusable solver/runtime
code, `AVWIS_APPLICATION_SOURCES` contains the executable and compatibility
dispatcher, and `AVWIS_TEST_MODE_SOURCES` contains `tests/contracts/*`. The
last set is linked because this executable deliberately exposes those test modes. In particular,
`contract_base_p2` still invokes the base check before the P2 check.

The solver owns the reusable numerical lifecycle: construction initializes the
AMReX geometry, grids, field storage, metrics, and boundary metadata, and
`initialize()` clears the persistent fields and applies the configured generic
boundary pipeline. A case or benchmark then supplies its physical initial
condition and any case-specific boundary setup through runner code (for
example, the lid velocity or manufactured shear). It does not reimplement the
solver or allocate a second solver; it only prepares the solver-owned state
before using the common advance and diagnostic APIs. This separation is why a
case may need setup even though the solver is already initialized: grid/field
allocation and physical flow initialization are different responsibilities.

P8 follows the same boundary. Production checkpoint/restart, point sampling,
plane CSV, and flow-statistics APIs remain in `src/`; the uninterrupted-versus-
restart comparison and manufactured sampling/schema assertions live in
`tests/contracts/AVWiSP8ContractChecks.cpp`. The narrow test-access bridge
coordinates private persistent state without putting P8 runner declarations
on the public `AVWiSSolver` API.

## P0--P4 data and lifecycle contract

The implementation is organized by responsibility: `CartesianBoundaryConfig.*`
owns input parsing and legacy-code mapping, `AVWiSBoundary.cpp` owns
physical ghosts and boundary-face fluxes, `AVWiSDiagnostics.cpp` owns runtime
diagnostics and schema output, and `tests/contracts/AVWiSContractChecks.cpp`
owns the P1--P5 checks, reductions, and manufactured test fields.  The solver header
and `AVWiSSolver.cpp` retain the data owner, lifecycle, halo exchange, and
Cartesian face/cell transforms.  Boundary kernels copy scalar configuration
into fixed-size GPU value arrays before launch; host strings and containers are
never captured by device lambdas.

`Geometry`, `BoxArray`, and `DistributionMapping` define one Cartesian level.
`P`, `Phi`, `Nvert`, `Ucat`, and `Ucat_old` are cell-centred. `Ucont`,
`Ucont_old`, and `Ucont_older` are three separate one-component face-centred
arrays: x/y/z faces respectively. Every field uses `avwis.nghost` grow cells
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
`avwis.metadata_file` is set. The P8 checkpoint path is separate: it writes a
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

The solver manifest declares `dx`, cell volume, Cartesian face areas and an unallocated
`legacy_Aj_equivalent=1/cell_volume` for unit-index computational coordinates;
this Cartesian schema remains unchanged even though the solver now owns a
read-only identity `MetricData`. The P2 inputs do not opt into mapped mode. `p2_contract.in`
checks periodic multi-box/MPI transforms, unique face count/flux sum, face
ghosts, and a linear net-flux/cell-volume divergence stencil.
`p2_boundary_face.in` checks non-periodic face ghosts and the explicit
boundary-face extrapolation across multiple boxes. These are contract and
manufactured algebra tests, not physical BC, projection, conservation, or CFD
validation.

## P5-003 C1/G0.2 identity adapter and C2.1/C2.2 orthogonal increment

`avwis_metric` is a production static library independent of the contract
runner and `AVWiSSolver` private state. `IdentityCoordinateMapping` fills
physical node coordinates from an explicit `LogicalGrid`; the current identity
parameterization uses the Cartesian `Geometry` origin and cell spacing, so
`mapping_jacobian_cc=1` while `cell_volume_cc=dx*dy*dz`. The two quantities are
separate fields and are never aliases.

`MetricData::define()` allocates nodal/cell/face fields on converted views of
one cell `BoxArray` and the same `DistributionMapping`. `build()` constructs
and validates the fields; later mutation is possible only through explicit
`rebuild()`. Production accessors return `const MultiFab&`. The frozen names
and meanings in this first increment are:

| Field | Meaning |
| --- | --- |
| `mapping_jacobian_cc` | Forward $J_x=\det(\partial x/\partial\xi)$ |
| `inverse_mapping_jacobian_cc` | Inverse $J_\xi=1/J_x$; legacy `Aj` semantics |
| `cell_volume_cc` | Discrete physical polyhedron volume; divergence volume source |
| `face_area_vector_fc[dir]` | Shared oriented integrated area vector toward increasing $\xi^{dir}$ |
| `grad_xi_cc` / `area_cofactor_cc` | $\nabla\xi^m$ and $J_x\nabla\xi^m$ |
| `face_gradient_metric_fc[dir]` | $Q_f^{mk}=S_f^m\cdot\nabla\xi_f^k$ |

The face geometry uses one fixed diagonal and two consistently oriented
triangles. Cell volume uses the same triangle moments and the divergence
theorem. Shared face fields call `OverrideSync` before halo fill; coordinate
ghosts preserve physical periodic translation rather than copying periodic
node positions. The dedicated `avwis_p5_metric_identity_contract` test checks
positive Jacobian/volume, reciprocal semantics, GCL closure, metric
reciprocity, `Ucont=u dot S`, zero divergence for constant velocity, exact
Cartesian identity values, multi-Box owner counts, and physical/inter-Box
ghost values.

At construction, `AVWiSSolver` defines and builds that identity metric from the
same `m_ba`, `m_dm`, and Cartesian `Geometry`, then freezes its successful
epoch. Public consumption is `const MetricData&` plus the recorded epoch; no
writable metric `MultiFab` is exposed. `compute_cartesian_divergence()` now
passes solver-owned integrated `Ucont` to `compute_identity_metric_divergence()`.
The adapter validates identity mapping, epoch, cell/face layouts and ownership,
then divides the net face flux exactly once by `cell_volume_cc`. Any future
non-identity mapping or stale epoch is rejected explicitly.

`avwis_p5_metric_identity_adapter_contract` compares the adapter with the old
Cartesian scalar-volume formula at
`512*epsilon*max(1, reference)` for one Box and many Boxes. It covers constant
`u dot S` face flux, zero constant-field divergence, nonzero affine-flux
divergence, overlapping-face layouts, and stale-epoch rejection. No runtime
input switch was added: existing Cartesian inputs retain their behavior and
cannot accidentally enable an unfinished mapped mode.

C1 is still not the complete G0 gate. `Ucat<->Ucont`, diagnostic divergence,
advection, viscosity, physical BC, pressure operator/correction, checkpoint
schema, and time advancement retain their established Cartesian metric sources.

`AnalyticOrthogonalCoordinateMapping` is a production separable provider with
per-axis positive `scale` and smooth sinusoidal `stretch`. Its strict
`abs(stretch)<1` invariant makes every directional derivative positive;
`scale=1,stretch=0` reproduces the C0 identity fields. The provider evaluates
nodal coordinates in GPU-safe kernels without capturing a host polymorphic
object. The strict factory accepts only `identity` and `analytic_orthogonal`,
and invalid finite/range/scale inputs fail before metric construction.

`avwis_p5_metric_analytic_orthogonal_contract` covers identity equivalence,
strong stretching, positive `Jx`/volume/oriented face areas, diagonal cell and
face metrics, reciprocity, GCL, constant-velocity `Ucont=u dot S` geometric
divergence, analytic node ghosts, shared-face owners, one/many Boxes, invalid
inputs, and three-grid second-order convergence of cell-centred `Jx`. Separable
face areas and polyhedral volumes are analytic-exact up to floating-point
roundoff for the frozen geometry construction.

C2.2 adds `MappingOperatorConfig` with explicit coordinate, mapping, and
projection modes. The default combination is `cartesian + identity +
cartesian_mlmg`. The only mapped combination is `mapped +
identity|analytic_orthogonal + orthogonal_mlmg`; stale epochs, mapping mismatch,
unknown modes, and `mapped + cartesian_mlmg` are rejected.

`AVWiSMappedOperators.*` implements centered metric cell gradients, integrated
face-flux divergence divided once by stored physical volume, diagonal face
pressure-gradient flux, and the separable orthogonal velocity transforms. The
solver path uses `MLABecLaplacian` on the volume-multiplied periodic pressure
equation and applies the same diagonal `Q` face flux in the correction. Enable
the tested path explicitly as follows:

```text
avwis.coordinates = mapped
avwis.mapping.type = analytic_orthogonal
avwis.mapping.scale = 1.2 0.85 1.1
avwis.mapping.stretch = 0.35 -0.25 0.2
avwis.projection.operator = orthogonal_mlmg
avwis.is_periodic = 1 1 1
```

This is not complete curvilinear solver support. C2.2 is limited to reusable
orthogonal operators and projection. C5.1/C5.2 extend it with the supported
mapped boundary vocabulary described below. Mapped advection/viscosity/time
advance, restart provenance, MPI/CUDA runtime, non-orthogonal cross terms, the
19-point pressure path, and a physical curved case remain outside the
increment. See
`_Docs/AMReX_P5-003_C2.2_orthogonal_operators_20260901.md`.

## P5-003 C5.1/C5.2 mapped physical boundaries

Mapped physical boundaries require an explicit, non-default configuration:

```text
avwis.coordinates = mapped
avwis.mapping.type = analytic_orthogonal
avwis.projection.operator = orthogonal_mlmg
avwisbcs.enabled = 1
avwisbcs.geometry = mapped_orthogonal
```

Non-periodic sides may use named `noslip`, `moving_wall`, `slip`, `symmetry`,
`inflow`, and `outflow`. The inlet/outlet topology remains exactly one of each;
closed domains use no inlet or outlet. Periodic directions use empty `lo/hi`
entries. Moving-wall velocity is a constant Cartesian vector and must be
tangential to every side carrying that mode. Mapped legacy integer BCs,
identity/unknown providers, coordinate/boundary mismatch, and general
non-orthogonal mappings remain rejected. Cartesian remains the default for
both coordinate and boundary geometry, so existing inputs retain their prior
path.

For logical face direction `m`, the boundary path reads the stored physical
area vector `S^m`, its magnitude, and its physical unit normal. Inlet profiles
are evaluated in physical tangential coordinates and globally normalized by
`sum(weight*|S^m|)`; authoritative `Ucont` is `u dot S^m`. Moving walls mirror
about the configured Cartesian wall vector, while slip and symmetry reflect
only the local physical-normal velocity. No-slip remains the C5.1 special case.
`Ucont` is already integrated, so area, Jacobian, and volume are not applied a
second time.

The orthogonal projection assigns homogeneous Neumann correction data to
prescribed-normal-flux boundaries and homogeneous Dirichlet correction data to
a fixed-pressure outlet. Consequently inlet/wall normal flux is preserved and
the outlet may be corrected to close divergence. Periodic/all-Neumann systems
retain the checked null space and volume-weighted gauge; outlet systems are
nonsingular. Corners preserve periodic wrapping first and then use the first
physical side in x/y/z order. This is deterministic, but it is not a general
multi-normal corner model.

The state order is `face owner/periodic halo -> physical cell ghost -> boundary
Ucont -> face owner/halo -> operator -> classified mapped projection ->
Ucont owner/halo -> Ucat reconstruction -> physical ghost -> diagnostics`.
Metric layout, ghost width, mapping ID, and epoch are checked before a mapped
boundary consumer runs. Focused inputs cover all six wall faces, mixed periodic
translation, corners, multi-Box ownership, stale ghost/metric and layout
rejection, inlet/outlet flux normalization and nonsingular projection,
moving/slip/symmetry states, unsafe normal wall motion, non-orthogonal and
legacy rejection, and projection compatibility. See the C5.1 report and
`_Docs/AMReX_P5-003_C5.2_mapped_boundary_modes_20260901.md` for exact
tolerances and acceptance results.

## P3 Cartesian physical boundaries

Set `avwisbcs.enabled=1` and explicitly name `avwisbcs.lo/hi` for every
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

`AVWiSDiagnostics.cpp` provides a cell-containing physical-coordinate
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
reference validation. The focused `avwis_lid_driven_cavity_sanity` CTest
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
avwis_physical_channel`. The runner writes `physical_channel.json`, with
one record per step containing timing, post-projection divergence, net mass
flux, momentum, kinetic energy, outlet flow, section means, centerline
velocity, pressure drop, and pressure statistics. It also writes arithmetic
time averages over the equally spaced post-step records. The report status is deliberately
`physical run / not yet validated` and `reference_available=false`: no legacy
or literature reference data is available, so a passing CTest means only that
the configured physical run and report contract completed.
