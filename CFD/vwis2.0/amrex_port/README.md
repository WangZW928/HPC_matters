# AVWiS (AMReX–VWiS) single-level mapped baseline

> Production increment (2026-10-12): opt-in face-authoritative incremental
> advancement now also supports periodic Cartesian and separable orthogonal
> grids, including stretched orthogonal metrics. One predictor/candidate/commit
> flow dispatches to Cartesian MLPoisson, orthogonal MLABecLaplacian, or the
> existing shear deferred solve. The stretched predictor uses volume-weighted
> interpolation to preserve total momentum after reconstruction. Runnable
> inputs and limits are described [below](#periodic-cartesian-and-orthogonal-incremental-pressure).

> Production increment (2026-10-12): `advance_one_step()` now accepts an optional
> cell acceleration field in addition to uniform body force. The new
> `case_mapped_pressure_wave` advances a decaying solenoidal flow with nonzero
> analytic pressure on affine shear, using continuum spatial forcing at time n.
> It exports per-step errors and an optional final field CSV. CPU 16/32/64
> refinement gives approximately second-order velocity, pressure, pressure-gradient
> and continuum momentum-balance errors. See [the runnable case](#mapped-pressure-wave-case).

> Latest production algorithm (2026-10-12): opt-in `face_authoritative + incremental_pressure` now runs an old-P face predictor in `advance_one_step()`, solves the shifted unknown Q=P_old+Phi with the existing deferred MLMG, and publishes pressure/velocity/history only after candidate validation. Phi is a pressure increment, not a dt-scaled impulse. Direct small-increment RHS stagnation was isolated without changing tolerances. The default dual-state and instantaneous P=Phi paths remain unchanged; physical pressure convergence and historical first-step pressure backend gates remain open. See [the implementation report](../_Docs/AMReX_M2_增量压力状态与移位求解_20261012.md).


> Latest increment (2026-10-12): two local stencil-rounding candidates failed to close the pressure gate safely and were reverted. The retained face-authoritative projection change computes each Ucat component maximum once per call, removing six duplicate local scans/global max reductions. CPU/MPI/CUDA focused CTests pass 18/19/18; all old-path fields and every face-authority snapshot remain exact against this round's baseline. First-step CPU layout/CUDA P/Phi remain BLOCKED at 1.11620e-12 / 1.19446e-12; MPI4 passes its 337-field slice. No timing speedup or completed M4 gate is claimed. See [the dated implementation report](../_Docs/AMReX_M2_后端舍入候选审计与M4范数复用_20261012.md).

> Latest opt-in increment (20261011): face-authoritative pressure RHS and deferred cross action now share direct signed net face flux, removing the net/V*V round trip. The default dual-state formulas and pressure/datum/failure contracts remain unchanged. CPU/MPI/CUDA builds and focused CTests pass 12/12 each; all 126 old-path fields remain exact against each backend's pre-change baseline. MPI4 passes all 337 fields, while CPU layout/CUDA first-step P/Phi remain BLOCKED (1.11620e-12 / 1.19446e-12) at the original <=1e-12 gate. M2/C3 remain BLOCKED, task counts stay 14/20/19. See [the implementation report](../_Docs/AMReX_M2_积分RHS统一与首步压力门_20261011.md); earlier notes below retain their historical scope.

## Naming convention

AVWiS means **AMReX–VWiS** and is the product name of this AMReX port. VWiS
continues to mean the original solver/algorithm. The AMReX implementation,
its executable (`avwis`), and its core C++ types use the `AVWiS` spelling.
The former `vwis.*` runtime namespace remains accepted as a compatibility
alias; new inputs and documentation use `avwis.*`. The boundary namespace
follows the same rule: `avwisbcs.*` is canonical and `vwisbcs.*` is legacy.

`amrex_port/` is independent of the original `vwis2.0/` PETSc/HYPRE solver.
It is a single-level Cartesian and opt-in separable-orthogonal mapped baseline
with an experimental periodic fixed weak-shear slice,
not a complete CFD solver. `project_cartesian()` forms a pressure
RHS from integrated face flux, solves a Cartesian cell-centred Poisson problem
with MLMG, corrects face flux, and synchronizes `Ucat`.
`compute_cartesian_advection_rhs()` and `compute_cartesian_viscous_rhs()` form
the P5-001 convective and P5-002 constant-coefficient viscous parts of the
momentum RHS. P5-004 adds a provisional explicit Euler predictor followed by
the P4 projection, with explicit `n/n-1/n-2` state rotation. This is not the
legacy SNES residual solve and does not implement semi-implicit or BDF2
advancement. P8-001/P8-002 and C6 provide a versioned single-level CPU
checkpoint/restart payload using AMReX `VisMF`, including all three fluid time
layers, strict Header validation, and mapped provenance. There is still no LES,
IBM/EB, FSI, general non-orthogonal operator, AMR, plotfile/HDF5 output,
MPI/GPU restart, legacy executable parity, or full CFD case validation. P5-003 C1 makes each solver own an immutable identity
`MetricData` on its exact `BoxArray/DistributionMapping` and uses a narrow
adapter as the cell-volume source for the existing integrated-flux divergence.
P5-003 C2.1 adds a production analytic orthogonal metric provider and geometry
contract. C2.2 adds explicit metric-aware gradient/divergence, the diagonal
orthogonal `Ucat<->Ucont` transform, and a periodic orthogonal MLMG projection
path. C2.3 adds conservative mapped advection, metric-aware constant-coefficient
viscosity, stability guards and explicit time advancement. The analytic path is
deliberate opt-in; Cartesian identity remains the default. C5.1 adds stationary no-slip walls and periodic pairs; C5.2 extends
that same analytic separable-orthogonal path with mapped inlet/fixed-pressure
outlet, tangential moving wall, slip, and symmetry modes. General non-orthogonal
surfaces and the legacy 19-point pressure path remain rejected. C3 now provides
a separate periodic weak-shear cross-term slice described below.
C8-G1 adds a CPU/single-rank pressure-gradient-driven Poiseuille reference on
the analytic stretched orthogonal mesh, including velocity, datum-free pressure,
divergence, flow-rate, wall-shear, and steady-residual convergence metrics.
P8-003 adds CPU/single-rank uniform-grid point probes,
plane summaries/CSV extraction, reusable flow statistics, and equal-spacing
post-step time averages in the physical channel report. These CSV/JSON files
are diagnostics and are explicitly not AMReX plotfiles.

The Cartesian boundary vocabulary also includes `moving_wall`, with
`avwisbcs.moving_wall_velocity` defaulting to `1 0 0`. Its normal component
must be zero. Cell ghosts use mirror Dirichlet data
`Ughost=2*Uwall-Uinterior`, tangential face-flux ghosts use the same wall
state after face-area scaling, and the physical normal face flux remains zero.
Existing `noslip` behavior is unchanged. Checkpoint schema version 3 records
and strictly checks coordinate mode, mapping type and parameters, projection,
metric discretization and a deterministic provenance checksum. The reader
still accepts schema-v1/v2 Cartesian checkpoints.

## Version and configuration contract

The local baseline is AMReX `26.04` at the git SHA recorded in
[amrex_version.lock](amrex_version.lock), CMake 3.20+, and C++17. The lock is
not proof that a compatible package has been built on a host: the test-result
record must name the install prefix and enabled backend. CMake gives a direct diagnostic if
`AMReXConfig.cmake` is unavailable and rejects a different exported AMReX
version unless the build is explicitly labelled exploratory.

Build the locked dependency from a Git checkout/worktree at the recorded SHA.
The upstream `26.04` tag's archive fallback reads the first heading of
`CHANGES.md`, which is still `26.03` at that commit; a source tree stripped of
`.git` therefore exports package version `26.03` and is correctly rejected by
this project. Do not hide that provenance loss with
`AVWIS_ALLOW_VERSION_MISMATCH` when producing validation evidence.

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
- `contract_p5_orthogonal_time`: run the C2.3 three-grid mapped advection,
  viscosity, non-trivial explicit-step, and free-stream contract.
- `contract_p5_mapped_boundary`: run the focused C5.1 analytic-orthogonal wall,
  periodic translation, ghost, ownership, validation, and projection contract.
- `contract_p5_mapped_boundary_c52`: run the C5.2 inlet/outlet,
  moving/slip/symmetry, mixed periodic, ownership, ghost-order, and matching
  pressure-projection contracts.
- `contract_p8_restart` and `contract_p8_sampling`: run the restart or uniform
  sampling/statistics contract.
- `contract_c3_temporal`: audit long mixed-mode advances with per-step numerical
  gates and canonical snapshots; pressure and peer acceptance are separate gates.
- `contract_c3_shear`: run the periodic weak-shear production-operator contracts
  and write numerical evidence to the required `metadata_file`.
- `contract_c7_backend`: export test-only canonical valid-field snapshots after
  a nontrivial orthogonal projection and three explicit steps for pointwise
  rank/layout/backend comparison. This is not a checkpoint or production I/O API.
- `benchmark_cartesian`, `benchmark_cartesian_mms`, and
  `benchmark_physical_channel`: run the baseline Cartesian report, the
  Taylor–Green MMS/legacy-stencil comparison, or physical plane-channel benchmark.
- `benchmark_mapped_channel_reference`: run the C8-G1 analytic-orthogonal
  pressure-gradient-driven Poiseuille reference and write JSON/CSV evidence.
- `case_lid_cavity`: run the lid-driven-cavity engineering case.
- `case_mapped_pressure_wave`: evolve the affine-shear analytic pressure wave
  with spatial forcing and export its physical-reference errors and optional CSV.

## Periodic Cartesian and orthogonal incremental pressure

The `face_authoritative + incremental_pressure` selection is now available
for Cartesian identity/`cartesian_mlmg` and mapped identity or
`analytic_orthogonal`/`orthogonal_mlmg`, in addition to the existing affine
shear/deferred path. It requires periodicity in all three directions and at
least one ghost cell. The default dual-state algorithms retain their existing
selection and arithmetic; instantaneous face-authoritative pressure remains
scoped to nonzero affine shear.

```bash
./build/c3-20261011-cpu/avwis amrex_port/inputs/incremental_cartesian.in
./build/c3-20261011-cpu/avwis amrex_port/inputs/incremental_orthogonal.in
```

Both inputs run the real solve loop for eight steps with nonzero uniform
three-component forcing. The orthogonal input includes stretch=(.1,-.05,.02).
Applications can also supply the spatial acceleration field described below.
Both paths use the same old-P predictor, shifted unknown Q=P_old+Phi,
pressure/face correction, cell reconstruction and candidate-only publication
as the shear implementation. Cartesian MLPoisson solves `Delta Q=-rhs/V`;
orthogonal MLABecLaplacian solves the integrated `(-N G_f)Q=rhs` using
`b_d=beta*Q_dd*h_d`. Residual validation uses the actual rounded P_old+Phi
and shared explicit face gradient, rather than MLMG's residual alone.

On stretched separable grids, the new state contract interpolates cell
acceleration with `(V_L*a_L+V_R*a_R)/(V_L+V_R)`. With the existing paired-face
reconstruction R this gives `sum(V R(I(a)))=sum(V a)` on the periodic
orthogonal grid. Arithmetic interpolation lost 3.81995e-8 of total momentum
in the eight-step mixed-mode case; the retained weighted interpolation reduces
that drift to 2.22045e-16. The uniform/affine and old default interpolations
keep their prior formulas. This is a conservative interpolation choice, not
a claim that I and R are inverse transforms or that it matches legacy stencils.

`avwis.projection.diagonal_max_iterations` controls only these new diagonal
incremental solves (1..1600, default 1600). Thin 64x64x4 solves exhausted a
200-cycle smoother budget; default bottom Krylov also failed on a later
16x16x4 wave step. The retained smoother/256 bottom sweeps and larger work
budget leave inner relative tolerance 1e-12, full relative tolerance 1e-11
and divergence gate 1e-10 unchanged. This establishes numerical operation,
not an optimized cost or speedup.

The existing pressure-wave case now also accepts Cartesian and affine
orthogonal inputs. For example:

```bash
./build/c3-20261011-cpu/avwis amrex_port/inputs/incremental_orthogonal.in \
  avwis.mode=case_mapped_pressure_wave 'avwis.mapping.stretch=0 0 0' \
  'avwis.body_force=0 0 0' avwis.final_time=0.002 \
  avwis.field_file=incremental_orthogonal_wave.csv
```

CPU 16/32/64 x 4 pressure-wave runs at T=.002, dt=.000125 give near-second-order
velocity, pressure, pressure-gradient and continuum momentum-balance errors.
Pressure errors are 1.81869e-3 / 4.53794e-4 / 1.13394e-4 on both families,
with ratios 4.008 / 4.002; maximum full relative residual stays below 9.84e-13.
The reused mixed-mode contract also exercises exact P_old+Phi updates, genuine
one-iteration solver exhaustion, full-state failure preservation, retry and zero
updates. It remains one existing contract, with no new CTest entries.

Physical boundaries, diagonal incremental checkpoint/restart, general
nonorthogonal mapping and pressure temporal convergence remain unsupported or
unverified. The first two are explicitly rejected; schema-v3 does not encode
this new state/pressure pair. Evidence is in `build/incremental-diagonal-20261012/`.

Both families also ran actual nonzero-force solve, mixed-mode state/failure
checks and the affine pressure wave on MPI4 and CUDA. Final MPI4 wave U/P/Phi
values are exact against CPU. CUDA final P relative L2 differences are
3.93e-14 / 4.69e-14 (Cartesian / orthogonal); the much smaller Phi differences
are 4.50e-10 / 5.75e-10. These observations do not close a tight all-field backend
gate or the historical shear pressure gate. Existing focused regression is CPU
10/10 (including physical BC and old restart), MPI executable 5/5 and CUDA 5/5.
The MPI CTest slice is single rank; the new application/state runs above use
four ranks. CPU's original shear numerical snapshots remain exact against this
round's executable baseline, excluding elapsed-time metadata.

## Mapped pressure wave case

This case supports affine Cartesian, orthogonal and shear grids and uses the real `face_authoritative + incremental_pressure` advance,
not a contract runner or a replay of a prescribed discrete RHS. On the affine
mapping `x=sx*xi+epsilon*sy*eta`, `y=sy*eta`, set
`theta=wx*xi+wy*eta`, where each logical direction has one periodic wave.
Its physical wave vector is `K=(wx/sx, wy/sy-epsilon*wx/sx, 0)` and
`B=(K_y,-K_x,0)/|K|`. The independent continuum solution is

```text
u(t) = 0.25 exp(-nu |K|^2 t) B sin(theta)
p(t) = 0.2 exp(-t) cos(theta)
f(t) = grad(p(t)) = -0.2 exp(-t) K sin(theta)
```

Because `B dot K=0`, both divergence and convective acceleration vanish;
`u_t=nu Laplacian(u)` and `-grad(p)+f=0`. The force is evaluated analytically
at cell centers at each step, independent of the production gradient stencil.
Initial face fluxes are exact face integrals of the analytic velocity, with
tangential sinc factors; cell velocity is reconstructed by the production R.
Initial P is the nonzero analytic pressure. No initialization projection is
used to manufacture that pressure or its forcing.

```bash
./build/c3-20261011-cpu/avwis amrex_port/inputs/mapped_pressure_wave.in
# Writes mapped_pressure_wave.json and mapped_pressure_wave.csv.
# Override metadata_file/field_file to select destinations; field_file= disables CSV.
```

The JSON series compares velocity at `t_(n+1)` and pressure at `t_n`, the time
of the explicit momentum RHS, and reports pressure gradient and continuum
momentum balance errors. It also reports the pressure error against `t_(n+1)`
to expose the time offset rather than silently relabel it. CSV columns contain
cell indices, physical coordinates, Ucat, P, Phi and analytic P at `t_n`.
CSV export gathers the final field on the IO rank and is intended for small
reference cases, not scalable production I/O. The case requires positive
viscosity, an integer `final_time/dt`, zero uniform body force, and no restart.

For application code, `advance_one_step(dt, nu, uniform_force, &force_field)`
adds a solver-layout cell-centered three-component acceleration at time n.
Only valid cells are read; the caller owns and updates the field. Invalid
layout/component count or nonfinite values are rejected before state changes.
The field is not a persistent checkpoint variable. A null pointer retains the
previous arithmetic in all three advancement paths.

CPU `16/32/64 x 4`, epsilon=.15, scale=(1.1,.9,1.05), nu=.01,
T=.002 and dt=.000125 produced the following final volume-weighted RMS errors:

|Reference error|16|32|64|Refinement ratios|
|---|---:|---:|---:|---|
|Velocity|4.48882e-3|1.12863e-3|2.82568e-4|3.977 / 3.994|
|Pressure at explicit n|2.71742e-3|6.83989e-4|1.71292e-4|3.973 / 3.993|
|Pressure gradient|5.23276e-2|1.32728e-2|3.33029e-3|3.942 / 3.985|
|Continuum momentum balance|5.32553e-2|1.35127e-2|3.39079e-3|3.941 / 3.985|

This closes a nonzero-pressure spatial reference for this forced affine wave.
It does not establish nonlinear-flow pressure accuracy, physical boundaries,
general mapping, physical pressure at n+1, or temporal convergence. The
historical first-step pressure backend gate remains separate. Numerical fields,
commands and logs are in `build/mapped-pressure-wave-20261012/`.

The 16x16x4 wave also ran on MPI4 and CUDA. Final MPI4 U/P/Phi CSV values
are exact against CPU. CUDA final P differs by at most 1.385e-14 (relative L2
3.480e-14); the much smaller Phi differs by at most 2.273e-14 (relative L2
4.119e-10). The near-zero w component has at most 1.042e-17 absolute
backend difference. These observations do not close the existing all-field
backend gate. Five existing focused CTests pass on CPU, MPI and CUDA; no
new CTest or large regression matrix was added. CPU unforced incremental
and instantaneous numerical snapshots remain exact against this round's
saved executable (wall-clock metadata excluded).

## Explicit solve input

For example:

```text
avwis.mode = solve
avwis.n_cell = 16 16 16
avwis.is_periodic = 1 1 1
avwis.run_steps = 8
avwis.dt = 1.0e-3
avwis.viscosity = 0.1
avwis.body_force = 0.0 0.0 0.0
```

The former independent `vwis.run_*` boolean selectors are not retained as a
compatibility interface. Inputs using the compatibility namespace must still
choose one `vwis.mode`; a missing or unknown mode is rejected with an explicit
diagnostic. This makes simultaneous operations impossible by construction.

`avwis.projection_time_coefficient` is configurable only in `contract_p4`,
`contract_p5_orthogonal`, `contract_p5_mapped_boundary`, and
`contract_p5_mapped_boundary_c52`, which pass it to a standalone projection.
All other modes accept only the default or explicit value `1`; another value
(including `1.5`, NaN, or infinity) is rejected before solver construction.
Explicit advancement fixes alpha to one. Setting this key cannot request BDF2
or modify accumulated-pressure semantics. The legacy `vwis.*` alias follows
the same rule. The [P5-005 implementation report](../_Docs/AMReX_P5-005_投影时间系数拒绝契约_20261011.md)
records CPU/MPI/CUDA focused validation; the pressure-time and fine-dt Phi
backend gates remain BLOCKED.

## Source ownership

`src/` owns the reusable production application core: solver state and
lifecycle, Cartesian operators and boundary handling, checkpoint/restart,
runtime diagnostic APIs, and the `CoordinateMapping`/`MetricData` geometry
module plus its explicit identity/orthogonal and periodic fixed-shear mapped
operator layer.
`benchmarks/` owns the manufactured Cartesian, physical channel, and mapped
Poiseuille reference runners, while `cases/` owns the lid-driven cavity runner.
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
first and physical cell ghosts/boundary faces second. Expensive P3 diagnostics
are explicit observation points rather than unconditional hot-path reductions.
Any valid write invalidates both freshness epochs. Long-lived temporaries must
be class-owned; do not allocate an owning `MultiFab` inside `MFIter`.

`write_metadata_manifest` writes rank-0 JSON schema only when
`avwis.metadata_file` is set. The P8 checkpoint path is separate: it writes a
versioned `Header` plus AMReX `VisMF` payloads. Cartesian and analytic
orthogonal mapped state are supported, including cross-Box decomposition
restore; execution is intentionally limited to one CPU rank and must not be
confused with a plotfile or MPI restart.

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
object. The C2 factory accepts `identity` and `analytic_orthogonal`,
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
cartesian_mlmg`. The C2 mapped combination is `mapped +
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

This is not complete curvilinear solver support. C2.3 is limited to separable
orthogonal operators, projection, conservative advection, constant-coefficient
viscosity and explicit Euler. C5.1/C5.2 supply the mapped boundary vocabulary,
and C6 supplies CPU restart provenance. C8-G1 supplies one analytic orthogonal
physical reference. MPI/GPU restart, general non-orthogonal operators, the 19-point pressure path,
general curved surfaces, and legacy executable parity remain outside this slice.

## C3 fixed weak shear and cross operators (2026-10-11)

The new opt-in `analytic_shear` provider composes the separable coordinates
`(X,Y,Z)` with `x=X+epsilon*(Y-ylo), y=Y, z=Z`. The finite bound
`abs(epsilon)<=0.25` controls the shear; only all-periodic geometry with
`avwisbcs.enabled=0` is accepted. This is a translated periodic parallelepiped,
not general curvilinear support or legacy 19-point operator parity.

```text
avwis.coordinates = mapped
avwis.mapping.type = analytic_shear
avwis.mapping.scale = 1.1 0.9 1.05
avwis.mapping.stretch = 0.25 -0.15 0.1
avwis.mapping.epsilon = 0.15
avwis.projection.operator = deferred_nonorthogonal
avwis.projection.max_iterations = 40
avwis.is_periodic = 1 1 1
avwisbcs.enabled = 0
avwis.time_integrator = explicit_euler
avwis.viscous_operator = component_laplacian
```

`metric_face_gradient_value()` supplies the normal two-cell derivative and
centered tangential derivatives to both pressure and component diffusion.
`compute_metric_face_gradient_flux()` exposes full/diagonal/cross integrated
fluxes. The pressure equation and Ucont correction use the same full flux,
while divergence divides once by stored physical volume. Deferred correction
solves the diagonal equation with MLMG and accepts only a full physical
relative infinity residual `<=1e-11`; incompatible integrated RHS or exhausted
outer iterations throw before changing velocity, Ucont, or pressure. Solver
workspaces are not rolled back. Bottom smoothing is deliberately conservative;
its cost and wider grid acceptance remain M4 work.

At exactly zero epsilon, viscosity and projection use the established
orthogonal specializations. The transform uses Cartesian velocity dot full
face area vectors; its inverse is the normalized six-face least-squares
triangular specialization for these fixed normals. Projection retains C2's
incremental cell-gradient velocity update, with Ucont authoritative for mass.
Reconstruction after projection and broader authority/freshness policy remain
an explicit C3/M2 follow-up; generic least-squares reconstruction is not claimed.
The current limited contract treats `Ucat` as the transported cell momentum
velocity and `Ucont` as the finite-volume mass flux. They receive their own
pressure corrections; halo freshness does not assert inverse interpolation.
Overwriting projected `Ucont` from `Ucat`, or reconstructing `Ucat` on every
step, would change this discretization and needs a separate M2 decision.

Nonzero shear now owns a persistent `DeferredProjectionWorkspace`: six face
and seven cell MultiFabs, including the cell pressure gradient. It binds to one
immutable metric object/epoch/layout and rejects a different owner even with
identical geometry. Only storage is reused; coefficients, RHS, gauge work and
Phi are recomputed, including after a failed solve. MLMG and its hierarchy are
still constructed per projection. All face/cell corrections are prepared
before committing flow valid values, followed by one valid-epoch increment
and owner/halo synchronization. Phi and scratch are outside rollback; complete
advance/history/predictor rollback remains unimplemented.

The inner solve retains relative tolerance `1e-12` and now permits 1600
iterations. A mixed-mode later step on `64x64x4` was still converging at the
previous 800 limit; diagnostics record outer/inner counts and initial/final
residuals. `last_step_projection()` exposes the last completed advance's
projection report. This storage/cost increment makes no performance claim.

`contract_c3_shear` and `tests/c3_shear_regression.py` verify three-grid operator,
gradient and advective MMS, signed shear scans, zero-shear projection regression,
independent affine 11-point action on an irregular periodic vector, diffusion
energy sign, conservation, pressure recovery, failure state protection, a
production step, four coupled mixed xy/xz-mode advances with nonzero initial
pressure, exact cell/face history rotation, volume-weighted momentum, cell and
face correction consistency, epoch/halo freshness, workspace reuse after
failure, epsilon-zero multistep orthogonal parity, and 100-step free stream. CTest also rejects wrong operators,
unsafe epsilon, physical BCs, unsupported BDF2/stress choices and shear
checkpoint output. Every numerical fixture also checks both checkpoint reader
and writer rejection before file access. The checkpoint schema has not been expanded for shear;
**shear checkpoint/restart remains rejected**, including zero epsilon.

Both algorithm selectors default to the existing explicit Euler/component
Laplacian baseline. Other choices are rejected explicitly. Legacy SNES uses a
nonlinear half-current/half-old RHS residual; legacy viscosity includes the
transpose-gradient stress contribution. These decisions do not establish
legacy algorithm equivalence or implement implicit advancement/full stress.
See [`AMReX_C3_弱剪切算子与算法决策_20261011.md`](../_Docs/AMReX_C3_弱剪切算子与算法决策_20261011.md)
for formulas, decision gates, commands, results and remaining scope. The
continuation evidence is recorded in
[`AMReX_C3_多步状态与workspace_20261011.md`](../_Docs/AMReX_C3_多步状态与workspace_20261011.md).

The third C3 increment adds `avwis.state_contract=experimental_dual_state`
(default for `analytic_shear`). This is an acceptance declaration: `Ucont`
authorizes mass, `Ucat` is transported independently, and `P` records accumulated
corrections. Requests such as `single_authority` or `time_converged_pressure`
are rejected before solver allocation. Other mappings reject this scoped key.
It does not select a new integrator, reconstruction, or per-field epoch policy.

`contract_c3_temporal` and `tests/c3_temporal_audit.py` now audit genuinely
coupled long advances. CPU/MPI/CUDA run 32/64/128 steps to `T=0.008` and 256 steps
to `T=0.064`; CUDA uses a single Box and a larger wall-clock budget, with a
matching CPU peer matrix. Every step checks full residual `<=1e-11`, face
divergence `<=1e-10`, momentum/mass, exact history, and halo freshness.
Canonical valid snapshots support full-field refinement and backend comparison.
The fixed-grid velocity/face error ratios are about two, but datum-free
accumulated pressure errors grow as dt decreases (ratio about 0.5). This blocks
physical-pressure acceptance; it is not a closed C3 or legacy CFD time gate.

The audit records independent `invariant_status`, `velocity_time_gate`,
`pressure_time_gate`, and `backend_field_gate`; CTest audit success means the
advance invariants and velocity refinement passed. `--require-pressure-convergence`
requires the pressure gate too. Peer comparison normally requires every field
at relative L2 `<=1e-12` and the established pointwise guard. Explicit
`--observe-peer` preserves failed field gates as `BLOCKED` without rejecting an
otherwise valid trajectory; it never labels those backend comparisons PASS.
MPI and CPU/CUDA Phi comparisons can exceed the L2 gate at the fine dt even when transported
fields, mass, and each physical residual pass. These unresolved gates are
reported in [the M2/time audit](../_Docs/AMReX_C3_时间审计与M2拒绝契约_20261011.md).

```bash
ctest --test-dir build/c3-20261011-cpu -R avwis_c3 --output-on-failure
ctest --test-dir build/c3-20261011-mpi -R avwis_c3 --output-on-failure
ctest --test-dir build/c3-20261011-cuda -R 'avwis_c3_(temporal|state_contract|pressure_contract)' --output-on-failure
```

## Pressure/Phi reference diagnosis (2026-10-11)

This opt-in test observes the existing experimental dual-state algorithm. It
records raw/projected/interpolated and predictor divergence, full/diagonal/cross
DG and DIC actions, P/Phi, metric arrays, gauge, ranks/layout, and every MLMG
inner initial/final residual and iteration count alongside the outer physical
residual. The optional trace adds no reductions or changed stopping rule to the
default solver. A spatial old-P perturbation tests the actual advance rather
than a surrogate pressure update.

The initial DG-DIC constraint defect is 0.085266493. After 16 fine-dt steps,
layout/MPI4/CUDA Phi differences remain about 2e-12 relative; identical RHS
replays agree to about 1e-16 with identical inner iteration counts. Solving each
candidate's own RHS on CPU reproduces its Phi to about 1e-16. This reference
isolates amplification of velocity/RHS roundoff by divergence and 1/dt; it does
not close the original long-trajectory backend gate or the pressure time gate.
There is no pressure algorithm repair in this increment. P remains accumulated
correction history and stronger pressure/state requests remain rejected.

Requires Python 3 and NumPy (observed 2.5.1), plus the locked existing builds.
Use the current c3_temporal input; the script opts into the diagnostic branch:

```bash
python3 -B amrex_port/tests/pressure_phi_reference.py \
  --cpu build/c3-20261011-cpu/avwis \
  --mpi build/c3-20261011-mpi/avwis \
  --cuda build/c3-20261011-cuda/avwis \
  --input amrex_port/inputs/c3_temporal.in \
  --work build/pressure-reference-20261011-matrix
python3 -B amrex_port/tests/test_pressure_phi_reference.py
```

Each invocation creates a fresh run directory containing commands, logs, full
canonical snapshots, frozen RHS and SHA-256, and evidence.json. Failures retain
INCOMPLETE/error evidence. `--require-backend` writes the evidence then exits
nonzero for the reproduced isolated backend failure; observation mode returns
DIAGNOSTICS_COMPLETE with BLOCKED results. Neither mode reports pressure time
convergence. `pressure_reference_warmup=0..16` and `pressure_reference_rhs`
are test-only keys in contract_c3_temporal. The frozen RHS is exactly one finite
value per cell in x-fastest order; incompatible integrals are rejected without
repair. Both periodic face endpoints and every diagnostic field are retained.

CPU 14/14, MPI 15/15, CUDA 14/14 focused CTests and one existing C3 single-grid
recovery contract on each backend passed. These counts exclude the diagnostic
matrix and six Python checks; the full long suites were not rerun. See
[dated implementation report](../_Docs/AMReX_Pressure-Phi_reference根因诊断_20261011.md).

## C7 rank, layout, and CPU/CUDA comparison

`tests/c7_backend_compare.py` runs periodic and mixed inlet/outlet
analytic-orthogonal contracts on `16x8x8` cells. Each run performs a nontrivial
projection, evaluates the production advective/viscous RHS, advances three
explicit steps, and gathers only valid values into canonical cell/face domains
on the IO rank. This small test-only snapshot copies GPU data to pinned host
storage; it does not extend production checkpoint or sampling support.

The comparison includes all three `Ucat/Ucont` time layers, `P/Phi`, initial
RHS, final divergence, cell volumes, and global mass/stability reductions.
Every field must satisfy relative L2 difference `<=1e-12`, plus a pointwise
`2e-10 + 2e-11*max(abs(reference),abs(candidate))` guard. The exactly zero
divergence reference uses physical `U/L=1` as its normalization scale.
Projection/final divergence must be below `1e-10`, and the divergence-theorem
defect below `1e-12`. Rank/backend/version/field shape and finite values are
checked explicitly; missing or failed runs cannot reuse a stale PASS report.

Python 3.8+ is required when `BUILD_TESTING=ON`. CPU and CUDA CTest suites
register the comparator's six negative/positive contracts and a three-layout
comparison (`max_grid_size=4/8/16`). An MPI AMReX package additionally
registers `avwis_c7_mpi_comparison`, launching 1/2/4 ranks with CMake's
`MPIEXEC_EXECUTABLE`, `MPIEXEC_NUMPROC_FLAG`, `MPIEXEC_PREFLAGS` and
`MPIEXEC_POSTFLAGS`. On hosts with fewer slots, configure
`-DMPIEXEC_PREFLAGS=--oversubscribe`. Grid size 4 exercises all four ranks;
grid size 8 also exercises idle ranks in the four-rank run.

```bash
ctest --test-dir build/c7-20261011-mpi -R avwis_c7 --output-on-failure
python3 -B amrex_port/tests/c7_backend_compare.py \
  --exe build/c7-20261011-cpu/avwis --inputs amrex_port/inputs \
  --work build/c7-20261011-cpu/c7_cuda_peer \
  --peer-exe build/c7-20261011-cuda/avwis
# A serial CPU executable may also be compared against the MPI executable.
python3 -B amrex_port/tests/c7_backend_compare.py \
  --exe build/c7-20261011-mpi/avwis --inputs amrex_port/inputs \
  --work build/c7-20261011-mpi/c7_serial_peer \
  --mpi-exec /usr/bin/mpiexec --mpi-preflag=--oversubscribe \
  --peer-exe build/c7-20261011-cpu/avwis --peer-backend CPU
```

The 2026-10-11 results and clean-build commands are recorded in
[`AMReX_C7_MPI_CPU-GPU对照_20261011.md`](../_Docs/AMReX_C7_MPI_CPU-GPU对照_20261011.md).
C7 remains open for detailed profiling, CUDA-aware MPI and MPI/GPU restart.
The recorded maximum-rank time covers three complete advances; it is not a
kernel/MLMG/halo/copy breakdown or a scaling/speedup claim.

## C8-G1 mapped channel physical reference

`benchmark_mapped_channel_reference` uses an analytic-orthogonal mesh stretched
only in the wall-normal `y` direction, periodic `x/z`, no-slip `y` walls,
constant viscosity `0.01`, and body force `f_z=0.12`. The exact steady solution
is `w(y)=6 y(1-y)`, with unit bulk flow and wall-shear magnitude `0.06`. The
`16/32/64` inputs start from the exact velocity and advance 100 steps to check
the production RHS/time path rather than a benchmark-only operator.

The validation also changed mapped no-slip/moving-wall viscous face fluxes to a
nonuniform one-sided quadratic wall derivative. Mirror ghosts remain the public
boundary-state contract; only the viscous wall-flux reconstruction changed.
The CTest requires near-second-order velocity, steady residual, and flow-rate
trends, at least first-order wall-shear convergence, and roundoff-level pressure
and divergence. The full values and legacy-source provenance boundary are in
`_Docs/AMReX_C8-G1_正交mapped通道物理reference_20260918.md`.

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
the semi-implicit/BDF2 solver decision remains P5-005 work. P8/C6 now provide a
separate restart payload for the validated single-level Cartesian and
separable-orthogonal CPU paths; this does not make the explicit integrator
equivalent to legacy SNES or extend restart to MPI, GPU, IBM, FSI, or AMR state.

## P8-001/P8-002 checkpoint and restart

The checkpoint directory contains a strict text `Header` and AMReX `VisMF`
payloads. Schema v3 records the locked AMReX version/SHA, single-rank CPU scope,
dimension/precision, domain/dx, boundary configuration, coordinate/mapping
parameters, projection and metric-discretization provenance, ghost width,
component/layout contract, time/step/history depth, and the fixed field list.
The payload stores `Ucat`, `Ucat_old`, `Ucat_older`, all three
`Ucont` history layers, `P`, `Phi`, and `Nvert`.

Restart validates the Header before reading any field. It rejects an invalid
magic/version, unsupported rank/backend, dimension or precision mismatch,
field/layout mismatch, wrong geometry, mapping or BC configuration, missing
payload, or unexpected extra field. Metrics are immutable derived state: the
configured mapping rebuilds and validates them before payload restore. A mapped
regression compares uninterrupted and checkpoint-at-K trajectories bitwise;
another restores the same payload into a different Box decomposition, and a
negative test rejects changed mapping parameters.

The original Cartesian evidence is recorded in the P8 report. The 2026-09-18
C6 extension and its current test commands/results are recorded in
`_Docs/AMReX_P10-001_C2.3_C6_实施与验证_20260918.md`. MPI/GPU restart and
plotfile/HDF5 output remain outside this increment.

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

The exact historical commands and error boundaries are recorded in
`_Docs/AMReX_P3_边界与诊断设计及测试_20260824.md`. A host session that permits
OpenMPI sockets is still required for the 2-rank result. The earlier CUDA
runtime blocker was closed on 2026-09-18 by the current `sm_89` evidence below.

## Current CPU and CUDA verification on 2026-09-18

Against locked AMReX 26.04 at the recorded SHA, the Release CPU build passes
45/45 CTests, including C8-G1. A separate CUDA 12.6.85 build for an RTX 4060
Laptop GPU (`sm_89`) previously passed 36/36 applicable CTests; after the C8-G1
wall-flux correction it compiles cleanly and the affected mapped C2.3/C5/C5.2
runtime subset passes 8/8. CMake does not register the explicitly CPU-only
checkpoint/restart, host sampling, and physical cases in a CUDA suite; this is
an honest scope boundary, not a skipped claim of GPU I/O support. Numerical
tables and commands are recorded in
`_Docs/AMReX_P10-001_C2.3_C6_实施与验证_20260918.md` and
`_Docs/AMReX_C8-G1_正交mapped通道物理reference_20260918.md`.

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


### 2026-10-11 M2 一致初态与失败事务诊断增量

新增 test-only affine shear 离散 streamfunction 初态：cell 权威一次构造 Ucont=I(Ucat)，
初态 DI/D 均约 6.62e-15，非零 DG multiplier 与独立 Fourier reference 通过；
DIC source residual 约0.14645，even grid有8个checkerboard null modes，不能把DG当DIC。
4/8/16步固定T=0.002的Ucat/三面差比近2，累计P约0.49999、末Phi约0.30083，仍BLOCKED。
直接face重投影的flow字段幂等PASS但Phi被清零，全字段幂等BLOCKED；先I重建再投影不幂等。
真实整步deferred exhaustion准确拒绝，current/history/epoch已改变，整步rollback BLOCKED。
新affine CPUlayout16/MPI4/CUDA六transport字段对照PASS；CUDA全部诊断字段仍13项BLOCKED。
这不是原stretched pressure/Phi门PASS：原门保留历史BLOCKED，本轮未重跑长矩阵。
本轮生产源码保持起始hash；定向CTest CPU13/13、MPI14/14、CUDA13/13通过，
数学决策strict driver准确exit=1并保留完整证据。当前主统计仍14/20/19（53项），
M2/P5-005/M4/C3/C7进行中、C4未开始。
详见 [M2一致初态与失败事务诊断](../_Docs/AMReX_M2_一致初态与失败事务诊断_20261011.md)。


```bash
python3 -B amrex_port/tests/consistent_state_reference.py \
  --cpu build/c3-20261011-cpu/avwis --mpi build/c3-20261011-mpi/avwis \
  --cuda build/c3-20261011-cuda/avwis --input amrex_port/inputs/c3_temporal.in \
  --work build/m2-consistent-20261011-matrix
```

`--require-decision` 在保存 evidence 后把数学决策 BLOCKED 转成非零退出。
`consistent_reference=1` 仅供 temporal contract；和 `pressure_reference=1` 互斥，
拒绝 non-affine mapping 和超出1..64的步数。初始P=lambda0是face约束multiplier，
随后仍为生产P history，不能解释为已经时间收敛的物理压力。


### 2026-10-11 M2 opt-in 面守恒权威生产增量

新增真实生产 `advance_one_step()` 分支，显式选择
`state_contract=face_authoritative`、`pressure_contract=instantaneous_pressure`、
`failure_contract=fatal`。仅支持非零 affine `analytic_shear`（stretch=0 0 0）、
全周期、无物理 BC、现有 explicit Euler/component Laplacian；默认仍为
`experimental_dual_state` + `accumulated_correction`，没有默认切换。
新面 predictor 为 Ucont[n]+dt*I(advection+viscosity+force)，保持面权威；成功 DG
投影后以现有六面归一化剪切重构 R 得 Ucat=R(Ucont)，不会先以 I(Ucat) 覆盖面状态。
`instantaneous_pressure` 严格指该次 DG 投影的离散 face 约束乘子，P=Phi；其 RHS
在显式 n 层评估，不能宣称发布于 n+1 的 P 已通过连续物理压力/时间层验收。
旧累计输出为 P+=Phi correction history，旧 P 不消费于 predictor。

投影面/cell 候选全部准备后才提交。真实 exhaustion 不提交 P/Ucat/投影修正，
但面 predictor、history、epoch 可改变；整步仍 fatal，`failure_contract=rollback`
准确拒绝。没有全 advance rollback、C4/SNES/BDF2/完整应力/IBM/FSI。
固定 T=.002 的 CPU 4/8/16 步速度/面差比近2，P/Phi 离散乘子差比2.00143；
限定时间子门通过不替代物理压力验收。layout16/MPI4/CUDA 完整337字段中
首步 P/Phi 原 relative<=1e-12 门仍 BLOCKED，其余335字段 PASS；原 stretched
压力/Phi门未重跑，保留 BLOCKED。主统计仍 **14 已完成、20 进行中、19 未开始**，
M2/P5-005/M4/C3/C7 进行中，C4 未开始。实现、最终验证及边界见
[面守恒权威生产增量报告](../_Docs/AMReX_M2_面守恒权威生产增量_20261011.md)。


```text
avwis.mode = solve
avwis.state_contract = face_authoritative
avwis.pressure_contract = instantaneous_pressure
avwis.failure_contract = fatal
avwis.coordinates = mapped
avwis.mapping.type = analytic_shear
avwis.mapping.epsilon = 0.15
avwis.mapping.scale = 1.1 0.9 1.05
avwis.mapping.stretch = 0 0 0
avwis.projection.operator = deferred_nonorthogonal
avwis.is_periodic = 1 1 1
avwisbcs.enabled = 0
```

保留现有网格、dt、黏性与运行步数配置。示例测试输入
`inputs/m2_face_authority.in` 使用 `contract_c3_temporal` +
`face_authority_reference=1` 生成非平凡初态并调用真实生产推进。
生产 `solve` 的零场初始化及既有 body_force 入口继续适用；未新增任意初场导入。
切回默认 pair 即恢复旧数值路径。shear checkpoint/restart 仍准确拒绝。

```bash
python3 -B amrex_port/tests/face_authority_audit.py \
  --cpu build/c3-20261011-cpu/avwis --mpi build/c3-20261011-mpi/avwis \
  --cuda build/c3-20261011-cuda/avwis --input amrex_port/inputs/m2_face_authority.in \
  --work build/m2-face-authority-20261011-matrix
```

默认 driver exit=0 仅表示证据收集完成；`--require-decision`/`--strict`
在保留完整 evidence 后为未通过总门返回非零。所有近零及 history/失败字段保留，
不以最后一步 P/Phi 的 PASS 替代首步完整后端 BLOCKED。

### 2026-10-12 Opt-in incremental pressure

For the existing periodic, nonzero affine `analytic_shear` slice only:

```ini
avwis.mode = solve
avwis.state_contract = face_authoritative
avwis.pressure_contract = incremental_pressure
avwis.failure_contract = fatal
avwis.time_integrator = explicit_euler
```

Keep the mapping/operator/geometry settings from `inputs/m2_pressure_increment.in`;
its reference selector belongs to the contract mode, so set
`avwis.pressure_increment_reference=0` when overriding it to `solve`.
`solve` starts from the existing zero-velocity, P=Phi=0 initialization; nontrivial
mixed-mode initialization in the acceptance runner is test-only. Run, for example:

```bash
build/c3-20261011-cpu/avwis amrex_port/inputs/m2_pressure_increment.in \
  avwis.mode=solve avwis.pressure_increment_reference=0 avwis.run_steps=4 \
  avwis.body_force="0.01 -0.02 0.03"
```

Let B=F_old+dt I(A_old), F*=B-dt G_f(P_old), with alpha=1. The increment
satisfies (-N G_f)Phi=-N(F*)/dt. Production solves the equivalent shifted system
(-N G_f)Q=-N(B)/dt, forms Phi=Q-P_old, and commits P=P_old+Phi exactly once.
The rounded combined P is rechecked at full relative residual <=1e-11 and
projected divergence <=1e-10 before publication. F_new=B-dt G_f(P), and
Ucat_new=R(F_new). Q uses the existing volume-weighted zero datum; P retains
that datum to rounding precision. All quantities are nondimensional; for beta=1,
restored dimensional P/Phi are kinematic pressure L²/T², so there is no extra
1/dt in the pressure update. P is the discrete face multiplier for explicit
n-layer forcing, published in the n+1 storage slot; continuous physical pressure
or its time level is not yet validated.

Standalone `project_mapped()` rejects this contract, so it cannot replay Phi.
Numerical/solver rejection leaves accepted P/Phi/velocities/histories and time
unchanged; dirty candidates are rebuilt. Device/MPI faults during publication
remain fatal. Only this opt-in allocates two extra scalar MultiFabs; registered
field order and existing checkpoint restrictions stay intact.

Rollback selects `pressure_contract=instantaneous_pressure` (P=Phi, no old-P
predictor), or the unchanged default `experimental_dual_state +
accumulated_correction`. Incremental Phi is not comparable to instantaneous Phi
as if they were the same quantity. The fixed-T audit checks discrete refinement,
not a continuous physical-pressure reference:

```bash
ctest --test-dir build/c3-20261011-cpu -R '^avwis_m2_pressure_increment$' --output-on-failure
```
