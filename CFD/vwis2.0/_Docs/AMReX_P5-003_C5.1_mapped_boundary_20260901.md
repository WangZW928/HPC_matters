# AMReX P5-003 C5.1 mapped physical boundary increment

_AVWiS analytic separable orthogonal wall/periodic boundary and ghost layer · 2026-09-01_

---

## 📋 Conclusion

C5.1 is complete for one narrow production subset: explicit mapped mode with
the `analytic_orthogonal` separable provider, stationary no-slip physical
walls, periodic direction pairs, and the matching singular orthogonal MLMG
projection. Cartesian coordinates and Cartesian boundary geometry remain the
default, and the complete existing Cartesian test suite is unchanged.

This result is not full C5 or G1. Mapped inlet/outlet, moving wall, slip,
symmetry, a general curved/non-orthogonal boundary, mapped time advancement,
restart, MPI/CUDA runtime, C3 cross terms, C4 19-point pressure, periodic hill,
and a physical curved validation case remain outside this increment.

## 🎯 Accepted API and rejected combinations

The opt-in contract is:

```text
avwis.coordinates = mapped
avwis.mapping.type = analytic_orthogonal
avwis.projection.operator = orthogonal_mlmg
avwisbcs.enabled = 1
avwisbcs.geometry = mapped_orthogonal
```

Each non-periodic side must use the named `noslip` condition. A periodic
direction must be periodic on both sides through `avwis.is_periodic`; its
`avwisbcs.lo/hi` entries remain empty. Existing inputs that omit
`avwisbcs.geometry` keep `cartesian` semantics.

| Combination | C5.1 result |
| --- | --- |
| Cartesian coordinates + default boundary geometry | Accepted through the unchanged Cartesian path |
| Mapped analytic orthogonal + stationary no-slip/periodic | Accepted through C5.1 |
| Mapped inflow/outflow | Rejected; physical profile normalization and pressure datum are not proven |
| Mapped moving/slip/symmetry | Rejected; no validated mapped semantics in this slice |
| Mapped legacy integer BC | Rejected; the mapped API requires an explicit named condition |
| Mapped identity/general mapping | Rejected for physical boundaries; C5.1 is analytic orthogonal only |
| Coordinate/boundary geometry mismatch | Rejected before initialization |

## ⚙️ Numerical semantics and ordering

For logical face direction $m$, `MetricData::face_area_vector_fc(m)` stores the
oriented physical area vector $mathbf S_f^m$. C5.1 obtains
$A_f=\|\mathbf S_f^m\|$ and the physical unit normal
$\mathbf n_f^m=\mathbf S_f^m/A_f$. The supported separable mapping is diagonal,
so the contracts additionally require the area vector to be axis aligned and
positive in its logical direction.

At a stationary no-slip wall, the boundary state is
$\mathbf u_w=\mathbf 0$. The Cartesian velocity ghost is mirror Dirichlet,

$$
\mathbf u_g=2\mathbf u_w-\mathbf u_i=-\mathbf u_i,
$$

which makes both the wall-normal and wall-tangential face averages zero. The
authoritative boundary flux is

$$
U_f^m=\mathbf u_w\cdot\mathbf S_f^m=0.
$$

`Ucont` already contains integrated physical flux. Boundary fill, divergence,
and pressure correction do not multiply it by face area or Jacobian again;
the mapped divergence divides the net flux by `cell_volume_cc` exactly once.

Pressure and pressure-correction ghosts are homogeneous Neumann at a wall.
The orthogonal MLMG operator uses periodic or Neumann domain BCs by direction,
checks the volume-integrated singular compatibility condition, fixes the
physical-volume-weighted pressure gauge, and uses the same explicit diagonal
face-gradient flux for correction. Projection therefore preserves the exact
wall-normal `Ucont` value.

```mermaid
flowchart LR
    accTitle: C5.1 Boundary Consumer Order
    accDescr: Fixed mapped boundary sequence from immutable metrics and face ownership through physical ghost fill, projection, velocity reconstruction, and diagnostics

    metric([⚙️ Validate metric epoch]) --> owner[🔄 Sync face owners]
    owner --> halo[🔄 Fill periodic halo]
    halo --> ghost[⚙️ Fill physical ghosts]
    ghost --> wall_flux[⚙️ Set wall flux]
    wall_flux --> resync[🔄 Resync face owners]
    resync --> projection[⚙️ Project with Neumann BC]
    projection --> reconstruct[🔄 Reconstruct Ucat]
    reconstruct --> diagnostics([📊 Check boundary diagnostics])

    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef terminal fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class owner,halo,ghost,wall_flux,resync,projection,reconstruct process
    class metric,diagnostics terminal
```

Physical cell corners preserve values supplied by periodic halo exchange, then
apply the first non-periodic side in x/y/z order. At an edge or corner with
multiple physical walls, one mirror is applied according to that precedence;
the result is deterministic and is not presented as a general multi-normal
corner construction. Face boxes use AMReX `OverrideSync` after boundary-face
writes so the lowest global box owner is authoritative before halo refresh.

## ✅ Focused contracts and tolerances

| Contract | Manufactured evidence and tolerance |
| --- | --- |
| Six physical faces | All low/high faces in x/y/z have positive axis-aligned stored area vectors; cross components at most `2e-12` relative scale |
| Wall normal flux | Boundary `Ucont` is exactly zero before projection and at most `2e-12` after projection |
| Wall tangential velocity | Every Cartesian component has zero face average within `2e-12` relative scale |
| Periodic translation | Analytic boundary nodes match the translation-periodic mapping within `2e-12` relative scale |
| Periodic/wall corners | Periodic image is retained before deterministic x/y/z physical-wall precedence; ghost values agree within `2e-12` |
| Multi-Box ownership | 18-Box layouts match the exact `OwnerMask` unique-face count in every direction |
| Freshness and layout | Stale flow halo, stale metric epoch, metric/solver layout mismatch, and insufficient metric layout/ghost contract are rejected |
| Unsafe configuration | Mapped inflow/outflow expected-failure input is rejected rather than using Cartesian behavior |
| All-wall projection | Maximum divergence decreases from `1.939845013` to `2.619916133e-11` |
| Mixed periodic/wall projection | Maximum divergence decreases from `7.312175192` to `2.416896131e-11` |

The two projection cases use integrated manufactured face flux. The all-wall
case uses a sine that vanishes on both physical end faces; the mixed case uses
a translation-periodic sine in x with physical no-slip walls in y and z. Both
are singular periodic/Neumann systems with compatible integrated RHS.

## 🧪 Build and acceptance results

The acceptance directory is `build/amrex_port_p5c51`. It is a Release build
against locked AMReX 26.04, GNU C/C++ 13.3.0, CPU double, MPI OFF, CUDA OFF.

| Verification | Result |
| --- | --- |
| Configure/build | PASS; all targets built |
| Focused C5.1 CTest | PASS; 3/3 |
| Full CTest | PASS; 35/35 |
| Lid-driven cavity sanity | PASS; included in full CTest and run separately |
| Static contract | PASS |
| `git diff --check` | PASS |
| MPI runtime | NOT TESTED |
| CUDA runtime | NOT TESTED |

Commands:

```bash
cmake -S amrex_port -B build/amrex_port_p5c51 \
  -DAMReX_DIR=/tmp/vwis-p5-amrex-git-install.Ra0ukN/lib/cmake/AMReX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/amrex_port_p5c51 -j2
ctest --test-dir build/amrex_port_p5c51 \
  -R 'avwis_p5_c5_' --output-on-failure
ctest --test-dir build/amrex_port_p5c51 --output-on-failure
ctest --test-dir build/amrex_port_p5c51 \
  -R avwis_lid_driven_cavity_sanity --output-on-failure
bash amrex_port/tests/static_contract_check.sh
git diff --check
```

## ⚠️ Limitations and next seam

C5.1 does not establish inlet mass-flow normalization on mapped faces, a
fixed-pressure mapped outlet, moving or slip/symmetry wall transforms, general
non-orthogonal face normals, multi-normal corner physics, mapped advection or
viscosity, time advance, restart provenance, MPI/GPU runtime, AMR, IBM/FSI, or
a physical curved-flow regression. The existing C2.2 report remains a record
of its own earlier operator scope; this report only records the new C5.1 seam.

The next safe seam is C6: persist mapping type, analytic parameters, metric
discretization version, checksum/epoch policy, and boundary geometry/side
configuration, then rebuild and validate metrics before restoring fields.
Mapped inlet/outlet should remain rejected until decision gate M3 supplies a
frozen pressure datum and a manufactured integrated-flow contract. C7/C8 must
add backend runtime and a physical mapped case before complete G1 can close.

## 🔗 Internal references

- [C2.2 orthogonal operator report](./AMReX_P5-003_C2.2_orthogonal_operators_20260901.md)
- [Migration plan](./AMReX迁移方案.md#当前状态剩余工作与下一步路线)
- [Task list](./AMReX移植任务清单.md#p5-003-曲线路线子模块状态)
- [Curvilinear implementation specification](./AVWiS曲线坐标实现规格.md)
- [Port README](../amrex_port/README.md#p5-003-c51-mapped-physical-boundaries)
