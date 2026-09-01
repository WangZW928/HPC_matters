# AMReX P5-003 C2.2 orthogonal operator increment

_AVWiS analytic separable orthogonal operator/projection increment · 2026-09-01_

---

## 📋 Conclusion

The smallest production C2.2 increment is complete. AVWiS now has an explicit
mapping/operator configuration, reusable metric-aware cell gradient and
integrated-flux divergence, diagonal orthogonal `Ucat↔Ucont` transforms, and a
fully periodic orthogonal pressure projection wired into `AVWiSSolver`.

Cartesian identity remains the default and retains its previous metadata and
numerical path. This increment does not close complete G1 and does not claim
general curvilinear support.

## 🎯 Scope and numerical contract

The accepted mapped runtime combination is:

```text
avwis.coordinates = mapped
avwis.mapping.type = identity|analytic_orthogonal
avwis.projection.operator = orthogonal_mlmg
```

The public sample input uses `analytic_orthogonal`. Unknown modes, stale metric
epochs, mapping/metric mismatch, layout mismatch, and `mapped +
cartesian_mlmg` are rejected. Mapped mode currently also requires at least one
ghost cell, no physical boundary configuration, and full periodicity.

The conservative divergence remains

$$
D_c(U)=\frac{\sum_m(U^m_{f+}-U^m_{f-})}{V_c},
$$

where `Ucont` already stores the integrated flux $\mathbf u_f\cdot\mathbf S_f$.
No face area, Jacobian, logical spacing, or cell volume is applied a second
time.

For the supported diagonal mapping, the pressure face flux is

$$
G_f^m(\phi)=\beta_fQ_f^{mm}\frac{\phi_R-\phi_L}{h_m}.
$$

`MLABecLaplacian` solves the volume-multiplied equation with
$b_f=\beta_fQ_f^{mm}h_m$. The face correction calls the same explicit
$G_f^m$ implementation. The singular compatibility check therefore uses the
integrated RHS, and the pressure gauge is physical-volume weighted.

## 📚 Files

| Area | Files |
| --- | --- |
| Production API/operators | `src/AVWiSMappedOperators.H/.cpp`, `src/AVWiSMetricData.H` |
| Solver integration | `src/AVWiSSolver.H/.cpp`, `src/AVWiSProjection.cpp`, `src/AVWiSTime.cpp`, `src/AVWiSCheckpoint.cpp`, `src/AVWiSDiagnostics.cpp`, `src/main.cpp`, `src/RunMode.H/.cpp` |
| Focused contracts | `tests/curvilinear/MetricOrthogonalOperatorContract.cpp`, `tests/contracts/AVWiSC2ContractChecks.cpp` |
| Inputs/build/static checks | `inputs/p5_analytic_orthogonal_projection.in`, `inputs/p5_mapped_cartesian_rejected.in`, `CMakeLists.txt`, `tests/static_contract_check.sh` |
| Documentation | `README.md`, `AMReX迁移方案.md`, `AMReX移植任务清单.md`, this report |

## ✅ Contract coverage and tolerances

| Contract | Evidence and tolerance |
| --- | --- |
| Constant/linear gradient | Constant gradient is exact zero; logical-linear gradient agrees with the discrete metric expression within `4096*epsilon` |
| Gradient MMS | Physical trigonometric field on `N=12,24,48`; $L_\infty$ errors `0.6899207645`, `0.2036489304`, `0.05325812737`; orders `1.760346445`, `1.93501063`; acceptance order at least `1.6` |
| Mapped divergence | Constant free stream at most `2e-11`; affine physical velocity gives divergence three within `2e-11`; physical volume is applied once |
| Orthogonal transforms | Constant `Ucat→Ucont→Ucat` maximum error at most `2e-13`; multi-Box face ownership/halo path used |
| Identity limit | Generic C2.2 divergence equals the C1 identity adapter and diagonal face gradient equals the Cartesian expression within `512*epsilon` |
| Invalid configuration | Mapping mismatch, stale epoch, incompatible/unknown operator, mapped Cartesian runtime input, and bad layouts are rejected |
| Projection | Analytic mapped divergence decreases from `7.510048977` to `8.003150917e-11`; reduction exceeds `9e10`; reported solve residual `5.126832292e-11` |

## 🧪 Acceptance results

The acceptance build is `build/amrex_port_p5c22_final`, configured as Release
against locked AMReX 26.04, CPU double, MPI OFF, CUDA OFF.

| Verification | Result |
| --- | --- |
| Clean configure/build | PASS; GNU C/C++ 13.3.0; all targets built |
| Focused C0/C1/C2.1/C2.2 and rejection CTest | 6/6 PASS |
| Full CTest | 32/32 PASS |
| Lid-driven cavity sanity | 1/1 PASS |
| Static contract check | PASS |
| `git diff --check` | PASS |
| MPI runtime | NOT TESTED; acceptance AMReX package is MPI OFF |
| CUDA runtime | NOT TESTED; acceptance AMReX package is CUDA OFF |

Commands:

```bash
cmake -S amrex_port -B build/amrex_port_p5c22_final \
  -DAMReX_DIR=/tmp/vwis-p5-amrex-git-install.Ra0ukN/lib/cmake/AMReX \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/amrex_port_p5c22_final -j2
ctest --test-dir build/amrex_port_p5c22_final \
  -R 'avwis_p5_(metric_identity|metric_analytic|metric_orthogonal|orthogonal_projection|mapped_cartesian_rejected)' \
  --output-on-failure
ctest --test-dir build/amrex_port_p5c22_final --output-on-failure
ctest --test-dir build/amrex_port_p5c22_final \
  -R avwis_lid_driven_cavity_sanity --output-on-failure
bash amrex_port/tests/static_contract_check.sh
git diff --check
```

## ⚠️ Explicit boundary and remaining seam

C2.2 does not implement mapped advection, viscosity, physical BC/ghosts, time
advance, diagnostic/checkpoint provenance, restart, MPI/CUDA runtime, AMR,
IBM/FSI, non-orthogonal cross terms, or the C4 19-point pressure operator. The
solver rejects the affected mapped entry points instead of using Cartesian
implementations.

The next safe integration seam is C5/C6: curved physical boundary/ghost
semantics and mapping-aware checkpoint/restart. C7/C8 must then supply backend
runtime and a physical curved case before complete G1 can close.

> **Subsequent status, 2026-09-01:** C5.1 now supplies the restricted
> analytic-orthogonal stationary no-slip wall/periodic boundary subset. Mapped
> inlet/outlet, general curved boundaries, C6 restart, C7 runtime, and C8
> physical-case evidence remain open. See the
> [C5.1 report](./AMReX_P5-003_C5.1_mapped_boundary_20260901.md).

## 🔗 Internal references

- [C2.1 analytic orthogonal geometry report](./AMReX_P5-003_C2_analytic_orthogonal_20260831.md)
- [Migration plan](./AMReX迁移方案.md#当前状态剩余工作与下一步路线)
- [Task list](./AMReX移植任务清单.md)
- [Curvilinear implementation specification](./AVWiS曲线坐标实现规格.md)
