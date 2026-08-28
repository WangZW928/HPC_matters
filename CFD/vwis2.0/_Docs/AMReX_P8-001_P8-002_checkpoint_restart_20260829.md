# AMReX P8-001/P8-002 checkpoint and restart

_VWiS to AMReX incremental implementation report — 2026-08-29_

---

## 📋 Scope and conclusion

This increment implements and validates the **single-level, uniform Cartesian,
single-process CPU** checkpoint/restart path for P8-001 and P8-002. It does not
claim plotfile/HDF5 output, MPI or GPU restart, curvilinear metrics, IBM/EB,
FSI, AMR, or full CFD validation.

The implementation is accepted for this declared scope. The checkpoint stores
real AMReX `VisMF` payloads rather than metadata alone, and restart validation
compares the complete persistent fluid state.

## 🔧 Implementation

### Versioned checkpoint contract

`VwisAmrExCheckpoint.cpp` writes a checkpoint directory containing a strict
`Header` manifest and one `VisMF` payload per persistent field. The manifest
records:

- schema magic/version and locked AMReX version/SHA
- single-rank CPU scope, dimension, precision, ghost width, components, and layout
- domain, `dx`, periodicity, and Cartesian boundary configuration
- `time`, `step`, and `history_depth`
- a fixed field list and field locations

The payload includes `Ucat`, `Ucat_old`, `Ucat_older`, the three directional
`Ucont` layers, `P`, `Phi`, and `Nvert`. Restart validates the manifest and
payload presence before reading fields. Invalid magic/version, unsupported rank
or backend, geometry/BC mismatch, field/layout mismatch, missing fields, and
unexpected extra fields are rejected with a `P8 checkpoint rejected:` diagnostic.

### Restart regression

The P8 contract creates a nonzero state, writes a checkpoint, reads it back,
and compares every persistent field and time-layer value. It then compares an
uninterrupted N-step trajectory with a trajectory checkpointed at K and resumed
to N. The comparison is bitwise on the validated CPU path.

## ✅ Verification evidence

All commands were run from `CFD/vwis2.0`:

```text
./amrex_port/tests/static_contract_check.sh
cmake --build build/amrex_port_p8 -j 16
ctest --test-dir build/amrex_port_p8 --output-on-failure
git diff --check
```

Results:

| Check | Result |
|---|---:|
| Static contract check | PASS |
| CPU rebuild/link | PASS |
| P8 restart consistency | PASS |
| P8 strict rejection | PASS |
| Complete CTest suite | 20/20 PASS |
| `git diff --check` | PASS |

The P8 tests use temporary runtime directories under the build tree and leave
the source tree free of checkpoint payloads.

## ⚠️ Boundaries and follow-up

- The implementation intentionally rejects multi-rank restart; MPI runtime evidence remains a separate task.
- AMReX `VisMF` payloads are not legacy PETSc binary/HDF5 files and no automatic conversion is provided.
- Plotfile/visualization output and sampling/statistics remain to be implemented.
- P5-005 semi-implicit/BDF2/SNES work remains separate from this explicit Euler baseline.
- IBM/EB, FSI, AMR, GPU runtime, real CFD cases, and release-level numerical acceptance remain open.

The working tree contains the uncommitted implementation and generated build
artifacts. No commit or push was performed in this increment.
