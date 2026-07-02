#ifndef included_Integrator
#define included_Integrator


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "petsctime.h"
#include "petscvec.h"
#include "petscdmda.h"
#include "petscksp.h"
#include "petscpc.h"
#include "petscsnes.h"

#include "CurvGrid.h"
#include "UData.h"
#include "RHSSolver.h"
#include "BcsUtility.h"
#include "WallModel.h"

using namespace std;

class Integrator
{
public:

    Integrator(
        const std::string& object_name,
        CurvGrid *grid,
        UData *data,
        RHSSolver *rhs,
        WallModel *wall,
        BcsUtility *bcs);

    ~Integrator();

    static
    PetscErrorCode SNESMonitor(
        SNES snes,
        PetscInt n,
        PetscReal rnorm,
        void *dummy);

    PetscErrorCode Solve(PetscInt ti);

    static
    PetscErrorCode SolveFunction(
        SNES snes,
        Vec Uconti,
        Vec Rhs,
        void *ptr);

    double CalculateMinimumDt();
    //Final Solver Residual
    PetscReal getResidual() {return d_norm;}
    //Maximum Velocity
    PetscReal getUNorm() {return d_unorm;}

    //These are created by SolveFunction needs to be static
    //to work with petsc
    CurvGrid *getGrid() {return d_grid;}
    UData *getData() {return d_data;}
    RHSSolver *getRHS() {return d_rhs;}
    WallModel *getWall() {return d_wall;}
    BcsUtility *getBcs() {return d_bcs;}

private:
    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;
    RHSSolver *d_rhs;
    WallModel *d_wall;
    BcsUtility *d_bcs;

    PetscBool d_snes_created;
    PetscReal d_imp_free_tol;
    PetscReal d_norm;
    PetscReal d_unorm;

    SNES d_snes;
    KSP d_ksp;
    PC d_pc;

    PetscReal d_dx_min;
    PetscReal d_di_min, d_dj_min, d_dk_min;
    PetscReal d_di_max, d_dj_max, d_dk_max;
};

#endif
