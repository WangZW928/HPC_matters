#ifndef included_FlowSolver
#define included_FlowSolver

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"
#include "RHSSolver.h"
#include "BcsUtility.h"
#include "LESModel.h"
#include "WallModel.h"
#include "Integrator.h"
#include "PoissonSolver.h"

class FlowSolver
{
public:
    FlowSolver(
        const std::string object_name,
        CurvGrid *grid,
        UData *data,
        RHSSolver *rhs,
        BcsUtility *bcs,
        WallModel *wall,
        LESModel *les,
        Integrator *integrate,
        PoissonSolver *poisson);

    ~FlowSolver();

    PetscErrorCode Solve(
       PetscInt ti,
       PetscReal time);

    PetscErrorCode CalculateDivergence(PetscInt ti);
    PetscErrorCode CalculateKE(PetscInt ti);

    PetscReal getMaxDivergence() {return d_maxdiv;}

private:

    PetscErrorCode ReadFromInput();

    const std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;
    RHSSolver *d_rhs;
    BcsUtility *d_bcs;
    WallModel *d_wall;
    LESModel *d_les;
    Integrator *d_integrate;
    PoissonSolver *d_poisson;


    PetscInt d_immersed;
    char d_path[256];
    //timers
    PetscReal t_solve_time;

    PetscReal d_maxdiv;
};

#endif
    
