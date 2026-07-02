#ifndef included_StructSolver
#define included_StructSolver

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "petsctime.h"
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"
#include "ImmersedBoundary.h"
#include "FSI.h"

class StructSolver
{
public:
   StructSolver(
       const std::string object_name,
       CurvGrid *grid,
       UData *data,
       ImmersedBoundary *ib,
       FSI *fsi);

   ~StructSolver();

    PetscErrorCode 
    Solve(
        PetscInt si, 
        PetscInt ti,
        PetscReal time,
        PetscBool *converged);

    PetscBool 
    CheckConvergence(PetscInt si);

private:
    
    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;
    ImmersedBoundary *d_ib;
    FSI *d_fsi;

    PetscInt d_sisteps;
    PetscInt d_immersed;
    PetscInt d_movefsi;
    PetscInt d_rotatefsi;
    PetscInt d_rotatefsi_noIBsearch;
    PetscInt d_changefsi;
    PetscInt d_rstart_fsi;
   
    PetscInt d_NumberOfBodies;
    PetscInt d_NumberOfRotatingBodies;


    PetscReal d_str_tol;
    PetscReal d_str_max;
    char d_path[256];
};

#endif
