#ifndef included_BcsUtility
#define included_BcsUtility

#define INLET 5
#define OUTLET 4
#define SOLIDWALL 1    
#define SYMMETRIC 3
#define FARFIELD 6

#include <stdlib.h>
#include <stdio.h>
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"
#include "PlaneExtraction.h"

using namespace std;

class BcsUtility
{
public:

    BcsUtility(
         const std::string& object_name,
         CurvGrid *grid,
         UData *data,
         PlaneExtraction *plane);

    ~BcsUtility();

    void CalculateInletArea();
    PetscErrorCode InflowFlux(PetscInt ti);
    PetscErrorCode OutflowFlux();
    PetscErrorCode FormBcs(
         PetscInt ti,
         int outflow_scale);
    PetscErrorCode InitializeFlowField();
    double randn_notrig();
    PetscErrorCode IbBC();
    PetscErrorCode CalculateInflowFlux();
    PetscErrorCode ScaleInitialFlow();
    PetscErrorCode ReadPlane(PetscInt ti);

    PetscErrorCode setUcatPlane(Cmpnts **uplane) {d_ucat_plane = uplane;}

private:
   
    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;
    PlaneExtraction *d_iplane;

    PetscReal d_FluxInSum, d_FluxOutSum;

    PetscReal d_threshold;
    PetscReal d_inletArea;
    PetscInt d_k_area_allocated;

    PetscReal d_inlet_flux;
    PetscInt d_inletprofile;
    PetscInt d_pseudo_periodic;
    PetscReal d_fluct_rms;
    PetscInt d_initial_perturbation;
    PetscInt d_initial_gaussian_perturbation;
    PetscReal d_magnitude_gaussian_perturbation;

    double *d_k_area;
    double *d_k_area_ibnode;

    PetscReal d_mean_k_area, d_mean_k_area_ibnode;

    Cmpnts **d_ucat_plane;
};

#endif
