#ifndef included_PlaneExtraction
#define included_PlaneExtraction

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"

using namespace std;

class PlaneExtraction
{
public:

    PlaneExtraction(
         const std::string& object_name,
         CurvGrid *grid,
         UData *data);

    ~PlaneExtraction();

    PetscErrorCode Save(PetscInt ti, PetscReal time);
    PetscErrorCode StoreSection(PetscInt kplane);
    PetscErrorCode Read(PetscInt ti);

    Cmpnts **getUcatPlane() {return d_ucat_plane;}
private:

    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;

    PetscInt d_nsavek;
    PetscInt d_ucat_plane_allocated;

    char d_path[256];
    char d_ipath[256];
    PetscInt d_save_inflow_period;
    PetscInt d_save_inflow_minus;
    PetscInt d_ti_lastsave;
    PetscInt d_inflow_recycle_period;
    PetscInt d_read_inflow_period;
    PetscReal d_scale_velocity;

    PetscInt *d_ksection;
    Cmpnts **d_ucat_plane;
    FILE *d_fp_inflow_u;
};

#endif 
