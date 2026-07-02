#ifndef included_PointProbe
#define included_PointProbe

#include <stdlib.h>
#include <stdio.h>
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"

using namespace std;

class PointProbe
{
public:

    PointProbe(
         const std::string& object_name,
         CurvGrid *grid,
         UData *data);

    ~PointProbe();

    PetscErrorCode Initialize();
    PetscErrorCode Probe(PetscInt ti, PetscReal dt, PetscReal time);

private:

    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;
   
    Cmpnts *d_savecoor;
    Index *d_saveindx;
    PetscInt d_npoints;

    char d_path[256];
    char d_fpath[256];
};

#endif

