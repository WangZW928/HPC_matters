#ifndef included_LESModel
#define included_LESModel

#include <stdlib.h>
#include <stdio.h>
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"

using namespace std;

class LESModel
{
public:

    LESModel(
         const std::string& object_name,
         CurvGrid *grid,
         UData *data);

    ~LESModel();

    void Initialize();
    void ComputeSmagorinksyConstant(PetscInt ti);
    void ComputeEddyViscosity();

    PetscBool useLES() {return d_use_les;}
    
    Vec getlNu_t() {return d_lNu_t;}

    //Get the maximum Nu_t
    PetscReal getMaxNorm() {return d_max_norm;}

    void WriteCs(PetscInt ti);
    void ReadCs();
private:

    PetscReal 
    integrate_testfilter_simpson(
       double val[3][3][3], 
       double w[3][3][3]);
    PetscReal 
    integrate_testfilter_ik(
       double val[3][3][3], 
       double vol[3][3][3]);

    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;

    Vec d_Nu_t;
    Vec d_lNu_t;
   
    Vec d_lCs, d_Cs;
    Vec d_lLM, d_lMM;
    Vec d_lLM_old, d_lMM_old;

    PetscBool d_use_les;
    PetscInt d_les;
    PetscInt d_i_homo_filter, d_j_homo_filter, d_k_homo_filter;
    PetscInt d_testfilter_ik;
    PetscReal d_les_eps;
    PetscReal d_max_cs;
    PetscReal d_wall_cs;
    PetscReal d_max_norm;
    PetscInt d_filter_size;
    PetscInt d_wallfunction;
    PetscInt d_tistart;
    PetscBool d_restart;
    char d_fieldpath[256];
    PetscInt d_hdf5, d_write_hdf5, d_read_hdf5;
    char d_wext[6], d_rext[6];
    PetscInt d_tiout;

};

#endif

