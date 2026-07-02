
#ifndef included_CurvGrid
#define included_CurvGrid

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include "petscvec.h"
#include "petscdmda.h"

#include "functions.h"
//#include "ibm_functions.h"

using namespace std;

class CurvGrid
{
public:

    CurvGrid(const std::string& object_name);

    ~CurvGrid();

    PetscErrorCode ReadGrid();
    PetscErrorCode ReadBC();
    PetscErrorCode InitializeVecs();
    PetscErrorCode CreateDM();

    //Form the Grid Metrics
    PetscErrorCode FormMetrics();

    DM getDA() {return d_da;}
    DM getFDA() {return d_fda;}

    //Helper functions
    Vec getlCsi() {return d_lCsi;}
    Vec getlEta() {return d_lEta;}
    Vec getlZet() {return d_lZet;}
    Vec getlAj() {return d_lAj;}

    Vec getlICsi() {return d_lICsi;}
    Vec getlIEta() {return d_lIEta;}
    Vec getlIZet() {return d_lIZet;}
    Vec getlIAj() {return d_lIAj;}

    Vec getlJCsi() {return d_lJCsi;}
    Vec getlJEta() {return d_lJEta;}
    Vec getlJZet() {return d_lJZet;}
    Vec getlJAj() {return d_lJAj;}

    Vec getlKCsi() {return d_lKCsi;}
    Vec getlKEta() {return d_lKEta;}
    Vec getlKZet() {return d_lKZet;}
    Vec getlKAj() {return d_lKAj;}

    Vec getlCent() {return d_lCent;}

    PetscInt isPeriodic() { return d_periodic;}
    PetscInt isIPeriodic() { return d_i_periodic;}
    PetscInt isJPeriodic() { return d_j_periodic;}
    PetscInt isKPeriodic() { return d_k_periodic;}
    PetscInt isIIPeriodic() { return d_ii_periodic;}
    PetscInt isJJPeriodic() { return d_jj_periodic;}
    PetscInt isKKPeriodic() { return d_kk_periodic;}
    
    PetscInt getBC(PetscInt i) {return d_bctype[i];}
   
    PetscInt *getIdx() {return d_idx_from;}

private:
   
    PetscErrorCode ReadFromInput();

    std::string d_object_name;

    //Type of grid file inputs
    PetscBool d_xyz_input, d_binary_input, d_uniform_input; 
   
    //Input Path
    char d_path[256], d_gridfile[256];

    //Characteristic length
    PetscReal d_cl;

    //Number of blocks
    //always one right now
    int d_block_number;

    //Size of Grid
    PetscInt d_IM, d_JM, d_KM;
    PetscReal d_Lx, d_Ly, d_Lz;

    PetscInt d_periodic;
    PetscInt d_i_periodic, d_j_periodic, d_k_periodic;
    PetscInt d_ii_periodic, d_jj_periodic, d_kk_periodic;

    //Boundary Condition types
    PetscInt d_bctype[6];

   /* Data structure for scalars 
      (include the grid geometry informaion, 
       to obtain the grid information use DMDAGetCoordinates) */
    DM d_da;  
    //Data structure for vectors 
    DM d_fda;

    // Coordinates of cell centers
    Vec d_Cent;
    // Grid Metrics at cell centers   
    Vec d_Csi, d_Eta, d_Zet, d_Aj;
    //Grid Metrics on cell faces
    Vec d_ICsi, d_IEta, d_IZet, d_IAj;
    Vec d_JCsi, d_JEta, d_JZet, d_JAj;
    Vec d_KCsi, d_KEta, d_KZet, d_KAj;
    Vec d_GridSpace;

    //Local Vecs
    Vec d_lCent;
    Vec d_lCsi, d_lEta, d_lZet, d_lAj;
    Vec d_lICsi, d_lIEta, d_lIZet, d_lIAj;
    Vec d_lJCsi, d_lJEta, d_lJZet, d_lJAj;
    Vec d_lKCsi, d_lKEta, d_lKZet, d_lKAj;
    Vec d_lGridSpace;

    PetscInt *d_idx_from;
 
};

#endif

