#ifndef included_UData
#define included_UData

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <cmath>
#include "petscvec.h"
#include "petscdmda.h"
#include "petscviewerhdf5.h"


#include "Timer.h"
#include "CurvGrid.h"
#include "WallFunctions.h"

using namespace std;

class UData
{
public:

    UData(const std::string& object_name, CurvGrid *grid);

    ~UData();

    void CopyLastStep();
    PetscErrorCode InitializeData();
    void ReadData();
    void WriteData(PetscInt ti);
    void Contra2Cart_single(
       Cmpnts &csi, 
       Cmpnts &eta, 
       Cmpnts &zet,
       Cmpnts &ucont, 
       Cmpnts *ucat);
    void Contra2Cart();
    void Average(PetscInt ti);

    Vec getUcont() {return d_Ucont;}
    Vec getUcont_o() {return d_Ucont_o;}
    Vec getUcont_rm1() {return d_Ucont_rm1;}
    Vec getRhs() {return d_Rhs;}
    Vec getRhs_o() {return d_Rhs_o;}
    Vec getUcat() {return d_Ucat;}
    Vec getUcat_o() {return d_Ucat_o;}
    Vec getUbcs() {return d_Ubcs;} //Get rid of this
    
    Vec getNvert() {return d_Nvert;}
    Vec getDp() {return d_Dp;}
    Vec getP() {return d_P;}
    Vec getP_o() {return d_P_o;}

    Vec getlUcont() {return d_lUcont;}
    Vec getlUcont_o() {return d_lUcont_o;}
    Vec getlUcat() {return d_lUcat;}
    Vec getlUcat_old() {return d_lUcat_old;}  //We can probably get rid of this 

    Vec getlNvert() {return d_lNvert;}
    Vec getlNvert_o() {return d_lNvert_o;}
    Vec getlNvert_o_fixed() {return d_lNvert_o_fixed;}
    Vec getlP() {return d_lP;}
    Vec getlUstar() {return d_lUstar;}

    Vec getUcat_sum() { return d_Ucat_sum;}
    Vec getUcat_cross_sum() {return d_Ucat_cross_sum;}
    Vec getUcat_square_sum() {return d_Ucat_square_sum;}

    Mat getJacobian() {return d_J;}
    PetscReal getRe() {return d_ren;}
    PetscReal getDt() {return d_dt;}
    PetscReal getSt() {return d_St;}

    //This is the time coefficent for integration
    PetscReal getTimeCoeff() {return 1.0;}

    //Get the starting iterations
    PetscInt get_tistart() {return d_tistart;}
    PetscBool isRestart() {return d_restart;}

    //This is inflow information needed
    PetscReal getMeanFlux() {return d_mean_flux;}
    PetscReal getMeanArea() {return d_mean_area;}
    PetscReal getMeanFluxIb() {return d_mean_flux_ib;}
    PetscReal getMeanAreaIb() {return d_mean_area_ib;}
 
    void setMeanFlux(PetscReal flux) {d_mean_flux = flux;}
    void setMeanArea(PetscReal area) {d_mean_area = area;}
    void setMeanFluxIb(PetscReal flux) {d_mean_flux_ib = flux;}
    void setMeanAreaIb(PetscReal area) {d_mean_area_ib = area;}
    void PhaseNumber(PetscInt ti, PetscInt *phase, PetscInt *previous_ti);

    void WriteFile(char *filen, Vec U);
    void ReadFile(char *filen, Vec U);
private:


    PetscErrorCode ReadFromInput();


    std::string d_object_name;
    CurvGrid *d_grid;

    Mat d_J;

    Vec d_Ucont;
    Vec d_Ucont_o;
    Vec d_Ucont_rm1;
    Vec d_Rhs;
    Vec d_Rhs_o;
    Vec d_Ucat;
    Vec d_Ucat_o;
    Vec d_Ubcs; //Get rid of this
  
    Vec d_Nvert;
    Vec d_Dp;
    Vec d_P;
    Vec d_Po;

    Vec d_lUcont;
    Vec d_lUcont_o;
    Vec d_lUcont_rm1;
    Vec d_lUcat;
    Vec d_lUcat_old;  //We can probably get rid of this 

    Vec d_lNvert;
    Vec d_lUstar;
    Vec d_lP;

    Vec d_Nvert_o;
    Vec d_lNvert_o;
    Vec d_lNvert_o_fixed;
    Vec d_P_o;

    //Averaging Data
    Vec d_Ucat_sum;
    Vec d_Ucat_cross_sum;
    Vec d_Ucat_square_sum;
    Vec d_P_sum;
    Vec d_P_square_sum;
    Vec d_Udp_sum;
    Vec d_dU2_sum;
    Vec d_UUU_sum;
    Vec d_Vort_sum;
    Vec d_Vort_square_sum;

    //Phase Averaging Data
    Vec d_Ucat_sum_phase;
    Vec d_Ucat_cross_sum_phase;
    Vec d_Ucat_square_sum_phase;
    Vec d_P_sum_phase;
    Vec d_P_square_sum_phase;
    Vec d_Udp_sum_phase;
    Vec d_dU2_sum_phase;
    Vec d_UUU_sum_phase;
    Vec d_Vort_sum_phase;
    Vec d_Vort_square_sum_phase;


    PetscInt d_tiout, d_tiout_ufield, d_tiend_ufield;
    PetscInt d_ti_lastsave;
    PetscInt d_immersed, d_movefsi, d_rotatefsi;

    PetscInt d_tistart;
    PetscBool d_restart;
    PetscInt d_tisteps;
  
    PetscInt d_averaging;
    PetscInt d_phase_averaging; 
    PetscInt d_hdf5, d_write_hdf5, d_read_hdf5;
    char d_wext[6], d_rext[6];   
 
    PetscReal d_ren, d_St;
    PetscReal d_dt, d_dt_inflow;
 
    PetscBool d_rough_set;
    PetscReal d_roughness_size;
    PetscInt d_dp_wm;

    PetscReal d_mean_flux;
    PetscReal d_mean_area;
    PetscReal d_mean_flux_ib;
    PetscReal d_mean_area_ib;

    char d_path[256];
    char d_fieldpath[256];
    char d_avepath[256];
    char d_phpath[256];

    WallFunctions *d_wallf;
};

#endif
     
