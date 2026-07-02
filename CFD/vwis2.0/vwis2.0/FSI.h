#ifndef included_FSI
#define included_FSI

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

class FSI
{
public:
    FSI(const std::string& object_name,
        CurvGrid *grid,
        UData *data,
        ImmersedBoundary *ib);

    ~FSI();

    PetscErrorCode Initialize();

    PetscErrorCode Restart(PetscInt ti);
    PetscErrorCode ReadFSI(PetscInt ti);
    PetscErrorCode WriteFSI(PetscInt ti);
    PetscErrorCode CopyLastStep();
    PetscErrorCode CopyToOld(PetscInt si);

    PetscErrorCode CalculatePosition(PetscInt ti,
                                     PetscReal time);

    PetscErrorCode CalculateRotation(PetscInt ti,
                                     PetscReal time);
    PetscErrorCode CalculateForces(PetscInt ti, PetscReal time);
  
    FSInfo *getFSInfo() {return d_fsi;} 
    PetscReal getS_ang_n(PetscInt ibi) {return d_fsi[ibi].S_ang_n[1];} 
    PetscReal getS_ang_o(PetscInt ibi) {return d_fsi[ibi].S_ang_o[1];} 
private:

    PetscErrorCode ReadFsiInput(FSInfo *FSinf, PetscInt ibi, PetscInt ti);
    PetscErrorCode WriteFSIOutput(FSInfo *FSinfo, PetscInt ibi, PetscInt ti);
    PetscErrorCode CalculateFSIPosition(FSInfo *FSinfo,
                                        PetscReal time);
    PetscErrorCode ElementMoveFSITranslation(FSInfo *FSinfo, IBMNodes *ibm);
    PetscErrorCode CollisionDetectionOfCylinders();
    PetscErrorCode CalculateFSIRotation(FSInfo *FSinfo);
    PetscErrorCode ElementMoveFSIRotation(FSInfo *FSinfo,
                                          IBMNodes *ibm,
                                          PetscInt ti);
    PetscErrorCode CalculateForces1(FSInfo *fsi, PetscInt ibi,
                                    PetscInt ti, PetscReal time);


    PetscErrorCode RotateXYZ(double ti, double dt, double angvel,
                             double x_c, double y_c, double z_c,
                             double x_bp0, double y_bp0, double z_bp0,
                             double *x_bp, double *y_bp, double *z_bp,
                             double *rot_angle);

    void ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;
    ImmersedBoundary *d_ib;

    IBMNodes *d_ibm;

    PetscReal d_red_vel;
    PetscReal d_damp;
    PetscReal d_mu_s;
    PetscReal d_x_c;
    PetscReal d_y_c;
    PetscReal d_z_c;
    PetscReal d_x_r;
    PetscReal d_y_r;
    PetscReal d_z_r;
    PetscReal d_Mx_applied;
    PetscReal d_My_applied;
    PetscReal d_Mz_applied;
    PetscReal d_Max_xbc, d_Max_ybc, d_Max_zbc;
    PetscReal d_Min_xbc, d_Min_ybc, d_Min_zbc;
    
    PetscInt d_NumberOfBodies;
    PetscInt d_NumberOfRotatingBodies;
   
    PetscInt d_sisteps; 
    PetscInt d_immersed;
    PetscInt d_movefsi;
    PetscInt d_rotatefsi;
    PetscInt d_rotatefsi_noIBsearch;
    PetscInt d_changefsi;
    PetscInt d_rstart_fsi;
    PetscInt d_dgf_z;
    PetscInt d_dgf_y;
    PetscInt d_dgf_x;
    PetscInt d_rotdir;
    PetscInt d_prescribed_rotation;
    
    PetscReal d_angvel;
 
    PetscInt d_tiout;
    PetscInt d_ti_lastsave;
    char d_path[256];
    char d_fsipath[256];

    FSInfo *d_fsi;
    
};



#endif
