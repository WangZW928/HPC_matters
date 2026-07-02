#ifndef included_ImmersedBoundary
#define included_ImmersedBoundary

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "petsctime.h"
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"
#include "WallFunctions.h"
#include "ibm_functions.h"

#define Dist(p1, p2) sqrt(( p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z))

#define Cross(Resu, v1, v2) \
        Resu.x = v1.y * v2.z - v1.z * v2.y; \
        Resu.y = v1.z * v2.x - v1.x * v2.z; \
        Resu.z = v1.x * v2.y - v1.y * v2.x;

#define VecAMinusB(C, A, B) \
        C.x = A.x - B.x; \
        C.y = A.y - B.y; \
        C.z = A.z - B.z;


class ImmersedBoundary
{
public:
     ImmersedBoundary(const std::string& object_name,
                      CurvGrid *grid,
                      UData *data);

     ~ImmersedBoundary();

     PetscErrorCode CopyLastStep();
     PetscErrorCode IBMSearchAdvanced(PetscInt ti);
     PetscErrorCode IBMSearchAdvanced1(IBMNodes *ibm,int ibi, PetscInt ti);
     PetscErrorCode IBMInterpolationAdvanced(PetscInt ti);
     PetscErrorCode IBMRead();
     PetscErrorCode IBMWrite(PetscInt ti);
     PetscErrorCode ReadUCD(IBMNodes *ibm, PetscInt ibi);
     PetscErrorCode WriteOutput1(IBMNodes *ibm, PetscInt ibi, PetscInt ti);

     IBMNodes *getIBMNodes() {return d_ibm;}
     IBMList *getIBMList() {return d_ibmlist;}
     PetscInt getNumberOfIBMBodies() {return d_NumberOfBodies;}

private:
  
     PetscErrorCode 
     BoundingSphere(
         IBMNodes *ibm);

     PetscErrorCode 
     NearestCell(
         Cmpnts p, 
         IBMNodes *ibm, 
         IBMInfo *ibminfo);

     PetscErrorCode 
     InterceptionPoint(
         Cmpnts p, 
         PetscInt i, PetscInt j, PetscInt k,
         IBMInfo *ibminfo);

     PetscInt 
     PointCellThin(
         Cmpnts p,Cmpnts p1,Cmpnts p2,
         Cmpnts p3, Cmpnts p4,
         PetscInt ip, PetscInt jp, PetscInt kp,
         IBMNodes *ibm, 
         PetscInt ncx, PetscInt ncy, PetscInt ncz, 
         PetscReal dcx, PetscReal dcy,
         PetscReal xbp_min, PetscReal ybp_min, PetscReal zbp_max, 
         LIST *cell_trg,
         PetscInt flg);

    PetscInt 
    PointCellAdvanced(
         Cmpnts p, 
         PetscInt ip, PetscInt jp, PetscInt kp,
         IBMNodes *ibm, 
         PetscInt ncx, PetscInt ncy, PetscInt ncz, 
         PetscReal dcx, PetscReal dcy,
         PetscReal xbp_min, PetscReal ybp_min, PetscReal zbp_max, 
         LIST *cell_trg,
         PetscInt flg);

    PetscErrorCode 
    ICP(
       Cmpnts p, Cmpnts pc[9],
       PetscReal nfx, PetscReal nfy, PetscReal nfz,
       IBMInfo *ibminfo,
       PetscInt *ip, PetscInt *jp, PetscInt *kp);

    double 
    ContravariantReynoldsStress(
        double uu, double uv, double uw,
        double vv, double vw, double ww,
        double csi0, double csi1, double csi2,
        double eta0, double eta1, double eta2);  

    void str_to_buffer(char *str, std::vector<char> &large_buffer); 
    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;

    char d_path[256];

    PetscInt d_IB_wm;
    PetscInt d_movefsi;
    PetscInt d_rotatefsi;
    PetscInt d_rotatefsi_noIBsearch;
    PetscInt d_changefsi;
    PetscInt d_thin;
    PetscInt d_immersed;

    PetscInt d_NumberOfBodies;
    PetscInt d_NumberOfRotatingBodies;

    PetscReal d_cl;
    PetscReal d_CMx_c;
    PetscReal d_CMy_c;
    PetscReal d_CMz_c;

    PetscInt d_tiout;
    PetscInt d_averaging;
    PetscInt d_wallfunction;
    PetscReal d_roughness_size;

    IBMNodes *d_ibm;
    IBMList *d_ibmlist;
};

#endif 
