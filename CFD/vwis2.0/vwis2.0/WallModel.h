#ifndef included_WallModel
#define included_WallModel

#include <stdlib.h>
#include <stdio.h>
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"
#include "LESModel.h"
#include "ImmersedBoundary.h"


using namespace std;

class WallModel
{
public:

    WallModel(
         const std::string& object_name,
         CurvGrid *grid,
         UData *data,
         LESModel *les,
         ImmersedBoundary *ib);

    ~WallModel();

    void Initialize();
    void CalculateVisc();
    void Solve(Vec Rhs, double coeff);
    PetscBool useWallModel() {return d_use_wall;}
    void setVisc(Vec Visc1, Vec Visc2, Vec Visc3) {
        d_lVisc1=Visc1; d_lVisc2=Visc2; d_lVisc3=Visc3;}
    void setFp(Vec Fp) { d_Fp = Fp; }

private:

    void Compute1_du_i(
        int i, int j, int k,
        int mx, int my, int mz,
        Cmpnts ***ucat, PetscReal ***nvert,
        double *dudc, double *dvdc, double *dwdc,
        double *dude, double *dvde, double *dwde,
        double *dudz, double *dvdz, double *dwdz);
    void Compute1_du_j(
        int i, int j, int k,
        int mx, int my, int mz,
        Cmpnts ***ucat, PetscReal ***nvert,
        double *dudc, double *dvdc, double *dwdc,
        double *dude, double *dvde, double *dwde,
        double *dudz, double *dvdz, double *dwdz);
    void Compute1_du_k(
        int i, int j, int k,
        int mx, int my, int mz,
        Cmpnts ***ucat, PetscReal ***nvert,
        double *dudc, double *dvdc, double *dwdc,
        double *dude, double *dvde, double *dwde,
        double *dudz, double *dvdz, double *dwdz);

    void Comput_du_wmlocal(
        double nx, double ny, double nz,
        double t1x, double t1y, double t1z,
        double t2x, double t2y, double t2z,
        double du_dx,double dv_dx,double dw_dx,
        double du_dy,double dv_dy,double dw_dy,
        double du_dz,double dv_dz,double dw_dz,
        double *dut1dn, double *dut2dn, double *dundn,
        double *dut1dt1, double *dut2dt1, double *dundt1,
        double *dut1dt2, double *dut2dt2, double *dundt2);

    void Comput_JacobTensor_i(
        int i, int j, int k,
        int mx, int my, int mz,
        Cmpnts ***coor,
        double *dxdc, double *dxde, double *dxdz,
        double *dydc, double *dyde, double *dydz,
        double *dzdc, double *dzde, double *dzdz);

    void Comput_JacobTensor_j(
        int i, int j, int k,
        int mx, int my, int mz,
        Cmpnts ***coor,
        double *dxdc, double *dxde, double *dxdz,
        double *dydc, double *dyde, double *dydz,
        double *dzdc, double *dzde, double *dzdz);

    void Comput_JacobTensor_k(
        int i, int j, int k,
        int mx, int my, int mz,
        Cmpnts ***coor,
        double *dxdc, double *dxde, double *dxdz,
        double *dydc, double *dyde, double *dydz,
        double *dzdc, double *dzde, double *dzdz);

    void Comput_du_Compgrid(
        double dxdc, double dxde, double dxdz, 
        double dydc, double dyde, double dydz, 
        double dzdc, double dzde, double dzdz, 
        double nx, double ny, double nz, 
        double t1x, double t1y, double t1z, 
        double t2x, double t2y, double t2z, 
        double dut1dn, double dut2dn, double dundn, 
        double dut1dt1, double dut2dt1, double dundt1, 
        double dut1dt2, double dut2dt2, double dundt2, 
        double *dudc, double *dvdc, double *dwdc, 
        double *dude, double *dvde, double *dwde, 
        double *dudz, double *dvdz, double *dwdz);

    void wallmodel_0424(
        double ks, PetscReal *ustar,
        double dpdx, double dpdy, double dpdz,
        double nu, double sb, double sc,
        Cmpnts *Ub, Cmpnts Uc, Cmpnts Ua,
        double nx, double ny, double nz,
        PetscReal alfa);

    void wallmodel_s(
        double nu, double sb, double sc,
        Cmpnts Uc, Cmpnts *Ub,  Cmpnts Ua,
        PetscInt bctype,
        double ks,
        double nx, double ny, double nz,
        double *tau_w, PetscReal *ustar,
        double dpdx, double dpdy, double dpdz,
        double *nut_2sb, double nut_c);


    double utau_powerlaw(double nu, double ut_mag, double sc);
    void innergrid( double *z_in, double h);

    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;
    LESModel *d_les;
    ImmersedBoundary *d_ib;

    PetscBool d_use_wall;
    PetscReal d_roughness_size;
    PetscReal d_alfa_wm;
    PetscReal d_les_eps;
    PetscInt d_powerlawwallmodel;
    PetscInt d_num_innergrid;
    PetscReal d_dhratio_wm;
    PetscReal d_dh1_wm;


    PetscInt d_imin_wm, d_imax_wm;
    PetscInt d_jmin_wm, d_jmax_wm;
    PetscInt d_kmin_wm, d_kmax_wm;
    PetscInt d_ib_wm, d_immersed;
    PetscInt d_infRe;

    Vec d_lVisc1_wm;
    Vec d_lVisc2_wm;
    Vec d_lVisc3_wm;
    Vec d_lTau;
    Vec d_lVisc1;
    Vec d_lVisc2;
    Vec d_lVisc3;
    Vec d_Fp;

 }; 
#endif
