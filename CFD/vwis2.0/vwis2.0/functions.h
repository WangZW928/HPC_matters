#ifndef included_functions
#define included_functions

#include "petscvec.h"

/*
 This is the struct for everything
*/
typedef struct {
    PetscScalar x, y, z;
} Cmpnts;

typedef struct {
    PetscInt i, j, k;
} Index;


void AxByC ( double a, Cmpnts &X, double b, Cmpnts &Y, Cmpnts *C);
void Subtract_Scale_AddTo ( Cmpnts &A, Cmpnts &B, double a, Cmpnts *C);
void Subtract_Scale_Set ( Cmpnts &A, Cmpnts &B, double a, Cmpnts *C);
void GlobalSum_Root(PetscReal* local,PetscReal* result,MPI_Comm comm);
void GlobalMax_Root(PetscReal* local,PetscReal* result,MPI_Comm comm);
void GlobalSum_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm);
void GlobalMax_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm);
void GlobalMin_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm);
void Set( Cmpnts *A, double a );
void AxC(double a, Cmpnts &X, Cmpnts *C);
void Calculate_Covariant_metrics(double g[3][3], double G[3][3]);
void Calculate_normal(Cmpnts csi, Cmpnts eta, Cmpnts zet, 
                      double ni[3], double nj[3], double nk[3]);
void Compute_dscalar_center(int i, int j, int k,  
                            int mx, int my, int mz, 
                            PetscReal ***K, PetscReal ***nvert, 
                            double *dkdc, double *dkde, double *dkdz);
void Compute_dscalar_dxyz(double csi0, double csi1, double csi2, 
                          double eta0, double eta1, double eta2, 
                          double zet0, double zet1, double zet2, 
                          double ajc,
                          double dkdc, double dkde, double dkdz, 
                          double *dk_dx, double *dk_dy, double *dk_dz);
void Compute_du_i(int i, int j, int k, 
                  int mx, int my, int mz,
                  Cmpnts ***ucat, 
                  PetscReal ***nvert, 
                  double *dudc, double *dvdc, double *dwdc, 
                  double *dude, double *dvde, double *dwde,
                  double *dudz, double *dvdz, double *dwdz);
void Compute_du_j(int i, int j, int k, 
                  int mx, int my, int mz, 
                  Cmpnts ***ucat, PetscReal ***nvert, 
                  double *dudc, double *dvdc, double *dwdc, 
                  double *dude, double *dvde, double *dwde,
                  double *dudz, double *dvdz, double *dwdz);
void Compute_du_k(int i, int j, int k, 
                  int mx, int my, int mz,  
                  Cmpnts ***ucat, PetscReal ***nvert, 
                  double *dudc, double *dvdc, double *dwdc, 
                  double *dude, double *dvde, double *dwde,
                  double *dudz, double *dvdz, double *dwdz);
void Compute_du_dxyz(double csi0, double csi1, double csi2,
                     double eta0, double eta1, double eta2,
                     double zet0, double zet1, double zet2, double ajc,
                     double dudc, double dvdc, double dwdc,
                     double dude, double dvde, double dwde,
                     double dudz, double dvdz, double dwdz,
                     double *du_dx, double *dv_dx, double *dw_dx,
                     double *du_dy, double *dv_dy, double *dw_dy,
                     double *du_dz, double *dv_dz, double *dw_dz );
void Compute_du_center(int i, int j, int k,
                       int mx, int my, int mz,
                       Cmpnts ***ucat, PetscReal ***nvert,
                       int i_p, int ii_p, int j_p, int jj_p, int k_p, int kk_p,
                       double *dudc, double *dvdc, double *dwdc,
                       double *dude, double *dvde, double *dwde,
                       double *dudz, double *dvdz, double *dwdz);
#endif
