
#include "functions.h"

 void AxByC ( double a, Cmpnts &X, double b, Cmpnts &Y, Cmpnts *C)
{
    (*C).x = a*X.x + b*Y.x;
    (*C).y = a*X.y + b*Y.y;
    (*C).z = a*X.z + b*Y.z;
}

 void Subtract_Scale_AddTo ( Cmpnts &A, Cmpnts &B, double a, Cmpnts *C)
{
    (*C).x += (A.x - B.x) * a;
    (*C).y += (A.y - B.y) * a;
    (*C).z += (A.z - B.z) * a;
}
 void Subtract_Scale_Set ( Cmpnts &A, Cmpnts &B, double a, Cmpnts *C)
{
    (*C).x = (A.x - B.x) * a;
    (*C).y = (A.y - B.y) * a;
    (*C).z = (A.z - B.z) * a;
}

 void GlobalSum_Root(PetscReal* local,PetscReal* result,MPI_Comm comm)
{
    MPI_Reduce(local,result,1,MPIU_REAL,MPI_SUM,0,comm);
}

 void GlobalMax_Root(PetscReal* local,PetscReal* result,MPI_Comm comm)
{
    MPI_Reduce(local,result,1,MPIU_REAL,MPI_MAX,0,comm);
}

 void GlobalSum_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm)
{
    MPI_Allreduce(local,result,1,MPIU_SCALAR,MPIU_SUM,comm);
}

 void GlobalMax_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm)
{
    MPI_Allreduce(local,result,1,MPIU_REAL,MPI_MAX,comm);
}

 void GlobalMin_All(PetscScalar* local,PetscScalar* result,MPI_Comm comm)
{
    MPI_Allreduce(local,result,1,MPIU_REAL,MPI_MIN,comm);
}

 void Set( Cmpnts *A, double a )
{
    (*A).x = a;
    (*A).y = a;
    (*A).z = a;
}

 void AxC(double a, Cmpnts &X, Cmpnts *C)
{
    (*C).x = a*X.x;
    (*C).y = a*X.y;
    (*C).z = a*X.z;
}


 void Calculate_Covariant_metrics(double g[3][3], double G[3][3])
{
   /*
     | csi.x  csi.y csi.z |-1        | x.csi  x.eta x.zet | 
     | eta.x eta.y eta.z |    =      | y.csi   y.eta  y.zet |
     | zet.x zet.y zet.z |           | z.csi  z.eta z.zet |
        
        */
    const double a11=g[0][0], a12=g[0][1], a13=g[0][2];
    const double a21=g[1][0], a22=g[1][1], a23=g[1][2];
    const double a31=g[2][0], a32=g[2][1], a33=g[2][2];

    double det= a11*(a33*a22-a32*a23)-
                a21*(a33*a12-a32*a13)+
                a31*(a23*a12-a22*a13);

    G[0][0] = (a33*a22-a32*a23)/det;
    G[0][1] =-(a33*a12-a32*a13)/det;
    G[0][2] = (a23*a12-a22*a13)/det;
    G[1][0] =-(a33*a21-a31*a23)/det;
    G[1][1] = (a33*a11-a31*a13)/det;
    G[1][2] =-(a23*a11-a21*a13)/det;
    G[2][0] = (a32*a21-a31*a22)/det;
    G[2][1] =-(a32*a11-a31*a12)/det;
    G[2][2] = (a22*a11-a21*a12)/det;
}


 void Calculate_normal(Cmpnts csi, Cmpnts eta, Cmpnts zet, 
                             double ni[3], double nj[3], double nk[3])
{
    double g[3][3];
    double G[3][3];

    g[0][0]=csi.x, g[0][1]=csi.y, g[0][2]=csi.z;
    g[1][0]=eta.x, g[1][1]=eta.y, g[1][2]=eta.z;
    g[2][0]=zet.x, g[2][1]=zet.y, g[2][2]=zet.z;

    Calculate_Covariant_metrics(g, G);
    double xcsi=G[0][0], ycsi=G[1][0], zcsi=G[2][0];
    double xeta=G[0][1], yeta=G[1][1], zeta=G[2][1];
    double xzet=G[0][2], yzet=G[1][2], zzet=G[2][2];

    double nx_i = xcsi, ny_i = ycsi, nz_i = zcsi;
    double nx_j = xeta, ny_j = yeta, nz_j = zeta;
    double nx_k = xzet, ny_k = yzet, nz_k = zzet;

    double sum_i=sqrt(nx_i*nx_i+ny_i*ny_i+nz_i*nz_i);
    double sum_j=sqrt(nx_j*nx_j+ny_j*ny_j+nz_j*nz_j);
    double sum_k=sqrt(nx_k*nx_k+ny_k*ny_k+nz_k*nz_k);

    nx_i /= sum_i, ny_i /= sum_i, nz_i /= sum_i;
    nx_j /= sum_j, ny_j /= sum_j, nz_j /= sum_j;
    nx_k /= sum_k, ny_k /= sum_k, nz_k /= sum_k;

    ni[0] = nx_i, ni[1] = ny_i, ni[2] = nz_i;
    nj[0] = nx_j, nj[1] = ny_j, nj[2] = nz_j;
    nk[0] = nx_k, nk[1] = ny_k, nk[2] = nz_k;
}


 void Compute_dscalar_center(int i, int j, int k,  
                                   int mx, int my, int mz, 
                                   PetscReal ***K, PetscReal ***nvert, 
                                   double *dkdc, double *dkde, double *dkdz)
{

    if (i==mx-1) *dkdc = ( K[k][j][i] - K[k][j][i-1] );
    else if (i==0) *dkdc = ( K[k][j][i+1] - K[k][j][i] );
    else if ((nvert[k][j][i+1])> 0.1)    
        *dkdc = ( K[k][j][i] - K[k][j][i-1] );
    else if ((nvert[k][j][i-1])> 0.1) 
        *dkdc = ( K[k][j][i+1] - K[k][j][i] );
    else 
        *dkdc = ( K[k][j][i+1] - K[k][j][i-1] ) * 0.5;

    if (j==my-1) *dkde = ( K[k][j][i] - K[k][j-1][i] );
    else if (j==0) *dkde = ( K[k][j+1][i] - K[k][j][i] );
    else if ((nvert[k][j+1][i])> 0.1 ) 
        *dkde = ( K[k][j][i] - K[k][j-1][i] );
    else if ((nvert[k][j-1][i])> 0.1 ) 
        *dkde = ( K[k][j+1][i] - K[k][j][i] );
    else 
        *dkde = ( K[k][j+1][i] - K[k][j-1][i] ) * 0.5;

    if (k==mz-1) *dkdz = ( K[k][j][i] - K[k-1][j][i] );
    else if (k==0) *dkdz = ( K[k+1][j][i] - K[k][j][i] );
    else if ((nvert[k+1][j][i])> 0.1 ) 
        *dkdz = ( K[k][j][i] - K[k-1][j][i] );
    else if ((nvert[k-1][j][i])> 0.1 ) 
        *dkdz = ( K[k+1][j][i] - K[k][j][i] );
    else 
        *dkdz = ( K[k+1][j][i] - K[k-1][j][i] ) * 0.5;
}

 void Compute_dscalar_dxyz(double csi0, double csi1, double csi2, 
                                 double eta0, double eta1, double eta2, 
                                 double zet0, double zet1, double zet2, 
                                 double ajc,
                                 double dkdc, double dkde, double dkdz, 
                                 double *dk_dx, double *dk_dy, double *dk_dz)
{
    *dk_dx = (dkdc * csi0 + dkde * eta0 + dkdz * zet0) * ajc;
    *dk_dy = (dkdc * csi1 + dkde * eta1 + dkdz * zet1) * ajc;
    *dk_dz = (dkdc * csi2 + dkde * eta2 + dkdz * zet2) * ajc;
}

 void Compute_du_i(int i, int j, int k, 
                         int mx, int my, int mz,
                         Cmpnts ***ucat, 
                         PetscReal ***nvert, 
                         double *dudc, double *dvdc, double *dwdc, 
                         double *dude, double *dvde, double *dwde,
                         double *dudz, double *dvdz, double *dwdz)
{

    *dudc = ucat[k][j][i+1].x - ucat[k][j][i].x;
    *dvdc = ucat[k][j][i+1].y - ucat[k][j][i].y;
    *dwdc = ucat[k][j][i+1].z - ucat[k][j][i].z;

    if ((nvert[k][j+1][i])> 0.1 || (nvert[k][j+1][i+1])> 0.1) {
        *dude = (ucat[k][j  ][i+1].x + ucat[k][j  ][i].x - 
                 ucat[k][j-1][i+1].x - ucat[k][j-1][i].x) * 0.5;
        *dvde = (ucat[k][j  ][i+1].y + ucat[k][j  ][i].y - 
                 ucat[k][j-1][i+1].y - ucat[k][j-1][i].y) * 0.5;
        *dwde = (ucat[k][j  ][i+1].z + ucat[k][j  ][i].z - 
                 ucat[k][j-1][i+1].z - ucat[k][j-1][i].z) * 0.5;
    }
    else if  ((nvert[k][j-1][i])> 0.1 || (nvert[k][j-1][i+1])> 0.1) {
        *dude = (ucat[k][j+1][i+1].x + ucat[k][j+1][i].x - 
                 ucat[k][j  ][i+1].x - ucat[k][j  ][i].x) * 0.5;
        *dvde = (ucat[k][j+1][i+1].y + ucat[k][j+1][i].y - 
                 ucat[k][j  ][i+1].y - ucat[k][j  ][i].y) * 0.5;
        *dwde = (ucat[k][j+1][i+1].z + ucat[k][j+1][i].z - 
                 ucat[k][j  ][i+1].z - ucat[k][j  ][i].z) * 0.5;
    }else {

        *dude = (ucat[k][j+1][i+1].x + ucat[k][j+1][i].x - 
                 ucat[k][j-1][i+1].x - ucat[k][j-1][i].x) * 0.25;
        *dvde = (ucat[k][j+1][i+1].y + ucat[k][j+1][i].y - 
                 ucat[k][j-1][i+1].y - ucat[k][j-1][i].y) * 0.25;
        *dwde = (ucat[k][j+1][i+1].z + ucat[k][j+1][i].z - 
                 ucat[k][j-1][i+1].z - ucat[k][j-1][i].z) * 0.25;
    }

    if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j][i+1])> 0.1) {
        *dudz = (ucat[k  ][j][i+1].x + ucat[k  ][j][i].x - 
                 ucat[k-1][j][i+1].x - ucat[k-1][j][i].x) * 0.5;
        *dvdz = (ucat[k  ][j][i+1].y + ucat[k  ][j][i].y - 
                 ucat[k-1][j][i+1].y - ucat[k-1][j][i].y) * 0.5;
        *dwdz = (ucat[k  ][j][i+1].z + ucat[k  ][j][i].z - 
                 ucat[k-1][j][i+1].z - ucat[k-1][j][i].z) * 0.5;
    }
    else if ((nvert[k-1][j][i])> 0.1 || (nvert[k-1][j][i+1])> 0.1) {
        *dudz = (ucat[k+1][j][i+1].x + ucat[k+1][j][i].x - 
                 ucat[k  ][j][i+1].x - ucat[k  ][j][i].x) * 0.5;
        *dvdz = (ucat[k+1][j][i+1].y + ucat[k+1][j][i].y - 
                 ucat[k  ][j][i+1].y - ucat[k  ][j][i].y) * 0.5;
        *dwdz = (ucat[k+1][j][i+1].z + ucat[k+1][j][i].z - 
                 ucat[k  ][j][i+1].z - ucat[k  ][j][i].z) * 0.5;
    } else {
        *dudz = (ucat[k+1][j][i+1].x + ucat[k+1][j][i].x - 
                 ucat[k-1][j][i+1].x - ucat[k-1][j][i].x) * 0.25;
        *dvdz = (ucat[k+1][j][i+1].y + ucat[k+1][j][i].y - 
                 ucat[k-1][j][i+1].y - ucat[k-1][j][i].y) * 0.25;
        *dwdz = (ucat[k+1][j][i+1].z + ucat[k+1][j][i].z - 
                 ucat[k-1][j][i+1].z - ucat[k-1][j][i].z) * 0.25;
    }
}



 void Compute_du_j(int i, int j, int k, 
                         int mx, int my, int mz, 
                         Cmpnts ***ucat, PetscReal ***nvert, 
                         double *dudc, double *dvdc, double *dwdc, 
                         double *dude, double *dvde, double *dwde,
                         double *dudz, double *dvdz, double *dwdz)
{
    if ((nvert[k][j][i+1])> 0.1 || (nvert[k][j+1][i+1])> 0.1) {
        *dudc = (ucat[k][j+1][i  ].x + ucat[k][j][i  ].x - 
                 ucat[k][j+1][i-1].x - ucat[k][j][i-1].x) * 0.5;
        *dvdc = (ucat[k][j+1][i  ].y + ucat[k][j][i  ].y - 
                 ucat[k][j+1][i-1].y - ucat[k][j][i-1].y) * 0.5;
        *dwdc = (ucat[k][j+1][i  ].z + ucat[k][j][i  ].z - 
                 ucat[k][j+1][i-1].z - ucat[k][j][i-1].z) * 0.5;
    }
    else if ((nvert[k][j][i-1])> 0.1 || (nvert[k][j+1][i-1])> 0.1) {
        *dudc = (ucat[k][j+1][i+1].x + ucat[k][j][i+1].x - 
                 ucat[k][j+1][i  ].x - ucat[k][j][i  ].x) * 0.5;
        *dvdc = (ucat[k][j+1][i+1].y + ucat[k][j][i+1].y - 
                 ucat[k][j+1][i  ].y - ucat[k][j][i  ].y) * 0.5;
        *dwdc = (ucat[k][j+1][i+1].z + ucat[k][j][i+1].z - 
                 ucat[k][j+1][i  ].z - ucat[k][j][i  ].z) * 0.5;
    } else {
        *dudc = (ucat[k][j+1][i+1].x + ucat[k][j][i+1].x - 
                 ucat[k][j+1][i-1].x - ucat[k][j][i-1].x) * 0.25;
        *dvdc = (ucat[k][j+1][i+1].y + ucat[k][j][i+1].y - 
                 ucat[k][j+1][i-1].y - ucat[k][j][i-1].y) * 0.25;
        *dwdc = (ucat[k][j+1][i+1].z + ucat[k][j][i+1].z - 
                 ucat[k][j+1][i-1].z - ucat[k][j][i-1].z) * 0.25;
    }

    *dude = ucat[k][j+1][i].x - ucat[k][j][i].x;
    *dvde = ucat[k][j+1][i].y - ucat[k][j][i].y;
    *dwde = ucat[k][j+1][i].z - ucat[k][j][i].z;
   
    if ((nvert[k+1][j][i])> 0.1 || (nvert[k+1][j+1][i])> 0.1) {
        *dudz = (ucat[k  ][j+1][i].x + ucat[k  ][j][i].x - 
                 ucat[k-1][j+1][i].x - ucat[k-1][j][i].x) * 0.5;
        *dvdz = (ucat[k  ][j+1][i].y + ucat[k  ][j][i].y - 
                 ucat[k-1][j+1][i].y - ucat[k-1][j][i].y) * 0.5;
        *dwdz = (ucat[k  ][j+1][i].z + ucat[k  ][j][i].z - 
                 ucat[k-1][j+1][i].z - ucat[k-1][j][i].z) * 0.5;
    }
    else if ((nvert[k-1][j][i])> 0.1 || (nvert[k-1][j+1][i])> 0.1) {
        *dudz = (ucat[k+1][j+1][i].x + ucat[k+1][j][i].x - 
                 ucat[k  ][j+1][i].x - ucat[k  ][j][i].x) * 0.5;
       *dvdz = (ucat[k+1][j+1][i].y + ucat[k+1][j][i].y - 
                ucat[k  ][j+1][i].y - ucat[k  ][j][i].y) * 0.5;
       *dwdz = (ucat[k+1][j+1][i].z + ucat[k+1][j][i].z - 
                ucat[k  ][j+1][i].z - ucat[k  ][j][i].z) * 0.5;
    } else {
       *dudz = (ucat[k+1][j+1][i].x + ucat[k+1][j][i].x - 
                ucat[k-1][j+1][i].x - ucat[k-1][j][i].x) * 0.25;
       *dvdz = (ucat[k+1][j+1][i].y + ucat[k+1][j][i].y - 
                ucat[k-1][j+1][i].y - ucat[k-1][j][i].y) * 0.25;
       *dwdz = (ucat[k+1][j+1][i].z + ucat[k+1][j][i].z - 
                ucat[k-1][j+1][i].z - ucat[k-1][j][i].z) * 0.25;
    }
}



 void Compute_du_k(int i, int j, int k, 
                         int mx, int my, int mz,  
                         Cmpnts ***ucat, PetscReal ***nvert, 
                         double *dudc, double *dvdc, double *dwdc, 
                         double *dude, double *dvde, double *dwde,
                         double *dudz, double *dvdz, double *dwdz)
{
    if ((nvert[k][j][i+1])> 0.1 || (nvert[k+1][j][i+1])> 0.1) {
        *dudc = (ucat[k+1][j][i  ].x + ucat[k][j][i  ].x - 
                 ucat[k+1][j][i-1].x - ucat[k][j][i-1].x) * 0.5;
        *dvdc = (ucat[k+1][j][i  ].y + ucat[k][j][i  ].y - 
                 ucat[k+1][j][i-1].y - ucat[k][j][i-1].y) * 0.5;
        *dwdc = (ucat[k+1][j][i  ].z + ucat[k][j][i  ].z - 
                 ucat[k+1][j][i-1].z - ucat[k][j][i-1].z) * 0.5;
    }
    else if ((nvert[k][j][i-1])> 0.1 || (nvert[k+1][j][i-1])> 0.1) {
        *dudc = (ucat[k+1][j][i+1].x + ucat[k][j][i+1].x - 
                 ucat[k+1][j][i  ].x - ucat[k][j][i  ].x) * 0.5;
        *dvdc = (ucat[k+1][j][i+1].y + ucat[k][j][i+1].y - 
                 ucat[k+1][j][i  ].y - ucat[k][j][i  ].y) * 0.5;
        *dwdc = (ucat[k+1][j][i+1].z + ucat[k][j][i+1].z - 
                 ucat[k+1][j][i  ].z - ucat[k][j][i  ].z) * 0.5;
    }
    else {
        *dudc = (ucat[k+1][j][i+1].x + ucat[k][j][i+1].x - 
                 ucat[k+1][j][i-1].x - ucat[k][j][i-1].x) * 0.25;
        *dvdc = (ucat[k+1][j][i+1].y + ucat[k][j][i+1].y - 
                 ucat[k+1][j][i-1].y - ucat[k][j][i-1].y) * 0.25;
        *dwdc = (ucat[k+1][j][i+1].z + ucat[k][j][i+1].z - 
                 ucat[k+1][j][i-1].z - ucat[k][j][i-1].z) * 0.25;
    }

    if ((nvert[k][j+1][i])> 0.1 || (nvert[k+1][j+1][i])> 0.1) {
        *dude = (ucat[k+1][j  ][i].x + ucat[k][j  ][i].x - 
                 ucat[k+1][j-1][i].x - ucat[k][j-1][i].x) * 0.5;
        *dvde = (ucat[k+1][j  ][i].y + ucat[k][j  ][i].y - 
                 ucat[k+1][j-1][i].y - ucat[k][j-1][i].y) * 0.5;
        *dwde = (ucat[k+1][j  ][i].z + ucat[k][j  ][i].z - 
                 ucat[k+1][j-1][i].z - ucat[k][j-1][i].z) * 0.5;
    }
    else if ((nvert[k][j-1][i])> 0.1 || (nvert[k+1][j-1][i])> 0.1) {
        *dude = (ucat[k+1][j+1][i].x + ucat[k][j+1][i].x - 
                 ucat[k+1][j  ][i].x - ucat[k][j  ][i].x) * 0.5;
        *dvde = (ucat[k+1][j+1][i].y + ucat[k][j+1][i].y - 
                 ucat[k+1][j  ][i].y - ucat[k][j  ][i].y) * 0.5;
        *dwde = (ucat[k+1][j+1][i].z + ucat[k][j+1][i].z - 
                 ucat[k+1][j  ][i].z - ucat[k][j  ][i].z) * 0.5;
    }
    else {
        *dude = (ucat[k+1][j+1][i].x + ucat[k][j+1][i].x - 
                 ucat[k+1][j-1][i].x - ucat[k][j-1][i].x) * 0.25;
        *dvde = (ucat[k+1][j+1][i].y + ucat[k][j+1][i].y - 
                 ucat[k+1][j-1][i].y - ucat[k][j-1][i].y) * 0.25;
        *dwde = (ucat[k+1][j+1][i].z + ucat[k][j+1][i].z - 
                 ucat[k+1][j-1][i].z - ucat[k][j-1][i].z) * 0.25;
    }
 
    *dudz = ucat[k+1][j][i].x - ucat[k][j][i].x;
    *dvdz = ucat[k+1][j][i].y - ucat[k][j][i].y;
    *dwdz = ucat[k+1][j][i].z - ucat[k][j][i].z;
}

void Compute_du_dxyz(double csi0, double csi1, double csi2, 
                     double eta0, double eta1, double eta2, 
                     double zet0, double zet1, double zet2, double ajc,
                     double dudc, double dvdc, double dwdc, 
                     double dude, double dvde, double dwde, 
                     double dudz, double dvdz, double dwdz,
                     double *du_dx, double *dv_dx, double *dw_dx, 
                     double *du_dy, double *dv_dy, double *dw_dy, 
                     double *du_dz, double *dv_dz, double *dw_dz )
{
    *du_dx = (dudc * csi0 + dude * eta0 + dudz * zet0) * ajc;
    *du_dy = (dudc * csi1 + dude * eta1 + dudz * zet1) * ajc;
    *du_dz = (dudc * csi2 + dude * eta2 + dudz * zet2) * ajc;
    *dv_dx = (dvdc * csi0 + dvde * eta0 + dvdz * zet0) * ajc;
    *dv_dy = (dvdc * csi1 + dvde * eta1 + dvdz * zet1) * ajc;
    *dv_dz = (dvdc * csi2 + dvde * eta2 + dvdz * zet2) * ajc;
    *dw_dx = (dwdc * csi0 + dwde * eta0 + dwdz * zet0) * ajc;
    *dw_dy = (dwdc * csi1 + dwde * eta1 + dwdz * zet1) * ajc;    
    *dw_dz = (dwdc * csi2 + dwde * eta2 + dwdz * zet2) * ajc;
}

void Compute_du_center(int i, int j, int k,  
                       int mx, int my, int mz, 
                       Cmpnts ***ucat, PetscReal ***nvert,
                       int i_p, int ii_p, int j_p, int jj_p, int k_p, int kk_p,  
                       double *dudc, double *dvdc, double *dwdc, 
                       double *dude, double *dvde, double *dwde,
                       double *dudz, double *dvdz, double *dwdz)
{
    if ((nvert[k][j][i+1])> 0.1 || (!i_p &&  !ii_p && i==mx-2) ) {
        *dudc = ( ucat[k][j][i].x - ucat[k][j][i-1].x );
        *dvdc = ( ucat[k][j][i].y - ucat[k][j][i-1].y );
        *dwdc = ( ucat[k][j][i].z - ucat[k][j][i-1].z );
    } else if ((nvert[k][j][i-1])> 0.1 || (!i_p &&  !ii_p && i==1) ) {
        *dudc = ( ucat[k][j][i+1].x - ucat[k][j][i].x );
        *dvdc = ( ucat[k][j][i+1].y - ucat[k][j][i].y );
        *dwdc = ( ucat[k][j][i+1].z - ucat[k][j][i].z );
    } else {
        *dudc = ( ucat[k][j][i+1].x - ucat[k][j][i-1].x ) * 0.5;
        *dvdc = ( ucat[k][j][i+1].y - ucat[k][j][i-1].y ) * 0.5;
        *dwdc = ( ucat[k][j][i+1].z - ucat[k][j][i-1].z ) * 0.5;
    }

    if ((nvert[k][j+1][i])> 0.1 || (!j_p &&  !jj_p && j==my-2) ) {
        *dude = ( ucat[k][j][i].x - ucat[k][j-1][i].x );
        *dvde = ( ucat[k][j][i].y - ucat[k][j-1][i].y );
        *dwde = ( ucat[k][j][i].z - ucat[k][j-1][i].z );
    } else if ((nvert[k][j-1][i])> 0.1 || (!j_p &&  !jj_p && j==1) ) {
        *dude = ( ucat[k][j+1][i].x - ucat[k][j][i].x );
        *dvde = ( ucat[k][j+1][i].y - ucat[k][j][i].y );
        *dwde = ( ucat[k][j+1][i].z - ucat[k][j][i].z );
    } else {
        *dude = ( ucat[k][j+1][i].x - ucat[k][j-1][i].x ) * 0.5;
        *dvde = ( ucat[k][j+1][i].y - ucat[k][j-1][i].y ) * 0.5;
        *dwde = ( ucat[k][j+1][i].z - ucat[k][j-1][i].z ) * 0.5;
    }

    if ((nvert[k+1][j][i])> 0.1 || ( !k_p &&  !kk_p && k==mz-2) ) {
        *dudz = ( ucat[k][j][i].x - ucat[k-1][j][i].x );
        *dvdz = ( ucat[k][j][i].y - ucat[k-1][j][i].y );
        *dwdz = ( ucat[k][j][i].z - ucat[k-1][j][i].z );
    } else if ((nvert[k-1][j][i])> 0.1 || (!k_p &&  !kk_p && k==1) ) {
        *dudz = ( ucat[k+1][j][i].x - ucat[k][j][i].x );
        *dvdz = ( ucat[k+1][j][i].y - ucat[k][j][i].y );
        *dwdz = ( ucat[k+1][j][i].z - ucat[k][j][i].z );
    } else {
        *dudz = ( ucat[k+1][j][i].x - ucat[k-1][j][i].x ) * 0.5;
        *dvdz = ( ucat[k+1][j][i].y - ucat[k-1][j][i].y ) * 0.5;
        *dwdz = ( ucat[k+1][j][i].z - ucat[k-1][j][i].z ) * 0.5;
    }
}
