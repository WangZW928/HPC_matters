#include "PointProbe.h"


PointProbe::PointProbe(
    const std::string& object_name,
    CurvGrid *grid,
    UData *data):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data)
{
    d_npoints = 0;
    sprintf(d_path, ".");
    sprintf(d_fpath, ".");

    ReadFromInput();
     
    char point_file[256]; 
    sprintf(point_file, "%s/savepoints", d_path);
    FILE *fp=fopen(point_file, "r");

    if (fp!=NULL) {
        int i=0;
        do {
              double x, y, z;
              fscanf(fp, "%le %le %le\n", &x, &y, &z);
              i++;
        } while(!feof(fp));
        d_npoints=i;
        fclose(fp);
        
        d_savecoor = (Cmpnts *) malloc(d_npoints * sizeof(Cmpnts));     
        d_saveindx = (Index *) malloc(d_npoints * sizeof(Index));  
      
        fp=fopen(point_file, "r");
        for (int i=0; i<d_npoints; i++)
            fscanf(fp, "%le %le %le\n", &d_savecoor[i].x, 
                                        &d_savecoor[i].y, 
                                        &d_savecoor[i].z);
        fclose(fp);
       
    }
}

PointProbe::~PointProbe()
{
    free(d_savecoor);
    free(d_saveindx);
}


PetscErrorCode PointProbe::Initialize()
{
    if (!d_npoints) return 0;

    PetscInt i,j,k;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();


    DMDALocalInfo info;
    int xs, xe, ys, ye, zs, ze; // Local grid information
    int mx, my, mz; // Dimensions in three directions
    int lxs, lxe, lys, lye, lzs, lze;
    DMDAGetLocalInfo(da, &info);
    mx = info.mx; my = info.my; mz = info.mz;
    xs = info.xs; xe = xs + info.xm;
    ys = info.ys; ye = ys + info.ym;
    zs = info.zs; ze = zs + info.zm;
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    Cmpnts ***cent;

    Vec lCent = d_grid->getlCent();

    DMDAVecGetArray(fda, lCent, &cent);

    Index *iclose;

    iclose = (Index *) malloc(d_npoints*sizeof(Index));

    for (int m=0; m<d_npoints; m++) {
        iclose[m].i=0;
        iclose[m].j=0;
        iclose[m].k=0;
    }

    PetscPrintf(PETSC_COMM_WORLD, "PointProve: save %d flow points\n\n", 
                                  d_npoints);
    int count[d_npoints], sum_count[d_npoints];
    for (int m=0; m<d_npoints; m++) {
        double XX=d_savecoor[m].x;
        double YY=d_savecoor[m].y;
        double ZZ=d_savecoor[m].z;

        double dis, dis_min;
        dis_min=1.e9;
                     
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) {
                    dis = pow(XX-cent[k][j][i].x,2) + 
                          pow(YY-cent[k][j][i].y,2) + 
                          pow(ZZ-cent[k][j][i].z,2);

                     if (dis<dis_min) {
                         dis_min=dis;
                         iclose[m].i=i;
                         iclose[m].j=j;
                         iclose[m].k=k;
                     }
                }

        double dmin_global;
        MPI_Allreduce (&dis_min, &dmin_global, 1, 
                       MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);
        count[m] = 1;
        double diff=fabs(dis_min-dmin_global);
        if (diff>1.e-9) {
            count[m]=0;
            iclose[m].i=0; iclose[m].j=0; iclose[m].k=0;
        }

    }

    PetscBarrier(PETSC_NULL);
    MPI_Allreduce(iclose, d_saveindx, d_npoints*3, MPI_INT, 
                  MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(count, sum_count, d_npoints, MPI_INT, 
                  MPI_SUM, PETSC_COMM_WORLD);

    for (int m=0; m<d_npoints; m++) {
        d_saveindx[m].i = d_saveindx[m].i/sum_count[m];
        d_saveindx[m].j = d_saveindx[m].j/sum_count[m];
        d_saveindx[m].k = d_saveindx[m].k/sum_count[m];

        PetscPrintf(PETSC_COMM_WORLD, 
                    "PointProbe: save flow points at x=%le y=%le z=%le\n", 
                    d_savecoor[m].x,  d_savecoor[m].y, d_savecoor[m].z );
        PetscPrintf(PETSC_COMM_WORLD, 
                    "PointProbe:save flow points at i=%d j=%d k=%d\n", 
                    d_saveindx[m].i,  d_saveindx[m].j, d_saveindx[m].k );
    }

    DMDAVecRestoreArray(fda, lCent, &cent);

    free(iclose);

}


PetscErrorCode PointProbe::Probe(PetscInt ti, PetscReal dt, PetscReal time)
{
    if (!d_npoints) return 0;

    PetscInt i,j,k;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    int xs, xe, ys, ye, zs, ze; // Local grid information
    int mx, my, mz; // Dimensions in three directions
    int lxs, lxe, lys, lye, lzs, lze;
    DMDAGetLocalInfo(da, &info);
    mx = info.mx; my = info.my; mz = info.mz;
    xs = info.xs; xe = xs + info.xm;
    ys = info.ys; ye = ys + info.ym;
    zs = info.zs; ze = zs + info.zm;
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;
 
    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    PetscReal ***p, ***aj, ***nvert;
    Cmpnts ***ucat, ***ucont, ***csi, ***eta, ***zet, ***cent;

    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();
    Vec lCent = d_grid->getlCent();

    Vec lUcat = d_data->getlUcat();
    Vec lUcont = d_data->getlUcont();
    Vec lP = d_data->getlP();
    Vec lNvert = d_data->getlNvert();

    PetscInt i_p = d_grid->isIPeriodic();
    PetscInt j_p = d_grid->isJPeriodic();
    PetscInt k_p = d_grid->isKPeriodic();
    PetscInt ii_p = d_grid->isIIPeriodic();
    PetscInt jj_p = d_grid->isJJPeriodic();
    PetscInt kk_p = d_grid->isKKPeriodic();

 

    DMDAVecGetArray(fda, lCent, &cent);
    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, Aj, &aj);

    DMDAVecGetArray(fda, lUcat, &ucat);
    DMDAVecGetArray(fda, lUcont, &ucont);
    DMDAVecGetArray(da, lP, &p);
    DMDAVecGetArray(da, lNvert, &nvert);

    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {    
                for (int m=0; m<d_npoints; m++) {
                    if (i==d_saveindx[m].i && 
                        j==d_saveindx[m].j && 
                        k==d_saveindx[m].k) {

                        double dudc, dvdc, dwdc; 
                        double dude, dvde, dwde;
                        double dudz, dvdz, dwdz;

                        double dpdc, dpde, dpdz;
                        double du_dx, du_dy, du_dz;
                        double dv_dx, dv_dy, dv_dz;
                        double dw_dx, dw_dy, dw_dz;
                        double dp_dx, dp_dy, dp_dz;

                        double csi0 = csi[k][j][i].x;
                        double csi1 = csi[k][j][i].y;
                        double csi2 = csi[k][j][i].z;
                        double eta0 = eta[k][j][i].x;
                        double eta1 = eta[k][j][i].y;
                        double eta2 = eta[k][j][i].z;
                        double zet0 = zet[k][j][i].x;
                        double zet1 = zet[k][j][i].y;
                        double zet2 = zet[k][j][i].z;
                        double ajc = aj[k][j][i];

                        double Ai = sqrt ( csi0*csi0 + csi1*csi1 + csi2*csi2 );
                        double Aj = sqrt ( eta0*eta0 + eta1*eta1 + eta2*eta2 );
                        double Ak = sqrt ( zet0*zet0 + zet1*zet1 + zet2*zet2 );

                        double U = 0.5*(ucont[k][j][i].x+ucont[k][j][i-1].x)/Ai;
                        double V = 0.5*(ucont[k][j][i].y+ucont[k][j-1][i].y)/Aj;
                        double W = 0.5*(ucont[k][j][i].z+ucont[k-1][j][i].z)/Ak;

                        Compute_du_center(i, j, k, 
                                          mx, my, mz, 
                                          ucat, nvert,
                                          i_p, ii_p, j_p, jj_p, k_p, kk_p, 
                                          &dudc, &dvdc, &dwdc, 
                                          &dude, &dvde, &dwde, 
                                          &dudz, &dvdz, &dwdz);
                        Compute_du_dxyz(csi0, csi1, csi2, 
                                        eta0, eta1, eta2, 
                                        zet0, zet1, zet2, ajc, 
                                        dudc, dvdc, dwdc, 
                                        dude, dvde, dwde, 
                                        dudz, dvdz, dwdz, 
                                        &du_dx, &dv_dx, &dw_dx, 
                                        &du_dy, &dv_dy, &dw_dy, 
                                        &du_dz, &dv_dz, &dw_dz );

                        Compute_dscalar_center(i, j, k, 
                                               mx, my, mz, 
                                               p, nvert, 
                                               &dpdc, &dpde, &dpdz );
                        Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                             eta0, eta1, eta2, 
                                             zet0, zet1, zet2, ajc, 
                                             dpdc, dpde, dpdz, 
                                             &dp_dx, &dp_dy, &dp_dz );

                        double vort_x = dw_dy - dv_dz;
                        double vort_y = du_dz - dw_dx;
                        double vort_z = dv_dx - du_dy;

                        FILE *f;
                        char filen[80];

                        sprintf(filen, "%s/Flow0_%.2e_%.2e_%.2e_dt_%g.dat", 
                                d_fpath, 
                                d_savecoor[m].x, d_savecoor[m].y, 
                                d_savecoor[m].z, dt);
                       f = fopen(filen, "a");
                       fprintf(f,"%d %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e "
                                 "%.7e %.7e %.7e %.7e %.7e \n",
                               (int)ti, 
                               cent[k][j][i].x,cent[k][j][i].y,cent[k][j][i].z,                                ucat[k][j][i].x,ucat[k][j][i].y,ucat[k][j][i].z,
                               p[k][j][i], U, V, W, 
                               vort_x, vort_y, vort_z);

                       fclose(f);

                       sprintf(filen, "%s/Flow1_%.2e_%.2e_%.2e_dt_%g.dat", 
                                d_fpath, 
                                d_savecoor[m].x, d_savecoor[m].y, 
                                d_savecoor[m].z, dt);
                       f = fopen(filen, "a");
                       fprintf(f, "%d %.7e %.7e %.7e %.7e %.7e %.7e %.7e %.7e "
                                  "%.7e %.7e %.7e %.7e %.7e %.7e %.7e\n", 
                               (int)ti, 
                               cent[k][j][i].x,cent[k][j][i].y,cent[k][j][i].z,
                               du_dx, du_dy, du_dz, 
                               dv_dx, dv_dy, dv_dz, 
                               dw_dx, dw_dy, dw_dz, 
                               dp_dx, dp_dy, dp_dz);
                       fclose(f);

                       break;
                    }
                }
            }

    DMDAVecRestoreArray(da, Aj, &aj);
    DMDAVecRestoreArray(da, lP, &p);
    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(fda, lUcat, &ucat);
    DMDAVecRestoreArray(fda, lUcont, &ucont);
    DMDAVecRestoreArray(fda, lCent, &cent);
     
    return 0;
}    
 

PetscErrorCode PointProbe::ReadFromInput()
{
    PetscOptionsGetString(PETSC_NULL,"-path", d_path, 256, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-flow_path", d_fpath, 256, PETSC_NULL);
} 
