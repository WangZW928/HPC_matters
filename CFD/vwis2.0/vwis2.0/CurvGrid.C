#include "CurvGrid.h"
#include "petscviewerhdf5.h"

CurvGrid::CurvGrid(const std::string& object_name):
    d_object_name(object_name)
{
    d_xyz_input = PETSC_FALSE;
    d_binary_input = PETSC_FALSE;
    d_uniform_input = PETSC_FALSE;

    d_cl = 1.0;
    sprintf(d_gridfile, "grid.dat"); 
    sprintf(d_path, ".");

    d_i_periodic = 0;
    d_j_periodic = 0;
    d_k_periodic = 0;
    d_ii_periodic = 0;
    d_jj_periodic = 0;
    d_kk_periodic = 0;

    ReadFromInput();
} 

CurvGrid::~CurvGrid()
{
    VecDestroy(&d_lCsi);
    VecDestroy(&d_lEta);
    VecDestroy(&d_lZet);
    VecDestroy(&d_lICsi);
    VecDestroy(&d_lIEta);
    VecDestroy(&d_lIZet);
    VecDestroy(&d_lJCsi);
    VecDestroy(&d_lJEta);
    VecDestroy(&d_lJZet);
    VecDestroy(&d_lKCsi);
    VecDestroy(&d_lKEta);
    VecDestroy(&d_lKZet);
    VecDestroy(&d_lGridSpace);
    VecDestroy(&d_lCent);

    VecDestroy(&d_lAj);
    VecDestroy(&d_lIAj);
    VecDestroy(&d_lJAj);
    VecDestroy(&d_lKAj);

    DMDestroy(&d_da);
}   


PetscErrorCode CurvGrid::ReadGrid()
{
    int i,j,k;

    FILE *fd;

  /* Read in number of blocks and allocate memory for UserCtx */

    char str[256];

    if (d_xyz_input) sprintf(str, "%s/%s", d_path, "xyz.dat");
    else if (d_uniform_input) {}
    else sprintf(str, "%s/%s", d_path, d_gridfile);


    if (!d_uniform_input) {
        fd = fopen(str, "r");
        PetscPrintf(PETSC_COMM_WORLD, "Reading: %s\n", str);
        if (fd==NULL) 
            printf("Cannot open %s !\n", str),exit(0);
    }

    if (d_xyz_input || d_uniform_input) {d_block_number=1;}
    else if (d_binary_input) fread(&d_block_number, sizeof(int), 1, fd);
    else fscanf(fd, "%i\n", &d_block_number);

    std::vector<double> X, Y, Z;
    double tmp;

    //Read in the size
    //XYZ_input reads everything
    if (d_xyz_input) {
        fscanf(fd, "%i %i %i\n", &d_IM, &d_JM, &d_KM);
        X.resize(d_IM);
        Y.resize(d_JM);
        Z.resize(d_KM);

        PetscPrintf(PETSC_COMM_WORLD, "Reading %s %dx%dx%d\n", 
                                       str, d_IM, d_JM, d_KM);
        for (i=0; i<d_IM; i++) fscanf(fd, "%le %le %le\n", &X[i], &tmp, &tmp);
        for (j=0; j<d_JM; j++) fscanf(fd, "%le %le %le\n", &tmp, &Y[j], &tmp);
        for (k=0; k<d_KM; k++) fscanf(fd, "%le %le %le\n", &tmp, &tmp, &Z[k]); 
    } else if (d_uniform_input) {
        X.resize(d_IM);
        Y.resize(d_JM);
        Z.resize(d_KM);

        double dx = d_Lx / (d_IM-1);
        double dy = d_Ly / (d_JM-1);
        double dz = d_Lz / (d_KM-1);
        for (i=0; i<d_IM; i++) X[i] = (double) i * dx;
        for (j=0; j<d_JM; j++) Y[j] = (double) j * dy;
        for (k=0; k<d_KM; k++) Z[k] = (double) k * dz;
    } else if(d_binary_input) {
        fread(&(d_IM), sizeof(int), 1, fd);
        fread(&(d_JM), sizeof(int), 1, fd);
        fread(&(d_KM), sizeof(int), 1, fd);
    } else { 
        fscanf(fd, "%i %i %i\n", &d_IM, &d_JM, &d_KM);

        PetscPrintf(PETSC_COMM_WORLD, "Reading %s %dx%dx%d\n", 
                                       d_gridfile, d_IM, d_JM, d_KM);
    }

    //Create the Grid Distribution
    CreateDM(); 
    MPI_Barrier(PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Created DM\n");


    DMDALocalInfo info;
    DMDAGetLocalInfo(d_da, &info);
    int xs = info.xs, xe = info.xs + info.xm;
    int ys = info.ys, ye = info.ys + info.ym;
    int zs = info.zs, ze = info.zs + info.zm;


    Vec Coor, gCoor;
    Cmpnts ***coor;

    //Get the Coordinates Vec
    DMGetCoordinatesLocal(d_da, &Coor);

    DMDAVecGetArray(d_fda, Coor, &coor);

    double buffer;

    //Read the grid and scale
    for (k=0; k<d_KM; k++)
        for (j=0; j<d_JM; j++)
            for (i=0; i<d_IM; i++) {

                if (d_xyz_input || d_uniform_input) {}
                else if (d_binary_input) fread(&buffer, sizeof(double), 1, fd);
                else fscanf(fd, "%le", &buffer);

                if ( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
                    if (d_xyz_input||d_uniform_input) 
                        coor[k][j][i].x = X[i]/d_cl;
                    else coor[k][j][i].x = buffer/d_cl;
                }
            }  
    for (k=0; k<d_KM; k++)
        for (j=0; j<d_JM; j++)
            for (i=0; i<d_IM; i++) {
                if (d_xyz_input|| d_uniform_input) {}
                else if(d_binary_input) fread(&buffer, sizeof(double), 1, fd);
                else fscanf(fd, "%le", &buffer);

                if ( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
                    if (d_xyz_input|| d_uniform_input) 
                        coor[k][j][i].y = Y[j]/d_cl;
                    else coor[k][j][i].y = buffer/d_cl;
                }
            }

    for (k=0; k<d_KM; k++)
        for (j=0; j<d_JM; j++)
            for (i=0; i<d_IM; i++) {
                if (d_xyz_input|| d_uniform_input) {}
                else if (d_binary_input) fread(&buffer, sizeof(double), 1, fd);
                else fscanf(fd, "%le", &buffer);

                if ( k>=zs && k<ze && j>=ys && j<ye && i>=xs && i<xe ) {
                   if (d_xyz_input|| d_uniform_input) 
                       coor[k][j][i].z = Z[k]/d_cl;
                   else coor[k][j][i].z = buffer/d_cl;
                }
            }

    DMDAVecRestoreArray(d_fda, Coor, &coor);
    DMGetCoordinates(d_da, &gCoor);

    DMLocalToGlobalBegin(d_fda, Coor, INSERT_VALUES, gCoor);
    DMLocalToGlobalEnd(d_fda, Coor, INSERT_VALUES, gCoor);

    DMGlobalToLocalBegin(d_fda, gCoor, INSERT_VALUES, Coor);
    DMGlobalToLocalEnd(d_fda, gCoor, INSERT_VALUES, Coor);

    
    //Clse the grid file
    if (!d_uniform_input) {
        fclose(fd);
    }
    MPI_Barrier(PETSC_COMM_WORLD);

    return 0;
}

PetscErrorCode CurvGrid::ReadBC()
{
    char str[256];
    sprintf(str, "%s/bcs.dat", d_path);
    FILE *fd = fopen(str, "r");
    if (!fd) PetscPrintf(PETSC_COMM_WORLD, "cannot open %s !\n", str),exit(0);

    fscanf(fd, "%i %i %i %i %i %i\n", 
           &d_bctype[0], &d_bctype[1], &d_bctype[2], 
           &d_bctype[3], &d_bctype[4],&d_bctype[5]);
    MPI_Bcast(d_bctype, 6, MPI_INT, 0, PETSC_COMM_WORLD);

    fclose(fd);
    return 0;
}

PetscErrorCode CurvGrid::InitializeVecs()
{
    PetscErrorCode ierr;
   
    //Global 3d vectors
    ierr = DMCreateGlobalVector(d_fda, &d_Csi);
    VecDuplicate(d_Csi, &d_Eta);
    VecDuplicate(d_Csi, &d_Zet);

    VecDuplicate(d_Csi, &d_ICsi);
    VecDuplicate(d_Csi, &d_IEta);
    VecDuplicate(d_Csi, &d_IZet);
    VecDuplicate(d_Csi, &d_JCsi);
    VecDuplicate(d_Csi, &d_JEta);
    VecDuplicate(d_Csi, &d_JZet);
    VecDuplicate(d_Csi, &d_KCsi);
    VecDuplicate(d_Csi, &d_KEta);
    VecDuplicate(d_Csi, &d_KZet);
    VecDuplicate(d_Csi, &d_Cent);
    VecDuplicate(d_Csi, &d_GridSpace);

    //Global 1D vectors
    ierr = DMCreateGlobalVector(d_da, &d_Aj); 
    VecDuplicate(d_Aj, &d_IAj);
    VecDuplicate(d_Aj, &d_JAj);
    VecDuplicate(d_Aj, &d_KAj);

    //Local 3d vector
    DMCreateLocalVector(d_fda, &(d_lCsi));
    VecDuplicate(d_lCsi, &d_lEta);
    VecDuplicate(d_lCsi, &d_lZet);
    VecDuplicate(d_lCsi, &d_lICsi);
    VecDuplicate(d_lCsi, &d_lIEta);
    VecDuplicate(d_lCsi, &d_lIZet);
    VecDuplicate(d_lCsi, &d_lJCsi);
    VecDuplicate(d_lCsi, &d_lJEta);
    VecDuplicate(d_lCsi, &d_lJZet);
    VecDuplicate(d_lCsi, &d_lKCsi);
    VecDuplicate(d_lCsi, &d_lKEta);
    VecDuplicate(d_lCsi, &d_lKZet);
    VecDuplicate(d_lCsi, &d_lGridSpace);
    VecDuplicate(d_lCsi, &d_lCent);

    //Local 1d vectors
    DMCreateLocalVector(d_da, &d_lAj);
    VecDuplicate(d_lAj, &d_lIAj);
    VecDuplicate(d_lAj, &d_lJAj);
    VecDuplicate(d_lAj, &d_lKAj);


    return ierr;

}

PetscErrorCode CurvGrid::CreateDM()
{
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    PetscInt m, n, p, s;
    DMDABoundaryType bx=DMDA_BOUNDARY_GHOSTED, 
                     by=DMDA_BOUNDARY_GHOSTED, 
                     bz=DMDA_BOUNDARY_GHOSTED;

    //The Grid Distrib on procs in 3D
    //Have petsc decide
    m = n = p = PETSC_DECIDE;

    if (d_i_periodic) m=1;
    if (d_j_periodic) n=1;
    if (d_k_periodic) p=1;

    s=3;
        
    if (d_ii_periodic) bx = DMDA_BOUNDARY_PERIODIC;
    if (d_jj_periodic) by = DMDA_BOUNDARY_PERIODIC;
    if (d_kk_periodic) bz = DMDA_BOUNDARY_PERIODIC;    

    DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, DMDA_STENCIL_BOX,
                 d_IM+1, d_JM+1, d_KM+1, m, n,
                 p, 1, s, PETSC_NULL, PETSC_NULL, PETSC_NULL,
                 &d_da);

    DMDAGetInfo(d_da, PETSC_NULL, PETSC_NULL, PETSC_NULL, 
                PETSC_NULL, &m, &n, &p, PETSC_NULL, PETSC_NULL, 
                PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
    PetscPrintf(PETSC_COMM_WORLD, "**DM 3D Proc Distribution: %i %i %i\n", 
                                  m, n, p);

    DMDASetUniformCoordinates(d_da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    DMGetCoordinateDM(d_da, &d_fda);

    return 0;
}


PetscErrorCode CurvGrid::FormMetrics()
{
    DM cda;
    Cmpnts  ***csi, ***eta, ***zet, ***coor;
    PetscScalar ***aj;
    Vec coords;

    DM  da = d_da, fda = d_fda;
    Vec Csi = d_Csi, Eta = d_Eta, Zet = d_Zet;
    Vec Aj = d_Aj;
    Vec ICsi = d_ICsi, IEta = d_IEta, IZet = d_IZet;
    Vec JCsi = d_JCsi, JEta = d_JEta, JZet = d_JZet;
    Vec KCsi = d_KCsi, KEta = d_KEta, KZet = d_KZet;
    Vec IAj = d_IAj, JAj = d_JAj, KAj = d_KAj;

  
    Cmpnts ***icsi, ***ieta, ***izet;
    Cmpnts ***jcsi, ***jeta, ***jzet;
    Cmpnts ***kcsi, ***keta, ***kzet;
    Cmpnts  ***gs;
    PetscReal ***iaj, ***jaj, ***kaj;

    Vec Cent = d_Cent; //local working array for storing cell center geometry

    Vec Centx, Centy, Centz, lCoor;
    Cmpnts ***cent, ***centx, ***centy, ***centz;

    DMDALocalInfo info;

    int i, j, k;
    int mx, my, mz;
    int xs, ys, zs, xe, ye, ze;
    int lxs, lxe, lys, lye, lzs, lze;
    int gxs, gxe, gys, gye, gzs, gze;

    PetscErrorCode ierr;

    //Get Local Info and Setup Bounds
    DMDAGetLocalInfo(da, &info);
    mx = info.mx; my = info.my; mz = info.mz;
    xs = info.xs; xe = xs + info.xm;
    ys = info.ys; ye = ys + info.ym;
    zs = info.zs; ze = zs + info.zm;

    gxs = info.gxs; gxe = gxs + info.gxm;
    gys = info.gys; gye = gys + info.gym;
    gzs = info.gzs; gze = gzs + info.gzm;

    //Get Coordinate Data
    DMGetCoordinateDM(da, &cda);
    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    ierr = DMDAVecGetArray(da, Aj, &aj); CHKERRQ(ierr);

    //Get Local Coordinates
    DMGetCoordinatesLocal(da, &coords);
    DMDAVecGetArray(fda, coords, &coor);


    DMGetLocalVector(fda, &Centx);

    VecDuplicate(Centx, &Centy);
    VecDuplicate(Centx, &Centz);

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;


    //Get Cell Center Coord
    DMDAVecGetArray(fda, d_Cent, &cent);
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++) 
            for (i=lxs; i<lxe; i++) 
            {
                cent[k][j][i].x = 0.125 *
                    (coor[k][j][i].x + coor[k][j-1][i].x +
                     coor[k-1][j][i].x + coor[k-1][j-1][i].x +
                     coor[k][j][i-1].x + coor[k][j-1][i-1].x +
                     coor[k-1][j][i-1].x + coor[k-1][j-1][i-1].x);
                cent[k][j][i].y = 0.125 *
                    (coor[k][j][i].y + coor[k][j-1][i].y +
                     coor[k-1][j][i].y + coor[k-1][j-1][i].y +
                     coor[k][j][i-1].y + coor[k][j-1][i-1].y +
                     coor[k-1][j][i-1].y + coor[k-1][j-1][i-1].y);
                cent[k][j][i].z = 0.125 *
                    (coor[k][j][i].z + coor[k][j-1][i].z +
                     coor[k-1][j][i].z + coor[k-1][j-1][i].z +
                     coor[k][j][i-1].z + coor[k][j-1][i-1].z +
                     coor[k-1][j][i-1].z + coor[k-1][j-1][i-1].z);
            }
    DMDAVecRestoreArray(fda, d_Cent, &cent);
    
    //Set Global Data of Cell Center
    DMGlobalToLocalBegin(fda, d_Cent, INSERT_VALUES, d_lCent);
    DMGlobalToLocalEnd(fda, d_Cent, INSERT_VALUES, d_lCent);
    

    Cmpnts ***csitmp, ***etatmp, ***zettmp;
    Vec lCsitmp, lEtatmp, lZettmp;
    DMCreateLocalVector(fda, &lCsitmp);
    DMCreateLocalVector(fda, &lEtatmp);
    DMCreateLocalVector(fda, &lZettmp);
    
    DMDAVecGetArray(fda, lCsitmp, &csitmp);
    DMDAVecGetArray(fda, lEtatmp, &etatmp);
    DMDAVecGetArray(fda, lZettmp, &zettmp);
    
    double dxde, dyde, dzde;
    double dxdz, dydz, dzdz;
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=xs; i<lxe; i++) 
            {
                /* csi = de X dz */
                dxde = 0.5*(coor[k][j][i].x + coor[k-1][j][i].x -
                                   coor[k][j-1][i].x - coor[k-1][j-1][i].x);
                dyde = 0.5*(coor[k][j][i].y + coor[k-1][j][i].y -
                            coor[k][j-1][i].y - coor[k-1][j-1][i].y);
                dzde = 0.5*(coor[k][j][i].z + coor[k-1][j][i].z -
                            coor[k][j-1][i].z - coor[k-1][j-1][i].z);

                dxdz = 0.5*(coor[k][j-1][i].x + coor[k][j][i].x -
                            coor[k-1][j-1][i].x - coor[k-1][j][i].x);
                dydz = 0.5*(coor[k][j-1][i].y + coor[k][j][i].y -
                            coor[k-1][j-1][i].y - coor[k-1][j][i].y);
                dzdz = 0.5*(coor[k][j-1][i].z + coor[k][j][i].z -
                            coor[k-1][j-1][i].z - coor[k-1][j][i].z);
        
                csitmp[k][j][i].x = dyde * dzdz - dzde * dydz;
                csitmp[k][j][i].y =-dxde * dzdz + dzde * dxdz;
                csitmp[k][j][i].z = dxde * dydz - dyde * dxdz;
        
            }
    
    // Need more work -- lg65
    /* calculating j direction metrics */
    double dxdc, dydc, dzdc;
    for (k=lzs; k<lze; k++)
        for (j=ys; j<lye; j++)
            for (i=lxs; i<lxe; i++) 
            {
                /* eta = dz X de */
                dxdc = 0.5*(coor[k][j][i].x + coor[k-1][j][i].x -
                            coor[k][j][i-1].x - coor[k-1][j][i-1].x);
                dydc = 0.5*(coor[k][j][i].y + coor[k-1][j][i].y -
                            coor[k][j][i-1].y - coor[k-1][j][i-1].y);
                dzdc = 0.5*(coor[k][j][i].z + coor[k-1][j][i].z -
                            coor[k][j][i-1].z - coor[k-1][j][i-1].z);
                                             
                dxdz = 0.5*(coor[k][j][i].x + coor[k][j][i-1].x -
                            coor[k-1][j][i].x - coor[k-1][j][i-1].x);
                dydz = 0.5*(coor[k][j][i].y + coor[k][j][i-1].y -
                            coor[k-1][j][i].y - coor[k-1][j][i-1].y);
                dzdz = 0.5*(coor[k][j][i].z + coor[k][j][i-1].z -
                            coor[k-1][j][i].z - coor[k-1][j][i-1].z);
        
                etatmp[k][j][i].x = dydz * dzdc - dzdz * dydc;
                etatmp[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
                etatmp[k][j][i].z = dxdz * dydc - dydz * dxdc;
        
            }

    /* calculating k direction metrics */
    for (k=zs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) 
            {
                dxdc = 0.5*(coor[k][j][i].x + coor[k][j-1][i].x -
                            coor[k][j][i-1].x - coor[k][j-1][i-1].x);
                dydc = 0.5*(coor[k][j][i].y + coor[k][j-1][i].y -
                            coor[k][j][i-1].y - coor[k][j-1][i-1].y);
                dzdc = 0.5*(coor[k][j][i].z + coor[k][j-1][i].z -
                            coor[k][j][i-1].z - coor[k][j-1][i-1].z);
                                     
                dxde = 0.5*(coor[k][j][i].x + coor[k][j][i-1].x -
                            coor[k][j-1][i].x - coor[k][j-1][i-1].x);
                dyde = 0.5*(coor[k][j][i].y + coor[k][j][i-1].y -
                            coor[k][j-1][i].y - coor[k][j-1][i-1].y);
                dzde = 0.5*(coor[k][j][i].z + coor[k][j][i-1].z -
                            coor[k][j-1][i].z - coor[k][j-1][i-1].z);
        
                zettmp[k][j][i].x = dydc * dzde - dzdc * dyde;
                zettmp[k][j][i].y =-dxdc * dzde + dzdc * dxde;
                zettmp[k][j][i].z = dxdc * dyde - dydc * dxde;
        
            }
    
    DMDAVecRestoreArray(fda, lCsitmp, &csitmp);
    DMDAVecRestoreArray(fda, lEtatmp, &etatmp);
    DMDAVecRestoreArray(fda, lZettmp, &zettmp);
    
    DMDALocalToLocalBegin(fda, lCsitmp, INSERT_VALUES, lCsitmp);
    DMDALocalToLocalEnd(fda, lCsitmp, INSERT_VALUES, lCsitmp);
    
    DMDALocalToLocalBegin(fda, lEtatmp, INSERT_VALUES, lEtatmp);
    DMDALocalToLocalEnd(fda, lEtatmp, INSERT_VALUES, lEtatmp);
    
    DMDALocalToLocalBegin(fda, lZettmp, INSERT_VALUES, lZettmp);
    DMDALocalToLocalEnd(fda, lZettmp, INSERT_VALUES, lZettmp);
    
    DMDAVecGetArray(fda, lCsitmp, &csitmp);
    DMDAVecGetArray(fda, lEtatmp, &etatmp);
    DMDAVecGetArray(fda, lZettmp, &zettmp);
    
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) 
            {
                AxByC(0.5,csitmp[k][j][i],0.5,csitmp[k][j][i-1],&csi[k][j][i]);
                AxByC(0.5,etatmp[k][j][i],0.5,etatmp[k][j-1][i],&eta[k][j][i]);
                AxByC(0.5,zettmp[k][j][i],0.5,zettmp[k-1][j][i],&zet[k][j][i]);
        
            }

    DMDAVecRestoreArray(fda, lCsitmp, &csitmp);
    DMDAVecRestoreArray(fda, lEtatmp, &etatmp);
    DMDAVecRestoreArray(fda, lZettmp, &zettmp);
    
    VecDestroy(&lCsitmp);
    VecDestroy(&lEtatmp);
    
    /* calculating Jacobian of the transformation */
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++)
            {

                if (i>0 && j>0 && k>0) 
                {
                    dxdc = 0.25*(coor[k][j][i].x + coor[k][j-1][i].x +
                                 coor[k-1][j][i].x + coor[k-1][j-1][i].x -
                                 coor[k][j][i-1].x - coor[k][j-1][i-1].x -
                                 coor[k-1][j][i-1].x - coor[k-1][j-1][i-1].x);
                    dydc = 0.25*(coor[k][j][i].y + coor[k][j-1][i].y +
                                 coor[k-1][j][i].y + coor[k-1][j-1][i].y -
                                 coor[k][j][i-1].y - coor[k][j-1][i-1].y -
                                 coor[k-1][j][i-1].y - coor[k-1][j-1][i-1].y);
                    dzdc = 0.25*(coor[k][j][i].z + coor[k][j-1][i].z +
                                 coor[k-1][j][i].z + coor[k-1][j-1][i].z -
                                 coor[k][j][i-1].z - coor[k][j-1][i-1].z -
                                 coor[k-1][j][i-1].z - coor[k-1][j-1][i-1].z);

                    dxde = 0.25*(coor[k][j][i].x + coor[k][j][i-1].x +
                                 coor[k-1][j][i].x + coor[k-1][j][i-1].x - 
                                 coor[k][j-1][i].x - coor[k][j-1][i-1].x -
                                 coor[k-1][j-1][i].x - coor[k-1][j-1][i-1].x);
                    dyde = 0.25*(coor[k][j][i].y + coor[k][j][i-1].y +
                                 coor[k-1][j][i].y + coor[k-1][j][i-1].y - 
                                 coor[k][j-1][i].y - coor[k][j-1][i-1].y -
                                 coor[k-1][j-1][i].y - coor[k-1][j-1][i-1].y);
                    dzde = 0.25*(coor[k][j][i].z + coor[k][j][i-1].z +
                                 coor[k-1][j][i].z + coor[k-1][j][i-1].z - 
                                 coor[k][j-1][i].z - coor[k][j-1][i-1].z -
                                 coor[k-1][j-1][i].z - coor[k-1][j-1][i-1].z);

                    dxdz = 0.25*(coor[k][j][i].x + coor[k][j-1][i].x +
                                 coor[k][j][i-1].x + coor[k][j-1][i-1].x -
                                 coor[k-1][j][i].x - coor[k-1][j-1][i].x -
                                 coor[k-1][j][i-1].x - coor[k-1][j-1][i-1].x);
                    dydz = 0.25*(coor[k][j][i].y + coor[k][j-1][i].y +
                                 coor[k][j][i-1].y + coor[k][j-1][i-1].y -
                                 coor[k-1][j][i].y - coor[k-1][j-1][i].y -
                                 coor[k-1][j][i-1].y - coor[k-1][j-1][i-1].y);
                    dzdz = 0.25*(coor[k][j][i].z + coor[k][j-1][i].z +
                                 coor[k][j][i-1].z + coor[k][j-1][i-1].z -
                                 coor[k-1][j][i].z - coor[k-1][j-1][i].z -
                                 coor[k-1][j][i-1].z - coor[k-1][j-1][i-1].z);
      
                   aj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
                                         dydc * (dxde * dzdz - dzde * dxdz) +
                                 dzdc * (dxde * dydz - dyde * dxdz);
                   aj[k][j][i] = 1./aj[k][j][i];

                   if (aj[k][j][i]<0) 
                       printf("Negative jacobian %d,%d,%d\n", i,j,k);
               }
           }
  
  

    // mirror grid outside the boundary
    if (xs==0) {
        i = xs;
        for (k=zs; k<ze; k++) 
            for (j=ys; j<ye; j++) {
                csi[k][j][i] = csi[k][j][i+1];
                eta[k][j][i] = eta[k][j][i+1];
                zet[k][j][i] = zet[k][j][i+1];
                aj[k][j][i] = aj[k][j][i+1];
            }
    }

    if (xe==mx) {
        i = xe-1;
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++) {
                csi[k][j][i] = csi[k][j][i-1];
                eta[k][j][i] = eta[k][j][i-1];
                zet[k][j][i] = zet[k][j][i-1];
                aj[k][j][i] = aj[k][j][i-1];
            }
    }
  

    if (ys==0) {
        j = ys;
        for (k=zs; k<ze; k++)
            for (i=xs; i<xe; i++) {
                eta[k][j][i] = eta[k][j+1][i];
                csi[k][j][i] = csi[k][j+1][i];
                zet[k][j][i] = zet[k][j+1][i];
                aj[k][j][i] = aj[k][j+1][i];
            }
    }
  

    if (ye==my) {
        j = ye-1;
        for (k=zs; k<ze; k++)
            for (i=xs; i<xe; i++) {
                eta[k][j][i] = eta[k][j-1][i];
                csi[k][j][i] = csi[k][j-1][i];
                zet[k][j][i] = zet[k][j-1][i];
                aj[k][j][i] = aj[k][j-1][i];
            }
    }
  

    if (zs==0) {
        k = zs;
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                zet[k][j][i] = zet[k+1][j][i];
                eta[k][j][i] = eta[k+1][j][i];
                csi[k][j][i] = csi[k+1][j][i];
                aj[k][j][i] = aj[k+1][j][i];
            }
    }
    

    if (ze==mz) {
        k = ze-1;
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                zet[k][j][i] = zet[k-1][j][i];
                eta[k][j][i] = eta[k-1][j][i];
                csi[k][j][i] = csi[k-1][j][i];
                aj[k][j][i] = aj[k-1][j][i];
            }
    }
    
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(da, Aj,  &aj);
    
    DMGlobalToLocalBegin(fda, d_Csi, INSERT_VALUES, d_lCsi);
    DMGlobalToLocalEnd(fda, d_Csi, INSERT_VALUES, d_lCsi);

    DMGlobalToLocalBegin(fda, d_Eta, INSERT_VALUES, d_lEta);
    DMGlobalToLocalEnd(fda, d_Eta, INSERT_VALUES, d_lEta);

    DMGlobalToLocalBegin(fda, d_Zet, INSERT_VALUES, d_lZet);
    DMGlobalToLocalEnd(fda, d_Zet, INSERT_VALUES, d_lZet);
  
    DMGlobalToLocalBegin(da, d_Aj, INSERT_VALUES, d_lAj);
    DMGlobalToLocalEnd(da, d_Aj, INSERT_VALUES, d_lAj);

    Cmpnts ***lcsi, ***leta, ***lzet;
    PetscScalar ***laj;
    
    
    DMDAVecGetArray(fda, d_Csi, &csi);
    DMDAVecGetArray(fda, d_Eta, &eta);
    DMDAVecGetArray(fda, d_Zet, &zet);
    DMDAVecGetArray(da, d_Aj,  &aj);

    DMDAVecGetArray(fda, d_lCsi, &lcsi);
    DMDAVecGetArray(fda, d_lEta, &leta);
    DMDAVecGetArray(fda, d_lZet, &lzet);
    DMDAVecGetArray(da, d_lAj,  &laj);

    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                int flag=0, a=i, b=j, c=k;
                int i_flag=0, j_flag=0, k_flag=0;
            
                if (d_i_periodic && i==0) a=mx-2, i_flag=1;
                else if (d_i_periodic && i==mx-1) a=1, i_flag=1;
            
                if (d_j_periodic && j==0) b=my-2, j_flag=1;
                else if (d_j_periodic && j==my-1) b=1, j_flag=1;
            
                if (d_k_periodic && k==0) c=mz-2, k_flag=1;
                else if (d_k_periodic && k==mz-1) c=1, k_flag=1;
            
                if (d_ii_periodic && i==0) a=-2, i_flag=1;
                else if (d_ii_periodic && i==mx-1) a=mx+1, i_flag=1;
                
                if (d_jj_periodic && j==0) b=-2, j_flag=1;
                else if (d_jj_periodic && j==my-1) b=my+1, j_flag=1;
            
                if (d_kk_periodic && k==0) c=-2, k_flag=1;
                else if (d_kk_periodic && k==mz-1) c=mz+1, k_flag=1;
            
                flag = i_flag + j_flag + k_flag;

                if (flag) {
                    lcsi[k][j][i] = lcsi[c][b][a];
                    leta[k][j][i] = leta[c][b][a];
                    lzet[k][j][i] = lzet[c][b][a];
                    laj[k][j][i] = laj[c][b][a];

                    csi[k][j][i] = lcsi[k][j][i];
                    eta[k][j][i] = leta[k][j][i];
                    zet[k][j][i] = lzet[k][j][i];
                    aj[k][j][i] = laj[k][j][i];

                }
            }

    DMDAVecRestoreArray(fda, d_Csi, &csi);
    DMDAVecRestoreArray(fda, d_Eta, &eta);
    DMDAVecRestoreArray(fda, d_Zet, &zet);
    DMDAVecRestoreArray(da, d_Aj,  &aj);

    DMDAVecRestoreArray(fda, d_lCsi, &lcsi);
    DMDAVecRestoreArray(fda, d_lEta, &leta);
    DMDAVecRestoreArray(fda, d_lZet, &lzet);
    DMDAVecRestoreArray(da, d_lAj,  &laj);


    DMGlobalToLocalBegin(fda, d_Csi, INSERT_VALUES, d_lCsi);
    DMGlobalToLocalEnd(fda, d_Csi, INSERT_VALUES, d_lCsi);

    DMGlobalToLocalBegin(fda, d_Eta, INSERT_VALUES, d_lEta);
    DMGlobalToLocalEnd(fda, d_Eta, INSERT_VALUES, d_lEta);

    DMGlobalToLocalBegin(fda, d_Zet, INSERT_VALUES, d_lZet);
    DMGlobalToLocalEnd(fda, d_Zet, INSERT_VALUES, d_lZet);
  
    DMGlobalToLocalBegin(da, d_Aj, INSERT_VALUES, d_lAj);
    DMGlobalToLocalEnd(da, d_Aj, INSERT_VALUES, d_lAj);

    
    DMDAVecGetArray(fda, d_lCsi, &lcsi);
    DMDAVecGetArray(fda, d_lEta, &leta);
    DMDAVecGetArray(fda, d_lZet, &lzet);
    DMDAVecGetArray(da, d_lAj,  &laj);
    
    DMDAVecGetArray(fda, ICsi, &icsi);
    DMDAVecGetArray(fda, IEta, &ieta);
    DMDAVecGetArray(fda, IZet, &izet);
    DMDAVecGetArray(da, IAj,  &iaj);

    
    
    DMDAVecGetArray(fda, d_GridSpace, &gs);    
    PetscReal xcp, ycp, zcp, xcm, ycm, zcm;
    for (k=lzs; k<lze; k++) 
        for (j=lys; j<lye; j++) 
            for (i=lxs; i<lxe; i++) 
            {
                xcp = 0.25*(coor[k][j][i].x + coor[k][j-1][i].x +
                              coor[k-1][j-1][i].x + coor[k-1][j][i].x);
                ycp = 0.25*(coor[k][j][i].y + coor[k][j-1][i].y +
                            coor[k-1][j-1][i].y + coor[k-1][j][i].y);
                zcp = 0.25*(coor[k][j][i].z + coor[k][j-1][i].z +
                            coor[k-1][j-1][i].z + coor[k-1][j][i].z);

                xcm = 0.25*(coor[k][j][i-1].x + coor[k][j-1][i-1].x +
                            coor[k-1][j-1][i-1].x + coor[k-1][j][i-1].x);
                ycm = 0.25*(coor[k][j][i-1].y + coor[k][j-1][i-1].y +
                            coor[k-1][j-1][i-1].y + coor[k-1][j][i-1].y);
                zcm = 0.25*(coor[k][j][i-1].z + coor[k][j-1][i-1].z +
                            coor[k-1][j-1][i-1].z + coor[k-1][j][i-1].z);

                gs[k][j][i].x = sqrt((xcp-xcm) * (xcp-xcm) +
                                     (ycp-ycm) * (ycp-ycm) +
                                     (zcp-zcm) * (zcp-zcm));

                xcp = 0.25*(coor[k][j][i].x + coor[k][j][i-1].x +
                            coor[k-1][j][i].x + coor[k-1][j][i-1].x);
                ycp = 0.25*(coor[k][j][i].y + coor[k][j][i-1].y +
                            coor[k-1][j][i].y + coor[k-1][j][i-1].y);
                zcp = 0.25*(coor[k][j][i].z + coor[k][j][i-1].z +
                            coor[k-1][j][i].z + coor[k-1][j][i-1].z);

                xcm = 0.25*(coor[k][j-1][i].x + coor[k][j-1][i-1].x +
                            coor[k-1][j-1][i].x + coor[k-1][j-1][i-1].x);
                ycm = 0.25*(coor[k][j-1][i].y + coor[k][j-1][i-1].y +
                            coor[k-1][j-1][i].y + coor[k-1][j-1][i-1].y);
                zcm = 0.25*(coor[k][j-1][i].z + coor[k][j-1][i-1].z +
                            coor[k-1][j-1][i].z + coor[k-1][j-1][i-1].z);

                gs[k][j][i].y = sqrt((xcp-xcm) * (xcp-xcm) +
                                     (ycp-ycm) * (ycp-ycm) +
                                     (zcp-zcm) * (zcp-zcm));

                xcp = 0.25*(coor[k][j][i].x + coor[k][j][i-1].x +
                            coor[k][j-1][i].x + coor[k][j-1][i-1].x);
                ycp = 0.25*(coor[k][j][i].y + coor[k][j][i-1].y +
                            coor[k][j-1][i].y + coor[k][j-1][i-1].y);
                zcp = 0.25*(coor[k][j][i].z + coor[k][j][i-1].z +
                            coor[k][j-1][i].z + coor[k][j-1][i-1].z);

                xcm = 0.25*(coor[k-1][j][i].x + coor[k-1][j][i-1].x +
                            coor[k-1][j-1][i].x + coor[k-1][j-1][i-1].x);
                ycm = 0.25*(coor[k-1][j][i].y + coor[k-1][j][i-1].y +
                            coor[k-1][j-1][i].y + coor[k-1][j-1][i-1].y);
                zcm = 0.25*(coor[k-1][j][i].z + coor[k-1][j][i-1].z +
                            coor[k-1][j-1][i].z + coor[k-1][j-1][i-1].z);

                gs[k][j][i].z = sqrt((xcp-xcm) * (xcp-xcm) +
                                     (ycp-ycm) * (ycp-ycm) +
                                     (zcp-zcm) * (zcp-zcm));

            }

    DMDAVecRestoreArray(fda, d_GridSpace, &gs);


    //i direction
    DMDAVecGetArray(fda, Centx, &centx);
    for (k=gzs+1; k<gze; k++) 
        for (j=gys+1; j<gye; j++) 
            for (i=gxs; i<gxe; i++) 
            {
                centx[k][j][i].x = 0.25*(coor[k][j][i].x + coor[k-1][j][i].x +
                                    coor[k][j-1][i].x + coor[k-1][j-1][i].x);
                centx[k][j][i].y = 0.25*(coor[k][j][i].y + coor[k-1][j][i].y +
                                    coor[k][j-1][i].y + coor[k-1][j-1][i].y);
                centx[k][j][i].z = 0.25*(coor[k][j][i].z + coor[k-1][j][i].z +
                                    coor[k][j-1][i].z + coor[k-1][j-1][i].z);

            }

    for (k=lzs; k<lze; k++) 
        for (j=lys; j<lye; j++) 
            for (i=xs; i<lxe; i++) 
            {  
    
                if (i==0) 
                {
                    dxdc = centx[k][j][i+1].x - centx[k][j][i].x;
                    dydc = centx[k][j][i+1].y - centx[k][j][i].y;
                    dzdc = centx[k][j][i+1].z - centx[k][j][i].z;
                } else if (i==mx-2) {
                    if (d_i_periodic) 
                    {
                        dxdc = (centx[k][j][1].x - centx[k][j][0].x + 
                                centx[k][j][i].x - centx[k][j][i-1].x)*0.5;
                        dydc = (centx[k][j][1].y - centx[k][j][0].y + 
                                centx[k][j][i].y - centx[k][j][i-1].y)*0.5;
                        dzdc = (centx[k][j][1].z - centx[k][j][0].z + 
                                centx[k][j][i].z - centx[k][j][i-1].z)*0.5;
                    } else if(d_ii_periodic) {
                        dxdc = (centx[k][j][mx+1].x - centx[k][j][mx+0].x + 
                                centx[k][j][i].x - centx[k][j][i-1].x)*0.5;
                        dydc = (centx[k][j][mx+1].y - centx[k][j][mx+0].y + 
                                centx[k][j][i].y - centx[k][j][i-1].y)*0.5;
                        dzdc = (centx[k][j][mx+1].z - centx[k][j][mx+0].z + 
                                centx[k][j][i].z - centx[k][j][i-1].z)*0.5;
                    } else {
                        dxdc = centx[k][j][i].x - centx[k][j][i-1].x;
                        dydc = centx[k][j][i].y - centx[k][j][i-1].y;
                        dzdc = centx[k][j][i].z - centx[k][j][i-1].z;
                    }
                } else {
                   dxdc = (centx[k][j][i+1].x - centx[k][j][i-1].x)*0.5;
                   dydc = (centx[k][j][i+1].y - centx[k][j][i-1].y)*0.5;
                   dzdc = (centx[k][j][i+1].z - centx[k][j][i-1].z)*0.5;
                }

    
                if (j==1) 
                {
                    dxde = centx[k][j+1][i].x - centx[k][j][i].x;
                    dyde = centx[k][j+1][i].y - centx[k][j][i].y;
                    dzde = centx[k][j+1][i].z - centx[k][j][i].z;
                } else if (j==my-2) {
                    dxde = centx[k][j][i].x - centx[k][j-1][i].x;
                    dyde = centx[k][j][i].y - centx[k][j-1][i].y;
                    dzde = centx[k][j][i].z - centx[k][j-1][i].z;
                } else {
                    dxde = (centx[k][j+1][i].x - centx[k][j-1][i].x)*0.5;
                    dyde = (centx[k][j+1][i].y - centx[k][j-1][i].y)*0.5;
                    dzde = (centx[k][j+1][i].z - centx[k][j-1][i].z)*0.5;
                }
    
                if (k==1) 
                {
                    dxdz = (centx[k+1][j][i].x - centx[k][j][i].x);
                    dydz = (centx[k+1][j][i].y - centx[k][j][i].y);
                    dzdz = (centx[k+1][j][i].z - centx[k][j][i].z);
                } else if (k==mz-2) {
                    dxdz = (centx[k][j][i].x - centx[k-1][j][i].x);
                    dydz = (centx[k][j][i].y - centx[k-1][j][i].y);
                    dzdz = (centx[k][j][i].z - centx[k-1][j][i].z);
                } else {
                    dxdz = (centx[k+1][j][i].x - centx[k-1][j][i].x)*0.5;
                    dydz = (centx[k+1][j][i].y - centx[k-1][j][i].y)*0.5;
                    dzdz = (centx[k+1][j][i].z - centx[k-1][j][i].z)*0.5;
                }
    
                icsi[k][j][i].x = dyde * dzdz - dzde * dydz;
                icsi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
                icsi[k][j][i].z = dxde * dydz - dyde * dxdz;

                ieta[k][j][i].x = dydz * dzdc - dzdz * dydc;
                ieta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
                ieta[k][j][i].z = dxdz * dydc - dydz * dxdc;

                izet[k][j][i].x = dydc * dzde - dzdc * dyde;
                izet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
                izet[k][j][i].z = dxdc * dyde - dydc * dxde;

                iaj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
                               dydc * (dxde * dzdz - dzde * dxdz) +
                               dzdc * (dxde * dydz - dyde * dxdz);
                iaj[k][j][i] = 1./iaj[k][j][i];
    
            }

    PetscPrintf(PETSC_COMM_WORLD, "Finished Metrics I\n");

    DMDAVecRestoreArray(fda, ICsi, &icsi);
    DMDAVecRestoreArray(fda, IEta, &ieta);
    DMDAVecRestoreArray(fda, IZet, &izet);
    DMDAVecRestoreArray(da, IAj,  &iaj);

    // j direction
    DMDAVecGetArray(fda, JCsi, &jcsi);
    DMDAVecGetArray(fda, JEta, &jeta);
    DMDAVecGetArray(fda, JZet, &jzet);
    DMDAVecGetArray(da, JAj,  &jaj);

    DMDAVecGetArray(fda, Centy, &centy);
    for (k=gzs+1; k<gze; k++)
       for (j=gys; j<gye; j++)
           for (i=gxs+1; i<gxe; i++) 
           {
               centy[k][j][i].x =0.25*(coor[k][j][i].x + coor[k-1][j][i].x +
                                       coor[k][j][i-1].x + coor[k-1][j][i-1].x);
               centy[k][j][i].y =0.25*(coor[k][j][i].y + coor[k-1][j][i].y +
                                       coor[k][j][i-1].y + coor[k-1][j][i-1].y);
               centy[k][j][i].z =0.25*(coor[k][j][i].z + coor[k-1][j][i].z +
                                       coor[k][j][i-1].z + coor[k-1][j][i-1].z);
           }

    for (k=lzs; k<lze; k++) 
        for (j=ys; j<lye; j++) 
            for (i=lxs; i<lxe; i++) 
            {
                if (i==1) 
                {
                    dxdc = centy[k][j][i+1].x - centy[k][j][i].x;
                    dydc = centy[k][j][i+1].y - centy[k][j][i].y;
                    dzdc = centy[k][j][i+1].z - centy[k][j][i].z;
                } else if (i==mx-2) {
                    dxdc = centy[k][j][i].x - centy[k][j][i-1].x;
                    dydc = centy[k][j][i].y - centy[k][j][i-1].y;
                    dzdc = centy[k][j][i].z - centy[k][j][i-1].z;
                } else {
                    dxdc = (centy[k][j][i+1].x - centy[k][j][i-1].x) * 0.5;
                    dydc = (centy[k][j][i+1].y - centy[k][j][i-1].y) * 0.5;
                    dzdc = (centy[k][j][i+1].z - centy[k][j][i-1].z) * 0.5;
                }

                if (j==0) 
                {
                    dxde = centy[k][j+1][i].x - centy[k][j][i].x;
                    dyde = centy[k][j+1][i].y - centy[k][j][i].y;
                    dzde = centy[k][j+1][i].z - centy[k][j][i].z;
                } else if (j==my-2) {
                    if (d_j_periodic) 
                    {
                        dxde = 0.5*(centy[k][1][i].x - centy[k][0][i].x + 
                                   centy[k][j][i].x - centy[k][j-1][i].x);
                        dyde = 0.5*(centy[k][1][i].y - centy[k][0][i].y + 
                                   centy[k][j][i].y - centy[k][j-1][i].y);
                        dzde = 0.5*(centy[k][1][i].z - centy[k][0][i].z + 
                                  centy[k][j][i].z - centy[k][j-1][i].z);
                    } else if(d_jj_periodic) {
                        dxde = 0.5*(centy[k][my+1][i].x - centy[k][my+0][i].x +
                                    centy[k][j][i].x - centy[k][j-1][i].x);
                        dyde = 0.5*(centy[k][my+1][i].y - centy[k][my+0][i].y +
                                    centy[k][j][i].y - centy[k][j-1][i].y);
                        dzde = 0.5*(centy[k][my+1][i].z - centy[k][my+0][i].z +
                                    centy[k][j][i].z - centy[k][j-1][i].z);
                    } else {
                        dxde = centy[k][j][i].x - centy[k][j-1][i].x;
                        dyde = centy[k][j][i].y - centy[k][j-1][i].y;
                        dzde = centy[k][j][i].z - centy[k][j-1][i].z;
                    }
                } else {
                    dxde = (centy[k][j+1][i].x - centy[k][j-1][i].x) * 0.5;
                    dyde = (centy[k][j+1][i].y - centy[k][j-1][i].y) * 0.5;
                    dzde = (centy[k][j+1][i].z - centy[k][j-1][i].z) * 0.5;
                }

                if (k==1) 
                {
                   dxdz = (centy[k+1][j][i].x - centy[k][j][i].x);
                   dydz = (centy[k+1][j][i].y - centy[k][j][i].y);
                   dzdz = (centy[k+1][j][i].z - centy[k][j][i].z);
                } else if (k==mz-2) {
                   dxdz = (centy[k][j][i].x - centy[k-1][j][i].x);
                   dydz = (centy[k][j][i].y - centy[k-1][j][i].y);
                   dzdz = (centy[k][j][i].z - centy[k-1][j][i].z);
                } else {
                   dxdz = (centy[k+1][j][i].x - centy[k-1][j][i].x) * 0.5;
                   dydz = (centy[k+1][j][i].y - centy[k-1][j][i].y) * 0.5;
                   dzdz = (centy[k+1][j][i].z - centy[k-1][j][i].z) * 0.5;
                }

                jcsi[k][j][i].x = dyde * dzdz - dzde * dydz;
                jcsi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
                jcsi[k][j][i].z = dxde * dydz - dyde * dxdz;
    
                jeta[k][j][i].x = dydz * dzdc - dzdz * dydc;
                jeta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
                jeta[k][j][i].z = dxdz * dydc - dydz * dxdc;

                jzet[k][j][i].x = dydc * dzde - dzdc * dyde;
                jzet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
                jzet[k][j][i].z = dxdc * dyde - dydc * dxde;

    
                jaj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
                                       dydc * (dxde * dzdz - dzde * dxdz) +
                                       dzdc * (dxde * dydz - dyde * dxdz);
                jaj[k][j][i] = 1./jaj[k][j][i];
            }
 
    PetscPrintf(PETSC_COMM_WORLD, "Finished Metrics J\n");

    DMDAVecRestoreArray(fda, JCsi, &jcsi);
    DMDAVecRestoreArray(fda, JEta, &jeta);
    DMDAVecRestoreArray(fda, JZet, &jzet);
    DMDAVecRestoreArray(da, JAj,  &jaj);

    // k direction
    DMDAVecGetArray(fda, KCsi, &kcsi);
    DMDAVecGetArray(fda, KEta, &keta);
    DMDAVecGetArray(fda, KZet, &kzet);
    DMDAVecGetArray(da, KAj,  &kaj);

    DMDAVecGetArray(fda, Centz, &centz);
    for (k=gzs; k<gze; k++) 
        for (j=gys+1; j<gye; j++) 
            for (i=gxs+1; i<gxe; i++) {
                centz[k][j][i].x = 0.25*(coor[k][j][i].x + coor[k][j-1][i].x +
                                        coor[k][j][i-1].x+ coor[k][j-1][i-1].x);
                centz[k][j][i].y = 0.25*(coor[k][j][i].y + coor[k][j-1][i].y +
                                        coor[k][j][i-1].y+ coor[k][j-1][i-1].y);
                centz[k][j][i].z = 0.25*(coor[k][j][i].z + coor[k][j-1][i].z +
                                        coor[k][j][i-1].z+ coor[k][j-1][i-1].z);
            }

    for (k=zs; k<lze; k++) 
        for (j=lys; j<lye; j++) 
            for (i=lxs; i<lxe; i++) {  
    
                if (i==1) {
                    dxdc = centz[k][j][i+1].x - centz[k][j][i].x;
                    dydc = centz[k][j][i+1].y - centz[k][j][i].y;
                    dzdc = centz[k][j][i+1].z - centz[k][j][i].z;
                } else if (i==mx-2) {
                    dxdc = centz[k][j][i].x - centz[k][j][i-1].x;
                    dydc = centz[k][j][i].y - centz[k][j][i-1].y;
                    dzdc = centz[k][j][i].z - centz[k][j][i-1].z;
                } else {
                    dxdc = (centz[k][j][i+1].x - centz[k][j][i-1].x) * 0.5;
                    dydc = (centz[k][j][i+1].y - centz[k][j][i-1].y) * 0.5;
                    dzdc = (centz[k][j][i+1].z - centz[k][j][i-1].z) * 0.5;
                }

    
                if (j==1) {
                    dxde = centz[k][j+1][i].x - centz[k][j][i].x;
                    dyde = centz[k][j+1][i].y - centz[k][j][i].y;
                    dzde = centz[k][j+1][i].z - centz[k][j][i].z;
                } else if (j==my-2) {
                    dxde = centz[k][j][i].x - centz[k][j-1][i].x;
                    dyde = centz[k][j][i].y - centz[k][j-1][i].y;
                    dzde = centz[k][j][i].z - centz[k][j-1][i].z;
                } else {
                    dxde = (centz[k][j+1][i].x - centz[k][j-1][i].x) * 0.5;
                    dyde = (centz[k][j+1][i].y - centz[k][j-1][i].y) * 0.5;
                    dzde = (centz[k][j+1][i].z - centz[k][j-1][i].z) * 0.5;
                }

    
                if (k==0) {
                    dxdz = (centz[k+1][j][i].x - centz[k][j][i].x);
                    dydz = (centz[k+1][j][i].y - centz[k][j][i].y);
                    dzdz = (centz[k+1][j][i].z - centz[k][j][i].z);
                } else if (k==mz-2) {
                    if (d_k_periodic) {
                        dxdz = 0.5*(centz[1][j][i].x - centz[0][j][i].x + 
                                    centz[k][j][i].x - centz[k-1][j][i].x);
                        dydz = 0.5*(centz[1][j][i].y - centz[0][j][i].y + 
                                    centz[k][j][i].y - centz[k-1][j][i].y);
                        dzdz = 0.5*(centz[1][j][i].z - centz[0][j][i].z + 
                                    centz[k][j][i].z - centz[k-1][j][i].z);
                    } else if (d_kk_periodic) {
                        dxdz = 0.5*(centz[mz+1][j][i].x - centz[mz+0][j][i].x +
                                    centz[k][j][i].x - centz[k-1][j][i].x);
                        dydz = 0.5*(centz[mz+1][j][i].y - centz[mz+0][j][i].y +
                                    centz[k][j][i].y - centz[k-1][j][i].y);
                        dzdz = 0.5*(centz[mz+1][j][i].z - centz[mz+0][j][i].z +
                                    centz[k][j][i].z - centz[k-1][j][i].z);
                    } else {
                        dxdz = (centz[k][j][i].x - centz[k-1][j][i].x);
                        dydz = (centz[k][j][i].y - centz[k-1][j][i].y);
                        dzdz = (centz[k][j][i].z - centz[k-1][j][i].z);
                    }
                } else {
                    dxdz = (centz[k+1][j][i].x - centz[k-1][j][i].x) * 0.5;
                    dydz = (centz[k+1][j][i].y - centz[k-1][j][i].y) * 0.5;
                    dzdz = (centz[k+1][j][i].z - centz[k-1][j][i].z) * 0.5;
                }

                kcsi[k][j][i].x = dyde * dzdz - dzde * dydz;
                kcsi[k][j][i].y =-dxde * dzdz + dzde * dxdz;
                kcsi[k][j][i].z = dxde * dydz - dyde * dxdz;

                keta[k][j][i].x = dydz * dzdc - dzdz * dydc;
                keta[k][j][i].y =-dxdz * dzdc + dzdz * dxdc;
                keta[k][j][i].z = dxdz * dydc - dydz * dxdc;

                kzet[k][j][i].x = dydc * dzde - dzdc * dyde;
                kzet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
                kzet[k][j][i].z = dxdc * dyde - dydc * dxde;


                kaj[k][j][i] = dxdc * (dyde * dzdz - dzde * dydz) -
                               dydc * (dxde * dzdz - dzde * dxdz) +
                               dzdc * (dxde * dydz - dyde * dxdz);
                kaj[k][j][i] = 1./kaj[k][j][i];
           }

    PetscPrintf(PETSC_COMM_WORLD, "Finished Metrics K\n");

    DMDAVecRestoreArray(fda, d_lCsi, &lcsi);
    DMDAVecRestoreArray(fda, d_lEta, &leta);
    DMDAVecRestoreArray(fda, d_lZet, &lzet);
    DMDAVecRestoreArray(da, d_lAj,  &laj);

    DMDAVecRestoreArray(fda, Centz, &centz);
    DMDAVecRestoreArray(fda, Centy, &centy);
    DMDAVecRestoreArray(fda, Centx, &centx);
  
    DMDALocalToLocalBegin(fda, Centx, INSERT_VALUES, Centx);
    DMDALocalToLocalEnd(fda, Centx, INSERT_VALUES, Centx);
    
    DMDALocalToLocalBegin(fda, Centy, INSERT_VALUES, Centy);
    DMDALocalToLocalEnd(fda, Centy, INSERT_VALUES, Centy);
    
    DMDALocalToLocalBegin(fda, Centz, INSERT_VALUES, Centz);
    DMDALocalToLocalEnd(fda, Centz, INSERT_VALUES, Centz);
    
    DMDAVecRestoreArray(fda, KCsi, &kcsi);
    DMDAVecRestoreArray(fda, KEta, &keta);
    DMDAVecRestoreArray(fda, KZet, &kzet);
    DMDAVecRestoreArray(da, KAj,  &kaj);
  

    DMDAVecRestoreArray(cda, coords, &coor);



    VecAssemblyBegin(Csi);
    VecAssemblyEnd(Csi);
    VecAssemblyBegin(Eta);
    VecAssemblyEnd(Eta);
    VecAssemblyBegin(Zet);
    VecAssemblyEnd(Zet);
    VecAssemblyBegin(Aj);
    VecAssemblyEnd(Aj);

    VecAssemblyBegin(d_Cent);
    VecAssemblyEnd(d_Cent);

    VecAssemblyBegin(d_ICsi);
    VecAssemblyEnd(d_ICsi);
    VecAssemblyBegin(d_IEta);
    VecAssemblyEnd(d_IEta);
    VecAssemblyBegin(d_IZet);
    VecAssemblyEnd(d_IZet);
    VecAssemblyBegin(d_IAj);
    VecAssemblyEnd(d_IAj);

    VecAssemblyBegin(d_JCsi);
    VecAssemblyEnd(d_JCsi);
    VecAssemblyBegin(d_JEta);
    VecAssemblyEnd(d_JEta);
    VecAssemblyBegin(d_JZet);
    VecAssemblyEnd(d_JZet);
    VecAssemblyBegin(d_JAj);
    VecAssemblyEnd(d_JAj);

    VecAssemblyBegin(d_KCsi);
    VecAssemblyEnd(d_KCsi);
    VecAssemblyBegin(d_KEta);
    VecAssemblyEnd(d_KEta);
    VecAssemblyBegin(d_KZet);
    VecAssemblyEnd(d_KZet);
    VecAssemblyBegin(d_KAj);
    VecAssemblyEnd(d_KAj);

    DMRestoreLocalVector(fda, &Centx);

    VecDestroy(&Centy);
    VecDestroy(&Centz);

    DMGlobalToLocalBegin(fda, d_Csi, INSERT_VALUES, d_lCsi);
    DMGlobalToLocalEnd(fda, d_Csi, INSERT_VALUES, d_lCsi);

    DMGlobalToLocalBegin(fda, d_Eta, INSERT_VALUES, d_lEta);
    DMGlobalToLocalEnd(fda, d_Eta, INSERT_VALUES, d_lEta);

    DMGlobalToLocalBegin(fda, d_Zet, INSERT_VALUES, d_lZet);
    DMGlobalToLocalEnd(fda, d_Zet, INSERT_VALUES, d_lZet);

    DMGlobalToLocalBegin(fda, d_ICsi, INSERT_VALUES, d_lICsi);
    DMGlobalToLocalEnd(fda, d_ICsi, INSERT_VALUES, d_lICsi);

    DMGlobalToLocalBegin(fda, d_IEta, INSERT_VALUES, d_lIEta);
    DMGlobalToLocalEnd(fda, d_IEta, INSERT_VALUES, d_lIEta);

    DMGlobalToLocalBegin(fda, d_IZet, INSERT_VALUES, d_lIZet);
    DMGlobalToLocalEnd(fda, d_IZet, INSERT_VALUES, d_lIZet);

    DMGlobalToLocalBegin(fda, d_JCsi, INSERT_VALUES, d_lJCsi);
    DMGlobalToLocalEnd(fda, d_JCsi, INSERT_VALUES, d_lJCsi);

    DMGlobalToLocalBegin(fda, d_JEta, INSERT_VALUES, d_lJEta);
    DMGlobalToLocalEnd(fda, d_JEta, INSERT_VALUES, d_lJEta);

    DMGlobalToLocalBegin(fda, d_JZet, INSERT_VALUES, d_lJZet);
    DMGlobalToLocalEnd(fda, d_JZet, INSERT_VALUES, d_lJZet);

    DMGlobalToLocalBegin(fda, d_KCsi, INSERT_VALUES, d_lKCsi);
    DMGlobalToLocalEnd(fda, d_KCsi, INSERT_VALUES, d_lKCsi);

    DMGlobalToLocalBegin(fda, d_KEta, INSERT_VALUES, d_lKEta);
    DMGlobalToLocalEnd(fda, d_KEta, INSERT_VALUES, d_lKEta);

    DMGlobalToLocalBegin(fda, d_KZet, INSERT_VALUES, d_lKZet);
    DMGlobalToLocalEnd(fda, d_KZet, INSERT_VALUES, d_lKZet);

    DMGlobalToLocalBegin(da, d_Aj, INSERT_VALUES, d_lAj);
    DMGlobalToLocalEnd(da, d_Aj, INSERT_VALUES, d_lAj);

    DMGlobalToLocalBegin(da, d_IAj, INSERT_VALUES, d_lIAj);
    DMGlobalToLocalEnd(da, d_IAj, INSERT_VALUES, d_lIAj);

    DMGlobalToLocalBegin(da, d_JAj, INSERT_VALUES, d_lJAj);
    DMGlobalToLocalEnd(da, d_JAj, INSERT_VALUES, d_lJAj);

    DMGlobalToLocalBegin(da, d_KAj, INSERT_VALUES, d_lKAj);
    DMGlobalToLocalEnd(da, d_KAj, INSERT_VALUES, d_lKAj);

    DMGlobalToLocalBegin(fda, d_GridSpace, INSERT_VALUES, d_lGridSpace);
    DMGlobalToLocalEnd(fda, d_GridSpace, INSERT_VALUES, d_lGridSpace);

    DMGlobalToLocalBegin(fda, d_Cent, INSERT_VALUES, d_lCent);
    DMGlobalToLocalEnd(fda, d_Cent, INSERT_VALUES, d_lCent);

    VecDestroy(&d_Csi);
    VecDestroy(&d_Eta);
    VecDestroy(&d_Zet);

    VecDestroy(&d_ICsi);
    VecDestroy(&d_IEta);
    VecDestroy(&d_IZet);

    VecDestroy(&d_JCsi);
    VecDestroy(&d_JEta);
    VecDestroy(&d_JZet);

    VecDestroy(&d_KCsi);
    VecDestroy(&d_KEta);
    VecDestroy(&d_KZet);

    VecDestroy(&d_Aj);
    VecDestroy(&d_IAj);
    VecDestroy(&d_JAj);
    VecDestroy(&d_KAj);

    //Do I need this at all
    VecDestroy(&Centx);
    VecDestroy(&Centy);
    VecDestroy(&Centz);

    DMDAGetGlobalIndices(d_da, PETSC_NULL, &d_idx_from);
 
    PetscPrintf(PETSC_COMM_WORLD, "Finished Metrics\n");
    PetscBarrier(PETSC_NULL);

    return 0;
}


PetscErrorCode CurvGrid::ReadFromInput()
{
    PetscOptionsGetReal(PETSC_NULL, "-chact_leng", &d_cl, PETSC_NULL);
    PetscOptionsGetBool(PETSC_NULL, "-xyz", &d_xyz_input, PETSC_NULL);
    PetscOptionsGetBool(PETSC_NULL, "-binary", &d_binary_input, PETSC_NULL);
    PetscOptionsGetBool(PETSC_NULL, "-uniform", &d_uniform_input, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-grid", d_gridfile, 256, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-path", d_path, 256, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-i_periodic", &d_i_periodic, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-j_periodic", &d_j_periodic, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-k_periodic", &d_k_periodic, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-ii_periodic", &d_ii_periodic, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-jj_periodic", &d_jj_periodic, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-kk_periodic", &d_kk_periodic, PETSC_NULL);

    if (d_uniform_input) {
        PetscOptionsGetInt(PETSC_NULL, "-IM", &d_IM, PETSC_NULL);
        PetscOptionsGetInt(PETSC_NULL, "-JM", &d_JM, PETSC_NULL);
        PetscOptionsGetInt(PETSC_NULL, "-KM", &d_KM, PETSC_NULL);

        PetscOptionsGetReal(PETSC_NULL, "-Lx", &d_Lx, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, "-Ly", &d_Ly, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, "-Lz", &d_Lz, PETSC_NULL);

        PetscPrintf(PETSC_COMM_WORLD, 
                    "Uniform Grid (%d x %d x %d) -> (%f x %f x %f)\n", 
                     d_IM, d_JM, d_KM, d_Lx, d_Ly, d_Lz);
    }

    if (d_i_periodic) PetscPrintf(PETSC_COMM_WORLD, "I-Periodic\n");
    if (d_ii_periodic) PetscPrintf(PETSC_COMM_WORLD, "II-Periodic\n");
    if (d_j_periodic) PetscPrintf(PETSC_COMM_WORLD, "J-Periodic \n");
    if (d_jj_periodic) PetscPrintf(PETSC_COMM_WORLD, "JJ-Periodic \n");
    if (d_k_periodic) PetscPrintf(PETSC_COMM_WORLD, "K-Periodic \n");
    if (d_kk_periodic) PetscPrintf(PETSC_COMM_WORLD, "KK-Periodic \n");

    d_periodic = d_i_periodic + d_j_periodic + d_k_periodic +
                 d_ii_periodic + d_jj_periodic + d_kk_periodic;
}
