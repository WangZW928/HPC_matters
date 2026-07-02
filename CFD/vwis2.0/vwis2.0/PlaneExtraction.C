#include "PlaneExtraction.h"


PlaneExtraction::PlaneExtraction(
    const std::string& object_name,
    CurvGrid *grid,
    UData *data):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data)
{
    d_nsavek = 0;
    sprintf(d_path, ".");
    sprintf(d_ipath, ".");
    d_scale_velocity = 1;
    d_save_inflow_period = 1000;
    d_read_inflow_period = 1000;
    d_save_inflow_minus = 0;
    d_ti_lastsave = 0;
    d_inflow_recycle_period = 20000;  

    d_ucat_plane_allocated = 0;

    ReadFromInput();

    char save_ksection_file[400];
    sprintf(save_ksection_file, "%s/savekplanes", d_path);
    FILE *fp=fopen(save_ksection_file, "r");
    if (fp!=NULL) {
        int i=0;
        int k;
        do {
            fscanf(fp, "%d\n", &k);
            i++;
        } while(!feof(fp));
        d_nsavek=i;
        fclose(fp);

        d_ksection = (PetscInt *) malloc(d_nsavek * sizeof(PetscInt)); 

        fp=fopen(save_ksection_file, "r");
        for (int i=0; i<d_nsavek; i++)
            fscanf(fp, "%d\n", d_ksection+i);
        fclose(fp);

    }
}

PlaneExtraction::~PlaneExtraction()
{

    if (d_ucat_plane_allocated) {
        DM da = d_grid->getDA();
        DM fda = d_grid->getFDA();
        DMDALocalInfo info;
        DMDAGetLocalInfo(da, &info);
        int mx = info.mx, my = info.my, mz = info.mz;
        for (int j=0; j<my; j++)  free(d_ucat_plane[j]);
        free(d_ucat_plane);
    }
   
    if (d_nsavek > 0)
       free(d_ksection);
}

PetscErrorCode PlaneExtraction::Save(PetscInt ti, PetscReal time)
{
 
    if (d_nsavek == 0) return 0;

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    int mx = info.mx, my = info.my, mz = info.mz;
    int j;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    int ti0 = ti-d_save_inflow_minus;
    
    for (int k=0; k<d_nsavek; k++) {

        int kplane = d_ksection[k];

        if (kplane<1 || kplane>mz-2) continue;
        
        StoreSection(kplane);

        if (!rank) {

            int tistart = d_data->get_tistart();
            double dt = d_data->getDt();
            char fname[256];
            int ti_name = ( (ti0-1) / d_save_inflow_period ) * 
                            d_save_inflow_period + 1;
  
            //Consider rewriting in hdf5 
            sprintf(fname, "%s%04d/inflow_%06d_dt=%g.dat", 
                    d_ipath, d_ksection[k], ti_name, dt);
            if (ti0==tistart || ti0%d_save_inflow_period==1) unlink(fname); 
                    
            FILE *fp = fopen(fname, "ab");
            if (!fp) printf("\n***Cannot open %s ! ***\n", fname), exit(0);
            
            for (j=0; j<my; j++) 
                fwrite(&d_ucat_plane[j][0], sizeof(Cmpnts), mx, fp);

            fclose(fp);
        
        }
    }

    return 0;
}

PetscErrorCode PlaneExtraction::StoreSection(PetscInt kplane)
{

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    int xs = info.xs, xe = info.xs + info.xm;
    int ys = info.ys, ye = info.ys + info.ym;
    int zs = info.zs, ze = info.zs + info.zm;
    int mx = info.mx, my = info.my, mz = info.mz;

    int i, j, k;
    
    if ( !d_ucat_plane_allocated ) {
        d_ucat_plane_allocated = 1;
        
        d_ucat_plane = (Cmpnts **)malloc( sizeof(Cmpnts *) * my );
        for(j=0; j<my; j++) 
            d_ucat_plane[j] = (Cmpnts *)malloc( sizeof(Cmpnts) * mx );
    }

    Vec Ucat = d_data->getUcat();    
    Cmpnts ***ucat;
    
    std::vector< std::vector<Cmpnts> > ucat_plane_tmp(my);
    for (j=0; j<my; j++) ucat_plane_tmp[j].resize(mx);
    
    for (j=0; j<my; j++)
        for (i=0; i<mx; i++) {
            ucat_plane_tmp[j][i].x = 0;
            ucat_plane_tmp[j][i].y = 0;
            ucat_plane_tmp[j][i].z = 0;
        }
    
    
    DMDAVecGetArray(fda, Ucat, &ucat);
    
    for (k=zs; k<ze; k++) {
        if (k==kplane) {
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) {
                    ucat_plane_tmp[j][i] = ucat[k][j][i];
                }
        }
    }
    
    DMDAVecRestoreArray(fda, Ucat, &ucat);

    for(j=0; j<my; j++) {
        MPI_Reduce(&ucat_plane_tmp[j][0], &d_ucat_plane[j][0], mx*3, 
                   MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    }

    return 0;

}

PetscErrorCode PlaneExtraction::Read(PetscInt ti)
{
    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    int mx = info.mx, my = info.my, mz = info.mz;
    int i, j;

    int rank;
    
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        
    if (!d_ucat_plane_allocated ) {
        d_ucat_plane_allocated = 1;
        
        d_ucat_plane = (Cmpnts **)malloc( sizeof(Cmpnts *) * my );

        for(j=0; j<my; j++) {
            d_ucat_plane[j] = (Cmpnts *)malloc( sizeof(Cmpnts) * mx );
        }
    }
    
    std::vector< std::vector<Cmpnts> > ucat_plane_tmp (my);
    for (j=0; j<my; j++) ucat_plane_tmp[j].resize(mx);
    
    for (j=0; j<my; j++)
        for (i=0; i<mx; i++) {
            ucat_plane_tmp[j][i].x = 0;
            ucat_plane_tmp[j][i].y = 0;
            ucat_plane_tmp[j][i].z = 0;
        }
    
    char fname[256];
    int ti2=ti+d_ti_lastsave;

    if (ti2==0) ti2=1;
        
    if (ti2>d_inflow_recycle_period) {
        ti2 -= (ti2/d_inflow_recycle_period) * d_inflow_recycle_period;
    }
    int ti_name = ( (ti2-1) / d_read_inflow_period ) * d_read_inflow_period + 1;

    int tistart = d_data->get_tistart();
    double dt = d_data->getDt();
    sprintf(fname, "%s/inflow_%06d_dt=%g.dat", d_ipath, ti_name, dt);
        
    if (!rank) {
        if (ti==tistart || (ti>tistart+90 && ti2%d_read_inflow_period==1)) {
            
            if (ti!=tistart) fclose(d_fp_inflow_u);
            d_fp_inflow_u=fopen(fname, "rb");

            if (!d_fp_inflow_u) 
                printf("\n**Cannot open %s ! ***\n", fname),exit(0);
            
            if (ti==tistart) {  
                for (int it=0; it<(ti2-1)%d_read_inflow_period; it++) {
                    for (j=0; j<my; j++) 
                        fread(&ucat_plane_tmp[j][0], sizeof(Cmpnts), 
                              mx, d_fp_inflow_u);
                }
            }
        }
        if (tistart==0 && ti==1) {}
        else 
           for(j=0; j<my; j++) 
              fread(&ucat_plane_tmp[j][0], sizeof(Cmpnts), mx, d_fp_inflow_u);

        for (j=0; j<my; j++)
           for (i=0; i<mx; i++) {
               ucat_plane_tmp[j][i].x = d_scale_velocity*ucat_plane_tmp[j][i].x;
               ucat_plane_tmp[j][i].y = d_scale_velocity*ucat_plane_tmp[j][i].y;
               ucat_plane_tmp[j][i].z = d_scale_velocity*ucat_plane_tmp[j][i].z;
        }
    
    }
    for (j=0; j<my; j++) 
        MPI_Allreduce(&ucat_plane_tmp[j][0], &d_ucat_plane[j][0], 
                      mx*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

    PetscPrintf(PETSC_COMM_WORLD, "\nRead inflow data from %s ... \n", fname);

    return 0;
}


PetscErrorCode PlaneExtraction::ReadFromInput()
{
    PetscOptionsGetString(PETSC_NULL,"-path", d_path, 256, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-inflow_path", d_ipath, 256, PETSC_NULL);

    PetscOptionsGetInt(PETSC_NULL, "-save_inflow_period", 
                       &d_save_inflow_period, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-save_inflow_minus", 
                       &d_save_inflow_minus, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-ti_lastsave", &d_ti_lastsave, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-recycle", &d_inflow_recycle_period, 
                       PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-read_inflow_period", 
                        &d_read_inflow_period, PETSC_NULL);

    PetscOptionsGetReal(PETSC_NULL, "-scale_velocity", 
                        &d_scale_velocity, PETSC_NULL);

    return 0;
}

