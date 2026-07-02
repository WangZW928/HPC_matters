#include "FSI.h"


FSI::FSI(
    const std::string& object_name,
    CurvGrid *grid,
    UData *data,
    ImmersedBoundary *ib):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data),
    d_ib(ib)
{
    sprintf(d_path, ".");
    sprintf(d_fsipath, ".");

    d_sisteps = 0;
    d_immersed = 0;
    d_movefsi = 0;
    d_rotatefsi=0;
    d_rotatefsi_noIBsearch=0;
    d_changefsi = 0;

    d_x_c=0.0;
    d_y_c=0.0;
    d_z_c=0.0;
    d_x_r=0.0;
    d_y_r=0.0;
    d_z_r=0.0;

    d_red_vel=0.52;
    d_damp=.02;
    d_mu_s=500.;

    d_Mx_applied=0.0;
    d_My_applied=0.0;
    d_Mz_applied=0.0;

    d_Max_xbc= 1e23; d_Max_ybc= 1e23; d_Max_zbc= 1e23;
    d_Min_xbc=-1e23; d_Min_ybc=-1e23; d_Min_zbc=-1e23;
   
    d_NumberOfBodies = 0;
    d_NumberOfRotatingBodies = 0;
   
    //FSI Move direction
    d_dgf_z=0;
    d_dgf_y=1;
    d_dgf_x=0;

    //FSI Rotation direction
    d_rotdir = 1;
    d_prescribed_rotation = 1;
    d_angvel = 0.0;

    d_ti_lastsave = 0;
    d_tiout = 1;
    d_rstart_fsi = 0;

    ReadFromInput();


    //Make directories
    struct stat st = {0};
        
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    if (!rank) {
        if (stat(d_fsipath, &st) == -1 && d_immersed) {
            mkdir(d_fsipath, 0754);
            printf("Creating Directory: %s\n", d_fsipath);
        }
    } 

    if (d_immersed)
        PetscMalloc(d_NumberOfBodies*sizeof(FSInfo), &d_fsi);

}

FSI::~FSI()
{
    if (d_immersed) PetscFree(d_fsi);
}

PetscErrorCode FSI::Initialize()
{
   
    if (!d_immersed) return 0;
 
    IBMNodes *ibm = d_ib->getIBMNodes(); 
    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++)
    {
        //n_elmt = 0; //ibm[ibi]->n_elmt; 
        FSInfo *fsi = &d_fsi[ibi];
        //PetscMalloc(n_elmt*sizeof(IBMInfo), &(fsi->fsi_intp));
        //PetscMalloc(n_elmt*sizeof(SurfElmtInfo), &(fsi->elmtinfo));

        for (int i=0;i<6;i++) {
            fsi->S_old[i]=0.;
            fsi->S_new[i]=0.;
            fsi->S_realm1[i]=0.;
            fsi->S_real[i]=0.;

            fsi->S_ang_n[i]=0.;
            fsi->S_ang_o[i]=0.;
            fsi->S_ang_r[i]=0.;
            fsi->S_ang_rm1[i]=0.;
        }

        fsi->F_x_old = 0.;
        fsi->F_y_old = 0.;
        fsi->F_z_old = 0.;

        fsi->F_x_real = 0.;
        fsi->F_y_real = 0.;
        fsi->F_z_real = 0.;

        fsi->M_x_old = 0.;
        fsi->M_y_old = 0.;
        fsi->M_z_old = 0.;

        fsi->M_x_real = 0.;
        fsi->M_y_real = 0.;
        fsi->M_z_real = 0.;

        fsi->x_c=d_x_c; fsi->y_c=d_y_c; fsi->z_c=d_z_c;

        fsi->red_vel=d_red_vel;
        fsi->damp=d_damp;
        fsi->mu_s=d_mu_s;

        fsi->Max_xbc=d_Max_xbc; fsi->Max_ybc=d_Max_ybc;fsi->Max_zbc=d_Max_zbc;
        fsi->Min_xbc=d_Min_xbc; fsi->Min_ybc=d_Min_ybc;fsi->Min_zbc=d_Min_zbc;

        fsi->Mx_applied=d_Mx_applied;
        fsi->My_applied=d_My_applied;
        fsi->Mz_applied=d_Mz_applied; 
    }
}

PetscErrorCode FSI::ReadFSI(PetscInt ti)
{
 
    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++)
    {
        FSInfo *fsi = &d_fsi[ibi];
        ReadFsiInput(fsi, ibi, ti);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "...Finished FSI Read\n");
   
    return 0;
}

PetscErrorCode FSI::WriteFSI(PetscInt ti)
{

    if (ti != (ti/d_tiout) * d_tiout && 
       (!d_changefsi)) return 0;

    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++)
    {
        FSInfo *fsi = &d_fsi[ibi];
        WriteFSIOutput(fsi, ibi, ti);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "...Finished FSI Write\n");
}

PetscErrorCode FSI::ReadFsiInput(FSInfo *FSinf, PetscInt ibi, PetscInt ti)
{
    int  i;
    PetscReal t;

    FILE *f;
    char filen[80];  
    sprintf(filen, "%s/DATA_FSI%5.5d_%2.2d.dat", d_fsipath, ti, ibi);

    f = fopen(filen, "r");
    if (!f) {
        PetscPrintf(PETSC_COMM_WORLD, 
                    "FSI_data cannot open file !!!!!!!!!!!!\n");
        exit(1);
    }
    PetscPrintf(PETSC_COMM_WORLD, "...Reading FSI %d %s\n",ti,filen);
    fscanf(f, "%le %le %le", &t, &t, &t);      
    PetscPrintf(PETSC_COMM_WORLD, 
                "......FSI V_red: %le, Damp: %le, Mu: %le \n",
                FSinf->red_vel,FSinf->damp,FSinf->mu_s);
    fscanf(f, "%le %le %le", &(FSinf->x_c), &(FSinf->y_c), &(FSinf->z_c));      
    fscanf(f, "%le %le %le \n", &(FSinf->F_x),&(FSinf->F_y), &(FSinf->F_z));     
    fscanf(f, "%le %le %le \n", &(FSinf->M_x),&(FSinf->M_y), &(FSinf->M_z));      

    for (i=0; i<6; i++) {
        fscanf(f, "%le %le %le %le",&(FSinf->S_new[i]), &(FSinf->S_old[i]), 
                                    &(FSinf->S_real[i]), &(FSinf->S_realm1[i]));
        fscanf(f, "%le %le %le %le", &(FSinf->S_ang_n[i]),&(FSinf->S_ang_o[i]),
                                  &(FSinf->S_ang_r[i]), &(FSinf->S_ang_rm1[i]));
    }
    fclose(f);
    PetscPrintf(PETSC_COMM_WORLD, "FSI_data input z, dz/dt  %le %le %le %le\n",
                FSinf->S_new[4],FSinf->S_new[5],FSinf->red_vel,FSinf->damp);
    PetscPrintf(PETSC_COMM_WORLD, "FSI_data input y, dy/dt  %le %le %le %le\n",
                FSinf->S_new[2],FSinf->S_new[3],FSinf->red_vel,FSinf->damp);
    return 0;
}

PetscErrorCode FSI::WriteFSIOutput(FSInfo *FSinfo, PetscInt ibi, PetscInt ti)
{
    int rank, i;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscBarrier(PETSC_NULL);
    if (!rank) {
        FILE *f;
        char filen[80];
        sprintf(filen, "%s/FSI_position%2.2d.dat", d_fsipath, ibi);
        f = fopen(filen, "a");
        PetscFPrintf(PETSC_COMM_WORLD, f, 
                     "%d %le %le %le %le %le %le %le %le %le\n",
                     ti, FSinfo->S_new[2],FSinfo->S_new[3],FSinfo->F_y, 
                     FSinfo->S_new[4],FSinfo->S_new[5],FSinfo->F_z,
                     FSinfo->S_new[0],FSinfo->S_new[1],FSinfo->F_x);
        fclose(f);


        sprintf(filen, "%s/FSI_Angle%2.2d.dat", d_fsipath, ibi);
        f = fopen(filen, "a");
        PetscFPrintf(PETSC_COMM_WORLD, f, 
                     "%d %le %le %le %le %le %le %le\n",
                     ti, FSinfo->S_ang_n[0],FSinfo->S_ang_n[1],FSinfo->M_x,
                     FSinfo->S_ang_r[0],FSinfo->S_ang_r[1],
                     FSinfo->S_ang_rm1[0],FSinfo->S_ang_rm1[1]);
        fclose(f);

        sprintf(filen, "%s/DATA_FSI%5.5d_%2.2d.dat", d_fsipath, ti, ibi);
        f = fopen(filen, "w");
        PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le \n", 
                     FSinfo->red_vel, FSinfo->damp, FSinfo->mu_s);      
        PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le \n", 
                     FSinfo->x_c, FSinfo->y_c, FSinfo->z_c);      
        PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le \n", 
                     FSinfo->F_x, FSinfo->F_y, FSinfo->F_z);      
        PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le \n", 
                     FSinfo->M_x, FSinfo->M_y, FSinfo->M_z);      
        for (i=0; i<6; i++) {
            PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le %le \n", 
                         FSinfo->S_new[i],FSinfo->S_old[i], 
                         FSinfo->S_real[i], FSinfo->S_realm1[i]);
            PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le %le \n", 
                         FSinfo->S_ang_n[i],FSinfo->S_ang_o[i], 
                         FSinfo->S_ang_r[i], FSinfo->S_ang_rm1[i]);
        }
        fclose(f);
    }
    return 0;
}

PetscErrorCode FSI::Restart(PetscInt ti)
{
    if (!d_immersed) return 0;
    if (!d_rstart_fsi) return 0;

    ReadFSI(ti);

    IBMNodes *ibm = d_ib->getIBMNodes(); 
    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++) {
        FSInfo *fsi = &d_fsi[ibi];
        if (d_movefsi) {
            ElementMoveFSITranslation(fsi, ibm+ibi);
            for (int i=0;i<6;i++){
                fsi[ibi].S_realm1[i]=fsi[ibi].S_real[i];
                fsi[ibi].S_real[i]=fsi[ibi].S_new[i];
            }
            for (int i=0; i<ibm[ibi].n_v; i++) {
                ibm[ibi].uold[i].x = fsi[ibi].S_real[1];
                ibm[ibi].uold[i].y = fsi[ibi].S_real[3];
                ibm[ibi].uold[i].z = fsi[ibi].S_real[5];
            }
            for (int i=0; i<ibm[ibi].n_v; i++) {
                ibm[ibi].urm1[i].x = fsi[ibi].S_realm1[1];
                ibm[ibi].urm1[i].y = fsi[ibi].S_realm1[3];
                ibm[ibi].urm1[i].z = fsi[ibi].S_realm1[5];
            }
        }
        if (d_rotatefsi || d_rotatefsi_noIBsearch) {
            fsi[ibi].x_c = d_x_r; 
            fsi[ibi].y_c = d_y_r;
            fsi[ibi].z_c = d_z_r;
    
            if (ibi==0 || ibi<d_NumberOfRotatingBodies) {
                ElementMoveFSIRotation(fsi, ibm+ibi, ti);
            } else {
                for (int i=0; i<ibm[ibi].n_v; i++) {
                    ibm[ibi].u[i].x = 0;
                    ibm[ibi].u[i].y = 0;
                    ibm[ibi].u[i].z = 0;
                    ibm[ibi].uold[i] = ibm[ibi].u[i];
                    ibm[ibi].urm1[i] = ibm[ibi].u[i];
               }
            }
            
            // if read ti, then will start for ti+1
            for (int i=0; i<6;i++){
                fsi[ibi].S_ang_rm1[i]=fsi[ibi].S_ang_r[i];
                fsi[ibi].S_ang_r[i]=fsi[ibi].S_ang_n[i];
            }

            fsi[ibi].F_x_real=fsi[ibi].F_x;
            fsi[ibi].F_y_real=fsi[ibi].F_y;
            fsi[ibi].F_z_real=fsi[ibi].F_z;

            fsi[ibi].M_x_rm3=fsi[ibi].M_x;
            fsi[ibi].M_y_rm3=fsi[ibi].M_y;
            fsi[ibi].M_z_rm3=fsi[ibi].M_z;

            fsi[ibi].M_x_rm2=fsi[ibi].M_x;
            fsi[ibi].M_y_rm2=fsi[ibi].M_y;
            fsi[ibi].M_z_rm2=fsi[ibi].M_z;

            fsi[ibi].M_x_real=fsi[ibi].M_x;
            fsi[ibi].M_y_real=fsi[ibi].M_y;
            fsi[ibi].M_z_real=fsi[ibi].M_z;
        }
    }

   return 0;
}


PetscErrorCode FSI::CopyToOld(PetscInt si)
{

    if (!d_changefsi) return 0;

    for (PetscInt ibi=0; ibi < d_NumberOfBodies; ibi++) {
        FSInfo *fsi = &d_fsi[ibi];
        for (int i=0;i<6;i++){
            fsi[ibi].S_old[i] = fsi[ibi].S_new[i];
            fsi[ibi].S_ang_o[i]=fsi[ibi].S_ang_n[i];
            if (si==1) {
                fsi[ibi].dS[i]=0.;
                fsi[ibi].atk=0.3;
            }
            fsi[ibi].dS_o[i]=fsi[ibi].dS[i];
           fsi[ibi].atk_o=fsi[ibi].atk;
        }
        if (si==2)
            fsi[ibi].atk_o=0.298;

    fsi[ibi].F_x_old=fsi[ibi].F_x;
    fsi[ibi].F_y_old=fsi[ibi].F_y;
    fsi[ibi].F_z_old=fsi[ibi].F_z;

    fsi[ibi].M_x_old=fsi[ibi].M_x;
    fsi[ibi].M_y_old=fsi[ibi].M_y;
    fsi[ibi].M_z_old=fsi[ibi].M_z;

    }

    return 0;
}


PetscErrorCode FSI::CopyLastStep()
{
    
    if (!d_changefsi) return 0;

    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++) {
        FSInfo *fsi = &d_fsi[ibi];

        for (int i=0;i<6;i++){
            fsi[ibi].S_realm1[i]=fsi[ibi].S_real[i];
            fsi[ibi].S_real[i]=fsi[ibi].S_new[i];

            fsi[ibi].S_ang_rm1[i]=fsi[ibi].S_ang_r[i];
            fsi[ibi].S_ang_r[i]=fsi[ibi].S_ang_n[i];
        }

        fsi[ibi].F_x_real=fsi[ibi].F_x;
        fsi[ibi].F_y_real=fsi[ibi].F_y;
        fsi[ibi].F_z_real=fsi[ibi].F_z;

        fsi[ibi].M_x_rm3=fsi[ibi].M_x_rm2;
        fsi[ibi].M_y_rm3=fsi[ibi].M_y_rm2;
        fsi[ibi].M_z_rm3=fsi[ibi].M_z_rm2;

        fsi[ibi].M_x_rm2=fsi[ibi].M_x_real;
        fsi[ibi].M_y_rm2=fsi[ibi].M_y_real;
        fsi[ibi].M_z_rm2=fsi[ibi].M_z_real;

        fsi[ibi].M_x_real=fsi[ibi].M_x;
        fsi[ibi].M_y_real=fsi[ibi].M_y;
        fsi[ibi].M_z_real=fsi[ibi].M_z;
    }

}

PetscErrorCode FSI::CalculatePosition(PetscInt ti, 
                                      PetscReal time)
{

    if (!d_movefsi) return 0;

    Vec lNvert = d_data->getlNvert();
    Vec Nvert = d_data->getNvert();

    //Calculate the updated position
    IBMNodes *ibm = d_ib->getIBMNodes(); 
    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++)
    {
        FSInfo *fsi = &d_fsi[ibi];
        CalculateFSIPosition(fsi, time);
    }

    //Detection
    CollisionDetectionOfCylinders();
   
    //Move the IBM elements
    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++)
    {
        FSInfo *fsi = &d_fsi[ibi];
        ElementMoveFSITranslation(fsi, ibm+ibi);
    }

    //This needs to be done before we do a new search
    VecSet(Nvert,0.);
    VecSet(lNvert,0.);
  
    //Now the IB Search only on rotating
    MPI_Barrier(PETSC_COMM_WORLD);
    d_ib->IBMSearchAdvanced(ti);
    MPI_Barrier(PETSC_COMM_WORLD);
    return 0;
}

PetscErrorCode FSI::CalculateFSIPosition(FSInfo *FSinfo,
                                         PetscReal time) 
{ 
    int i,j;
    int itr=23;
    PetscReal pi=3.141592654;
    PetscReal S_new[6],S_old[6],S_real[6],S_realm1[6];  
    PetscReal red_vel, damp, mu_s; // reduced vel, damping coeff, mass coeff
    PetscReal F_x,F_y,F_z; //Forces and Area
  
    PetscReal dtime = d_data->getDt();
    PetscReal dt = 0.5*d_data->getDt();

    // init values
    for (i=0;i<6;i++) {
        S_new[i] = FSinfo->S_real[i];
        S_real[i] = FSinfo->S_real[i];
        S_realm1[i] = FSinfo->S_realm1[i];
    }
  
    red_vel = FSinfo->red_vel;
    damp = FSinfo->damp;
    mu_s = FSinfo->mu_s;

    F_x = FSinfo->F_x;
    F_y = FSinfo->F_y;
    F_z = FSinfo->F_z;

    PetscPrintf(PETSC_COMM_WORLD, "...FSI  %le %le %le %le %le %le %le\n",
                red_vel,damp,mu_s,F_y,F_z, S_real[2],S_realm1[2] );

    // solve lin mom equ
    for (i=0; i<itr;i++) { 
        for (j=0;j<6;j++) {
            S_old[j]=S_new[j];
        }

        if (d_dgf_x) {
            S_new[0] = S_new[0] - dt/2./dtime*
                      (3.*S_new[0] - 4.*S_real[0]+S_realm1[0])+S_new[1]*dt; // x
            S_new[1] = S_new[1] - dt/2./dtime*
                       (3.*S_new[1]-4.*S_real[1]+S_realm1[1])+
                       dt*(-2.*damp*(red_vel)*S_new[1]
                       -(red_vel*red_vel)*S_old[0]
                       + mu_s*F_x); // dx/dt
        }
        if (d_dgf_y) {
            S_new[2] = S_new[2]-dt/2./dtime*
                      (3.*S_new[2]-4.*S_real[2]+S_realm1[2])+S_new[3]*dt; // y
            // dy/dt
            S_new[3] = S_new[3]-dt/2./dtime*
                       (3.*S_new[3]-4.*S_real[3]+S_realm1[3])+
                       dt*(-2.*damp*(red_vel)*S_new[3]
                       -(red_vel*red_vel)*S_old[2]
                       + mu_s*F_y);
        }
        if (d_dgf_z) {
            S_new[4] = S_new[4]-dt/2./dtime*
                       (3*S_new[4]-4*S_real[4]+S_realm1[4])+S_new[5]*dt; //z
            S_new[5] = S_new[5]-dt/2./dtime*(3*S_new[5]-4*S_real[5]+S_realm1[5])
                       +dt*(-2.*damp*(red_vel)*S_new[5]
                       -(red_vel*red_vel)*(S_old[4])
                       + mu_s*F_z); //dz/dt
        }

        // FSI convergence
        PetscPrintf(PETSC_COMM_WORLD, "FSI convergence y: %le  u_y:%le\n", 
                    S_new[2]-S_old[2],S_new[3]-S_old[3]);

    }
    
    // store results
    for (i=0;i<6;i++){
        FSinfo->S_new[i]=S_new[i];
    }

    // output values
    PetscPrintf(PETSC_COMM_WORLD, "z, dz/dt %le %le %le\n",
                S_new[4],S_new[5], F_z);
    PetscPrintf(PETSC_COMM_WORLD, "y, dy/dt %le %le %le\n",
                S_new[2],S_new[3], F_y);
    PetscPrintf(PETSC_COMM_WORLD, "x, dx/dt %le %le %le\n",
                S_new[0],S_new[1], F_x);

    return 0;
}

PetscErrorCode FSI::ElementMoveFSITranslation(FSInfo *FSinfo, IBMNodes *ibm)
{
    int n_v = ibm->n_v, n_elmt = ibm->n_elmt;

    PetscPrintf(PETSC_COMM_WORLD, "MOVE BODY x: %le  y:%le z:%le\n", 
                FSinfo->S_new[0],FSinfo->S_new[2],FSinfo->S_new[4]);

    for (int i=0; i<n_v; i++) {
        ibm->x_bp[i] = ibm->x_bp0[i]+(FSinfo->S_new[0]);
        ibm->y_bp[i] = ibm->y_bp0[i]+(FSinfo->S_new[2]);
        ibm->z_bp[i] = ibm->z_bp0[i]+(FSinfo->S_new[4]);
    }
  
    for (int i=0; i<n_v; i++) {
        ibm->u[i].x = FSinfo->S_new[1];
        ibm->u[i].y = FSinfo->S_new[3];
        ibm->u[i].z = FSinfo->S_new[5];
    }

    return 0;
}


PetscErrorCode FSI::CollisionDetectionOfCylinders()

{
    PetscReal x_c,y_c,z_c;
    PetscReal x_c2,y_c2,z_c2;
    PetscReal l_c;
    PetscReal n_x,n_y,n_z; //collision direction
    PetscReal v_x=0.,v_y=0.,v_z=0.;
    PetscReal v_x2=0.,v_y2=0.,v_z2=0.;
    PetscReal v_n1, v_t1; //vel in collision direction
    PetscReal v_n2, v_t2;

    int ibi,ibi2;

    for (ibi=0;ibi<d_NumberOfBodies;ibi++) {
        for (ibi2=ibi+1;ibi2<d_NumberOfBodies;ibi2++){
      
            x_c=d_fsi[ibi].x_c  ; y_c=d_fsi[ibi].y_c  ; z_c=d_fsi[ibi].z_c;
            x_c2=d_fsi[ibi2].x_c; y_c2=d_fsi[ibi2].y_c; z_c2=d_fsi[ibi2].z_c;

            l_c=sqrt((x_c-x_c2)*(x_c-x_c2) + 
                     (y_c-y_c2)*(y_c-y_c2) +
                     (z_c-z_c2)*(z_c-z_c2));

            if (l_c < 1.) { 
                PetscPrintf(PETSC_COMM_WORLD, 
                            "Collision Detected!!!! cylinder %d with %d\n", 
                            ibi, ibi2);
    
                // Collision Direction
                n_x = 0.;//(x_c-x_c2)/l_c;
                n_y = (y_c-y_c2)/l_c;
                n_z = (z_c-z_c2)/l_c;
    
                /* Move the 2nd cyl to the 1D distance of 1st Cyl */
                d_fsi[ibi2].x_c= x_c + n_x;
                d_fsi[ibi2].y_c= y_c + n_y;
                d_fsi[ibi2].z_c= z_c + n_z;

                /* Change the Vel to the collsion Vel! */

                v_x=d_fsi[ibi].S_new[1];
                v_y=d_fsi[ibi].S_new[3];
                v_z=d_fsi[ibi].S_new[5];
                v_n1 = v_x*n_x + v_y*n_y + v_z*n_z;
                v_t1 = v_x*n_x + v_y*n_z - v_z*n_y;
      
                v_x2=d_fsi[ibi2].S_new[1];
                v_y2=d_fsi[ibi2].S_new[3];
                v_z2=d_fsi[ibi2].S_new[5];
                v_n2 = v_x2*n_x + v_y2*n_y + v_z2*n_z;
                v_t2 = v_x2*n_x + v_y2*n_z - v_z2*n_y;

                PetscPrintf(PETSC_COMM_WORLD,
                            "Velocity: cyl1 %le %le %le  cyl2 %le %le %le\n", 
                            v_x,v_y,v_z,v_x2,v_y2,v_z2);

                v_x = v_n2*n_x + v_t1 *n_x;
                v_y = v_n2*n_y + v_t1 *n_z;
                v_z = v_n2*n_z - v_t1 *n_y;

                v_x2 = v_n1*n_x + v_t2 *n_x;
                v_y2 = v_n1*n_y + v_t2 *n_z;
                v_z2 = v_n1*n_z - v_t2 *n_y;

                d_fsi[ibi].S_new[1]=v_x;
                d_fsi[ibi].S_new[3]=v_y;
                d_fsi[ibi].S_new[5]=v_z;

                d_fsi[ibi2].S_new[1]=v_x2;
                d_fsi[ibi2].S_new[3]=v_y2;
                d_fsi[ibi2].S_new[5]=v_z2;

                PetscPrintf(PETSC_COMM_WORLD, 
                            "Collision Velocity: cyl1 %le %le %le "
                            " cyl2 %le %le %le\n", 
                            v_x,v_y,v_z,v_x2,v_y2,v_z2);
            }
        }
    }
    return 0;
}

PetscErrorCode FSI::CalculateRotation(PetscInt ti, 
                                      PetscReal time)
{

    if (!d_rotatefsi || d_rotatefsi_noIBsearch) return 0;

    Vec lNvert = d_data->getlNvert();
    Vec Nvert = d_data->getNvert();

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    IBMNodes *ibm = d_ib->getIBMNodes(); 

    //This needs to be done before we do a new search
    VecSet(Nvert,0.);
    VecSet(lNvert,0.);

    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++)
    {
        FSInfo *fsi = &d_fsi[ibi];
        
        if (ibi==0 || ibi<d_NumberOfRotatingBodies) {
            CalculateFSIRotation(fsi);
            ElementMoveFSIRotation(fsi, ibm+ibi, ti);
  
            //Write some output 
            if (!rank) {
                PetscReal S_ang_n[6];
                FILE *f;
                char filen[80];
                sprintf(filen, "%s/ang_rfsi.dat", d_fsipath);
                if (ti==1 && ibi==0) {
                    f = fopen(filen, "w");
                    PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=\"I\", \"time\", \"ang_x\", \"ang_y\", \"ang_z\", \"angvel_x\", \"angvel_y\", \"angvel_z\", \"M_x\", \"M_y\", \"M_z\" \n");
                } else f = fopen(filen, "a");
                for (int i=0;i<6;i++){
                     S_ang_n[i]=fsi->S_ang_n[i];
                }
                double M_x = 0.5*(fsi->M_x + fsi->M_x_real)+fsi->Mx_applied;
                double M_y = 0.5*(fsi->M_y + fsi->M_y_real)+fsi->My_applied;
                double M_z = 0.5*(fsi->M_z + fsi->M_z_real)+fsi->Mx_applied;

                PetscFPrintf(PETSC_COMM_WORLD, f, 
                             "%d %d %le %le %le %le %le %le %le %le %le \n",
                             ibi, ti, S_ang_n[0], S_ang_n[2], S_ang_n[4], 
                             S_ang_n[1], S_ang_n[3], S_ang_n[5], 
                             M_x, M_y, M_z);
                fclose(f);
            }

           //Now the IB Search only on rotating
           MPI_Barrier(PETSC_COMM_WORLD);
           d_ib->IBMSearchAdvanced1(ibm+ibi, ibi, ti);
           MPI_Barrier(PETSC_COMM_WORLD);
        }
    }
   
    return 0;
}

PetscErrorCode FSI::CalculateFSIRotation(FSInfo *FSinfo)
{  
    int i,itr=12,j,nv;
    PetscReal pi=3.141592654;
    PetscReal S_ang_n[6],S_ang_r[6],S_ang_rm1[6],S_ang_o[6];  
    PetscReal red_vel, damp, mu_s; // reduced vel, damping coeff, mass coeff
    PetscReal M_x,M_y,M_z; //Forces and Area
    PetscReal rx,ry,rz;
    PetscReal x_c=FSinfo->x_c, y_c=FSinfo->y_c, z_c=FSinfo->z_c;
    PetscReal wx=0., wy=0., wz=0.;
    PetscReal w=.5, wf=1.;
    PetscReal Mdpdn_x;

    PetscReal dt = d_data->getDt(); 

    // init values
    for (i=0;i<6;i++) {
        S_ang_o[i] = FSinfo->S_ang_o[i];
        S_ang_r[i] = FSinfo->S_ang_r[i];
        S_ang_rm1[i] = FSinfo->S_ang_rm1[i];
    }
  
    red_vel = FSinfo->red_vel;
    damp = FSinfo->damp;
    mu_s = FSinfo->mu_s;
    Mdpdn_x = FSinfo->Mdpdn_x;

    M_x = 0.5*(FSinfo->M_x + FSinfo->M_x_real);
    M_y = 0.5*(FSinfo->M_y + FSinfo->M_y_real);
    M_z = 0.5*(FSinfo->M_z + FSinfo->M_z_real);

    double Mx_a=FSinfo->Mx_applied;
    double My_a=FSinfo->My_applied;
    double Mz_a=FSinfo->Mz_applied;

    M_x+=Mx_a;
    M_y+=My_a;
    M_z+=Mz_a;

    // solve Ang mom equ
    if (d_rotdir==0) {
        S_ang_n[1] = (1-damp*dt)/(1.+damp*dt)*S_ang_r[1]
                      + dt/(1.+damp*dt)*(mu_s*M_x); // w=w_r + int(M/Idt)
        //ang=ang_r+w_avedt
        S_ang_n[0] = S_ang_r[0]+0.5*(S_ang_n[1]+S_ang_r[1])*dt; 
    }
    if (d_rotdir==1) {
        S_ang_n[3] = S_ang_r[3]+ dt*(mu_s*M_y); // w=w_r + int(M/Idt)
        //ang=ang_r+w_avedt
        S_ang_n[2] = S_ang_r[2]+0.5*(S_ang_n[3]+S_ang_r[3])*dt; 
    }
    if (d_rotdir==2) {
        S_ang_n[5] = S_ang_r[5]+ dt*(mu_s*M_z); // w=w_r + int(M/Idt)

        //ang=ang_r+w_avedt
        S_ang_n[4] = S_ang_r[4]+0.5*(S_ang_n[5]+S_ang_r[5])*dt; 
    }

  
    // Relaxation
    if (d_sisteps) {
        FSinfo->atk=0.;
        for (i=1;i<6;i+=2) {
            FSinfo->dS[i]=S_ang_o[i]-S_ang_n[i];
    
            if (fabs(FSinfo->dS[i]-FSinfo->dS_o[i])>1e-8  &&
                FSinfo->atk_o!=0.3) {

                FSinfo->atk+=(FSinfo->dS[i])/
                             (FSinfo->dS_o[i]-FSinfo->dS[i]);
            }
        }
        FSinfo->atk=FSinfo->atk_o+(FSinfo->atk_o-1)*FSinfo->atk;
        if (FSinfo->atk>.9) FSinfo->atk=.9;
        if (FSinfo->atk<-.2) FSinfo->atk=-0.2;
    
        w=1.-FSinfo->atk;
        for (i=1;i<6;i+=2){
             S_ang_n[i]=w*S_ang_n[i]+(1.-w)*S_ang_o[i];
             S_ang_n[i-1]=S_ang_r[i-1]+0.5*(S_ang_n[i]+S_ang_r[i])*dt;
        }
    }

    // store results
    for (i=0;i<6;i++){
        FSinfo->S_ang_n[i]=S_ang_n[i];
    }
 
    return 0;
}

PetscErrorCode FSI::ElementMoveFSIRotation(FSInfo *FSinfo, 
                                           IBMNodes *ibm, 
                                           PetscInt ti)
{

    int i;
    int n1e, n2e, n3e;
    int n_v = ibm->n_v, n_elmt = ibm->n_elmt;
    PetscReal dx12, dy12, dz12, dx13, dy13, dz13, dr;
    PetscReal rx,ry,rz;
    PetscReal x_c=FSinfo->x_c, y_c=FSinfo->y_c, z_c=FSinfo->z_c;
    PetscReal rot_angle;
    PetscReal dt = d_data->getDt();

    if (!d_prescribed_rotation) {
        if (d_rotdir==0) rot_angle = FSinfo->S_ang_n[0]; 
        if (d_rotdir==1) rot_angle = FSinfo->S_ang_n[2]; 
        if (d_rotdir==2) rot_angle = FSinfo->S_ang_n[4]; 
    } else {
        rot_angle = d_angvel * dt * (ti+d_ti_lastsave);        

        if (ti==d_data->get_tistart() && d_rstart_fsi) {
            if (d_rotdir==0) rot_angle = FSinfo->S_ang_n[0]; 
            if (d_rotdir==1) rot_angle = FSinfo->S_ang_n[2]; 
            if (d_rotdir==2) rot_angle = FSinfo->S_ang_n[4]; 
        }

        if (d_rotdir==0) FSinfo->S_ang_n[0]=rot_angle; 
        if (d_rotdir==1) FSinfo->S_ang_n[2]=rot_angle; 
        if (d_rotdir==2) FSinfo->S_ang_n[4]=rot_angle; 

        if (d_rotdir==0) FSinfo->S_ang_n[1]=d_angvel; 
        if (d_rotdir==1) FSinfo->S_ang_n[3]=d_angvel; 
        if (d_rotdir==2) FSinfo->S_ang_n[5]=d_angvel; 
    }

    for (i=0; i<n_v; i++) {
        RotateXYZ(ti+d_ti_lastsave, dt, d_angvel, x_c, y_c, z_c, 
                  ibm->x_bp0[i], ibm->y_bp0[i], ibm->z_bp0[i], 
                  &ibm->x_bp[i], &ibm->y_bp[i], &ibm->z_bp[i], 
                  &rot_angle);

        double x1, y1, z1;
        double x2, y2, z2;
        double tmp, eps=1.e-6;

        double rot_angle1 = d_angvel * dt * (ti+d_ti_lastsave-1);
        double rot_angle2 = d_angvel * dt * (ti+d_ti_lastsave+1);

        RotateXYZ(ti+d_ti_lastsave-1, dt, d_angvel, x_c, y_c, z_c, 
                  ibm->x_bp0[i], ibm->y_bp0[i], ibm->z_bp0[i], 
                  &x1, &y1, &z1, 
                  &rot_angle1);
        RotateXYZ(ti+d_ti_lastsave+1, dt, d_angvel, x_c, y_c, z_c, 
                  ibm->x_bp0[i], ibm->y_bp0[i], ibm->z_bp0[i], 
                  &x2, &y2, &z2, 
                  &rot_angle2);
        ibm->u[i].x = (x2 - x1) / dt * 0.5;
        ibm->u[i].y = (y2 - y1) / dt * 0.5;
        ibm->u[i].z = (z2 - z1) / dt * 0.5;
        
        if (ti==d_data->get_tistart()) {
            double rot_angle1 = d_angvel * dt * (ti+d_ti_lastsave-1);
            double rot_angle2 = d_angvel * dt * (ti+d_ti_lastsave+1);

            RotateXYZ(ti+d_ti_lastsave-1-eps, dt, d_angvel, x_c, y_c, z_c, 
                      ibm->x_bp0[i], ibm->y_bp0[i], ibm->z_bp0[i], 
                      &x1, &y1, &z1, 
                      &rot_angle1);
            RotateXYZ(ti+d_ti_lastsave-1+eps, dt, d_angvel, x_c, y_c, z_c, 
                      ibm->x_bp0[i], ibm->y_bp0[i], ibm->z_bp0[i], 
                      &x2, &y2, &z2, 
                      &rot_angle2);

            ibm->uold[i].x = (x2 - x1) / dt * 0.5;
            ibm->uold[i].y = (y2 - y1) / dt * 0.5;
            ibm->uold[i].z = (z2 - z1) / dt * 0.5;
        }
    }
    
    for (i=0; i<n_elmt; i++) {

       n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; n3e =ibm->nv3[i];
       dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e]; 
       dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e]; 
       dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e]; 
    
       dx13 = ibm->x_bp[n3e] - ibm->x_bp[n1e]; 
       dy13 = ibm->y_bp[n3e] - ibm->y_bp[n1e]; 
       dz13 = ibm->z_bp[n3e] - ibm->z_bp[n1e]; 
    
       ibm->nf_x[i] = dy12 * dz13 - dz12 * dy13;
       ibm->nf_y[i] = -dx12 * dz13 + dz12 * dx13;
       ibm->nf_z[i] = dx12 * dy13 - dy12 * dx13;

       dr = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
                 ibm->nf_y[i]*ibm->nf_y[i] + 
                 ibm->nf_z[i]*ibm->nf_z[i]);
    
       ibm->nf_x[i]/=dr; ibm->nf_y[i]/=dr; ibm->nf_z[i]/=dr;
    
       // ns = nf x k
       if ((((1.-ibm->nf_z[i])<=1e-6 )&&((-1.+ibm->nf_z[i])<1e-6))||
           (((ibm->nf_z[i]+1.)<=1e-6 )&&((-1.-ibm->nf_z[i])<1e-6))) {

            ibm->ns_x[i] = 1.;
            ibm->ns_y[i] = 0.;
            ibm->ns_z[i] = 0 ;

            // nt = ns x nf
            ibm->nt_x[i] = 0.;
            ibm->nt_y[i] = 1.;
            ibm->nt_z[i] = 0.;
        } else {
            ibm->ns_x[i] =  ibm->nf_y[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
                                               ibm->nf_y[i]*ibm->nf_y[i]);
            ibm->ns_y[i] = -ibm->nf_x[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
                                               ibm->nf_y[i]*ibm->nf_y[i]);
            ibm->ns_z[i] = 0 ;

            // nt = ns x nf
            ibm->nt_x[i] = -ibm->nf_x[i]*ibm->nf_z[i] / 
                            sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
                                 ibm->nf_y[i]*ibm->nf_y[i]);
            ibm->nt_y[i] = -ibm->nf_y[i]*ibm->nf_z[i] / 
                            sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
                                 ibm->nf_y[i]*ibm->nf_y[i]);
            ibm->nt_z[i] = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
                                ibm->nf_y[i]*ibm->nf_y[i]);
        }

        ibm->dA[i] = dr/2.;

        // Calc the center of the element
        ibm->cent_x[i]= (ibm->x_bp[n1e]+ibm->x_bp[n2e]+ibm->x_bp[n3e])/3.;
        ibm->cent_y[i]= (ibm->y_bp[n1e]+ibm->y_bp[n2e]+ibm->y_bp[n3e])/3.;
        ibm->cent_z[i]= (ibm->z_bp[n1e]+ibm->z_bp[n2e]+ibm->z_bp[n3e])/3.;

    }
  
    for (i=0; i<n_v; i++) {
        rx = ibm->x_bp[i]-x_c;
        ry = ibm->y_bp[i]-y_c;
        rz = ibm->z_bp[i]-z_c;  
    }
  
    return 0;
}

PetscErrorCode FSI::RotateXYZ(double ti, double dt, double angvel, 
                              double x_c, double y_c, double z_c, 
                              double x_bp0, double y_bp0, double z_bp0, 
                              double *x_bp, double *y_bp, double *z_bp, 
                              double *rot_angle)
{
    if(d_rotdir==0) { // rotate around x-axis
        double rot_x = angvel * dt * ti;
        
        *x_bp = x_bp0;
        *y_bp = y_c + (y_bp0-y_c)*cos(rot_x) - (z_bp0-z_c)*sin(rot_x);
        *z_bp = z_c + (y_bp0-y_c)*sin(rot_x) + (z_bp0-z_c)*cos(rot_x);
        
        *rot_angle = rot_x;
    }
    else if(d_rotdir==1) { // rotate around y-axis
        double rot_y = angvel * dt * ti;

        *y_bp = y_bp0;
        *z_bp = z_c + (z_bp0-z_c)*cos(rot_y) - (x_bp0-x_c)*sin(rot_y);
        *x_bp = x_c + (z_bp0-z_c)*sin(rot_y) + (x_bp0-x_c)*cos(rot_y);
        
        *rot_angle = rot_y;
    }
    else { // rotate around z-axis
        double rot_z = angvel * dt * ti;
        
        *z_bp = z_bp0;
        *x_bp = x_c + (x_bp0-x_c)*cos(rot_z) - (y_bp0-y_c)*sin(rot_z);
        *y_bp = y_c + (x_bp0-x_c)*sin(rot_z) + (y_bp0-y_c)*cos(rot_z);
        
        *rot_angle = rot_z;
    }

    return 0;
}

PetscErrorCode FSI::CalculateForces(PetscInt ti, PetscReal time)
{
    if (!d_immersed) return 0;

    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++)
    {
        FSInfo *fsi = &d_fsi[ibi];
        CalculateForces1(fsi, ibi, ti, time);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "...Finished Calculate Forces\n");
   
    return 0;
} 


PetscErrorCode FSI::CalculateForces1(FSInfo *fsi, PetscInt ibi, 
                                     PetscInt ti, PetscReal time)
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
    int lxs, lxe, lys, lye, lzs, lze;

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    PetscReal     rei= 1./d_data->getRe();

    int i, j, k, elmt;
    int nv1,nv2,nv3;
    Cmpnts uelmt;
    PetscReal sb;
    PetscReal Tow_ws,Tow_wt,Tow_wn;
    PetscReal Tow_x, Tow_y, Tow_z;
    Cmpnts ***coor, ***ucat, ***cent;
    PetscReal ***p, ***nvert;
    PetscReal dx,dy,dz, dx1,dy1,dz1;
    PetscReal dwdz,dwdy,dwdx;
    PetscReal dvdz,dvdy,dvdx;
    PetscReal dudz,dudy,dudx;
    PetscReal Txx,Tyy,Tzz;
    PetscReal Tzy,Tzx,Tyx;
    PetscReal nfx,nfy,nfz;
    PetscReal nsx,nsy,nsz;
    PetscReal ntx,nty,ntz;
    PetscReal F_px,F_py,F_pz,Ap_x,Ap_y,Ap_z,Ap_t; //Forces and Area
    PetscReal F_nx,F_ny,F_nz,An_x,An_y,An_z,An_t; 
    PetscReal Cp_nx,Cp_ny,Cp_nz; //Pressure Forces - side
    PetscReal Cs_nx,Cs_ny,Cs_nz; //Surface Forces - side
    PetscReal Cp_px,Cp_py,Cp_pz; //Pressure Forces + side
    PetscReal Cs_px,Cs_py,Cs_pz; //Surface Forces + side
    PetscReal F_xSum,F_ySum,F_zSum,A_totSum; //Surface Force
    PetscReal F_pxSum,F_pySum,F_pzSum; //Surface Force
    PetscReal F_nxSum,F_nySum,F_nzSum; //Surface Force
    PetscReal Ap_xSum,Ap_ySum,Ap_zSum,Ap_tSum; // + side
    PetscReal An_xSum,An_ySum,An_zSum,An_tSum; // - side
    PetscReal A_xSum,A_ySum,A_zSum,A_tSum;
    PetscReal Cp_xSum,Cp_ySum,Cp_zSum; //Pressure Force
    PetscReal Cp_pxSum,Cp_pySum,Cp_pzSum; //Pressure Force
    PetscReal Cp_nxSum,Cp_nySum,Cp_nzSum; //Pressure Force

    // Moments
    PetscReal MF_px,MF_py,MF_pz; //Forces for Moment Calc
    PetscReal MF_nx,MF_ny,MF_nz; //Forces for Moment Calc
    PetscReal M_nx,M_ny,M_nz;   //Moments
    PetscReal M_px,M_py,M_pz;   //Moments
    PetscReal r_x,r_y,r_z;   //Anchor dist
    PetscReal x, y, z;       //cell coord
    PetscReal X_c,Y_c,Z_c;   //center of rotation coord
    PetscReal M_xSum,M_ySum,M_zSum; //Surface Mom on all processors
    PetscReal M_pxSum,M_pySum,M_pzSum; //Surface Mom on all processors
    PetscReal M_nxSum,M_nySum,M_nzSum; //Surface Mom on all processors
    PetscReal Iap_x,Iap_y,Iap_z;
    PetscReal Ian_x,Ian_y,Ian_z;
    PetscReal Iap_xSum,Iap_ySum,Iap_zSum; // + side
    PetscReal Ian_xSum,Ian_ySum,Ian_zSum; // - side
    PetscReal Ia_xSum,Ia_ySum,Ia_zSum;
    PetscReal Pw_nx,Pw_ny,Pw_nz;   //Power
    PetscReal Pw_px,Pw_py,Pw_pz;   //Power
    PetscReal Pw_nxSum,Pw_nySum,Pw_nzSum;   //Power
    PetscReal Pw_pxSum,Pw_pySum,Pw_pzSum;   //Power
    PetscReal Pw_xSum,Pw_ySum,Pw_zSum;

    PetscReal A_x,A_y,A_z,Atot;
    PetscReal u_x,u_y,u_z;
    Cmpnts ***csi,***eta,***zet;
    PetscReal csi1,csi2,csi3;
    PetscReal eta1,eta2,eta3;
    PetscReal zet1,zet2,zet3;
    PetscReal ***iaj,***jaj,***kaj;

    PetscReal dr;
    PetscReal MFdpdn_px,MFdpdn_py,MFdpdn_pz;
    PetscReal Mdpdn_px,Mdpdn_py,Mdpdn_pz;
    PetscReal Mdpdn_pxSum,Mdpdn_pySum,Mdpdn_pzSum;
    PetscReal MFdpdn_nx,MFdpdn_ny,MFdpdn_nz;
    PetscReal Mdpdn_nx,Mdpdn_ny,Mdpdn_nz;
    PetscReal Mdpdn_nxSum,Mdpdn_nySum,Mdpdn_nzSum;
    PetscReal Mdpdn_xSum,Mdpdn_ySum,Mdpdn_zSum; 

    IBMInfo *ibminfo;
    IBMListNode *current;

    PetscReal pi = 3.141592653589793;
    PetscReal v_side, dzz, az0, Pw_side, Thrust, Drag;
    PetscReal Pw_sideSum, ThrustSum, DragSum, efficiency;

    /*   Init var */
    F_px=0.;F_py=0.;F_pz=0.;
    F_nx=0.;F_ny=0.;F_nz=0.;
    Ap_x=0.;Ap_y=0.;Ap_z=0.;Ap_t=0.;
    An_x=0.;An_y=0.;An_z=0.;An_t=0.;
    Cp_px=0.;Cp_py=0.;Cp_pz=0.;
    Cs_px=0.;Cs_py=0.;Cs_pz=0.;
    Cp_nx=0.;Cp_ny=0.;Cp_nz=0.;
    Cs_nx=0.;Cs_ny=0.;Cs_nz=0.;

    M_px=0.;M_py=0.;M_pz=0.;
    M_nx=0.;M_ny=0.;M_nz=0.;
    Iap_x=0.;Iap_y=0.;Iap_z=0.;
    Ian_x=0.;Ian_y=0.;Ian_z=0.;
    Pw_px=0.;Pw_py=0.;Pw_pz=0.;
    Pw_nx=0.;Pw_ny=0.;Pw_nz=0.;

    Pw_side=0.; Thrust=0.; Drag=0.;
  
    Mdpdn_px=0.;Mdpdn_py=0.;Mdpdn_pz=0.;
    Mdpdn_nx=0.;Mdpdn_ny=0.;Mdpdn_nz=0.;
    MFdpdn_px=0.;MFdpdn_py=0.;MFdpdn_pz=0.;
    MFdpdn_nx=0.;MFdpdn_ny=0.;MFdpdn_nz=0.;

    X_c=fsi->x_c; Y_c=fsi->y_c; Z_c=fsi->z_c;
    X_c=d_x_r, Y_c=d_y_r, Z_c=d_z_r;

    PetscPrintf(PETSC_COMM_WORLD, "...Calculuting Forces\n");
    PetscPrintf(PETSC_COMM_WORLD, 
                "Mu: %le X_c: %le %le %le\n",
                 rei, X_c, Y_c, Z_c);

    Vec Coor;
    Vec lUcat = d_data->getlUcat();
    Vec lNvert = d_data->getlNvert();
    Vec lP = d_data->getlP();
    
    Vec lCent = d_grid->getlCent();
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec IAj = d_grid->getlIAj();
    Vec JAj = d_grid->getlJAj();
    Vec KAj = d_grid->getlKAj();
   
 

    DMGetCoordinatesLocal(da, &Coor);
    DMDAVecGetArray(fda, Coor, &coor);
    DMDAVecGetArray(fda, lCent, &cent);
    DMDAVecGetArray(fda, lUcat, &ucat);
    DMDAVecGetArray(da, lP, &p);
    DMDAVecGetArray(da, lNvert, &nvert);

    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, IAj, &iaj);
    DMDAVecGetArray(da, JAj, &jaj);
    DMDAVecGetArray(da, KAj, &kaj);
  
    IBMNodes *ibm = d_ib->getIBMNodes();
    ibm += ibi;

    /* Loop around all ibm nodes */
    IBMList *ibmlist = d_ib->getIBMList();
    current = ibmlist[ibi].head;
    while (current) {
 
        ibminfo = &current->ibm_intp;
        current = current->next; 
        i = ibminfo->ni; j= ibminfo->nj; k = ibminfo->nk;
        elmt = ibminfo->cell; // closest ibm element
        sb = ibminfo->d_s;

        // normal 
        nfx=ibm->nf_x[elmt];
        nfy=ibm->nf_y[elmt];
        nfz=ibm->nf_z[elmt];

        // 1st bi-normal of nf
        nsx=ibm->ns_x[elmt];
        nsy=ibm->ns_y[elmt];
        nsz=ibm->ns_z[elmt];

        // 2nd bi-normal of nf
        ntx=ibm->nt_x[elmt];
        nty=ibm->nt_y[elmt];
        ntz=ibm->nt_z[elmt];

        // nodes of closest ibm elmnt
        nv1=ibm->nv1[elmt];
        nv2=ibm->nv2[elmt];
        nv3=ibm->nv3[elmt];

        //velocity of the closest elmnt
        uelmt.x = (ibm->u[nv1].x+ibm->u[nv2].x+ibm->u[nv3].x)/3.;
        uelmt.y = (ibm->u[nv1].y+ibm->u[nv2].y+ibm->u[nv3].y)/3.;
        uelmt.z = (ibm->u[nv1].z+ibm->u[nv2].z+ibm->u[nv3].z)/3.;
    
        if (i>=lxs && i<lxe && j>=lys && j<lye && k>=lzs && k<lze) {         
            rei= 1./d_data->getRe();
            MF_px=0.;MF_py=0.;MF_pz=0.;
            MF_nx=0.;MF_ny=0.;MF_nz=0.;
            MFdpdn_px=0.;MFdpdn_py=0.;MFdpdn_pz=0.;
            MFdpdn_nx=0.;MFdpdn_ny=0.;MFdpdn_nz=0.;


            /* Shear Stresses (2nd & 1st order) and Shear Force */

            if (nvert[k+1][j][i]<2.5 && nvert[k-1][j][i]<2.5 && 
                k+1<mz-1 && k-1>0) {

                zet1 = 0.25*(zet[k][j][i].x+zet[k-1][j][i].x)*
                              (kaj[k][j][i]+kaj[k-1][j][i]);
                zet2 = 0.25*(zet[k][j][i].y+zet[k-1][j][i].y)*
                              (kaj[k][j][i]+kaj[k-1][j][i]);
                zet3 = 0.25*(zet[k][j][i].z+zet[k-1][j][i].z)*
                              (kaj[k][j][i]+kaj[k-1][j][i]);
                dwdz = (ucat[k+1][j][i].z - ucat[k-1][j][i].z)/2.*zet3;
                dvdz = (ucat[k+1][j][i].y - ucat[k-1][j][i].y)/2.*zet3;
                dudz = (ucat[k+1][j][i].x - ucat[k-1][j][i].x)/2.*zet3;

                dwdy = (ucat[k+1][j][i].z - ucat[k-1][j][i].z)/2.*zet2;
                dvdy = (ucat[k+1][j][i].y - ucat[k-1][j][i].y)/2.*zet2;
                dudy = (ucat[k+1][j][i].x - ucat[k-1][j][i].x)/2.*zet2;

                dwdx = (ucat[k+1][j][i].z - ucat[k-1][j][i].z)/2.*zet1;
                dvdx = (ucat[k+1][j][i].y - ucat[k-1][j][i].y)/2.*zet1;
                dudx = (ucat[k+1][j][i].x - ucat[k-1][j][i].x)/2.*zet1;

            } else if (nvert[k+1][j][i]<2.5 && k+1<mz-1) {
                zet1 = (zet[k][j][i].x)*kaj[k][j][i];
                zet2 = (zet[k][j][i].y)*kaj[k][j][i];
                zet3 = (zet[k][j][i].z)*kaj[k][j][i];

                dwdz = (ucat[k+1][j][i].z - ucat[k][j][i].z)*zet3;
                dvdz = (ucat[k+1][j][i].y - ucat[k][j][i].y)*zet3;
                dudz = (ucat[k+1][j][i].x - ucat[k][j][i].x)*zet3;

                dwdy = (ucat[k+1][j][i].z - ucat[k][j][i].z)*zet2;
                dvdy = (ucat[k+1][j][i].y - ucat[k][j][i].y)*zet2;
                dudy = (ucat[k+1][j][i].x - ucat[k][j][i].x)*zet2;

                dwdx = (ucat[k+1][j][i].z - ucat[k][j][i].z)*zet1;
                dvdx = (ucat[k+1][j][i].y - ucat[k][j][i].y)*zet1;
                dudx = (ucat[k+1][j][i].x - ucat[k][j][i].x)*zet1;

            } else if (nvert[k-1][j][i]<2.5 && k-1>0){
                zet1 = (zet[k-1][j][i].x)*kaj[k-1][j][i];
                zet2 = (zet[k-1][j][i].y)*kaj[k-1][j][i];
                zet3 = (zet[k-1][j][i].z)*kaj[k-1][j][i];

                dwdz = (ucat[k][j][i].z - ucat[k-1][j][i].z)*zet3;
                dvdz = (ucat[k][j][i].y - ucat[k-1][j][i].y)*zet3;
                dudz = (ucat[k][j][i].x - ucat[k-1][j][i].x)*zet3;

                dwdy = (ucat[k][j][i].z - ucat[k-1][j][i].z)*zet2;
                dvdy = (ucat[k][j][i].y - ucat[k-1][j][i].y)*zet2;
                dudy = (ucat[k][j][i].x - ucat[k-1][j][i].x)*zet2;

                dwdx = (ucat[k][j][i].z - ucat[k-1][j][i].z)*zet1;
                dvdx = (ucat[k][j][i].y - ucat[k-1][j][i].y)*zet1;
                dudx = (ucat[k][j][i].x - ucat[k-1][j][i].x)*zet1;

            } else {
                dwdz = 0.;
                dvdz = 0.;
                dudz = 0.;

                dwdy = 0.;
                dvdy = 0.;
                dudy = 0.;

                dwdx = 0.;
                dvdx = 0.;
                dudx = 0.;
            }


            if (nvert[k][j+1][i]<2.5 && j+1<my-1 && 
                nvert[k][j-1][i]<2.5 && j-1>0) {

                eta1 = 0.25*(eta[k][j][i].x+eta[k][j-1][i].x)*
                              (jaj[k][j][i]+jaj[k][j-1][i]);
                eta2 = 0.25*(eta[k][j][i].y+eta[k][j-1][i].y)*
                              (jaj[k][j][i]+jaj[k][j-1][i]);
                eta3 = 0.25*(eta[k][j][i].z+eta[k][j-1][i].z)*
                              (jaj[k][j][i]+jaj[k][j-1][i]);
                dwdz += (ucat[k][j+1][i].z - ucat[k][j-1][i].z)/2.*eta3;
                dvdz += (ucat[k][j+1][i].y - ucat[k][j-1][i].y)/2.*eta3;
                dudz += (ucat[k][j+1][i].x - ucat[k][j-1][i].x)/2.*eta3;

                dwdy += (ucat[k][j+1][i].z - ucat[k][j-1][i].z)/2.*eta2;
                dvdy += (ucat[k][j+1][i].y - ucat[k][j-1][i].y)/2.*eta2;
                dudy += (ucat[k][j+1][i].x - ucat[k][j-1][i].x)/2.*eta2;

                dwdx += (ucat[k][j+1][i].z - ucat[k][j-1][i].z)/2.*eta1;
                dvdx += (ucat[k][j+1][i].y - ucat[k][j-1][i].y)/2.*eta1;
                dudx += (ucat[k][j+1][i].x - ucat[k][j-1][i].x)/2.*eta1;

            } else if (nvert[k][j+1][i]<2.5 && j+1<my-1) {
                eta1 = eta[k][j][i].x*jaj[k][j][i];
                eta2 = eta[k][j][i].y*jaj[k][j][i];
                eta3 = eta[k][j][i].z*jaj[k][j][i];

                dwdz += (ucat[k][j+1][i].z - ucat[k][j][i].z)*eta3;
                dvdz += (ucat[k][j+1][i].y - ucat[k][j][i].y)*eta3;
                dudz += (ucat[k][j+1][i].x - ucat[k][j][i].x)*eta3;

                dwdy += (ucat[k][j+1][i].z - ucat[k][j][i].z)*eta2;
                dvdy += (ucat[k][j+1][i].y - ucat[k][j][i].y)*eta2;
                dudy += (ucat[k][j+1][i].x - ucat[k][j][i].x)*eta2;

                dwdx += (ucat[k][j+1][i].z - ucat[k][j][i].z)*eta1;
                dvdx += (ucat[k][j+1][i].y - ucat[k][j][i].y)*eta1;
                dudx += (ucat[k][j+1][i].x - ucat[k][j][i].x)*eta1;

            } else if (nvert[k][j-1][i]<2.5 && j-1>0){
                eta1 = eta[k][j-1][i].x*jaj[k][j-1][i];
                eta2 = eta[k][j-1][i].y*jaj[k][j-1][i];
                eta3 = eta[k][j-1][i].z*jaj[k][j-1][i];

                dwdz += (ucat[k][j][i].z - ucat[k][j-1][i].z)*eta3;
                dvdz += (ucat[k][j][i].y - ucat[k][j-1][i].y)*eta3;
                dudz += (ucat[k][j][i].x - ucat[k][j-1][i].x)*eta3;

                dwdy += (ucat[k][j][i].z - ucat[k][j-1][i].z)*eta2;
                dvdy += (ucat[k][j][i].y - ucat[k][j-1][i].y)*eta2;
                dudy += (ucat[k][j][i].x - ucat[k][j-1][i].x)*eta2;

                dwdx += (ucat[k][j][i].z - ucat[k][j-1][i].z)*eta1;
                dvdx += (ucat[k][j][i].y - ucat[k][j-1][i].y)*eta1;
                dudx += (ucat[k][j][i].x - ucat[k][j-1][i].x)*eta1;
            } 

            if (nvert[k][j][i+1]<2.5 && i+1<mx-1 && 
                 nvert[k][j][i-1]<2.5 && i-1>0) {

                csi1 = 0.25*(csi[k][j][i].x+csi[k][j][i-1].x)*
                              (iaj[k][j][i]+iaj[k][j][i-1]);
                csi2 = 0.25*(csi[k][j][i].y+csi[k][j][i-1].y)*
                              (iaj[k][j][i]+iaj[k][j][i-1]);
                csi3 = 0.25*(csi[k][j][i].z+csi[k][j][i-1].z)*
                              (iaj[k][j][i]+iaj[k][j][i-1]);

                dwdz += (ucat[k][j][i+1].z - ucat[k][j][i-1].z)/2.*csi3;
                dvdz += (ucat[k][j][i+1].y - ucat[k][j][i-1].y)/2.*csi3;
                dudz += (ucat[k][j][i+1].x - ucat[k][j][i-1].x)/2.*csi3;

                dwdy += (ucat[k][j][i+1].z - ucat[k][j][i-1].z)/2.*csi2;
                dvdy += (ucat[k][j][i+1].y - ucat[k][j][i-1].y)/2.*csi2;
                dudy += (ucat[k][j][i+1].x - ucat[k][j][i-1].x)/2.*csi2;

                dwdx += (ucat[k][j][i+1].z - ucat[k][j][i-1].z)/2.*csi1;
                dvdx += (ucat[k][j][i+1].y - ucat[k][j][i-1].y)/2.*csi1;
                dudx += (ucat[k][j][i+1].x - ucat[k][j][i-1].x)/2.*csi1;

            } else if (nvert[k][j][i+1]<2.5 && i+1<mx-1) {
                csi1 = csi[k][j][i].x*iaj[k][j][i];
                csi2 = csi[k][j][i].y*iaj[k][j][i];
                csi3 = csi[k][j][i].z*iaj[k][j][i];

                dwdz += (ucat[k][j][i+1].z - ucat[k][j][i].z)*csi3;
                dvdz += (ucat[k][j][i+1].y - ucat[k][j][i].y)*csi3;
                dudz += (ucat[k][j][i+1].x - ucat[k][j][i].x)*csi3;

                dwdy += (ucat[k][j][i+1].z - ucat[k][j][i].z)*csi2;
                dvdy += (ucat[k][j][i+1].y - ucat[k][j][i].y)*csi2;
                dudy += (ucat[k][j][i+1].x - ucat[k][j][i].x)*csi2;

                dwdx += (ucat[k][j][i+1].z - ucat[k][j][i].z)*csi1;
                dvdx += (ucat[k][j][i+1].y - ucat[k][j][i].y)*csi1;
                dudx += (ucat[k][j][i+1].x - ucat[k][j][i].x)*csi1;

            } else if (nvert[k][j][i-1]<2.5 && i-1>0){
                csi1 = csi[k][j][i-1].x*iaj[k][j][i-1];
                csi2 = csi[k][j][i-1].y*iaj[k][j][i-1];
                csi3 = csi[k][j][i-1].z*iaj[k][j][i-1];

                dwdz += (ucat[k][j][i].z - ucat[k][j][i-1].z)*csi3;
                dvdz += (ucat[k][j][i].y - ucat[k][j][i-1].y)*csi3;
                dudz += (ucat[k][j][i].x - ucat[k][j][i-1].x)*csi3;

                dwdy += (ucat[k][j][i].z - ucat[k][j][i-1].z)*csi2;
                dvdy += (ucat[k][j][i].y - ucat[k][j][i-1].y)*csi2;
                dudy += (ucat[k][j][i].x - ucat[k][j][i-1].x)*csi2;

                dwdx += (ucat[k][j][i].z - ucat[k][j][i-1].z)*csi1;
                dvdx += (ucat[k][j][i].y - ucat[k][j][i-1].y)*csi1;
                dudx += (ucat[k][j][i].x - ucat[k][j][i-1].x)*csi1;
            }
      
            Tzz = rei * (dwdz + dwdz);
            Tyy = rei * (dvdy + dvdy);
            Txx = rei * (dudx + dudx);
            Tzy = rei * (dwdy + dvdz);
            Tzx = rei * (dwdx + dudz);
            Tyx = rei * (dvdx + dudy);

            MF_px=0.;MF_py=0.;MF_pz=0.;
            MF_nx=0.;MF_ny=0.;MF_nz=0.;

            r_x = 0.;
            r_y = 0.;
            r_z = 0.;

            /*       K +  */

            if (nvert[k+1][j][i]<0.9 ){
                x = 0.5*(cent[k][j][i].x+cent[k+1][j][i].x);
                y = 0.5*(cent[k][j][i].y+cent[k+1][j][i].y);
                z = 0.5*(cent[k][j][i].z+cent[k+1][j][i].z);

                r_x = x-X_c;
                r_y = y-Y_c;
                r_z = z-Z_c;

                A_z=zet[k  ][j][i].z;
                A_y=zet[k  ][j][i].y;
                A_x=zet[k  ][j][i].x;
                  

                Cp_pz += -p[k+1][j][i]*A_z;
                MF_pz += -p[k+1][j][i]*A_z;
                Ap_z  += A_z;
                Iap_x -=   r_y*A_z;
                Iap_y -= - r_x*A_z;
                MFdpdn_pz += sb*(nfy*r_z-nfz*r_y)*A_z;

                if (MF_pz<0) 
                  Thrust += MF_pz;
                else 
                  Drag += MF_pz;

                Cp_py += -p[k+1][j][i]*A_y;
                MF_py += -p[k+1][j][i]*A_y;
                Ap_y  += A_y;
                Iap_x -=  - r_z*A_y;
                Iap_z -=    r_x*A_y;
                MFdpdn_py += sb*(nfy*r_z-nfz*r_y)*A_y;

                Cp_px += -p[k+1][j][i]*A_x;
                MF_px += -p[k+1][j][i]*A_x;
                Ap_x  += A_x;
                Iap_z -=   - r_y*A_x;
                Iap_y -= -(- r_z*A_x);
                MFdpdn_px += sb*(nfy*r_z-nfz*r_y)*A_x;

                Cs_pz += Tzz*A_z ;
                Cs_py += Tzy*A_z ;
                Cs_px += Tzx*A_z ;
              
                MF_pz += Tzz*A_z ;
                MF_py += Tzy*A_z ;
                MF_px += Tzx*A_z ;

                if (Tzz*A_z<0) 
                  Thrust += Tzz*A_z;
                else 
                  Drag += Tzz*A_z;

                Cs_pz += Tzy*A_y ;
                Cs_py += Tyy*A_y ;
                Cs_px += Tyx*A_y ;
              
                MF_pz += Tzy*A_y ;
                MF_py += Tyy*A_y ;
                MF_px += Tyx*A_y ;

                if (Tzy*A_y<0) 
                  Thrust += Tzy*A_y;
                else 
                  Drag += Tzy*A_y;

                Cs_pz += Tzx*A_x;
                Cs_py += Tyx*A_x;
                Cs_px += Txx*A_x;
              
                MF_pz += Tzx*A_x;
                MF_py += Tyx*A_x;
                MF_px += Txx*A_x;

                if (Tzx*A_x<0) 
                  Thrust += Tzx*A_x;
                else 
                  Drag += Tzx*A_x;

                M_px -=   r_y*MF_pz - r_z*MF_py;//
                M_py -= -(r_x*MF_pz - r_z*MF_px);
                M_pz -=   r_x*MF_py - r_y*MF_px;
    
                Atot = sqrt(A_x*A_x+A_y*A_y+A_z*A_z);
                Ap_t += Atot;

                u_x = uelmt.x;
                u_y = uelmt.y;
                u_z = uelmt.z;

                Pw_px += MF_px * u_x ;//tot;
                Pw_py += MF_py * u_y ;//* Atot;
                Pw_pz += MF_pz * u_z ;//* Atot;

            }   

            MF_px=0.;MF_py=0.;MF_pz=0.;
            MF_nx=0.;MF_ny=0.;MF_nz=0.;

            /*       K -  */
   
            if (nvert[k-1][j][i]<0.9) {
                A_z=zet[k-1][j][i].z;
                A_y=zet[k-1][j][i].y;
                A_x=zet[k-1][j][i].x;
                  
                x = 0.5*(cent[k][j][i].x+cent[k-1][j][i].x);
                y = 0.5*(cent[k][j][i].y+cent[k-1][j][i].y);
                z = 0.5*(cent[k][j][i].z+cent[k-1][j][i].z);

                r_x = x-X_c;
                r_y = y-Y_c;
                r_z = z-Z_c;

                Cp_nz +=  p[k-1][j][i]*A_z;
                MF_nz +=  p[k-1][j][i]*A_z;
                An_z  += A_z;
                Ian_x -=   r_y*A_z;
                Ian_y -= - r_x*A_z;
                MFdpdn_nz += -sb*(nfy*r_z-nfz*r_y)*A_z;

                if (MF_nz<0) 
                  Thrust += MF_nz;
                else 
                  Drag += MF_nz;

                Cp_ny +=  p[k-1][j][i]*A_y;
                MF_ny +=  p[k-1][j][i]*A_y;
                An_y  += A_y;
                Ian_x -=  - r_z*A_y;
                Ian_z -=    r_x*A_y;
                MFdpdn_ny += -sb*(nfy*r_z-nfz*r_y)*A_y;

                Cp_nx +=  p[k-1][j][i]*A_x;
                MF_nx +=  p[k-1][j][i]*A_x;
                An_x  += A_x;
                Ian_z -=   - r_y*A_x;
                Ian_y -= -(- r_z*A_x);
                MFdpdn_nx += -sb*(nfy*r_z-nfz*r_y)*A_x;

                Cs_nz -= Tzz*A_z ;
                Cs_ny -= Tzy*A_z ;
                Cs_nx -= Tzx*A_z ;
            
                MF_nz -= Tzz*A_z ;
                MF_ny -= Tzy*A_z ;
                MF_nx -= Tzx*A_z ;
            
                if (-Tzz*A_z<0) 
                  Thrust -= Tzz*A_z;
                else 
                  Drag -= Tzz*A_z;

                Cs_nz -= Tzy*A_y ;
                Cs_ny -= Tyy*A_y ;
                Cs_nx -= Tyx*A_y ;
            
                MF_nz -= Tzy*A_y ;
                MF_ny -= Tyy*A_y ;
                MF_nx -= Tyx*A_y ;

                if (-Tzy*A_y<0) 
                  Thrust -= Tzy*A_y;
                else 
                  Drag -= Tzy*A_y;
            
                Cs_nz -= Tzx*A_x;
                Cs_ny -= Tyx*A_x;
                Cs_nx -= Txx*A_x;
            
                MF_nz -= Tzx*A_x;
                MF_ny -= Tyx*A_x;
                MF_nx -= Txx*A_x;

                if (-Tzx*A_x<0) 
                  Thrust -= Tzx*A_x;
                else 
                  Drag -= Tzx*A_x;

                M_nx -=   r_y*MF_nz - r_z*MF_ny;//
                M_ny -= -(r_x*MF_nz - r_z*MF_nx);
                M_nz -=   r_x*MF_ny - r_y*MF_nx;

                Atot = sqrt(A_x*A_x+A_y*A_y+A_z*A_z);
                An_t += Atot;

                u_x = uelmt.x;
                u_y = uelmt.y;
                u_z = uelmt.z;

                Pw_nx += MF_nx * u_x ;//* Atot;
                Pw_ny += MF_ny * u_y ;//* Atot;
                Pw_nz += MF_nz * u_z ;//* Atot;

            }
     
            MF_px=0.;MF_py=0.;MF_pz=0.;
            MF_nx=0.;MF_ny=0.;MF_nz=0.;

            /*       j +  */

            if (nvert[k][j+1][i]<0.9 ){
                A_z=eta[k][j  ][i].z;
                A_y=eta[k][j  ][i].y;
                A_x=eta[k][j  ][i].x;
                  

                x = 0.5*(cent[k][j][i].x+cent[k][j+1][i].x);
                y = 0.5*(cent[k][j][i].y+cent[k][j+1][i].y);
                z = 0.5*(cent[k][j][i].z+cent[k][j+1][i].z);

                r_x = x-X_c;
                r_y = y-Y_c;
                r_z = z-Z_c;

                Cp_pz += -p[k][j+1][i]*A_z;
                MF_pz += -p[k][j+1][i]*A_z;
                Ap_z  += A_z;
                Iap_x -=   r_y*A_z;
                Iap_y -= - r_x*A_z;
                MFdpdn_pz += sb*(nfy*r_z-nfz*r_y)*A_z;


                Cp_py += -p[k][j+1][i]*A_y;
                MF_py += -p[k][j+1][i]*A_y;
                Ap_y  += A_y;
                Iap_x -=  - r_z*A_y;
                Iap_z -=    r_x*A_y;
                MFdpdn_py += sb*(nfy*r_z-nfz*r_y)*A_y;


                Cp_px += -p[k][j+1][i]*A_x;
                MF_px += -p[k][j+1][i]*A_x;
                Ap_x  += A_x;
                Iap_z -=   - r_y*A_x;
                Iap_y -= -(- r_z*A_x);
                MFdpdn_px += sb*(nfy*r_z-nfz*r_y)*A_x;

                Cs_pz += Tzz*A_z ;
                Cs_py += Tzy*A_z ;
                Cs_px += Tzx*A_z ;
              
                MF_pz += Tzz*A_z ;
                MF_py += Tzy*A_z ;
                MF_px += Tzx*A_z ;

                if (Tzz*A_z<0) 
                  Thrust += Tzz*A_z;
                else 
                  Drag += Tzz*A_z;

                Cs_pz += Tzy*A_y ;
                Cs_py += Tyy*A_y ;
                Cs_px += Tyx*A_y ;
              
                MF_pz += Tzy*A_y ;
                MF_py += Tyy*A_y ;
                MF_px += Tyx*A_y ;

                if (Tzy*A_y<0) 
                  Thrust += Tzy*A_y;
                else 
                  Drag += Tzy*A_y;

                Cs_pz += Tzx*A_x;
                Cs_py += Tyx*A_x;
                Cs_px += Txx*A_x;
              
                MF_pz += Tzx*A_x;
                MF_py += Tyx*A_x;
                MF_px += Txx*A_x;
    
                if (Tzx*A_x<0) 
                  Thrust += Tzx*A_x;
                else 
                  Drag += Tzx*A_x;

                M_px -=   r_y*MF_pz - r_z*MF_py;//
                M_py -= -(r_x*MF_pz - r_z*MF_px);
                M_pz -=   r_x*MF_py - r_y*MF_px;
    
                Atot = sqrt(A_x*A_x+A_y*A_y+A_z*A_z);
                Ap_t += Atot;
    
                u_x = uelmt.x;
                u_y = uelmt.y;
                u_z = uelmt.z;
    
                Pw_px += MF_px * u_x ;//* Atot;
                Pw_py += MF_py * u_y ;//* Atot;
                Pw_pz += MF_pz * u_z ;//* Atot;
    
            }

            MF_px=0.;MF_py=0.;MF_pz=0.;
            MF_nx=0.;MF_ny=0.;MF_nz=0.;

            /*       j -  */

            if (nvert[k][j-1][i]<0.9) {  
                A_z=eta[k][j-1][i].z;
                A_y=eta[k][j-1][i].y;
                A_x=eta[k][j-1][i].x;
                  
                x = 0.5*(cent[k][j][i].x+cent[k][j-1][i].x);
                y = 0.5*(cent[k][j][i].y+cent[k][j-1][i].y);
                z = 0.5*(cent[k][j][i].z+cent[k][j-1][i].z);

                r_x = x-X_c;
                r_y = y-Y_c;
                r_z = z-Z_c;

                Cp_nz +=  p[k][j-1][i]*A_z;
                MF_nz +=  p[k][j-1][i]*A_z;
                An_z  += A_z;
                Ian_x -=   r_y*A_z;
                Ian_y -= - r_x*A_z;
                MFdpdn_nz += -sb*(nfy*r_z-nfz*r_y)*A_z;

                Cp_ny +=  p[k][j-1][i]*A_y;
                MF_ny +=  p[k][j-1][i]*A_y;
                An_y  += A_y;
                Ian_x -=  - r_z*A_y;
                Ian_z -=    r_x*A_y;
                MFdpdn_ny += -sb*(nfy*r_z-nfz*r_y)*A_y;

                Cp_nx +=  p[k][j-1][i]*A_x;
                MF_nx +=  p[k][j-1][i]*A_x;
                An_x  += A_x;
                Ian_z -=   - r_y*A_x;
                Ian_y -= -(- r_z*A_x);
                MFdpdn_nx += -sb*(nfy*r_z-nfz*r_y)*A_x;

                Cs_nz -= Tzz*A_z ;
                Cs_ny -= Tzy*A_z ;
                Cs_nx -= Tzx*A_z ;
            
                MF_nz -= Tzz*A_z ;
                MF_ny -= Tzy*A_z ;
                MF_nx -= Tzx*A_z ;
                
                if (-Tzz*A_z<0) 
                  Thrust -= Tzz*A_z;
                else 
                  Drag -= Tzz*A_z;

                Cs_nz -= Tzy*A_y ;
                Cs_ny -= Tyy*A_y ;
                Cs_nx -= Tyx*A_y ;
            
                MF_nz -= Tzy*A_y ;
                MF_ny -= Tyy*A_y ;
                MF_nx -= Tyx*A_y ;

                if (-Tzy*A_y<0) 
                  Thrust -= Tzy*A_y;
                else 
                  Drag -= Tzy*A_y;
            
                Cs_nz -= Tzx*A_x;
                Cs_ny -= Tyx*A_x;
                Cs_nx -= Txx*A_x;
            
                MF_nz -= Tzx*A_x;
                MF_ny -= Tyx*A_x;
                MF_nx -= Txx*A_x;

                if (-Tzx*A_x<0) 
                  Thrust -= Tzx*A_x;
                else 
                  Drag -= Tzx*A_x;

                M_nx -=   r_y*MF_nz - r_z*MF_ny;//
                M_ny -= -(r_x*MF_nz - r_z*MF_nx);
                M_nz -=   r_x*MF_ny - r_y*MF_nx;
    
                Atot = sqrt(A_x*A_x+A_y*A_y+A_z*A_z);
                An_t += Atot;

                u_x = uelmt.x;
                u_y = uelmt.y;
                u_z = uelmt.z;

                Pw_nx += MF_nx * u_x ;//* Atot;
                Pw_ny += MF_ny * u_y ;//* Atot;
                Pw_nz += MF_nz * u_z ;//* Atot;

            }

            MF_px=0.;MF_py=0.;MF_pz=0.;
            MF_nx=0.;MF_ny=0.;MF_nz=0.;

            /*       i +  */

            if (nvert[k][j][i+1]<0.9){
                A_z=csi[k][j][i].z;
                A_y=csi[k][j][i].y;
                A_x=csi[k][j][i].x;
                      

                x = 0.5*(cent[k][j][i].x+cent[k][j][i+1].x);
                y = 0.5*(cent[k][j][i].y+cent[k][j][i+1].y);
                z = 0.5*(cent[k][j][i].z+cent[k][j][i+1].z);

                r_x = x-X_c;
                r_y = y-Y_c;
                r_z = z-Z_c;

                Cp_pz += -p[k][j][i+1]*A_z;
                MF_pz += -p[k][j][i+1]*A_z;
                Ap_z  += A_z;
                Iap_x -=   r_y*A_z;
                Iap_y -= - r_x*A_z;
                MFdpdn_pz += sb*(nfy*r_z-nfz*r_y)*A_z;


                Cp_py += -p[k][j][i+1]*A_y;
                MF_py += -p[k][j][i+1]*A_y;
                Ap_y  += A_y;
                Iap_x -=  - r_z*A_y;
                Iap_z -=    r_x*A_y;
                MFdpdn_py += sb*(nfy*r_z-nfz*r_y)*A_y;


                Cp_px += -p[k][j][i+1]*A_x;
                MF_px += -p[k][j][i+1]*A_x;
                Ap_x  += A_x;
                Iap_z -=   - r_y*A_x;
                Iap_y -= -(- r_z*A_x);
                MFdpdn_px += sb*(nfy*r_z-nfz*r_y)*A_x;

                Cs_pz += Tzz*A_z ;
                Cs_py += Tzy*A_z ;
                Cs_px += Tzx*A_z ;
              
                MF_pz += Tzz*A_z ;
                MF_py += Tzy*A_z ;
                MF_px += Tzx*A_z ;

                if (Tzz*A_z<0) 
                  Thrust += Tzz*A_z;
                else 
                  Drag += Tzz*A_z;

                Cs_pz += Tzy*A_y ;
                Cs_py += Tyy*A_y ;
                Cs_px += Tyx*A_y ;
              
                MF_pz += Tzy*A_y ;
                MF_py += Tyy*A_y ;
                MF_px += Tyx*A_y ;

                if (Tzy*A_y<0) 
                  Thrust += Tzy*A_y;
                else 
                  Drag += Tzy*A_y;

                Cs_pz += Tzx*A_x;
                Cs_py += Tyx*A_x;
                Cs_px += Txx*A_x;
              
                MF_pz += Tzx*A_x;
                MF_py += Tyx*A_x;
                MF_px += Txx*A_x;

                if (Tzx*A_x<0) 
                  Thrust += Tzx*A_x;
                else 
                  Drag += Tzx*A_x;

                M_px -=   r_y*MF_pz - r_z*MF_py;//
                M_py -= -(r_x*MF_pz - r_z*MF_px);
                M_pz -=   r_x*MF_py - r_y*MF_px;

                Atot = sqrt(A_x*A_x+A_y*A_y+A_z*A_z);
                Ap_t += Atot;

                u_x = uelmt.x;
                u_y = uelmt.y;
                u_z = uelmt.z;

                Pw_px += MF_px * u_x ;//* Atot;
                Pw_py += MF_py * u_y ;//* Atot;
                Pw_pz += MF_pz * u_z ;//* Atot;

            }

            MF_px=0.;MF_py=0.;MF_pz=0.;
            MF_nx=0.;MF_ny=0.;MF_nz=0.;

            /*       i -  */

            if  (nvert[k][j][i-1]<0.9) { 
                A_z=csi[k][j][i-1].z;
                A_y=csi[k][j][i-1].y;
                A_x=csi[k][j][i-1].x;
                  

                x = 0.5*(cent[k][j][i].x+cent[k][j][i-1].x);
                y = 0.5*(cent[k][j][i].y+cent[k][j][i-1].y);
                z = 0.5*(cent[k][j][i].z+cent[k][j][i-1].z);

                r_x = x-X_c;
                r_y = y-Y_c;
                r_z = z-Z_c;

                Cp_nz +=  p[k][j][i-1]*A_z;
                MF_nz +=  p[k][j][i-1]*A_z;
                An_z  += A_z;
                Ian_x -=   r_y*A_z;
                Ian_y -= - r_x*A_z;
                MFdpdn_nz += -sb*(nfy*r_z-nfz*r_y)*A_z;

                Cp_ny +=  p[k][j][i-1]*A_y;
                MF_ny +=  p[k][j][i-1]*A_y;
                An_y  += A_y;
                Ian_x -=  - r_z*A_y;
                Ian_z -=    r_x*A_y;
                MFdpdn_ny += -sb*(nfy*r_z-nfz*r_y)*A_y;

                Cp_nx +=  p[k][j][i-1]*A_x;
                MF_nx +=  p[k][j][i-1]*A_x;
                An_x  += A_x;
                Ian_z -=   - r_y*A_x;
                Ian_y -= -(- r_z*A_x);
                MFdpdn_nx += -sb*(nfy*r_z-nfz*r_y)*A_x;

                Cs_nz -= Tzz*A_z ;
                Cs_ny -= Tzy*A_z ;
                Cs_nx -= Tzx*A_z ;
            
                MF_nz -= Tzz*A_z ;
                MF_ny -= Tzy*A_z ;
                MF_nx -= Tzx*A_z ;
            
                if (-Tzz*A_z<0) 
                  Thrust -= Tzz*A_z;
                else 
                  Drag -= Tzz*A_z;

                Cs_nz -= Tzy*A_y ;
                Cs_ny -= Tyy*A_y ;
                Cs_nx -= Tyx*A_y ;
            
                MF_nz -= Tzy*A_y ;
                MF_ny -= Tyy*A_y ;
                MF_nx -= Tyx*A_y ;

                if (-Tzy*A_y<0) 
                  Thrust -= Tzy*A_y;
                else 
                  Drag -= Tzy*A_y;
            
                Cs_nz -= Tzx*A_x;
                Cs_ny -= Tyx*A_x;
                Cs_nx -= Txx*A_x;
            
                MF_nz -= Tzx*A_x;
                MF_ny -= Tyx*A_x;
                MF_nx -= Txx*A_x;

                if (-Tzx*A_x<0) 
                  Thrust -= Tzx*A_x;
                else 
                  Drag -= Tzx*A_x;
            
                M_nx -=   r_y*MF_nz - r_z*MF_ny;//
                M_ny -= -(r_x*MF_nz - r_z*MF_nx);
                M_nz -=   r_x*MF_ny - r_y*MF_nx;

                Atot = sqrt(A_x*A_x+A_y*A_y+A_z*A_z);
                An_t += Atot;

                u_x = uelmt.x;
                u_y = uelmt.y;
                u_z = uelmt.z;

                Pw_nx += MF_nx * u_x ;//* Atot;
                Pw_ny += MF_ny * u_y ;//* Atot;
                Pw_nz += MF_nz * u_z ;//* Atot;

            }

            Mdpdn_px -=   r_y*MFdpdn_pz - r_z*MFdpdn_py;//
            Mdpdn_py -= -(r_x*MFdpdn_pz - r_z*MFdpdn_px);
            Mdpdn_pz -=   r_x*MFdpdn_py - r_y*MFdpdn_px;

            Mdpdn_nx -=   r_y*MFdpdn_nz - r_z*MFdpdn_ny;//
            Mdpdn_ny -= -(r_x*MFdpdn_nz - r_z*MFdpdn_nx);
            Mdpdn_nz -=   r_x*MFdpdn_ny - r_y*MFdpdn_nx;
      
        } // if ibm node in CPU
         
    } //End of Loop ibm nodes 


    /*   Total Force on each processor */
    F_px = Cp_px + Cs_px; 
    F_py = Cp_py + Cs_py;
    F_pz = Cp_pz + Cs_pz;

    F_nx = Cp_nx + Cs_nx; 
    F_ny = Cp_ny + Cs_ny;
    F_nz = Cp_nz + Cs_nz;

    /*   Global Sum */
  
    GlobalSum_Root(&F_px, &F_pxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&F_py, &F_pySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&F_pz, &F_pzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&F_nx, &F_nxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&F_ny, &F_nySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&F_nz, &F_nzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Ap_x, &Ap_xSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Ap_y, &Ap_ySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Ap_z, &Ap_zSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&An_x, &An_xSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&An_y, &An_ySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&An_z, &An_zSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Cp_nx, &Cp_nxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Cp_ny, &Cp_nySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Cp_nz, &Cp_nzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Cp_px, &Cp_pxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Cp_py, &Cp_pySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Cp_pz, &Cp_pzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&M_px, &M_pxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&M_py, &M_pySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&M_pz, &M_pzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&M_nx, &M_nxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&M_ny, &M_nySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&M_nz, &M_nzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Iap_x, &Iap_xSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Iap_y, &Iap_ySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Iap_z, &Iap_zSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Ian_x, &Ian_xSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Ian_y, &Ian_ySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Ian_z, &Ian_zSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Mdpdn_px, &Mdpdn_pxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Mdpdn_py, &Mdpdn_pySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Mdpdn_pz, &Mdpdn_pzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Mdpdn_nx, &Mdpdn_nxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Mdpdn_ny, &Mdpdn_nySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Mdpdn_nz, &Mdpdn_nzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Pw_px, &Pw_pxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Pw_py, &Pw_pySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Pw_pz, &Pw_pzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Pw_nx, &Pw_nxSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Pw_ny, &Pw_nySum, PETSC_COMM_WORLD);
    GlobalSum_Root(&Pw_nz, &Pw_nzSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Ap_t, &Ap_tSum, PETSC_COMM_WORLD);
    GlobalSum_Root(&An_t, &An_tSum, PETSC_COMM_WORLD);

    GlobalSum_Root(&Thrust, &ThrustSum , PETSC_COMM_WORLD);
    GlobalSum_Root(&Drag  , &DragSum   , PETSC_COMM_WORLD);
    GlobalSum_Root(&Pw_side,&Pw_sideSum, PETSC_COMM_WORLD);

    /*   Scale Check later !!!!! */

    A_xSum = 0.5 * (Ap_xSum + An_xSum);
    A_ySum = 0.5 * (Ap_ySum + An_ySum);
    A_zSum = 0.5 * (Ap_zSum + An_zSum);
    A_tSum = (Ap_tSum + An_tSum);

    F_xSum = F_pxSum + F_nxSum;
    F_ySum = F_pySum + F_nySum;
    F_zSum = F_pzSum + F_nzSum;

    if (fabs(A_xSum)>1e-6)
        F_xSum=F_xSum/A_xSum*2.;
    if (fabs(A_ySum)>1e-6)
        F_ySum=F_ySum/A_ySum*2.;
    if (fabs(A_zSum)>1e-6)
        F_zSum=F_zSum/A_zSum*2.;

    Cp_xSum = Cp_pxSum + Cp_nxSum;
    Cp_ySum = Cp_pySum + Cp_nySum;
    Cp_zSum = Cp_pzSum + Cp_nzSum;

    if (fabs(A_xSum)>1e-6)
        Cp_xSum=Cp_xSum/A_xSum*2.;
    if (fabs(A_ySum)>1e-6)
        Cp_ySum=Cp_ySum/A_ySum*2.;
    if (fabs(A_zSum)>1e-6)
        Cp_zSum=Cp_zSum/A_zSum*2.;    

    Ia_xSum = 0.5 * (Iap_xSum - Ian_xSum);
    Ia_ySum = 0.5 * (Iap_ySum - Ian_ySum);
    Ia_zSum = 0.5 * (Iap_zSum - Ian_zSum);

    M_xSum = M_pxSum + M_nxSum;
    M_ySum = M_pySum + M_nySum;
    M_zSum = M_pzSum + M_nzSum;

    Pw_xSum = (Pw_pxSum + Pw_nxSum);/// A_tSum;
    Pw_ySum = (Pw_pySum + Pw_nySum);/// A_tSum;
    Pw_zSum = (Pw_pzSum + Pw_nzSum);/// A_tSum;

    efficiency= Cp_zSum/(Cp_zSum+Pw_xSum);


    Mdpdn_xSum = Mdpdn_pxSum + Mdpdn_nxSum;
    Mdpdn_ySum = Mdpdn_pySum + Mdpdn_nySum;
    Mdpdn_zSum = Mdpdn_pzSum + Mdpdn_nzSum;


    A_totSum = Ap_xSum + Ap_ySum + Ap_zSum;

    /*   store results in fsi */
    fsi->F_x = F_xSum; fsi->F_y = F_ySum; fsi->F_z = F_zSum;
    fsi->A_tot = A_totSum;
    fsi->M_x = M_xSum; fsi->M_y = M_ySum; fsi->M_z = M_zSum;
    fsi->Mdpdn_x = Mdpdn_xSum; 
    fsi->Mdpdn_y = Mdpdn_ySum; 
    fsi->Mdpdn_z = Mdpdn_zSum;

    /*   output values */
    PetscPrintf(PETSC_COMM_WORLD, 
                "F_x,F_y,F_z:, %le %le %le Az %le %le Ay %le %le\n",
                F_xSum,F_ySum,F_zSum,Ap_zSum,An_zSum,Ap_ySum,An_ySum);
    PetscPrintf(PETSC_COMM_WORLD, 
                "M_x,M_y,M_z:, %le %le %le Ia_x %le %le Ip_y %le %le\n",
                M_xSum,M_ySum,M_zSum,Iap_xSum,Ian_xSum,Iap_ySum,Ian_ySum);
    PetscPrintf(PETSC_COMM_WORLD, 
                "Mdpdn_x,Mdpdn_y,Mdpdn_z:, %le %le %le\n",
                Mdpdn_xSum,Mdpdn_ySum,Mdpdn_zSum);

    int rank=0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (!rank) {
        FILE *f;
        char filen[80];
        sprintf(filen, "%s/Force_Coeff_SI%2.2d_0.dat",d_fsipath,ibi);
        f = fopen(filen, "a");
        PetscFPrintf(PETSC_COMM_WORLD, f, 
                     "%d %le %le %le %le %le %le %le %le %le\n",
                     ti, F_xSum, F_ySum, F_zSum, 
                     Cp_xSum,Cp_ySum,Cp_zSum, A_xSum,A_ySum,A_zSum);
        fclose(f);

        sprintf(filen, "%s/Momt_Coeff_SI%2.2d_0.dat",d_fsipath,ibi);
        f = fopen(filen, "a");
        PetscFPrintf(PETSC_COMM_WORLD, f, 
                     "%d %le %le %le %le %le %le\n",
                     ti, M_xSum, M_ySum, M_zSum, 
                     fsi->M_x_old, fsi->M_x_real, fsi->Mdpdn_x);
        fclose(f);

      sprintf(filen, "%s/Power_SI%2.2d_0.dat",d_fsipath,ibi);
      f = fopen(filen, "a");
      PetscFPrintf(PETSC_COMM_WORLD, f,
                   "%d %le %le %le %le %le %le %le\n",
                   ti, efficiency,Pw_sideSum,
                   Pw_xSum,Pw_ySum,Pw_zSum,F_zSum,A_tSum);
      fclose(f);

    }

    DMDAVecRestoreArray(fda, lCent, &cent);
    DMDAVecRestoreArray(fda, Coor, &coor);
    DMDAVecRestoreArray(fda, lUcat, &ucat);
    DMDAVecRestoreArray(da, lP, &p);
    DMDAVecRestoreArray(da, lNvert, &nvert);

    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(da, IAj, &iaj);
    DMDAVecRestoreArray(da, JAj, &jaj);
    DMDAVecRestoreArray(da, KAj, &kaj);


    return 0;
}

void FSI::ReadFromInput()
{
    PetscOptionsGetReal(PETSC_NULL, "-red_vel", &d_red_vel, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-damp", &d_damp, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-mu_s", &d_mu_s, PETSC_NULL);

    //Center of the fsi
    PetscOptionsGetReal(PETSC_NULL, "-x_c", &d_x_c, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-y_c", &d_y_c, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-z_c", &d_z_c, PETSC_NULL);

    //Center of rotation
    PetscOptionsGetReal(PETSC_NULL, "-x_r", &d_x_r, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-y_r", &d_y_r, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-z_r", &d_z_r, PETSC_NULL);


    PetscOptionsGetReal(PETSC_NULL, "-Mx_a", &d_Mx_applied, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-My_a", &d_My_applied, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-Mz_a", &d_Mz_applied, PETSC_NULL);

    PetscOptionsGetReal(PETSC_NULL, "-Max_xbc", &d_Max_xbc, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-Min_xbd", &d_Min_xbc, PETSC_NULL);

    PetscOptionsGetInt(PETSC_NULL, "-body", &d_NumberOfBodies, PETSC_NULL);
    //Rotating Bodies should be the first bodies in the list
    PetscOptionsGetInt(PETSC_NULL, "-rbody", &d_NumberOfRotatingBodies, 
                       PETSC_NULL);

    PetscOptionsGetInt(PETSC_NULL, "-str", &d_sisteps, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-imm", &d_immersed, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-fsi", &d_movefsi, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-rfsi", &d_rotatefsi, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-rfsi_noIBsearch",
                       &d_rotatefsi_noIBsearch, PETSC_NULL);
    //A Single check to see if anything is moving
    d_changefsi = d_movefsi + d_rotatefsi + d_rotatefsi_noIBsearch;


    //Are we restarting
    PetscOptionsGetInt(PETSC_NULL, "-rs_fsi", &d_rstart_fsi, PETSC_NULL);

    //Directions of fsi movement
    PetscOptionsGetInt(PETSC_NULL, "-dgf_z", &d_dgf_z, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-dgf_y", &d_dgf_y, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-dgf_x", &d_dgf_x, PETSC_NULL);
  
    //Rotation Direction
    PetscOptionsGetInt(PETSC_NULL, "-rotdir", &d_rotdir, PETSC_NULL);
    //Is the rotation prescribed
    PetscOptionsGetInt(PETSC_NULL, "-d_prescribed_rotation", 
                       &d_prescribed_rotation, PETSC_NULL);
    //Prescribed Angualr velocity
    PetscOptionsGetReal(PETSC_NULL, "-angvel", &d_angvel, PETSC_NULL);
    
    PetscOptionsGetInt(PETSC_NULL, "-tio", &d_tiout, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-ti_lastsave", &d_ti_lastsave, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-path", d_path, 256, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-fsi_path", d_fsipath, 256, PETSC_NULL);
}
