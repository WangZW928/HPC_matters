#include "FlowSolver.h"

FlowSolver::FlowSolver(
    const std::string object_name,
    CurvGrid *grid,
    UData *data, 
    RHSSolver *rhs,
    BcsUtility *bcs,
    WallModel *wall, 
    LESModel *les,
    Integrator *integrate,
    PoissonSolver *poisson):
    
    d_object_name(object_name),
    d_grid(grid),
    d_data(data),
    d_rhs(rhs),
    d_bcs(bcs),
    d_les(les),
    d_wall(wall),
    d_integrate(integrate),
    d_poisson(poisson)
{
    sprintf(d_path, ".");
    d_immersed = 0;
    d_maxdiv = 1e10;
    ReadFromInput();
}
    
          
FlowSolver::~FlowSolver()
{}


PetscErrorCode FlowSolver::Solve(PetscInt ti, 
                                 PetscReal time)
{
    DM fda = d_grid->getFDA();
    Vec lUcont = d_data->getlUcont();
    Vec Ucont = d_data->getUcont();

    PetscReal ts, te;

    PetscTime(&ts);

    //Calculate the Minimum Dt
    d_integrate->CalculateMinimumDt();

    //Apply BC if first time step
    if (ti==d_data->get_tistart()) 
    {
         d_bcs->IbBC();

         DMLocalToGlobalBegin(fda, lUcont, INSERT_VALUES, Ucont);
         DMLocalToGlobalEnd(fda, lUcont, INSERT_VALUES, Ucont);
    }

    //Read inflow plane will return if inletprofile!=100
    d_bcs->ReadPlane(ti);

    //Calculate the inflow flux 
    d_bcs->CalculateInflowFlux();


    //Setup RHS if first time
    if (ti==d_data->get_tistart())
    {
        PetscPrintf(PETSC_COMM_WORLD, "Initializing RHS\n");
        d_wall->Initialize();
        d_rhs->Initialize();
    }

    //Setup LES
    if (d_les->useLES())
    {
        //if (ti==d_data->get_tistart()) 
        d_les->ComputeSmagorinksyConstant(ti);
        d_les->ComputeEddyViscosity();
    }

    //Compute RHS_o
    VecSet(d_data->getRhs_o(), 0.0);
    d_rhs->Solve(d_data->getRhs_o(), 1.0);

    //Calculate Pressure Gradent
    d_rhs->CalculatePressureGradient();

    //Here is where interpolation_advanced
    //Here is where rotor and nacelle advance are (later)

    //Calculate Wall Model
    d_wall->CalculateVisc();

    //Here is where Temperature is calculated (later)

    PetscBarrier(PETSC_NULL);

    //Integrate Momentum
    d_integrate->Solve(ti);

    PetscBarrier(PETSC_NULL);

    //Setup Poisson if first time
    if (ti==d_data->get_tistart())
        d_poisson->Initialize();

    //Solve Poisson Equation + Velocity Correction
    d_poisson->Solve(ti, time);

    //Check Divergence
    CalculateDivergence(ti);
    
    //Apply Boundary Conditions 
    d_bcs->IbBC();
    DMLocalToGlobalBegin(fda, lUcont, INSERT_VALUES, Ucont);
    DMLocalToGlobalEnd(fda, lUcont, INSERT_VALUES, Ucont);
    d_data->Contra2Cart();


    //Calculate Write KE
    CalculateKE(ti);

    PetscBarrier(PETSC_NULL);

    PetscTime(&te);

    t_solve_time = te - ts;

    return 0;
}


PetscErrorCode FlowSolver::CalculateDivergence(PetscInt ti)
{
   int    i, j, k;

    //Get the DMs 
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo  info;
    DMDAGetLocalInfo(da, &info);

    PetscInt xs = info.xs, xe = info.xs + info.xm;
    PetscInt ys = info.ys, ye = info.ys + info.ym;
    PetscInt zs = info.zs, ze = info.zs + info.zm;
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt gxs = info.gxs;
    PetscInt gxe = gxs + info.gxm;
    PetscInt gys = info.gys;
    PetscInt gye = gys + info.gym;
    PetscInt gzs = info.gzs;
    PetscInt gze = gzs + info.gzm;
    PetscInt gxm = info.gxm, gym = info.gym, gzm = info.gzm;

    PetscInt lxs, lys, lzs, lxe, lye, lze;

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    Vec Div;
    PetscReal ***div, ***aj, ***nvert;
    Cmpnts ***ucont;
    PetscReal maxdiv;

    Vec lUcont = d_data->getlUcont();
    Vec lNvert = d_data->getlNvert();
    Vec Aj = d_grid->getlAj();
    Vec P = d_data->getP();

    DMDAVecGetArray(fda, lUcont, &ucont);
    DMDAVecGetArray(da, Aj, &aj);
    VecDuplicate(P, &Div);

    DMDAVecGetArray(da, Div, &div);
    DMDAVecGetArray(da, lNvert, &nvert);
    for (k=lzs; k<lze; k++) {
        for (j=lys; j<lye; j++) {
            for (i=lxs; i<lxe; i++) {
   
                maxdiv = fabs((ucont[k][j][i].x - ucont[k][j][i-1].x + 
                               ucont[k][j][i].y - ucont[k][j-1][i].y + 
                               ucont[k][j][i].z - ucont[k-1][j][i].z) * 
                         aj[k][j][i]);

                if (nvert[k][j][i] + nvert[k+1][j][i] + nvert[k-1][j][i] + 
                    nvert[k][j+1][i] + nvert[k][j-1][i] + 
                    nvert[k][j][i+1] + nvert[k][j][i-1] > 0.1) maxdiv = 0.;
   
                div[k][j][i] = maxdiv;
   
            }
        }
    }

    if (zs==0) {
        k=0;
        for (j=ys; j<ye; j++) {
            for (i=xs; i<xe; i++) {
                div[k][j][i] = 0.;
            }
        }
    }

    if (ze == mz) {
        k=mz-1;
        for (j=ys; j<ye; j++) {
            for (i=xs; i<xe; i++) {
                div[k][j][i] = 0.;
            }
        }
    }

    if (xs==0) {
        i=0;
        for (k=zs; k<ze; k++) {
            for (j=ys; j<ye; j++) {
                div[k][j][i] = 0.;
            }
        }
    }

    if (xe==mx) {
        i=mx-1;
        for (k=zs; k<ze; k++) {
            for (j=ys; j<ye; j++) {
                div[k][j][i] = 0;
            }
        }
    }

    if (ys==0) {
        j=0;
        for (k=zs; k<ze; k++) {
            for (i=xs; i<xe; i++) {
                 div[k][j][i] = 0.;
            }
        }
    }

    if (ye==my) {
        j=my-1;
        for (k=zs; k<ze; k++) {
            for (i=xs; i<xe; i++) {
                 div[k][j][i] = 0.;
            }
        }
    }
    DMDAVecRestoreArray(da, Div, &div);
    VecMax(Div, &i, &d_maxdiv);
    PetscPrintf(PETSC_COMM_WORLD, 
                "Maxdiv %d %d %e\n", (int)ti, i, d_maxdiv);
    int mi;
  
    PetscInt *idx = d_grid->getIdx();

    for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
            for (mi=xs; mi<xe; mi++) {
                int indx = (idx[(k-gzs)*(gxm*gym) + (j-gys)*(gxm) + (mi-gxs)]);
 
                if (indx ==i) {
                   PetscPrintf(PETSC_COMM_SELF, 
                               "Max Div Loc %d %d %d\n", mi,j, k);
                }
            }
        }
    }
  
   
    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(fda, lUcont, &ucont);
    DMDAVecRestoreArray(da, Aj, &aj);

    VecDestroy(&Div);

    return 0;
}

PetscErrorCode FlowSolver::CalculateKE(PetscInt ti)
{
   int    i, j, k;

    //Get the DMs 
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo  info;
    DMDAGetLocalInfo(da, &info);

    PetscInt xs = info.xs, xe = info.xs + info.xm;
    PetscInt ys = info.ys, ye = info.ys + info.ym;
    PetscInt zs = info.zs, ze = info.zs + info.zm;
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt lxs, lys, lzs, lxe, lye, lze;

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    PetscReal ***aj;
    Cmpnts ***ucat;

    Vec lUcat = d_data->getlUcat();
    Vec Aj = d_grid->getlAj();

    DMDAVecGetArray(fda, lUcat, &ucat);
    DMDAVecGetArray(da, Aj, &aj);

    double lsum=0, sum=0;
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
                lsum += 0.5*ucat[k][j][i].x * ucat[k][j][i].x / aj[k][j][i];
                lsum += 0.5*ucat[k][j][i].y * ucat[k][j][i].y / aj[k][j][i];
                lsum += 0.5*ucat[k][j][i].z * ucat[k][j][i].z / aj[k][j][i];
            }
    GlobalSum_All(&lsum, &sum, PETSC_COMM_WORLD);

    DMDAVecRestoreArray(fda, lUcat, &ucat);
    DMDAVecRestoreArray(da, Aj, &aj);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (!rank) {
        char filen[80];
        sprintf(filen, "%s/Kinetic_Energy.dat", d_path);
        FILE *f = fopen(filen, "a");
        PetscFPrintf(PETSC_COMM_WORLD, f, "%d\t%.7e\n", ti, sum);
        fclose(f);
    }

    return 0;
}

PetscErrorCode FlowSolver::ReadFromInput()
{
    PetscOptionsGetInt(PETSC_NULL, "-imm", &d_immersed, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-path", d_path, 256, PETSC_NULL);
}
 
 
 
   

 
    

    
