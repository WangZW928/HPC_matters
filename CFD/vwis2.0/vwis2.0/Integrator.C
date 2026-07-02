#include "Integrator.h"

Integrator::Integrator(
   const std::string& object_name,
    CurvGrid *grid,
    UData *data,
    RHSSolver *rhs,
    WallModel *wall, 
    BcsUtility *bcs):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data),
    d_rhs(rhs),
    d_wall(wall),
    d_bcs(bcs)
{
    d_imp_free_tol = 1e-4;
    d_norm = 1e10;
    d_unorm = 1e10;

    d_snes_created = PETSC_FALSE;
    ReadFromInput();
}

Integrator::~Integrator()
{
    SNESDestroy(&d_snes);
}

PetscErrorCode Integrator::SNESMonitor(
    SNES snes, 
    PetscInt n,
    PetscReal rnorm,
    void *dummy)
{
    PetscPrintf(PETSC_COMM_WORLD,
                    "     (%D) SNES Residual norm %14.12e \n",n,rnorm);
    return 0;
}

PetscErrorCode Integrator::Solve(PetscInt ti)
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
    

    Cmpnts ***ucont, ***lucont;
    Cmpnts ***ucat, ***lucat;
    Cmpnts ***coor, ***csi, ***eta, ***zet;
    PetscInt i, j, k;
   
    Vec U, Coor;
    Vec Rhs = d_data->getRhs(); 
    Vec Ucont = d_data->getUcont();
    Vec Ucat = d_data->getUcat();
    Vec lUcat = d_data->getlUcat();
    Vec lUcont = d_data->getlUcont();
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();    


    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscReal ts,te,cput;
    PetscTime(&ts);

    PetscReal ts1, te1;
    
    PetscTime(&ts1);

    if (!d_snes_created) {
        SNESCreate(PETSC_COMM_WORLD,&d_snes);
        SNESMonitorSet (d_snes, SNESMonitor, PETSC_NULL, PETSC_NULL);
        PetscPrintf(PETSC_COMM_WORLD, "Solver Object Created \n");
    }
    
    SNESSetFunction(d_snes, Rhs, SolveFunction, static_cast<void*>(this));

    PetscTime(&te1);

    PetscPrintf(PETSC_COMM_WORLD, "Time for SNESSetFunction %le\n", te1-ts1);

    PetscTime(&ts1);

    Mat J = d_data->getJacobian();
    #if defined(PETSC_HAVE_ADIC___)
    DAGetMatrix(da,MATAIJ,&J);
    ISColoring iscoloring;
    DAGetColoring(da,IS_COLORING_GHOSTED,&iscoloring);
    MatSetColoring(J, iscoloring);
    ISColoringDestroy(iscoloring);
    SNESSetJacobian(d_snes, J, J, SNESDAComputeJacobianWithAdic, PETSC_NULL);
    #else
    MatCreateSNESMF(d_snes, &J);
    SNESSetJacobian(d_snes, J, J,MatMFFDComputeJacobian, PETSC_NULL);
    #endif
        
    SNESSetType(d_snes, SNESNEWTONTR);            //SNESTR,SNESLS    : SNESLS is better for stiff PDEs such as the one including IB but slower
        //SNESSetType(user[bi].snes, SNESNEWTONLS);            //SNESTR,SNESLS    : SNESLS is better for stiff PDEs such as the one including IB but slower

    SNESSetMaxLinearSolveFailures(d_snes,10000);
    SNESSetMaxNonlinearStepFailures(d_snes,10000);        
    SNESKSPSetUseEW(d_snes, PETSC_TRUE);
    SNESKSPSetParametersEW(d_snes,3,
                           PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,
                           PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
    SNESSetTolerances(d_snes, PETSC_DEFAULT, d_imp_free_tol,
                      PETSC_DEFAULT,50,50000);
        
        
    SNESGetKSP(d_snes, &d_ksp);
    KSPGetPC(d_ksp,&d_pc);
        
    if(d_snes_created) {
        KSPSetType(d_ksp, KSPGMRES);
    }
    #if defined(PETSC_HAVE_ADIC___)
    PCSetType(d_pc,PCBJACOBI);
    #else
    PCSetType(d_pc,PCNONE);
    #endif
    int maxits=1000;    
    double rtol=d_imp_free_tol, atol=PETSC_DEFAULT, dtol=PETSC_DEFAULT;

    KSPSetTolerances(d_ksp,rtol,atol,dtol,maxits);
        
    d_snes_created = PETSC_TRUE;
        
    PetscTime(&te1);
    
    PetscPrintf(PETSC_COMM_WORLD, 
                "Time for set up momentum solver %le\n", te1-ts1);


    PetscTime(&ts1);
    
    //Create Vecs for Momentum Solve
    VecDuplicate(Ucont, &U);
    VecDuplicate(Ucont, &Rhs);

    //Calculate Inflow Flux
    d_bcs->InflowFlux(ti);
 
    //Calculate Boundary Conditions
    d_bcs->FormBcs(ti, 0);
 
    //Copy Values of Ucont to working Vec       
    VecCopy(Ucont, U);
        
    PetscTime(&te1);
        
    PetscPrintf(PETSC_COMM_WORLD, 
                "Time for duplicating and copying vector %le\n", te1-ts1);

    
    //Solve Momentum Equations
    SNESSolve(d_snes, PETSC_NULL, U);
        
    PetscPrintf(PETSC_COMM_WORLD, "\nMomentum eqs computed ...\n");
    
    SNESGetFunctionNorm(d_snes, &d_norm);
    PetscPrintf(PETSC_COMM_WORLD, "\nSNES residual norm=%.5e\n\n", d_norm);
    
    //Copy final working Vector back    
    VecCopy(U, Ucont);
    
    //Send Global to Local    
    DMGlobalToLocalBegin(fda, Ucont, INSERT_VALUES, lUcont);
    DMGlobalToLocalEnd(fda, Ucont, INSERT_VALUES, lUcont);
    
    DMGetCoordinatesLocal(da, &Coor);
    DMDAVecGetArray(fda, Coor, &coor);
    DMDAVecGetArray(fda, Ucont, &ucont);
    DMDAVecGetArray(fda, Ucat, &ucat);
    DMDAVecGetArray(fda, lUcont, &lucont);
    DMDAVecGetArray(fda, lUcat, &lucat);
    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);

    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
      
                if ( (d_grid->getBC(0)==1 || d_grid->getBC(0)==-1 || 
                      d_grid->getBC(0)==-2 || d_grid->getBC(0)==10) && 
                      i==0) 
                    ucont[k][j][i].x = 0;
                if ( (d_grid->getBC(1)==1 || d_grid->getBC(1)==-1 || 
                      d_grid->getBC(1)==-2 || d_grid->getBC(1)==10) && 
                      i==mx-2) 
                    ucont[k][j][i].x = 0;
                if ( (d_grid->getBC(2)==1 || d_grid->getBC(2)==-1 || 
                      d_grid->getBC(2)==-2 || d_grid->getBC(2)==10) && 
                      j==0) 
                    ucont[k][j][i].y = 0;
                if ( (d_grid->getBC(3)==1 || d_grid->getBC(3)==-1 || 
                      d_grid->getBC(3)==-2 || d_grid->getBC(3)==10 || 
                      d_grid->getBC(3)==2) && j==my-2) 
                    ucont[k][j][i].y = 0;
                if ( (d_grid->getBC(4)==1 || d_grid->getBC(4)==-1 || 
                      d_grid->getBC(4)==-2 || d_grid->getBC(4)==10) && 
                      k==0) 
                    ucont[k][j][i].z = 0;
                if ( (d_grid->getBC(5)==1 || d_grid->getBC(5)==-1 || 
                      d_grid->getBC(5)==-2 || d_grid->getBC(5)==10) && 
                      k==mz-2) 
                    ucont[k][j][i].z = 0;

                /*
                if (wallmodel_test) 
                {
                    if ( (d_grid->getBC(0)==-1 || d_grid->getBC(0)==-2) && 
                          i==1) 
                        ucont[k][j][i].x = 0;
                    if ( (d_grid->getBC(1)==-1 || d_grid->getBC(1)==-2) && 
                          i==mx-3) 
                        ucont[k][j][i].x = 0;
                    if ( (d_grid->getBC(2)==-1 || d_grid->getBC(2)==-2) && 
                          j==1) 
                        ucont[k][j][i].y = 0;
                    if ( (d_grid->getBC(3)==-1 || d_grid->getBC(3)==-2) && 
                          j==my-3) 
                        ucont[k][j][i].y = 0;
                    if ( (d_grid->getBC(4)==-1 || d_grid->getBC(4)==-2) && 
                          k==1) 
                        ucont[k][j][i].z = 0;
                    if ( (d_grid->getBC(5)==-1 || d_grid->getBC(5)==-2) && 
                          k==mz-3) 
                        ucont[k][j][i].z = 0;
                }
                */

                if ( d_grid->getBC(3)==4 && j==my-2 ) 
                {
                    ucat[k][j+1][i] = ucat[k][j][i];
                    lucat[k][j+1][i] = ucat[k][j+1][i];
                    ucont[k][j][i].y = 
                     0.5*(lucat[k][j][i].x+lucat[k][j+1][i].x)*eta[k][j][i].x +
                     0.5*(lucat[k][j][i].y+lucat[k][j+1][i].y)*eta[k][j][i].y +
                     0.5*(lucat[k][j][i].z+lucat[k][j+1][i].z)*eta[k][j][i].z;
                }
        
                if ( ti && d_grid->getBC(4)==4 && k==0 ) 
                {
                    if (ucont[k][j][i].z<0) 
                    {
                        ucont[k][j][i].z = 0;
                        ucat[k][j][i].x = ucat[k][j][i].y = ucat[k][j][i].z = 0;
                        ucat[k+1][j][i] = ucat[k][j][i];
                    }
                }
        
                if ( ti && (d_grid->getBC(5)==4 || d_grid->getBC(5)==5) && 
                     k==mz-2 ) 
                { 
                    ucat[k+1][j][i] = ucat[k][j][i]; 
                    lucat[k+1][j][i] = ucat[k+1][j][i];
                    ucont[k][j][i].z = 
                     0.5*(lucat[k][j][i].x+lucat[k+1][j][i].x)*zet[k][j][i].x +
                     0.5*(lucat[k][j][i].y+lucat[k+1][j][i].y)*zet[k][j][i].y +
                     0.5*(lucat[k][j][i].z+lucat[k+1][j][i].z)*zet[k][j][i].z;
          
                    if (ucont[k][j][i].z<0) {
                        ucont[k][j][i].z = 0;
                        ucat[k][j][i].x = ucat[k][j][i].y = ucat[k][j][i].z = 0;
                        ucat[k+1][j][i] = ucat[k][j][i];
                    }
                }
                if (d_grid->getBC(0)==11 && i==0 && 
                   (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) 
                {
                    double zc = (coor[k][j][i+1].z + coor[k-1][j][i+1].z + 
                             coor[k][j-1][i+1].z + coor[k-1][j-1][i+1].z)* 0.25;
                    if( zc > 0 ) {
                        ucont[k][j][i].x = lucat[k][j][i].z * csi[k][j][i].z;
                    }
                }
            }

    DMDAVecRestoreArray(fda, Coor, &coor);
    DMDAVecRestoreArray(fda, Ucont, &ucont);
    DMDAVecRestoreArray(fda, Ucat, &ucat);
    DMDAVecRestoreArray(fda, lUcont, &lucont);
    DMDAVecRestoreArray(fda, lUcat, &lucat);
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    
    //Calculate some more BCs
    d_bcs->FormBcs(ti, 1);


    PetscTime(&te);

    //Destroy Working Vecs
    VecDestroy(&U);
    VecDestroy(&Rhs);
    MatDestroy(&J);
    
    //Update Ucont 
    DMGlobalToLocalBegin(fda, Ucont, INSERT_VALUES, lUcont);
    DMGlobalToLocalEnd(fda, Ucont, INSERT_VALUES, lUcont);

    //Find the Max velocity 
    VecMax(Ucat, &i, &d_unorm);
    PetscPrintf(PETSC_COMM_WORLD, "*** Max Ucat = %e \n", d_unorm);
    
    //Convarvariant to Cartesian Velocity
    d_data->Contra2Cart();
    return(0);
}


PetscErrorCode Integrator::SolveFunction(
    SNES snes, 
    Vec Uconti, 
    Vec Rhs, 
    void *ptr)
{
  
    //We have to do this because SolveFunction is called 
    //be SNES which needs to be in the form of 
    //PetscErrorCode SNESFunction(SNES snes,Vec x,Vec f,void *ctx) 
    //So can't be a object method.  
    //So sending *this* to static method instead
    Integrator *iptr = static_cast<Integrator *>(ptr);    
  
    //Now we need all of these from iptr (this) 
    CurvGrid *d_grid = iptr->getGrid();
    UData *d_data = iptr->getData();
    RHSSolver *d_rhs = iptr->getRHS();
    WallModel *d_wall = iptr->getWall();
    BcsUtility *d_bcs = iptr->getBcs();

    //Get the DMs 
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo  info; 
    DMDAGetLocalInfo(da, &info);
    int xs = info.xs, xe = info.xs + info.xm;
    int ys = info.ys, ye = info.ys + info.ym;
    int zs = info.zs, ze = info.zs + info.zm;
    int mx = info.mx, my = info.my, mz = info.mz;
    int lxs, lxe, lys, lye, lzs, lze;
    int i, j, k;
    
    Cmpnts ***ucont;
    PetscReal ***nvert;
    
    lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

    if (lxs==0) lxs++;
    if (lxe==mx) lxe--;
    if (lys==0) lys++;
    if (lye==my) lye--;
    if (lzs==0) lzs++;
    if (lze==mz) lze--;

    //Get the Vecs I need
    Vec Ucont = d_data->getUcont();    
    Vec Ucont_o = d_data->getUcont_o();
    Vec Rhs_o = d_data->getRhs_o();
    Vec Dp = d_data->getDp();
    Vec lNvert = d_data->getlNvert();
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();

    //Not sure if we need this cause we just copied it
    //Could check for speed up
    VecCopy(Uconti, Ucont);


    DMDAVecGetArray(fda, Ucont, &ucont);
    DMDAVecGetArray(da, lNvert, &nvert);

    Cmpnts ***csi, ***eta, ***zet;

    DMDAVecGetArray(fda, Csi,  &csi);
    DMDAVecGetArray(fda, Eta,  &eta);
    DMDAVecGetArray(fda, Zet,  &zet);

    //set some Bcs
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                // noslip BC 
                if(i==0 && d_grid->getBC(0)==1) ucont[k][j][i].x = 0;
                if(i==mx-1 && d_grid->getBC(1)==1) ucont[k][j][i-1].x = 0;
                if(j==0 && d_grid->getBC(2)==1) ucont[k][j][i].y = 0;
                if(j==my-1 && d_grid->getBC(3)==1) ucont[k][j-1][i].y = 0;
                if(k==0 && d_grid->getBC(4)==1) ucont[k][j][i].z = 0;
                if(k==mz-1 && d_grid->getBC(5)==1) ucont[k-1][j][i].z = 0;
        
                // wall model
                if(i==0 && d_grid->getBC(0)==-1) ucont[k][j][i].x = 0;
                if(i==mx-1 && d_grid->getBC(1)==-1) ucont[k][j][i-1].x = 0;
                if(j==0 && d_grid->getBC(2)==-1) ucont[k][j][i].y = 0;
                if(j==my-1 && d_grid->getBC(3)==-1) ucont[k][j-1][i].y = 0;
                if(k==0 && d_grid->getBC(4)==-1) ucont[k][j][i].z = 0;
                if(k==mz-1 && d_grid->getBC(5)==-1) ucont[k-1][j][i].z = 0;
        
                //cavity problem 
                if (j==my-1 && d_grid->getBC(3)==2) ucont[k][j-1][i].y = 0;
        
                // couette flow j=0
                if (j==0 && d_grid->getBC(2)==12) ucont[k][j][i].y = 0;
        
                // couette flow j=my-1
                if (j==my-1 && d_grid->getBC(3)==12) ucont[k][j-1][i].y = 0;
        
                //slip BC
                if ( d_grid->getBC(0)==10 && i==0 && 
                     (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) 
                    ucont[k][j][i].x = 0;
                if ( d_grid->getBC(1)==10 && i==mx-1 && 
                     (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) 
                    ucont[k][j][i-1].x = 0;
                if ( d_grid->getBC(2)==10 && j==0 && 
                     (i!=0 && i!=mx-1 && k!=0 && k!=mz-1) ) 
                    ucont[k][j][i].y = 0;
                if ( std::abs(d_grid->getBC(3))==10 && j==my-1 && 
                     (i!=0 && i!=mx-1 && k!=0 && k!=mz-1) ) 
                    ucont[k][j-1][i].y = 0;
            }
    
    DMDAVecRestoreArray(fda, Ucont, &ucont);
    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(fda, Csi,  &csi);
    DMDAVecRestoreArray(fda, Eta,  &eta);
    DMDAVecRestoreArray(fda, Zet,  &zet);

    //Update Ucont
    Vec lUcont = d_data->getlUcont();
    DMGlobalToLocalBegin(fda, Ucont, INSERT_VALUES, lUcont);
    DMGlobalToLocalEnd(fda, Ucont, INSERT_VALUES, lUcont);

    //Convarvariant to Cartesian Velocity
    d_data->Contra2Cart();

    //Get IB Boundary Condtions
    d_bcs->IbBC();
    
    VecSet(Rhs,0);
    
    const double dt = d_data->getDt();

    //Get the type of integration we are doing
    double coeff = d_data->getTimeCoeff();

    //Add time derivative to Rhs
    if( coeff>0.9 && coeff<1.1 ) {
        VecAXPY(Rhs, -1./dt, Ucont);
        VecAXPY(Rhs, 1./dt, Ucont_o);
    }
    else {
        VecAXPY(Rhs, -1.5/dt, Ucont);
        VecAXPY(Rhs, 2./dt, Ucont_o);
        VecAXPY(Rhs, -0.5/dt, d_data->getUcont_rm1());
    }
    
    //Solving for the RHS
    if ( coeff>0.9 && coeff<1.1 ) {
        d_rhs->Solve(Rhs, 0.5);    // careful ! adding values to Rhs
        VecAXPY(Rhs, 0.5, Rhs_o);
    }
    else d_rhs->Solve(Rhs, 1.0);

    //Including Pressure Gradient
    VecAXPY(Rhs, -1, Dp); 

    //Here is where F_eul from rotor added (later)

    //Including Wall Model
    if (d_wall->useWallModel()) {

        Vec Rhs_wm;
        VecDuplicate(Rhs, &Rhs_wm);
        VecSet(Rhs_wm, 0.0);
    
        d_wall->setFp(d_rhs->getFp());
        d_wall->setVisc(d_rhs->getVisc1(), 
                        d_rhs->getVisc2(), 
                        d_rhs->getVisc3()); 

        d_wall->Solve(Rhs_wm, 1.0);
        VecAXPY(Rhs, 1, Rhs_wm);

        VecDestroy(&Rhs_wm);
    }

    if ( coeff>1.1 && coeff<2.0 ) 
        VecScale(Rhs, 1./1.5);

    return 0;
}


double Integrator::CalculateMinimumDt()
{
    int    i, j, k;

    //Get the DMs 
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo  info; 
    DMDAGetLocalInfo(da, &info);
    int xs, xe, ys, ye, zs, ze; 
    int mx, my, mz; 
    int lxs, lxe, lys, lye, lzs, lze;

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
    
    PetscReal ***aj;
    Cmpnts ***ucont, ***cent;
    Cmpnts ***csi, ***eta, ***zet;
    PetscReal ***nvert;

    Vec lUcont = d_data->getlUcont();
    Vec lNvert = d_data->getlNvert();

    Vec lCent = d_grid->getlCent();
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();
  
    DMDAVecGetArray(fda, lUcont,  &ucont);
    DMDAVecGetArray(fda, lCent,  &cent);
    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, Aj, &aj);
    DMDAVecGetArray(da, lNvert, &nvert);
    
    double ldt=1.e7, dt=0;
    double ldx=1.e7;
    double ldi_min=1.e7, ldj_min=1.e7, ldk_min=1.e7;
    double ldi_max=0, ldj_max=0, ldk_max=0;
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {

                ldt = PetscMin ( fabs(1./ucont[k][j][i].x/aj[k][j][i]), ldt );
                ldt = PetscMin ( fabs(1./ucont[k][j][i].y/aj[k][j][i]), ldt );
                ldt = PetscMin ( fabs(1./ucont[k][j][i].z/aj[k][j][i]), ldt );
        

                double A1 = sqrt( csi[k][j][i].x*csi[k][j][i].x +
                                  csi[k][j][i].y*csi[k][j][i].y + 
                                  csi[k][j][i].z*csi[k][j][i].z ); 
                double A2 = sqrt( eta[k][j][i].x*eta[k][j][i].x +
                                  eta[k][j][i].y*eta[k][j][i].y + 
                                  eta[k][j][i].z*eta[k][j][i].z ); 
                double A3 = sqrt( zet[k][j][i].x*zet[k][j][i].x +
                                  zet[k][j][i].y*zet[k][j][i].y + 
                                  zet[k][j][i].z*zet[k][j][i].z ); 

                double ldi = 1./aj[k][j][i]/A1;
                double ldj = 1./aj[k][j][i]/A2;
                double ldk = 1./aj[k][j][i]/A3;
        
                ldi_min = PetscMin ( ldi_min, ldi );
                ldj_min = PetscMin ( ldj_min, ldj );
                ldk_min = PetscMin ( ldk_min, ldk );
        
                ldi_max = PetscMax ( ldi_max, ldi );
                ldj_max = PetscMax ( ldj_max, ldj );
                ldk_max = PetscMax ( ldk_max, ldk );
        
                if (nvert[k][j][i]<0.1) 
                    ldx = PetscMin ( ldi_min, PetscMin ( ldj_min, ldk_min ) );
            }
    
    GlobalMin_All(&ldt, &dt, PETSC_COMM_WORLD);
    GlobalMin_All(&ldx, &d_dx_min, PETSC_COMM_WORLD);
    GlobalMin_All(&ldi_min, &d_di_min, PETSC_COMM_WORLD);
    GlobalMin_All(&ldj_min, &d_dj_min, PETSC_COMM_WORLD);
    GlobalMin_All(&ldk_min, &d_dk_min, PETSC_COMM_WORLD);
    
    GlobalMax_All(&ldi_max, &d_di_max, PETSC_COMM_WORLD);
    GlobalMax_All(&ldj_max, &d_dj_max, PETSC_COMM_WORLD);
    GlobalMax_All(&ldk_max, &d_dk_max, PETSC_COMM_WORLD);
    
    DMDAVecRestoreArray(fda, lUcont,  &ucont);
    DMDAVecRestoreArray(fda, lCent,  &cent);
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(da, Aj, &aj);
    DMDAVecRestoreArray(da, lNvert, &nvert);
    
    PetscPrintf(PETSC_COMM_WORLD, 
                "CFL 1.0 time step=%.6f, dx_min=%.6f\n", dt, d_dx_min);
    
    return dt;
}

PetscErrorCode Integrator::ReadFromInput()
{
    PetscOptionsGetReal(PETSC_NULL, "-imp_tol", &d_imp_free_tol, PETSC_NULL);
}
