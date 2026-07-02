#include "StructSolver.h"

StructSolver::StructSolver(
    const std::string object_name,
    CurvGrid *grid,
    UData *data,
    ImmersedBoundary *ib,
    FSI *fsi):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data),
    d_ib(ib),
    d_fsi(fsi)
{
    sprintf(d_path, ".");
    d_immersed = 0;
    d_str_tol = 1e-5;
    d_str_max = 1e-10;  

    d_sisteps = 1;
    
    d_movefsi = 0;
    d_rotatefsi=0;
    d_rotatefsi_noIBsearch=0;
    d_changefsi = 0;

    d_rstart_fsi = 0;

    d_NumberOfBodies = 0;
    d_NumberOfRotatingBodies = 0;

 
    ReadFromInput();
}

StructSolver::~StructSolver()
{}

PetscErrorCode StructSolver::Solve(PetscInt si, 
                                   PetscInt ti, 
                                   PetscReal time, 
                                   PetscBool *converged)
{
    //We don't need this at all if not using immersed boundry
    //Set Converged to true just to make sure
    if (!d_immersed) { 
        *converged = PETSC_TRUE;
        return 0;
    }

    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();
 
    Vec Ucat = d_data->getUcat();
    Vec Ucat_o = d_data->getUcat_o();
    Vec lUcat = d_data->getlUcat();
    Vec Ucont = d_data->getUcont();
    Vec Ucont_o = d_data->getUcont_o();
    Vec lUcont = d_data->getlUcont();
    Vec P = d_data->getP();
    Vec P_o = d_data->getP_o();
    Vec lP = d_data->getlP();
    Vec lNvert = d_data->getlNvert();
    Vec Nvert = d_data->getNvert();
    

    //Store old fsi to check convergence for sc
    d_fsi->CopyToOld(si);

    //Calculate Forces
    d_fsi->CalculateForces(ti, time);

    //Copy Ucat to the old
    if (si == 1) VecCopy(Ucat, Ucat_o);

    if (d_changefsi and si > 1){  
        VecCopy(Ucont_o, Ucont);
        VecCopy(P_o, P);

        DMGlobalToLocalBegin(fda, Ucont, INSERT_VALUES, lUcont);
        DMGlobalToLocalEnd(fda, Ucont, INSERT_VALUES, lUcont);

        DMGlobalToLocalBegin(da, P, INSERT_VALUES, lP);
        DMGlobalToLocalEnd(da, P, INSERT_VALUES, lP);

        d_data->Contra2Cart();
    }

    MPI_Barrier(PETSC_COMM_WORLD);

    //Calculate any translation movement and IB Search
    d_fsi->CalculatePosition(ti, time);

    //Calculate any rotation movement and IB Search
    d_fsi->CalculateRotation(ti, time);
    
    //Check Convergence
    *converged = CheckConvergence(si);

    //Write something
    if (d_sisteps > 1)
        PetscPrintf(PETSC_COMM_WORLD, 
                "StructSolver Convergence: %d %d %le %le %le\n", 
                 ti, si, d_str_max, d_fsi->getS_ang_n(0), d_fsi->getS_ang_o(0));

    if (*converged)
        d_fsi->WriteFSI(ti);
        
}


PetscBool StructSolver::CheckConvergence(PetscInt si)
{
    PetscBool converge = PETSC_FALSE;
    PetscReal dS_sc;  

    FSInfo *fsi = d_fsi->getFSInfo();
    for (PetscInt ibi=0;ibi<d_NumberOfBodies;ibi++) {
       
        if (d_movefsi) {
            for (int i=0; i<6;i++) {
                dS_sc = fabs(fsi[ibi].S_new[i]-fsi[ibi].S_old[i]);
                if (dS_sc > d_str_tol) converge = PETSC_TRUE;
                if (dS_sc > d_str_max) d_str_max = dS_sc;
            }
            //Run another step to make sure!!Maybe remove
            if (si < 2) converge = PETSC_FALSE;
        } else if (d_rotatefsi) {
            dS_sc = fabs(fsi[ibi].S_ang_n[0]-fsi[ibi].S_ang_o[0]);
            if (dS_sc > d_str_tol) converge = PETSC_TRUE;
            dS_sc = fabs(fsi[ibi].S_ang_n[1]-fsi[ibi].S_ang_o[1]);
            if (fabs(fsi[ibi].S_ang_n[1]+fsi[ibi].S_ang_o[1])>2.)
            dS_sc /= 0.5*fabs(fsi[ibi].S_ang_n[1]+fsi[ibi].S_ang_o[1]);

            if (dS_sc > d_str_tol) d_str_tol = PETSC_TRUE;
            if (dS_sc > d_str_max) d_str_max = dS_sc;
            //Run another step to make sure!!Maybe remove
            if (si < 2) converge = PETSC_FALSE;
   
        } else {
           //If not moving then it is converged
           converge = PETSC_TRUE;
        }
    }

    return converge;
}

PetscErrorCode StructSolver::ReadFromInput()
{
    PetscOptionsGetInt(PETSC_NULL, "-fsi", &d_movefsi, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-rfsi", &d_rotatefsi, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-rfsi_noIBsearch",
                       &d_rotatefsi_noIBsearch, PETSC_NULL);

    PetscOptionsGetInt(PETSC_NULL, "-body", &d_NumberOfBodies, PETSC_NULL);
    //Rotating Bodies should be the first bodies in the list
    PetscOptionsGetInt(PETSC_NULL, "-rbody", &d_NumberOfRotatingBodies,
                       PETSC_NULL);


    //A Single check to see if anything is moving
    d_changefsi = d_movefsi + d_rotatefsi + d_rotatefsi_noIBsearch;

    //Are we restarting
    PetscOptionsGetInt(PETSC_NULL, "-rs_fsi", &d_rstart_fsi, PETSC_NULL);

    PetscOptionsGetInt(PETSC_NULL, "-str", &d_sisteps, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-imm", &d_immersed, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-path", d_path, 256, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-str_tol", &d_str_tol, PETSC_NULL);
}

  

