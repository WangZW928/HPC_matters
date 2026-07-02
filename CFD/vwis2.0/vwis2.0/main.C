/*
 * \file main.c
 * \author Danny Foti
 * \email dvfoti@memphis.edu
 * \description Main driver function for VWiS-ROM
 */
  
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"
#include "RHSSolver.h"
#include "BcsUtility.h"
#include "LESModel.h"
#include "WallModel.h"
#include "PoissonSolver.h"
#include "Integrator.h"
#include "FlowSolver.h"
#include "ImmersedBoundary.h"
#include "FSI.h"
#include "StructSolver.h"
#include "PointProbe.h"
#include "PlaneExtraction.h"

using namespace std;

int main(int argc, char **argv)
{

    static char help[] = "VWiS 2.0 with interface for ROM\n"
                         "Input file is control.dat";

    int rank, size;
    PetscInt ti, tistart=0, tisteps=1;
    PetscInt si, sisteps = 1;

    PetscReal dt, time=0.0;
    PetscBool restart;   
    PetscErrorCode ierr;

    PetscInitialize(&argc, &argv, (char *)0, help);
    MPI_Barrier(PETSC_COMM_WORLD);

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    //Set the input file
    PetscOptionsInsertFile(PETSC_COMM_WORLD, "control.dat", PETSC_TRUE);
    
    //Read the Inputs we need
    PetscOptionsGetInt(PETSC_NULL, "-rstart", &tistart, &restart);
    PetscOptionsGetInt(PETSC_NULL, "-totalsteps", &tisteps, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-str", &sisteps, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-dt", &dt, PETSC_NULL);

    //Create Grid object
    CurvGrid *grid = new CurvGrid("Curvilinear Grid");
    //Create Data object
    UData *data = new UData("User Data", grid);
   
    //Create Immersed boundary and FSI
    ImmersedBoundary *ibm = new ImmersedBoundary("Discrete IB Method",
                                                 grid, data);
    FSI *fsi = new FSI("Fluid-Structure Interations", grid, data, ibm);

    //Create some helper objects
    PointProbe *probe = new PointProbe("Point Probe", grid, data);
    PlaneExtraction *explane = new PlaneExtraction("Plane Extraction", 
                                                    grid, data);

    //Create the low level solvers
    BcsUtility *bcs = new BcsUtility("BCS", grid, data, explane);
    PoissonSolver *poisson = new PoissonSolver("Poisson", grid, data);
    LESModel *les = new LESModel("LES", grid, data);
    WallModel *wall = new WallModel("Wall", grid, data, les, ibm);    

    //Create Momentum RHS Solver
    RHSSolver *rhs = new RHSSolver("RHS", grid, data, les);

    //Create Momentum Integrator
    Integrator *integrate = new Integrator("Time Integration",  
                                           grid, data, rhs, wall, bcs);


    //Create Flow Solver
    FlowSolver *flow = new FlowSolver("Flow Solver", 
                                      grid, data, rhs, bcs, wall, 
                                      les, integrate, poisson);

    //Create Struct Solver
    StructSolver *struc = new StructSolver("Struct Solver", 
                                           grid, data, ibm, fsi);

    //Here is where we create Turbine Solvers

   

    /*
     * Start the setup of the solvers
     */

    /*
     * Read the grid/bcs/setup
     * This needs to be done first to create grid distribution
     */
    grid->ReadGrid();
    grid->ReadBC();
    grid->InitializeVecs();

    //Form the grid metrics
    grid->FormMetrics();

    //Create the solver data
    data->InitializeData();     

    //IBM/FSI are initialized
    ibm->IBMRead();
    fsi->Initialize();

    //Here is where Turbine is initialized

    //Initialize Point probe
    probe->Initialize();


    //Read data input if from restart
    data->ReadData();
   
    //Read ibm input if from restart
    fsi->Restart(tistart);
 
    //First ibm search and interp (only one if not moving)
    ibm->IBMSearchAdvanced(tistart);
    MPI_Barrier(PETSC_COMM_WORLD);
    ibm->IBMInterpolationAdvanced(tistart);    

    //Calculate the inlet area
    bcs->CalculateInletArea();

    //Now apply initial conditions if not restart
    if (!restart) {
        bcs->InitializeFlowField();
        bcs->ScaleInitialFlow();
    }

    //Copy n step to n-1 step
    data->CopyLastStep();

    MPI_Barrier(PETSC_COMM_WORLD);

    time = dt*tistart;

    /*
     * The main outer loop (time)
     */
    for (ti=tistart; ti<= tisteps; ti++) {     
 
        PetscPrintf(PETSC_COMM_WORLD, "Time: %d  %f\n", ti, time);
        PetscReal ts,te,cput;
        PetscTime(&ts);
    
        /*
         * Inner Loop for Strong Coupling
         * Weak Coupling is sistep =1
         */
 
        PetscBool isConverged = PETSC_FALSE;       
        for (si=0; si<sisteps; si++) {

            //Check Convergence if Strong Coupling
            if (isConverged) break;

            PetscPrintf(PETSC_COMM_WORLD, "Inner Loop # %d\n",  si);  
     
            //Here is the Struc Solver
            struc->Solve(si, ti, time, &isConverged);

            /*
             * Solve Flow
             */
            flow->Solve(ti, time); 
            
        }

        //Increment Time
        time += dt;

        //Copy Last Step
        data->CopyLastStep();
        ibm->CopyLastStep(); 
  
        //Average Data
        data->Average(ti);

        //Write Output here
        data->WriteData(ti);
        les->WriteCs(ti);
        ibm->IBMWrite(ti);

        //Help objects do their work
        probe->Probe(ti, dt, time);
        explane->Save(ti, time);
 
    }


    /* 
     * Clean up
     */
    delete probe;
    delete struc;
    delete flow;
    delete fsi;
    delete ibm;
    delete integrate;
    delete rhs;
    delete bcs;
    delete les;
    delete wall;
    delete poisson;
    delete data;
    delete grid;

    PetscFinalize();

    return 0;

}

