#ifndef included_PoissonSolver
#define included_PoissonSolver

#define iCP 0
#define iEP 1
#define iWP 2
#define iNP 3
#define iSP 4
#define iTP 5
#define iBP 6
#define iNE 7
#define iSE 8
#define iNW 9
#define iSW 10
#define iTN 11
#define iBN 12
#define iTS 13
#define iBS 14
#define iTE 15
#define iBE 16
#define iTW 17
#define iBW 18

#include <stdlib.h>
#include <stdio.h>
#include "petsctime.h"
#include "petscvec.h"
#include "petscdmda.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_IJ_mv.h"

#include "CurvGrid.h"
#include "UData.h"


using namespace std;

class PoissonSolver
{
public:

    PoissonSolver(const std::string& object_name,
                  CurvGrid *grid,
                  UData *data);

    ~PoissonSolver();


    void Initialize();
    void DestroyHypreSolver();
    void CreateHypreSolver();
    void RemoveNullspace(HYPRE_IJVector &B, 
                         int i_lower);

    void PoissonRHS2_hypre(HYPRE_IJVector &B, 
                           PetscInt i_lower);    

    void PetsctoHypreVector(Vec A,    
                            HYPRE_IJVector &B, 
                            PetscInt i_lower);
    void HypretoPetscVector(HYPRE_IJVector &B, 
                            Vec A, 
                            PetscInt i_lower);
    void DestroyHypreMatrix();
    void DestroyHypreVector();
    void CreateHypreMatrix();
    void CreateHypreVector();

    void SolvePoisson(PetscInt ti);
    void Solve(PetscInt ti, PetscReal time);    

    void Setup(bool start);
    void ConvertPhi2();
   
    void PoissonLHS(HYPRE_IJMatrix &Ap); 

    void VolumeFlux(Vec lUcor,
                    PetscReal *ibm_Flux,          
                    PetscReal *ibm_Area);
    void AddIBMFluxtoOutlet(PetscReal ibm_Flux);
    void VolumeFlux(PetscReal *ibm_Flux, 
                               PetscReal *ibm_Area, 
                               int flg);
    void Projection();

    void UpdatePressure();

    PetscReal getResidual() {return d_norm;}
private:

    PetscErrorCode ReadFromInput();

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;

    HYPRE_Solver d_pcg_solver_p, d_precon_p;
    HYPRE_IJMatrix d_Ap;
    HYPRE_ParCSRMatrix d_par_d_Ap;
    HYPRE_IJVector d_Vec_p, d_Vec_p_rhs;
    HYPRE_ParVector d_par_Vec_p, d_par_Vec_p_rhs;

    Vec d_Gid;
    Vec d_Phi2;
    Vec d_Phi, d_lPhi;

    PetscReal d_poisson_threshold;

    PetscInt d_amg_agg;
    PetscInt d_amg_coarsentype;
    PetscReal d_amg_thresh;
    PetscInt d_poisson_it;
    PetscReal d_poisson_tol;
    PetscInt d_immersed;

    PetscInt d_rhs_count;
    PetscInt d_p_global_begin;
    PetscInt d_reduced_p_size;
    PetscInt d_local_Phi2_size;

    PetscReal d_norm;
};

#endif
