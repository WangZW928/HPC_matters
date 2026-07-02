#ifndef included_RHSSolver
#define included_RHSSolver

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "petscvec.h"
#include "petscdmda.h"

#include "CurvGrid.h"
#include "UData.h"
#include "LESModel.h"

using namespace std;

class RHSSolver
{
public:

    RHSSolver(
        const std::string& object_name,
        CurvGrid *grid,
        UData *data,
        LESModel *les);

    ~RHSSolver();

    PetscErrorCode Initialize();

    void CalculatePressureGradient();

    PetscErrorCode Solve(Vec Rhs, double scale);

    void Calculate_dP_dc_de_dz(
        double dp_dx, double dp_dy, double dp_dz,
        Cmpnts csi, Cmpnts eta, Cmpnts zet,
        double aj,
        double *dpdc, double *dpde, double *dpdz);

    Vec getVisc1(){return d_Visc1;}
    Vec getVisc2(){return d_Visc2;}
    Vec getVisc3(){return d_Visc3;}
    Vec getFp(){return d_Fp;}

private:
    PetscErrorCode ReadFromInput();    

    std::string d_object_name;
    CurvGrid *d_grid;
    UData *d_data;
    LESModel *d_les;

    Vec d_Div1;
    Vec d_Div2;
    Vec d_Div3;
    Vec d_Visc1;
    Vec d_Visc2;
    Vec d_Visc3;
    Vec d_Fp;

    PetscReal d_mean_pressure_gradient;
    PetscBool d_dpdz_set;
    PetscReal d_inlet_flux;
    PetscInt d_second_order;
    PetscInt d_immersed;

};

#endif


