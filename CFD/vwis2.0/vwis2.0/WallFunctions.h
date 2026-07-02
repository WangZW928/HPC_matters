#ifndef included_WallFunctions
#define included_WallFunctions


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "functions.h"

class WallFunctions
{
public:
    WallFunctions(const std::string& object_name);
    
    ~WallFunctions();


    //Static Functions
    static void 
    wall_function_s(
        double nu, double ks, 
        double sc, double sb, 
        Cmpnts Ua, Cmpnts Uc, Cmpnts *Ub, 
        PetscReal *ustar, 
        double nx, double ny, double nz);

    static void 
    noslip(
        double Re, double sc, double sb,
        Cmpnts Ua, Cmpnts Uc, Cmpnts *Ub,
        PetscReal *ustar,
        double nx, double ny, double nz);

    static void 
    freeslip(
        double sc, double sb,
        Cmpnts Ua, Cmpnts Uc, Cmpnts *Ub,
        double nx, double ny, double nz);

    static double 
    utau_wf(
       double nu, double ks, double sb, double Ut_mag);

    //Methods
    double find_utau_Cabot(
        double nu, double u, 
        double y, double guess, 
        double dpdn);
 
    double find_utau_Cabot_roughness(
        double nu, double u, 
        double y, double guess, 
        double dpdn, double ks);


private:
    double u_Cabot(double nu, double y, double utau, double dpdn);
    double f_Cabot(double nu, double u, double y, double utau, double dpdn);
    double df_Cabot(double nu, double u, double y, double utau, double dpdn);

    double u_Cabot_roughness(double nu, double y, double utau,
                             double dpdn, double ks);
    double f_Cabot_roughness(double nu, double u, double y,
                             double utau, double dpdn, double ks);
    double df_Cabot_roughness(double nu, double u, double y,
                              double utau, double dpdn, double ks);
    
    double near_wall_eddy_viscosity(double yplus);
    double near_wall_eddy_viscosity(double yplus, double yp_shift);

    void pre_integrate();
    double integrate_F(double nu, double utau, double yb);
    double integrate_F(double nu, double utau, double yb, double ks);
    

    std::string d_object_name;
    double d_kappa;

    int d_pre_integrate_flag;
    int d_n_yp;
    int d_interval_yp;
    double *d_integration_buffer;
    double *d_integration_buffer_rough;
    

};



#endif
