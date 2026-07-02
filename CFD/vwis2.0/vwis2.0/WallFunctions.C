#include "WallFunctions.h"


WallFunctions::WallFunctions(
    const std::string& object_name):
    d_object_name(object_name)
{
    d_pre_integrate_flag=0;
    d_n_yp=0;
    d_interval_yp=2;
    d_kappa=0.41;
}
  
WallFunctions::~WallFunctions()
{
    free(d_integration_buffer);
    free(d_integration_buffer_rough); 
}

void WallFunctions::wall_function_s(
    double nu, double ks, 
    double sc, double sb, 
    Cmpnts Ua, Cmpnts Uc, Cmpnts *Ub, 
    PetscReal *ustar, 
    double nx, double ny, double nz)
{
    double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
    double un = u_c * nx + v_c * ny + w_c * nz;
    double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
    double ut_mag = sqrt( ut*ut + vt*vt + wt*wt );

    double A=8.3;
    double B=1.0/7.0; 
        *ustar = pow( ut_mag * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));  /// 

    double kappa_rans = 0.4;
    double _ks=ks*0.033;
    if (_ks>1.e-9) {
        *ustar = ut_mag*kappa_rans/log(sc/_ks);
    }

    double ybp = sb*(*ustar)/nu;
    double ycp = sc*(*ustar)/nu;
    double ut_mag_modeled;
    if (ybp>12.0) {
        ut_mag_modeled = A*(*ustar)*pow(ybp,B);
        if (_ks>1.e-9) ut_mag_modeled = (*ustar)*log(sb/_ks)/kappa_rans;
    } 
    else {
        ut_mag_modeled = ut_mag*sb/sc;    
    }


    if(ut_mag>1.e-10) {
        ut *= ut_mag_modeled/ut_mag;
        vt *= ut_mag_modeled/ut_mag;
        wt *= ut_mag_modeled/ut_mag;
    }
    else ut=vt=wt=0;
                    
    // u = ut + (u.n)n
    (*Ub).x = ut + sb/sc * un * nx;
    (*Ub).y = vt + sb/sc * un * ny;
    (*Ub).z = wt + sb/sc * un * nz;
    
    (*Ub).x += Ua.x;
    (*Ub).y += Ua.y;
    (*Ub).z += Ua.z;
}

void WallFunctions::noslip(
    double Re, double sc, double sb, 
    Cmpnts Ua, Cmpnts Uc, Cmpnts *Ub, 
    PetscReal *ustar, 
    double nx, double ny, double nz)
{
    double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
    double un = u_c * nx + v_c * ny + w_c * nz;
    double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
    double ut_mag = sqrt( ut*ut + vt*vt + wt*wt );
    
    *ustar = sqrt ( ut_mag / sc / Re );
    
    (*Ub).x = sb/sc * u_c;
    (*Ub).y = sb/sc * v_c;
    (*Ub).z = sb/sc * w_c;
    
    (*Ub).x += Ua.x;
    (*Ub).y += Ua.y;
    (*Ub).z += Ua.z;
}

void WallFunctions::freeslip(
    double sc, double sb, 
    Cmpnts Ua, Cmpnts Uc, Cmpnts *Ub, 
    double nx, double ny, double nz)
{
  
    double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
    double un = u_c * nx + v_c * ny + w_c * nz;
    double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
    
    (*Ub).x = ut + sb/sc * un * nx;
    (*Ub).y = vt + sb/sc * un * ny;
    (*Ub).z = wt + sb/sc * un * nz;
    
    (*Ub).x += Ua.x;
    (*Ub).y += Ua.y;
    (*Ub).z += Ua.z;
    
}

//Can be static
double WallFunctions::utau_wf(double nu, double ks, double sb, double Ut_mag)
{

    double kappa_rans = 0.4;
    double _ks=ks*0.033;
    if (_ks>1.e-13) {
        return Ut_mag*kappa_rans/log(sb/_ks);
    } else {
        double A=8.3;
        double B=1.0/7.0; 
    }

}


double WallFunctions::u_Cabot(double nu, double y, double utau, double dpdn)
{
    return utau * utau * integrate_F( nu, utau, y );
}

double WallFunctions::u_Cabot_roughness(double nu, double y, double utau, 
                                        double dpdn, double ks)
{
  return utau * utau * integrate_F( nu, utau, y, ks );
};

double WallFunctions::f_Cabot(double nu, double u, double y, 
                              double utau, double dpdn)
{
    return utau * utau * integrate_F( nu, utau, y ) - u;
}

double WallFunctions::f_Cabot_roughness(double nu, double u, double y, 
                                        double utau, double dpdn, double ks)
{
    return utau * utau * integrate_F( nu, utau, y, ks ) - u;
}

double WallFunctions::df_Cabot(double nu, double u, double y, 
                               double utau, double dpdn)
{
    double eps=1.e-7;
    return ( f_Cabot(nu, u, y, utau+eps, dpdn) - 
             f_Cabot(nu, u, y, utau-eps, dpdn) ) / ( 2*eps ) ;
}

double WallFunctions::df_Cabot_roughness(double nu, double u, double y, 
                                         double utau, double dpdn, double ks)
{
    double eps=1.e-7;
    return ( f_Cabot_roughness(nu, u, y, utau+eps, dpdn, ks) - 
             f_Cabot_roughness (nu, u, y, utau-eps, dpdn, ks) ) / ( 2*eps ) ;
}


double WallFunctions::near_wall_eddy_viscosity(double yplus)// in fact, nu_t/nu
{
    return d_kappa * yplus * pow ( 1. - exp( - yplus / 19. ) , 2.0 );
};

double WallFunctions::near_wall_eddy_viscosity(double yplus, double yp_shift)
// in fact, near_wall_eddy_viscosity(/nu
{
    return d_kappa * (yplus+yp_shift) * pow (1.-exp(-(yplus+yp_shift)/19.), 2.0);
}


void WallFunctions::pre_integrate()
{
    if (d_pre_integrate_flag) return;
    else d_pre_integrate_flag=1;

    int max_yp=1e7;
    d_n_yp = ( max_yp / d_interval_yp ) ;
    
    d_integration_buffer = new double [ d_n_yp + 1 ];
    d_integration_buffer_rough = new double [ d_n_yp + 1 ];
    
    d_integration_buffer[0] = 0.;
    d_integration_buffer_rough[0] = 0.;
    
    for (int i=1; i<=d_n_yp; i++) {
        int N=24;
        double ya_p = (double)(i-1) * d_interval_yp;
        double yb_p = (double)(i) * d_interval_yp;
        double ydiff = yb_p - ya_p, dy_p = ydiff / (double)N;
        std::vector<double> E(N+1);
        double val=0, ybegin_p=ya_p;
        
        for (int k=0; k<=N; k++) 
            E[k] =  1. / ( 1. + near_wall_eddy_viscosity(ybegin_p + dy_p*k ) );
        
        for (int k=0; k<N; k++) {
            double F[4];
            F[0] = E[k];
            F[1] = 1./ ( 1. + near_wall_eddy_viscosity(ybegin_p+dy_p*1./3.) );
            F[2] = 1./ ( 1. + near_wall_eddy_viscosity(ybegin_p+dy_p*2./3.) );
            F[3] = E [k+1];
            val += dy_p  / 3.* ( 3*F[0] + 9*F[1] + 9*F[2] + 3*F[3] ) / 8.;
            ybegin_p += dy_p;
        }
        
        d_integration_buffer[i] = d_integration_buffer[i-1] + val;
        d_integration_buffer_rough[i] = d_integration_buffer_rough[i-1] + val;
    }
}

double WallFunctions::integrate_F(double nu, double utau, double yb)
{
    double val=0;
    
    pre_integrate();
    
    double ya_plus = 0 * utau / nu;
    double yb_plus = yb * utau / nu;
    
    
    int max_yp=1e7;
    int ib = (int) ( yb_plus / (double) d_interval_yp );
    int N=4;
    
    if ( yb_plus <= (double) max_yp ) {
        double int_a = 0;
        double int_b = (d_integration_buffer[ib+1] - d_integration_buffer[ib])/
                       (double) (d_interval_yp) * 
                           ( yb_plus - (double)(ib) * d_interval_yp ) + 
                       d_integration_buffer[ib];
        
        return ( int_b - int_a ) / utau;
    }
    else {
        val = d_integration_buffer[d_n_yp];
        ya_plus = max_yp;
    }
    
    
    double ydiff = yb_plus - ya_plus, dy = ydiff / (double)N;
    std::vector<double> E(N+1);
    double ybegin=ya_plus;
    
    for (int i=0; i<=N; i++) {
        E[i] =  1./ ( 1. + near_wall_eddy_viscosity(ybegin + dy*i ) );
    }
    
    for (int i=0; i<N; i++) {
            double F[4];
            F[0] = E[i];
        F[1] = 1./ ( 1. + near_wall_eddy_viscosity(ybegin+dy*1./3.) );
        F[2] = 1./ ( 1. + near_wall_eddy_viscosity(ybegin+dy*2./3.) );
        F[3] = E[i+1];
        val += dy / 3.* ( 3*F[0] + 9*F[1] + 9*F[2] + 3*F[3] ) / 8.;
        ybegin += dy;
    }
    
    val /= utau;
    return val;
}

double WallFunctions::integrate_F(double nu, double utau, double yb, double ks)
{
    double ks_plus = /*fabs*/(utau * ks / nu);
    double yp_shift = 0.9*(sqrt(ks_plus)-(ks_plus)*exp(-ks_plus/6.));
    
    if (yp_shift<0) return 0;
    
    double val=0;
        
    double ya_plus = 0 * utau / nu;
    double yb_plus = yb * utau / nu;
        
    int ib = (int) ( yb_plus / (double) d_interval_yp );
    int N=10;
    val=0;
    
    double ydiff = yb_plus - ya_plus;
    double  dy = ydiff / (double)N;
    std::vector<double> E(N+1);
    double ybegin=ya_plus;
    
    for (int i=0; i<=N; i++) {
        E[i] =  1./ (1. + near_wall_eddy_viscosity(ybegin + dy*i, yp_shift ));
    }
    
    for (int i=0; i<N; i++) {
        double F[4];
        F[0] = E[i];
        F[1] = 1./ (1. + near_wall_eddy_viscosity(ybegin+dy*1./3., yp_shift));
        F[2] = 1./ (1. + near_wall_eddy_viscosity(ybegin+dy*2./3., yp_shift));
        F[3] = E[i+1];
        val += dy / 3.* ( 3*F[0] + 9*F[1] + 9*F[2] + 3*F[3] ) / 8.;
        ybegin += dy;
    }
    
    val /= utau;
    return val;
};

double WallFunctions::find_utau_Cabot(double nu, double u, 
                                      double y, double guess, double dpdn)
{
    double x, x0=guess;
    int i;
    
    for (i=0; i<30; i++) {
        x = x0 - f_Cabot(nu, u, y, x0, dpdn)/df_Cabot(nu, u, y, x0, dpdn);
        if( fabs(x0 - x) < 1.e-10) break;
        x0 = x;
    }
    
    if (fabs(x0 - x) > 1.e-5 && i>=29 ) 
         printf("\nWallFunctions::find_utau_Cabot Iteration Failed \n");
        
    return x;
}

double WallFunctions::find_utau_Cabot_roughness(
    double nu, double u, 
    double y, double guess, 
    double dpdn, double ks)
{
    double x, x0=guess;
    
    int i;
    
    for (i=0; i<30; i++) {
        x = x0 - f_Cabot_roughness(nu, u, y, x0, dpdn, ks)/
                 df_Cabot_roughness(nu, u, y, x0, dpdn, ks);
        if ( fabs(x0 - x) < 1.e-10) break;
        x0 = x;
    }
    
    if ( fabs(x0 - x) > 1.e-5 && i>=29 ) 
       printf("\nWallFunctions::find_utau_Cabot_roughtness Iteration Failed\n");
        
    return x;
}

