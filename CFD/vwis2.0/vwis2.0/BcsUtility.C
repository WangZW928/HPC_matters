#include "BcsUtility.h"

BcsUtility::BcsUtility(
   const std::string& object_name,
    CurvGrid *grid,
    UData *data,
    PlaneExtraction *plane):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data),
    d_iplane(plane)
{
    d_inletprofile = 1;
    d_inlet_flux = -1;
    d_threshold = 0.1;
    d_inletArea = 0;
    d_k_area_allocated = 0;
    d_initial_perturbation = 0;
    d_initial_gaussian_perturbation = 0;
    d_magnitude_gaussian_perturbation = 0;
    d_fluct_rms = 0.005;

    ReadFromInput();
}

BcsUtility::~BcsUtility()
{
}

void BcsUtility::CalculateInletArea()
{
    PetscInt i, j, k;

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    PetscInt xs = info.xs, xe = info.xs + info.xm;
    PetscInt ys = info.ys, ye = info.ys + info.ym;
    PetscInt zs = info.zs, ze = info.zs + info.zm;
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt lxs, lxe, lys, lye, lzs, lze;
 
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;
  
    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    Cmpnts ***csi, ***eta, ***zet;
    PetscReal ***nvert, ***level, ***rho, ***aj;
   
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();
    
    Vec lNvert = d_data->getlNvert();
 
    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, Aj, &aj);
    DMDAVecGetArray(da, lNvert, &nvert);

    if(!d_k_area_allocated) 
    {
        d_k_area_allocated = 1;
        d_k_area = new double[mz];
        d_k_area_ibnode = new double[mz];
    }
        
    std::vector<double> lArea(mz), lArea_ibm(mz);
    
    std::fill ( lArea.begin(), lArea.end(), 0 );
    std::fill ( lArea_ibm.begin(), lArea_ibm.end(), 0 );

    for (k=lzs; k<lze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) 
            {
                if (j>=1 && j<=my-2 && i>=1 && i<=mx-2) 
                {
                    double area = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                        zet[k][j][i].y*zet[k][j][i].y + 
                                        zet[k][j][i].z*zet[k][j][i].z );
                    if (nvert[k+1][j][i]+nvert[k][j][i] < 0.1) 
                        lArea[k] += area;
                    else if (nvert[k+1][j][i]+nvert[k][j][i] < 2.1) 
                        lArea_ibm[k] += area;
                }
            }

    MPI_Allreduce(&lArea[0], &d_k_area[0], mz, 
                  MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&lArea_ibm[0], &d_k_area_ibnode[0], mz, 
                  MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    
    d_k_area[0] = d_k_area[1];

    d_mean_k_area = 0;
    d_mean_k_area_ibnode = 0;
    
    for (k=1; k<=mz-2; k++) 
    {
        d_mean_k_area += d_k_area[k];
        d_mean_k_area_ibnode += d_k_area_ibnode[k];
    }
    
    d_mean_k_area /= (double)(mz-2);
    d_mean_k_area_ibnode /= (double)(mz-2);
   
    //Set this in data
    d_data->setMeanArea(d_mean_k_area);
    d_data->setMeanAreaIb(d_mean_k_area_ibnode);

    PetscInt k_periodic = d_grid->isKPeriodic();
    PetscInt kk_periodic = d_grid->isKKPeriodic();
 
    if (d_grid->getBC(4) == INLET || k_periodic || kk_periodic) 
    {
        double lArea=0;
        if (zs==0) 
        {
            k = 1;
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) 
                {
                    double k_area = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                    zet[k][j][i].y*zet[k][j][i].y + 
                                    zet[k][j][i].z*zet[k][j][i].z );
                
                if (nvert[k][j][i] < d_threshold) {                  
                    lArea += k_area;
                }
            }
        }
        GlobalSum_All(&lArea, &d_inletArea, PETSC_COMM_WORLD);    
    }
    
    if (d_grid->getBC(5) == INLET) 
    {
        double lArea=0;
        if (ze==mz) 
        {
            k = mz-2;
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) 
                {
                    double k_area = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                          zet[k][j][i].y*zet[k][j][i].y + 
                                          zet[k][j][i].z*zet[k][j][i].z );

                    if (nvert[k][j][i] < d_threshold) {                  
                        lArea += k_area;
                    } 
                }
        }
        GlobalSum_All(&lArea, &d_inletArea, PETSC_COMM_WORLD);    
    }
       
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(da, Aj, &aj);
    
    PetscPrintf(PETSC_COMM_WORLD, 
                "\n...(Fluid) Inlet Area:%f, (Fluid) Inlet Flux: %f\n\n", 
                d_inletArea, d_inlet_flux);
}


PetscErrorCode BcsUtility::InflowFlux(PetscInt ti)
{
  
    PetscInt i, j, k;
    PetscReal r, uin, xc, yc;
    Vec Coor;
    Cmpnts    ***ucont, ***ubcs, ***ucat, ***coor, ***csi, ***eta, ***zet;
    Cmpnts ***icsi;
    
    PetscReal Umax=1.5;

    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    PetscInt xs = info.xs, xe = info.xs + info.xm;
    PetscInt ys = info.ys, ye = info.ys + info.ym;
    PetscInt zs = info.zs, ze = info.zs + info.zm;
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt lxs, lxe, lys, lye, lzs, lze;
    
    PetscReal ***nvert, ***level, ***aj; 
    
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;
  
    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;
  
    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;
    
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();
    Vec ICsi = d_grid->getlICsi();
    
    Vec lNvert = d_data->getlNvert();
    Vec Ucont = d_data->getUcont();
    Vec Ucat = d_data->getUcat();
    Vec Ubcs = d_data->getUbcs();
  
    DMDAVecGetArray(da, lNvert, &nvert); 
    DMGetCoordinatesLocal(da, &Coor);

    //DMDAGetGhostedCoordinates(da, &Coor);
    DMDAVecGetArray(fda, Coor, &coor);
    DMDAVecGetArray(fda, Ucont, &ucont);
    DMDAVecGetArray(fda, Ubcs, &ubcs);
    DMDAVecGetArray(fda, Ucat,  &ucat);
  
    DMDAVecGetArray(fda, Csi,  &csi);
    DMDAVecGetArray(fda, Eta,  &eta);
    DMDAVecGetArray(fda, Zet,  &zet);
    DMDAVecGetArray(da, Aj,  &aj);

    DMDAVecGetArray(fda, ICsi,  &icsi);
  
    PetscReal FluxIn=0.;
    
    double lFluxIn0=0, sumFluxIn0=0;
    double lFluxIn1=0, sumFluxIn1=0;
  
    srand( time(NULL)); 
    int fluct=0;
      
    if (d_grid->getBC(4) == INLET) 
    {
        if (zs==0) 
        {
            k = 0;
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) 
                {
                    xc = (coor[k+1][j][i].x + coor[k+1][j-1][i].x + 
                          coor[k+1][j][i-1].x + coor[k+1][j-1][i-1].x) * 0.25;
                    yc = (coor[k+1][j][i].y + coor[k+1][j-1][i].y + 
                          coor[k+1][j][i-1].y + coor[k+1][j-1][i-1].y) * 0.25;
                    r = sqrt(xc * xc + yc * yc);
                    double area = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                        zet[k][j][i].y*zet[k][j][i].y + 
                                        zet[k][j][i].z*zet[k][j][i].z );
                
                    //uin is flux in
                    if (d_inletprofile == 0) uin = 1;
                    //uin is flux/area
                    else if (d_inletprofile==1) 
                    {
                        if (d_inlet_flux<0) uin=1.;
                        else uin = d_inlet_flux/d_inletArea;
                    // uniform flow with noise
                    } else if (d_inletprofile==2) {   
                        if(d_inlet_flux<0) uin=1.;
                        else uin = d_inlet_flux/d_inletArea;
                        fluct=1;
                    // parabolic with channel height 2, bulk = 1
                    } else if (d_inletprofile == 3) { 
                        if(d_inlet_flux<0) uin=1.;
                        else uin = d_inlet_flux/d_inletArea;
                    
                        uin *= 1.5 * (2 * yc - yc * yc);
                    // parabolic with channel height 1, bulk = 1
                    } else if (d_inletprofile == 4) { 
                        if(d_inlet_flux<0) uin=1.;
                        else uin = d_inlet_flux/d_inletArea;

                        uin *= 6 * (yc - yc * yc);
                    // Power law for hemisphere case
                    } else if (d_inletprofile==10) {    
                        double delta = 0.45263, a=5.99;
                        if( yc>=delta ) uin=1.0;
                        else if(yc<=0) uin=0.0;
                        else uin = pow( yc/delta, 1./a );
                    // backward facing step 
                    }  else if (d_inletprofile==11) {   
                        fluct=1;
                        yc -= 3;
                        double delta=1.2;
                        if ( 2-fabs(yc)<delta ) 
                            uin = pow( (2-fabs(yc))/delta, 1./7 );
                        else uin=1;
                    // pipe shear stress test
                    }  else if (d_inletprofile==12) {    
                       uin = 2*(1 - pow(r/0.5,2.0) );
                    // periodic channel flow
                    } else if (d_inletprofile==13) { 
                    // curved pipe flow (Anwer)   
                    } else if(d_inletprofile==14) {   
                        double R[11] = {0.000,0.062,0.125,0.188,0.250,
                                        0.312,0.354,0.399,0.438,0.469,0.500};
                        double W[11] = {1.120,1.115,1.110,1.093,1.050,
                                        1.010,0.963,0.915,0.825,0.730,0.000};
                    
                        int ii;
                        for(ii=1; ii<11; ii++) {
                            if( R[ii]>=r && R[ii-1]<r ) break;
                        }
                        uin = ( W[ii] - W[ii-1] ) / ( R[ii] - R[ii-1] ) * 
                              ( r - R[ii-1] ) + W[ii-1];
                    
                        if(r>0.5) uin = 0;
                    // round jet (Longmire)
                    } else if(d_inletprofile==15) {    
                        if(r>0.5) uin = 0;
                        else uin = 1;
                    // periodic pipe flow
                    } else if (d_inletprofile==16) {    
                    // enright test
                    } else if(d_inletprofile==20){
                    // laminar poiseulle flow profile
                    } else if (d_inletprofile==21) {   
                        double w_bulk=1.0;//d_inlet_flux/SumArea;
                        uin = 1.5 * w_bulk * yc * ( 2 - yc );
                    //2D Mixing layer
                    } else if (d_inletprofile==25) {
                        fluct = 1;
                        double gam = 3.0;
                        double U1 = 1.0;
                        double U2 = U1/gam;
                        double del = 1.0;
                        uin = ( (U1+U2)/2.0 + 
                                        (U1-U2)/2.0*tanh((yc-10)/del) );
                    //2D jet
                    } else if (d_inletprofile==26) {
                        fluct = 1;
                        double Sigma = 7.5;
                        double Radius = 5;
 
                        uin = (.091+1.091)/2. + 
                              (1./2.)*tanh(Sigma*(1-2*abs(yc-60)/Radius)); 
                    // saved data for LES
                    } else if (d_inletprofile == 100) {    
                    } else {
                        PetscPrintf(PETSC_COMM_SELF, 
                                    "WRONG Inlet Profile %d "
                                    "Setting: U_in = 0\n", d_inletprofile);
                        uin = 0.;
                    }
               
                    // pseudo-periodic BC in k-direction 
                    if (d_pseudo_periodic || d_inletprofile==100) {    
                        fluct=0;
                    }
                
                    if (nvert[k+1][j][i] < d_threshold) {   
                        if (d_pseudo_periodic || d_inletprofile==100) 
                        {
                            double u = d_ucat_plane[j][i].x;
                            double v = d_ucat_plane[j][i].y;
                            double w = d_ucat_plane[j][i].z;
                        
                            ucat[k][j][i].x = u;
                            ucat[k][j][i].y = v;
                            ucat[k][j][i].z = w;
                            ucont[k][j][i].z = 
                                0.5*(u+ucat[k+1][j][i].x) * zet[k][j][i].x + 
                                0.5*(v+ucat[k+1][j][i].y) * zet[k][j][i].y + 
                                0.5*(w+ucat[k+1][j][i].z) * zet[k][j][i].z;
                        } else {
                            ucont[k][j][i].z = uin * area;
                            ucont[k][j][i].x = 0;
                            ucont[k][j][i].y = 0;
                        
                            if (fluct) 
                            {
                                double areai = 
                                       sqrt( csi[k][j][i].x*csi[k][j][i].x + 
                                             csi[k][j][i].y*csi[k][j][i].y + 
                                             csi[k][j][i].z*csi[k][j][i].z );
                                double areaj = 
                                       sqrt( eta[k][j][i].x*eta[k][j][i].x +
                                             eta[k][j][i].y*eta[k][j][i].y + 
                                             eta[k][j][i].z*eta[k][j][i].z );
                                double n1 = randn_notrig();
                                double n2 = randn_notrig();
                                double n3 = randn_notrig();
                                ucont[k][j][i].x += 
                                 n1 * d_magnitude_gaussian_perturbation * areai;
                                ucont[k][j][i].y += 
                                 n2 * d_magnitude_gaussian_perturbation * areaj;
                                ucont[k][j][i].z += 
                                  n3 *d_magnitude_gaussian_perturbation * area;
                            }                        
                            Cmpnts u;
                            d_data->Contra2Cart_single(
                                         csi[k][j][i], eta[k][j][i], 
                                         zet[k][j][i], ucont[k][j][i], 
                                          &u);
                        
                            if (fluct) 
                            {
                                //double f = d_fluct_rms;            // 1% noise
                                //int n1 = rand() % 20000 - 10000
                                //int n2 = rand() % 20000 - 10000;
                                //int n3 = rand() % 20000 - 10000;
                                //u.x += ((double)n1)/10000. * f * uin;
                                //u.y += ((double)n2)/10000. * f * uin;
                                //u.z += ((double)n3)/10000. * f * uin;
                            }
                            ucat[k][j][i].x = - ucat[k+1][j][i].x + 2*u.x;
                            ucat[k][j][i].y = - ucat[k+1][j][i].y + 2*u.y;
                            ucat[k][j][i].z = - ucat[k+1][j][i].z + 2*u.z;
                                        
                        
                        } 
                        ubcs[k][j][i] = ucat[k][j][i];
                    } else {
                        ucat[k][j][i].z = 0;    //seokkoo
                        ubcs[k][j][i].z = 0;
                        ucont[k][j][i].z = 0;
                    }
                    lFluxIn0 += ucont[k][j][i].z;

                    //if (fluct) {
                    //    double f = d_fluct_rms;       // 1% noise
                    //    int n1 = rand() % 20000 - 10000; // RAND_MAX = 65535
                    //    ucont[k][j][i].z *= ( 1 + ((double)n1)/10000. * f );
                    //}
                
                    lFluxIn1 += ucont[k][j][i].z;
                }
        }
        
        GlobalSum_All(&lFluxIn0, &sumFluxIn0, PETSC_COMM_WORLD);
        GlobalSum_All(&lFluxIn1, &sumFluxIn1, PETSC_COMM_WORLD);
        
        if (d_inlet_flux<0) sumFluxIn0=1*d_inletArea;
        else sumFluxIn0=d_inlet_flux;

        if (d_pseudo_periodic || d_inletprofile==100 || d_inletprofile==14) {
            PetscPrintf(PETSC_COMM_WORLD,  
                        "\nConstant Flux is %f !\n\n", d_inlet_flux);
        }
        
        PetscPrintf(PETSC_COMM_WORLD, 
                    "\n...Fluxin0:%f, Fluxin1:%f, Area:%f\n\n", 
                    sumFluxIn0, sumFluxIn1, d_inletArea);
        
        if (zs==0 && fabs(sumFluxIn0 - sumFluxIn1)>1.e-9 && d_inlet_flux>0) 
        {
            
            PetscPrintf(PETSC_COMM_WORLD, 
                        "The inflow is corrected to Flux=%f \n\n", sumFluxIn0);
            k = 0;
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {
                double vf=1.0;

                double A = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                 zet[k][j][i].y*zet[k][j][i].y + 
                                 zet[k][j][i].z*zet[k][j][i].z );
                if (nvert[k+1][j][i] < d_threshold) 
                    ucont[k][j][i].z += (sumFluxIn0 - sumFluxIn1) * 
                                         A * vf / d_inletArea;
            }
        }
        
        FluxIn = 0;
        if (zs==0) {
            k = 0;
            for (j=lys; j<lye; j++) 
                for (i=lxs; i<lxe; i++) {
                    if (nvert[k+1][j][i] < d_threshold) {
                        FluxIn += ucont[k][j][i].z;
                    }
                }
        }
    // made just for subcritical levelset flow    
    } else if (d_grid->getBC(5) == INLET) { 
        if (ze==mz) 
        {
            k = mz-1;
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) 
                {
                    double area = sqrt( zet[k-1][j][i].x*zet[k-1][j][i].x + 
                                        zet[k-1][j][i].y*zet[k-1][j][i].y + 
                                        zet[k-1][j][i].z*zet[k-1][j][i].z );
      
                    if (d_inletprofile == 1) {
                        if (d_inlet_flux<0) uin=1.;
                        else uin = d_inlet_flux/d_inletArea;
                    }
                    if (ti==0) {
                        ucat[k][j][i].z = uin;
                        ubcs[k][j][i].z = uin;
                        ucont[k-1][j][i].z = uin * area;
                        ucont[k-1][j][i].x = 0;
                        ucont[k-1][j][i].y = 0;
                    // treated in implicitsolverand correct with outflow scale
                    }else {}
                
                    lFluxIn0 += ucont[k-1][j][i].z;

                    lFluxIn1 += ucont[k-1][j][i].z;
               }
        }
        
        GlobalSum_All(&lFluxIn0, &sumFluxIn0, PETSC_COMM_WORLD);
        GlobalSum_All(&lFluxIn1, &sumFluxIn1, PETSC_COMM_WORLD);
        PetscPrintf(PETSC_COMM_WORLD, 
                    "\n...Fluxin0:%f, Fluxin1:%f, Area:%f\n\n", 
                    sumFluxIn0, sumFluxIn1, d_inletArea);
        
        FluxIn = 0;
        if (ze==mz) 
        {
            k = mz-2;
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {
                    if (nvert[k][j][i] < d_threshold) 
                        FluxIn += ucont[k][j][i].z;
                }
        }
    } else if(d_grid->getBC(0)==11) {
        FluxIn = 0;
        if (xs==0) 
        {
            i = 0;
            for (j=lys; j<lye; j++) 
                for (k=lzs; k<lze; k++) {
                    if (nvert[k][j][i+1] < d_threshold) {
                        double zc = 0.25*
                                   (coor[k][j][i+1].z+coor[k-1][j][i+1].z +
                                    coor[k][j-1][i+1].z+coor[k-1][j-1][i+1].z);
                    
                        if (zc <= 0 ) {
                            double u=0, v=0, w=1.;
                            ucont[k][j][i].x = u * icsi[k][j][i].x + 
                                               v * icsi[k][j][i].y + 
                                               w * icsi[k][j][i].z;
                        
                            FluxIn += ucont[k][j][i].x;
                        } else { }    // outflow
                    }
                }
        }
    }
    GlobalSum_All(&FluxIn, &d_FluxInSum, PETSC_COMM_WORLD);
    
  
    DMDAVecRestoreArray(fda, Coor, &coor);
    DMDAVecRestoreArray(fda, Ucont, &ucont);
    DMDAVecRestoreArray(fda, Ubcs, &ubcs);
    DMDAVecRestoreArray(fda, Ucat,  &ucat);
  
    DMDAVecRestoreArray(fda, Csi,  &csi);
    DMDAVecRestoreArray(fda, Eta,  &eta);
    DMDAVecRestoreArray(fda, Zet,  &zet);
    DMDAVecRestoreArray(da, Aj,  &aj);

    DMDAVecRestoreArray(fda, ICsi,  &icsi);
    
    DMDAVecRestoreArray(da, lNvert, &nvert);   
    
    return 0;
}

PetscErrorCode BcsUtility::OutflowFlux() 
{
  
    PetscInt i, j, k;
    PetscReal FluxOut;
    Vec Coor;
    Cmpnts ***ucont, ***ubcs, ***ucat, ***coor;
  

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    PetscInt xs = info.xs, xe = info.xs + info.xm;
    PetscInt ys = info.ys, ye = info.ys + info.ym;
    PetscInt zs = info.zs, ze = info.zs + info.zm;
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt lxs, lxe, lys, lye, lzs, lze;
  
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;
  
    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;
  
    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;
 
    DMGetCoordinatesLocal(da,&Coor);

    Vec lNvert = d_data->getlNvert();
    Vec Ucont = d_data->getUcont();
    Vec Ucat = d_data->getUcat();
    Vec Ubcs = d_data->getUbcs();
 
    DMDAVecGetArray(fda, Coor, &coor);
    DMDAVecGetArray(fda, Ucont, &ucont);
    DMDAVecGetArray(fda, Ubcs, &ubcs);
    DMDAVecGetArray(fda, Ucat,  &ucat);
 
    PetscReal ***nvert; 
    DMDAVecGetArray(da, lNvert, &nvert);   
    

    FluxOut = 0;
  
    if (d_grid->getBC(5) == 4) 
    {    
        if (ze==mz) 
        {
            k = mz-2;
            for (j=lys; j<lye; j++) 
                for (i=lxs; i<lxe; i++) {
                    if (nvert[k][j][i] < d_threshold) 
                        FluxOut += ucont[k][j][i].z;
                }
        } else 
            FluxOut = 0;
    } else if (d_grid->getBC(4) == 4) {    
        if (zs==0) 
        {
            k = 0;
            for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
                    if (nvert[k+1][j][i] < d_threshold)
                        FluxOut += ucont[k][j][i].z;
                }
            }
        } else 
            FluxOut = 0;
    }
    GlobalSum_All(&FluxOut, &d_FluxOutSum, PETSC_COMM_WORLD);

    DMDAVecRestoreArray(fda, Coor, &coor);
    DMDAVecRestoreArray(fda, Ucont, &ucont);
    DMDAVecRestoreArray(fda, Ubcs, &ubcs);
    DMDAVecRestoreArray(fda, Ucat,  &ucat);
  
    DMDAVecRestoreArray(da, lNvert, &nvert); 
    return 0;
}

/* Boundary condition definition (array d_grid->getBC(0-5]):
   0:    interpolation
   1:    solid wall (not moving)
   2:    moving solid wall (U=1)
   5:    Inlet
   4:    Outlet
   8:   Characteristic BC
*/
PetscErrorCode BcsUtility::FormBcs(PetscInt ti, int outflow_scale) //FSInfo *fsi
{
    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    PetscInt xs = info.xs, xe = info.xs + info.xm;
    PetscInt ys = info.ys, ye = info.ys + info.ym;
    PetscInt zs = info.zs, ze = info.zs + info.zm;
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt lxs, lxe, lys, lye, lzs, lze;
    PetscInt i, j, k;

    Vec Coor;
    Cmpnts ***ucont, ***ubcs, ***ucat, ***coor, ***csi, ***eta, ***zet;
     Cmpnts ***kzet;
    PetscScalar FluxIn, FluxOut, ratio;
    PetscScalar lArea, AreaSum, ***level;
    PetscScalar FarFluxIn=0., FarFluxOut=0., FarFluxInSum, FarFluxOutSum;
    PetscScalar FarAreaIn=0., FarAreaOut=0., FarAreaInSum, FarAreaOutSum;
    PetscScalar FluxDiff, VelDiffIn, VelDiffOut;
    Cmpnts V_frame;
    PetscInt moveframe=1;

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    double ibm_Flux=0, ibm_Area=0;

    Vec lNvert = d_data->getlNvert();
    Vec Ucont = d_data->getUcont();
    Vec lUcont = d_data->getlUcont();
    Vec Ucat = d_data->getUcat();
    Vec lUcat = d_data->getlUcat();
    Vec Ubcs = d_data->getUbcs();

    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();
    Vec KZet = d_grid->getlKZet();

    DMGetCoordinatesLocal(da, &Coor);
   
    DMDAVecGetArray(fda, Coor, &coor);
    DMDAVecGetArray(fda, Ubcs, &ubcs);

    DMDAVecGetArray(fda, Csi,  &csi);
    DMDAVecGetArray(fda, Eta,  &eta);
    DMDAVecGetArray(fda, Zet,  &zet);
    DMDAVecGetArray(fda, KZet,  &kzet);


    d_data->Contra2Cart();
    DMDAVecGetArray(fda, Ucat,  &ucat);
    DMDAVecGetArray(fda, Ucont, &ucont);

/* ========================================             */
/*   FAR-FIELD BC */
/* ========================================             */


    if (d_grid->getBC(0)==6) {
        if (xs == 0) {
            i= xs;
            for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                    ubcs[k][j][i].x = ucat[k][j][i+1].x;
                    ubcs[k][j][i].y = ucat[k][j][i+1].y;
                    ubcs[k][j][i].z = ucat[k][j][i+1].z;    
                    ucont[k][j][i].x = ubcs[k][j][i].x * csi[k][j][i+1].x;
                    FarFluxIn += ucont[k][j][i].x;
                    FarAreaIn += csi[k][j][i].x;
                }
            }
        }
    }
  
    if (d_grid->getBC(1)==6) {
        if (xe==mx) {
            i= xe-1;
            for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                    ubcs[k][j][i].x = ucat[k][j][i-1].x;
                    ubcs[k][j][i].y = ucat[k][j][i-1].y;
                    ubcs[k][j][i].z = ucat[k][j][i-1].z;
                    ucont[k][j][i-1].x = ubcs[k][j][i].x * csi[k][j][i-1].x;
                    FarFluxOut += ucont[k][j][i-1].x;
                    FarAreaOut += csi[k][j][i-1].x;
                }
            }
        }
    }

    if (d_grid->getBC(2)==6) {
        if (ys==0) {
            j= ys;
            for (k=lzs; k<lze; k++) {
                for (i=lxs; i<lxe; i++) {
                    ubcs[k][j][i].x = ucat[k][j+1][i].x;
                    ubcs[k][j][i].y = ucat[k][j+1][i].y;
                    ubcs[k][j][i].z = ucat[k][j+1][i].z;
                    ucont[k][j][i].y = ubcs[k][j][i].y * eta[k][j+1][i].y;
                    FarFluxIn += ucont[k][j][i].y;
                    FarAreaIn += eta[k][j][i].y;
                }
            }
        }
    }
  
    if (d_grid->getBC(3)==6) {
        if (ye==my) {
            j=ye-1;
            for (k=lzs; k<lze; k++) {
                for (i=lxs; i<lxe; i++) {
                    ubcs[k][j][i].x = ucat[k][j-1][i].x;
                    ubcs[k][j][i].y = ucat[k][j-1][i].y;
                    ubcs[k][j][i].z = ucat[k][j-1][i].z;
                    ucont[k][j-1][i].y = ubcs[k][j][i].y * eta[k][j-1][i].y;
                    FarFluxOut += ucont[k][j-1][i].y;
                    FarAreaOut += eta[k][j-1][i].y;
                }
            }
        }
    }

    if (d_grid->getBC(4)==6) {
        if (zs==0) {
            k = 0;
            for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {  
                     ubcs[k][j][i].x = ucat[k+1][j][i].x;
                     ubcs[k][j][i].y = ucat[k+1][j][i].y;
                     ubcs[k][j][i].z = ucat[k+1][j][i].z;
                     ucont[k][j][i].z = ubcs[k][j][i].z * zet[k+1][j][i].z;
                     FarFluxIn += ucont[k][j][i].z;
                     FarAreaIn += zet[k][j][i].z;
                }
            }
        } 
    }

    if (d_grid->getBC(5)==6) {
        if (ze==mz) {
            k = ze-1;
            for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {  
                    ubcs[k][j][i].x = ucat[k-1][j][i].x;
                    ubcs[k][j][i].y = ucat[k-1][j][i].y;
                    ubcs[k][j][i].z = ucat[k-1][j][i].z;
                    ucont[k-1][j][i].z = ubcs[k][j][i].z * zet[k-1][j][i].z;
                    FarFluxOut += ucont[k-1][j][i].z;
                    FarAreaOut += zet[k-1][j][i].z;
                }
            }
        }
    }

    GlobalSum_All(&FarFluxIn, &FarFluxInSum, PETSC_COMM_WORLD);
    GlobalSum_All(&FarFluxOut, &FarFluxOutSum, PETSC_COMM_WORLD);

    GlobalSum_All(&FarAreaIn, &FarAreaInSum, PETSC_COMM_WORLD);
    GlobalSum_All(&FarAreaOut, &FarAreaOutSum, PETSC_COMM_WORLD);

    if (d_grid->getBC(5)==6) {
        FluxDiff = 0.5*(FarFluxInSum - FarFluxOutSum) ;
        VelDiffIn  = FluxDiff / FarAreaInSum ;
        if (fabs(FluxDiff) < 1.e-6) VelDiffIn = 0.;
        if (fabs(FarAreaInSum) <1.e-6) VelDiffIn = 0.;

        VelDiffOut  = FluxDiff / FarAreaOutSum ;
        if (fabs(FluxDiff) < 1.e-6) VelDiffOut = 0.;
        if (fabs(FarAreaOutSum) <1.e-6) VelDiffOut = 0.;

        PetscPrintf(PETSC_COMM_WORLD, 
                    "Far Flux Diff %le %le %le %le %le %le %le\n", 
                     FarFluxInSum, FarFluxOutSum, FluxDiff, 
                     FarAreaInSum, FarAreaOutSum, VelDiffIn, VelDiffOut);
        
    }


    // scale global mass conservation

    if (d_grid->getBC(5)==6) {
        if (ze==mz) {
            k = ze-1;
            for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
                    ubcs[k][j][i].z = ucat[k-1][j][i].z + VelDiffOut;
                    ucont[k-1][j][i].z = ubcs[k][j][i].z * zet[k-1][j][i].z;
                }
             }
        }
    }

    if (d_grid->getBC(3)==6) {
        if (ye==my) {
            j=ye-1;
            for (k=lzs; k<lze; k++) {
                for (i=lxs; i<lxe; i++) {
                    ubcs[k][j][i].y = ucat[k][j-1][i].y + VelDiffOut;
                    ucont[k][j-1][i].y = ubcs[k][j][i].y * eta[k][j-1][i].y;
                }
            }
        }
    }
    
    if (d_grid->getBC(1)==6) {
        if (xe==mx) {
            i= xe-1;
            for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                    ubcs[k][j][i].x = ucat[k][j][i-1].x + VelDiffOut;
                    ucont[k][j][i-1].x = ubcs[k][j][i].x * csi[k][j][i-1].x;
                }
            }
        }
    }


    if (d_grid->getBC(0)==6) {
        if (xs == 0) {
            i= xs;
            for (k=lzs; k<lze; k++) {
                for (j=lys; j<lye; j++) {
                    ubcs[k][j][i].x = ucat[k][j][i+1].x - VelDiffIn;
                    ucont[k][j][i].x = ubcs[k][j][i].x * csi[k][j][i+1].x;
                }
            }
        }
    }
  

    if (d_grid->getBC(2)==6) {
        if (ys==0) {
            j= ys;
            for (k=lzs; k<lze; k++) {
                for (i=lxs; i<lxe; i++) {
                     ubcs[k][j][i].y = ucat[k][j+1][i].y - VelDiffIn;
                     ucont[k][j][i].y = ubcs[k][j][i].y * eta[k][j+1][i].y;
                }
            }
        }
    }
  

    if (d_grid->getBC(4)==6) {
        if (zs==0) {
            k = 0;
            for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
                     ubcs[k][j][i].z = ucat[k+1][j][i].z - VelDiffIn;
                     ucont[k][j][i].z = ubcs[k][j][i].z * zet[k+1][j][i].z;
                }
            }
        }
    }


/* =====================================             */
/*     CHARACTERISTIC OUTLET BC :8 */
/* =====================================             */

    if (d_grid->getBC(5)==8) {
        if (ze == mz) {
            k = ze-2;
            FluxOut = 0;
            for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {
                    FluxOut += ucont[k][j][i].z;
                }
            }
        } else {
            FluxOut = 0.;
        }
    
        FluxIn = d_FluxInSum + FarFluxInSum;
        GlobalSum_All(&FluxOut, &d_FluxOutSum, PETSC_COMM_WORLD);

        ratio = FluxIn / d_FluxOutSum;
        if (fabs(d_FluxOutSum) < 1.e-6) ratio = 1.;
        if (fabs(FluxIn) <1.e-6) ratio = 0.;
        PetscPrintf(PETSC_COMM_WORLD, 
                    "Char Ratio %le %le %le %le %d %d\n", 
                    ratio, FluxIn, d_FluxOutSum, FarFluxInSum,zs, ze);

        if (ze==mz) {
            k = ze-1;
            for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {  
                    ubcs[k][j][i].x = ucat[k-1][j][i].x;
                    ubcs[k][j][i].y = ucat[k-1][j][i].y;
                    if (ti==0 || ti==1) 
                        if (d_inletprofile<0) 
                            ubcs[k][j][i].z = -1.;
                        else if (d_grid->getBC(4)==6) 
                            ubcs[k][j][i].z = 0.;
                        else
                            ubcs[k][j][i].z = 1.;//ubcs[0][j][i].z;//-1.;//1.;
      
                    else 
                        ucont[k-1][j][i].z = ucont[k-1][j][i].z*ratio;
                    ubcs[k][j][i].z = ucont[k-1][j][i].z / zet[k-1][j][i].z;
                }
            }
        }
    }

/* ================================             */
/*     OUTLET BC :4 */
/* ================================             */

  
    if (d_grid->getBC(0)==11) {
        PetscReal ***nvert;
        DMDAVecGetArray(da, lNvert, &nvert);
        
        lArea=0.;
        if (xs==0) {
            i = 0;
            
            FluxOut = 0;
            for (j=lys; j<lye; j++) 
                for (k=lzs; k<lze; k++) {
                    double zc = 0.25*(coor[k][j][i+1].z + coor[k-1][j][i+1].z + 
                                  coor[k][j-1][i+1].z + coor[k-1][j-1][i+1].z);
                    if ( zc > 0 && nvert[k][j][i+1] < d_threshold) {
                    
                        double u=ucat[k][j][i+1].x, 
                               v=ucat[k][j][i+1].y, 
                               w=ucat[k][j][i+1].z;
                        ucat[k][j][i].x=u;
                        ucat[k][j][i].y=v;
                        ucat[k][j][i].z=w;
                    
                        ucont[k][j][i].x = u*csi[k][j][i].x + 
                                           v*csi[k][j][i].y + w*csi[k][j][i].z;
                    
                        FluxOut +=  ucont[k][j][i].x;
                        lArea += fabs(csi[k][j][i].z);
                    }
                }
        }
        else FluxOut = 0.;
        
        FluxIn = d_FluxInSum + FarFluxInSum;
        GlobalSum_All(&FluxOut, &d_FluxOutSum, PETSC_COMM_WORLD);
        GlobalSum_All(&lArea, &AreaSum, PETSC_COMM_WORLD);
         
        d_FluxOutSum *= -1;
        ratio = (d_FluxInSum - d_FluxOutSum) / AreaSum;
        
        double FluxOut_new=0, FluxOut_new_sum;
        
        if (outflow_scale) {
            PetscPrintf(PETSC_COMM_WORLD, 
                "Time %d, Vel correction=%e, FluxIn=%e, FluxOut=%e, Area=%f\n", 
                ti, ratio, d_FluxInSum, d_FluxOutSum, AreaSum);
        
            if (xs==0) {
                i=0;
                for (j=lys; j<lye; j++) 
                    for (k=lzs; k<lze; k++) {
                        double zc, Area;
                        zc = 0.25*(coor[k][j][i+1].z + coor[k-1][j][i+1].z + 
                                   coor[k][j-1][i+1].z + coor[k-1][j-1][i+1].z);
                        if ( zc > 0 && nvert[k][j][i+1] < d_threshold) {
                            Area = csi[k][j][i+1].z;
                            ucont[k][j][i].x += (d_FluxInSum - d_FluxOutSum) * 
                                                 Area / AreaSum;
                            FluxOut_new += ucont[k][j][i].x;
                        }
                    }
            }
        }        
        GlobalSum_All(&FluxOut_new, &FluxOut_new_sum, PETSC_COMM_WORLD);
        PetscPrintf(PETSC_COMM_WORLD, "Corrected FluxOut=%e\n", 
                                       FluxOut_new_sum);
        DMDAVecRestoreArray(da, lNvert, &nvert);
    }
        
    if (d_grid->getBC(5)==4 || d_grid->getBC(5)==5) {
        PetscReal    ***nvert;   
        DMDAVecGetArray(da, lNvert, &nvert);
        
        lArea=0.;
        if (ze==mz) {
            k = ze-1;
            for (j=lys; j<lye; j++) 
                for (i=lxs; i<lxe; i++) {  
                    ubcs[k][j][i].x = ucat[k-1][j][i].x;
                    ubcs[k][j][i].y = ucat[k-1][j][i].y;
                    if (nvert[k-1][j][i] < d_threshold) 
                        ubcs[k][j][i].z = ucat[k-1][j][i].z;
                    else ubcs[k][j][i].z = 0;
                }
                        
            FluxOut = 0;
            for (j=lys; j<lye; j++) 
                for (i=lxs; i<lxe; i++) {
                    if (nvert[k-1][j][i] < d_threshold) //seokkoo
                    {
                        FluxOut +=  ucont[k-1][j][i].z;
                        lArea += sqrt( zet[k-1][j][i].x*zet[k-1][j][i].x + 
                                       zet[k-1][j][i].y*zet[k-1][j][i].y + 
                                       zet[k-1][j][i].z*zet[k-1][j][i].z );
                    }  
                }
        }
        else    FluxOut = 0.;
        
        FluxIn = d_FluxInSum + FarFluxInSum;
        GlobalSum_All(&FluxOut, &d_FluxOutSum, PETSC_COMM_WORLD);
        
        GlobalSum_All(&lArea, &AreaSum, PETSC_COMM_WORLD);
         
        ratio = (d_FluxInSum - d_FluxOutSum) / AreaSum;
        
        if(outflow_scale) {
            PetscPrintf(PETSC_COMM_WORLD, 
            "Time %d, Vel correction=%e, FluxIn=%e, FluxOut=%e, Area=%f\n", 
             ti, ratio, d_FluxInSum, d_FluxOutSum, AreaSum);
        
            if (ze==mz) {
                k = ze-1;
                for (j=lys; j<lye; j++) 
                    for (i=lxs; i<lxe; i++) {
                        double Area = sqrt( zet[k-1][j][i].x*zet[k-1][j][i].x +
                                            zet[k-1][j][i].y*zet[k-1][j][i].y +
                                            zet[k-1][j][i].z*zet[k-1][j][i].z );
                    
                        if (nvert[k-1][j][i] < d_threshold) {
                            ucont[k-1][j][i].z += (d_FluxInSum - d_FluxOutSum) * 
                                                   Area / AreaSum;
                        }
                    }
            }
        }        
        DMDAVecRestoreArray(da, lNvert, &nvert);    //seokkoo 
    } else if (d_grid->getBC(5)==0) {
        if (ze==mz) {
            k = ze-1;
            for (j=lys; j<lye; j++) {
                for (i=lxs; i<lxe; i++) {  
                    ubcs[k][j][i].x = ucat[k-1][j][i].x;
                    ubcs[k][j][i].y = ucat[k-1][j][i].y;
                    ubcs[k][j][i].z = ucat[k-1][j][i].z;
                }
            }
        }
    }  else if (d_grid->getBC(5)==2) {
       /* Designed for driven cavity problem (top(k=kmax) wall moving)
          u_x = 1 at k==kmax */
       if (ze==mz) {
           k = ze-1;
           for (j=lys; j<lye; j++) {
               for (i=lxs; i<lxe; i++) {
                   ubcs[k][j][i].x = 1.;
                   ubcs[k][j][i].y = 0.;
                   ubcs[k][j][i].z = 0.;
               }
           }
       }
    }
    
    // slip
    Cmpnts ***lucont;    // for use of ucont[k-1] etc..
    DMDAVecGetArray(fda, lUcont, &lucont);
  
  
    if ( (d_grid->getBC(0)==1 || d_grid->getBC(0)==10) && xs==0) {
        i= 0;
        for (k=lzs; k<lze; k++)     /* use lzs */
            for (j=lys; j<lye; j++) {
                ucont[k][j][i].x=0;
            }
    }
      
    if ( (d_grid->getBC(1)==1 || d_grid->getBC(1)==10) && xe==mx) {
        i= xe-1;
        for (k=lzs; k<lze; k++)
            for (j=lys; j<lye; j++) {
                ucont[k][j][i-1].x=0;
            }
    }
    
    if ( (d_grid->getBC(2)==1 || d_grid->getBC(2)==10) && ys==0) {
        j= 0;
        for (k=lzs; k<lze; k++)     /* use lzs */
            for (i=lxs; i<lxe; i++) {
                ucont[k][j][i].y=0;
                //if (d_inletprofile==25) 
                //    ucont[k][j][i].z = 1.*kzet[k][j][i].z;
            }
    }
    
    if ( (d_grid->getBC(3)==1 || d_grid->getBC(3)==10) && ye==my) {
        j= ye-1;
        for (k=lzs; k<lze; k++)     /* use lzs */
            for (i=lxs; i<lxe; i++) {
                ucont[k][j-1][i].y=0;
                //if (d_inletprofile==25) 
                //    ucont[k][j-1][i].z = (1./3.)*kzet[k][j-1][i].z;
            }
    }
    
    DMDAVecRestoreArray(fda, lUcont, &lucont);
    DMDAVecRestoreArray(fda, KZet,  &kzet);
    //  end slip
  
    DMDAVecRestoreArray(fda, Ucont, &ucont);
    DMGlobalToLocalBegin(fda, Ucont, INSERT_VALUES, lUcont);
    DMGlobalToLocalEnd(fda, Ucont, INSERT_VALUES, lUcont);
    
    DMDAVecRestoreArray(fda, Ucat, &ucat);
  
    d_data->Contra2Cart();
    DMDAVecGetArray(fda, Ucat, &ucat);
  



/* ==========================================             */
/*   SYMMETRY BC */
/* ==========================================             */
    if (d_grid->getBC(0)==3) {
      
        if (xs==0) {
            i= xs;

            for (k=zs; k<ze; k++) {
                for (j=ys; j<ye; j++) {
                    ubcs[k][j][i].x = 0.;
                    ubcs[k][j][i].y = ucat[k][j][i+1].y;
                    ubcs[k][j][i].z = ucat[k][j][i+1].z;
                }
            }
        }
    }

    if (d_grid->getBC(1)==3) {
        if (xe==mx) {
            i= xe-1;

            for (k=zs; k<ze; k++) {
                for (j=ys; j<ye; j++) {
                    ubcs[k][j][i].x = 0.;
                    ubcs[k][j][i].y = ucat[k][j][i-1].y;
                    ubcs[k][j][i].z = ucat[k][j][i-1].z;
                }
            }
        }
    }

    if (d_grid->getBC(2)==3) {
        if (ys==0) {
            j= ys;

            for (k=zs; k<ze; k++) {
                for (i=xs; i<xe; i++) {
                    ubcs[k][j][i].x = ucat[k][j+1][i].x;
                    ubcs[k][j][i].y = 0.;
                    ubcs[k][j][i].z = ucat[k][j+1][i].z;
                }
            }
        }
    }

    if (d_grid->getBC(3)==3) {
        if (ye==my) {
            j=ye-1;

            for (k=zs; k<ze; k++) {
                for (i=xs; i<xe; i++) {
                    ubcs[k][j][i].x = ucat[k][j-1][i].x;
                    ubcs[k][j][i].y = 0.;
                    ubcs[k][j][i].z = ucat[k][j-1][i].z;
                }
            }
        }
    }

  
    // 0 velocity on the corner point
    if (zs==0) {
        k=0;
        if (xs==0) {
            i=0;
            for (j=ys; j<ye; j++) {
                 ucat[k][j][i].x = 0.;
                 ucat[k][j][i].y = 0.;
                 ucat[k][j][i].z = 0.;
            }
        }
        if (xe == mx) {
            i=mx-1;
            for (j=ys; j<ye; j++) {
                ucat[k][j][i].x = 0.;
                ucat[k][j][i].y = 0.;
                ucat[k][j][i].z = 0.;
            }
        }

        if (ys==0) {
            j=0;
            for (i=xs; i<xe; i++) {
                ucat[k][j][i].x = 0.;
                ucat[k][j][i].y = 0.;
                ucat[k][j][i].z = 0.;
            }
        }

        if (ye==my) {
            j=my-1;
            for (i=xs; i<xe; i++) {
                 ucat[k][j][i].x = 0.;
                 ucat[k][j][i].y = 0.;
                 ucat[k][j][i].z = 0.;
            }
        }

    }

    if (ze==mz) {
        k=mz-1;
        if (xs==0) {
            i=0;
            for (j=ys; j<ye; j++) {
                 ucat[k][j][i].x = 0.;
                 ucat[k][j][i].y = 0.;
                 ucat[k][j][i].z = 0.;
            }
        }
        if (xe == mx) {
            i=mx-1;
            for (j=ys; j<ye; j++) {
                ucat[k][j][i].x = 0.;
                ucat[k][j][i].y = 0.;
                ucat[k][j][i].z = 0.;
            }
        }

        if (ys==0) {
            j=0;
            for (i=xs; i<xe; i++) {
                 ucat[k][j][i].x = 0.;
                 ucat[k][j][i].y = 0.;
                 ucat[k][j][i].z = 0.;
            }
        }

        if (ye==my) {
            j=my-1;
            for (i=xs; i<xe; i++) {
                 ucat[k][j][i].x = 0.;
                 ucat[k][j][i].y = 0.;
                 ucat[k][j][i].z = 0.;
            }
        }

    }

    if (ys==0) {
        j=0;
        if (xs==0) {
            i=0;
            for (k=zs; k<ze; k++) {
                ucat[k][j][i].x = 0.;
                ucat[k][j][i].y = 0.;
                ucat[k][j][i].z = 0.;
            }
        } 

        if (xe==mx) {
            i=mx-1;
            for (k=zs; k<ze; k++) {
                ucat[k][j][i].x = 0.;
                ucat[k][j][i].y = 0.;
                ucat[k][j][i].z = 0.;
            }
        }
    }

    if (ye==my) {
        j=my-1;
        if (xs==0) {
            i=0;
            for (k=zs; k<ze; k++) {
                ucat[k][j][i].x = 0.;
                ucat[k][j][i].y = 0.;
                ucat[k][j][i].z = 0.;
            }
        }

        if (xe==mx) {
            i=mx-1;
            for (k=zs; k<ze; k++) {
                 ucat[k][j][i].x = 0.;
                 ucat[k][j][i].y = 0.;
                 ucat[k][j][i].z = 0.;
            }
        }
    }
    DMDAVecRestoreArray(fda, Ucat,  &ucat);
  

    DMDAVecRestoreArray(fda, Ubcs, &ubcs);
    DMDAVecRestoreArray(fda, Coor, &coor);

    DMDAVecRestoreArray(fda, Csi,  &csi);
    DMDAVecRestoreArray(fda, Eta,  &eta);
    DMDAVecRestoreArray(fda, Zet,  &zet);
  
    DMGlobalToLocalBegin(fda, Ucat, INSERT_VALUES, lUcat);
    DMGlobalToLocalEnd(fda, Ucat, INSERT_VALUES, lUcat);
    return 0;
}


PetscErrorCode BcsUtility::InitializeFlowField()
{
    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    PetscInt xs = info.xs, xe = info.xs + info.xm;
    PetscInt ys = info.ys, ye = info.ys + info.ym;
    PetscInt zs = info.zs, ze = info.zs + info.zm;
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt lxs, lxe, lys, lye, lzs, lze;

    Cmpnts ***ucont, ***cent;
    Cmpnts ***icsi, ***jeta, ***kzet, ***zet, ***eta, ***csi;
  
    PetscInt i, j, k;

    PetscReal ***nvert, ***p, ***level;   
    
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;
  
    Vec Coor;
    Vec lNvert = d_data->getlNvert();
    Vec Ucont = d_data->getUcont();
    Vec Ucat = d_data->getUcat();
    Vec lUcat = d_data->getlUcat();
    Vec P = d_data->getP();

    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec ICsi = d_grid->getlICsi();
    Vec JEta = d_grid->getlJEta();
    Vec KZet = d_grid->getlKZet();
    Vec Cent = d_grid->getlCent();

    DMGetCoordinatesLocal(da, &Coor);
    Cmpnts ***coor;

    DMGetCoordinatesLocal(da,&Coor);
    
    DMDAVecGetArray(da, lNvert, &nvert);
    DMDAVecGetArray(da, P, &p);
    DMDAVecGetArray(fda, Coor, &coor);
    DMDAVecGetArray(fda, Ucont, &ucont);
    DMDAVecGetArray(fda, ICsi,  &icsi);
    DMDAVecGetArray(fda, JEta,  &jeta);
    DMDAVecGetArray(fda, KZet,  &kzet);
    DMDAVecGetArray(fda, Zet,  &zet);
    DMDAVecGetArray(fda, Eta,  &eta);
    DMDAVecGetArray(fda, Csi,  &csi);
    DMDAVecGetArray(fda, Cent,  &cent);
  
    double lArea=0, SumArea;
    if (zs==0) {
        k=0;
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) 
                if(nvert[k][j][i]+nvert[k+1][j][i]<0.1) 
                    lArea+=sqrt( kzet[k][j][i].x*kzet[k][j][i].x + 
                                 kzet[k][j][i].y*kzet[k][j][i].y + 
                                 kzet[k][j][i].z*kzet[k][j][i].z );
    }
    GlobalSum_All(&lArea, &SumArea, PETSC_COMM_WORLD);
              

    // Test the instability of the time-averaged turbine wake 
    //  with/without rotation
    double r[500], uk[500], utheta[500];
    int numberofuinlet, ijk_xyz, ijk_yzx, ijk_zxy;
    double x_c, y_c, z_c, radius;
    
    // this is only for x,y and z are corresponding to i, j, k, respectively. 
    if (d_inletprofile==22 || d_inletprofile==23) {
        FILE *fd;
        char filen[80];  
        sprintf(filen,"./uinlet" );
 
        fd = fopen(filen, "r"); 
        if (!fd) printf("Cannot open %s !!", filen),exit(0);
        else printf("Opened %s !\n", filen);


        if (fd) {
            fscanf(fd, "%i ",&numberofuinlet);
            //fscanf(fd, "%i %i %i",&ijk_xyz, &ijk_yzx, &ijk_zxy);
            fscanf(fd, "%le %le %le %le",&x_c, &y_c, &z_c, &radius);

            for (k=0;k<numberofuinlet;k++) {
                fscanf(fd, "%le %le %le ",&r[k],&uk[k],&utheta[k]);
            }

            fclose(fd);
        }
    }    
    
    double a = 2.*M_PI;
    double lambda = d_data->getRe()/2. - 
                    sqrt ( pow(d_data->getRe()/2., 2.0) + pow(a, 2.0) );
   
    //x velocity 
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=xs; i<lxe; i++) {
                double xi = (coor[k][j][i].x + coor[k-1][j][i].x + 
                             coor[k][j-1][i].x + coor[k-1][j-1][i].x) * 0.25;   
                double yi = (coor[k][j][i].y + coor[k-1][j][i].y + 
                             coor[k][j-1][i].y + coor[k-1][j-1][i].y) * 0.25;   
                double zi = (coor[k][j][i].z + coor[k-1][j][i].z + 
                             coor[k][j-1][i].z + coor[k-1][j-1][i].z) * 0.25; 
        
                if (d_inletprofile==22 || d_inletprofile==23) {
                    double rr;
                    if (d_inletprofile==22) 
                        rr=sqrt(pow(xi-x_c,2)+pow(yi-y_c,2));
                    if (d_inletprofile==23) 
                        rr=sqrt(pow(zi-z_c,2)+pow(yi-y_c,2));
            
                    double tx=-(yi-y_c)/(rr+1.e-19);
 
                    double ty=(xi-x_c)/(rr+1.e-19);
            
                    int kk;
                   double fac1, fac2;
                   for (kk=1;kk<numberofuinlet;kk++) {
                       if (rr-r[numberofuinlet-1]>-1.e-9) {
                           ucont[k][j][i].x = utheta[numberofuinlet-1]*tx * 
                                               icsi[k][j][i].x +
                                              utheta[numberofuinlet-1]*ty * 
                                                icsi[k][j][i].y;

                        } else if (rr-r[kk-1]>-1.e-9 && rr-r[kk]<1.e-9) {
                           fac1=(r[kk]-rr)/(r[kk]-r[kk-1]);
                           fac2=(-r[kk-1]+rr)/(r[kk]-r[kk-1]);

                           ucont[k][j][i].x = 
                               (utheta[kk-1]*fac1+utheta[kk]*fac2)*
                                             tx * icsi[k][j][i].x +
                               (utheta[kk-1]*fac1+utheta[kk]*fac2)*
                                             ty * icsi[k][j][i].y;
                        }                
                    }
                 // 2D Taylor-Green vortex
                } else if(d_inletprofile==17) {   
                    ucont[k][j][i].x = - cos(xi)*sin(yi*a)*icsi[k][j][i].x;
                 // 3D Taylor-Green vortex
                } else if(d_inletprofile==24) {   
                    ucont[k][j][i].x = sin(xi)*cos(yi)*cos(zi)*
                                         icsi[k][j][i].x;
                //2D Mixing layer
                } else if (d_inletprofile==25) {
                    ucont[k][j][i].x = 0.0;
                 // 2D Kovasznay flow
                } else if(d_inletprofile==18) {  
                    ucont[k][j][i].x = (1.0 - exp ( lambda*xi )*cos( a*yi ) ) * 
                                    icsi[k][j][i].x;
                 // Enright test
                } else if(d_inletprofile==20) {
                    ucont[k][j][i].x = 2 * pow( sin (M_PI*xi), 2.) * 
                                       sin(2.*M_PI*yi) * sin(2.*M_PI*zi) * 
                                       icsi[k][j][i].x;
                } else if( d_inletprofile==0 ) ucont[k][j][i].x = 0;
            }
    //y velocity 
    for (k=lzs; k<lze; k++)
        for (j=ys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {    
                double xj = (coor[k][j][i].x + coor[k-1][j][i].x + 
                             coor[k][j][i-1].x + coor[k-1][j][i-1].x) * 0.25;
                double yj = (coor[k][j][i].y + coor[k-1][j][i].y + 
                             coor[k][j][i-1].y + coor[k-1][j][i-1].y) * 0.25;
                double zj = (coor[k][j][i].z + coor[k-1][j][i].z + 
                             coor[k][j][i-1].z + coor[k-1][j][i-1].z) * 0.25;

    
                if (d_inletprofile==22 || d_inletprofile==23) {
                    double rr;
                    if (d_inletprofile==22) 
                        rr=sqrt(pow(xj-x_c,2)+pow(yj-y_c,2));
                    if (d_inletprofile==23) 
                        rr=sqrt(pow(zj-z_c,2)+pow(yj-y_c,2));

                    double tx=-(yj-y_c)/(rr+1.e-19);
                    double ty=(xj-x_c)/(rr+1.e-19);
                    int kk;
                    double fac1, fac2;
                    for (kk=1;kk<numberofuinlet;kk++) {
                        if (rr-r[numberofuinlet-1]>-1.e-9) {

                            ucont[k][j][i].y = utheta[numberofuinlet-1]*tx * 
                                                 jeta[k][j][i].x +
                                               utheta[numberofuinlet-1]*ty * 
                                                 jeta[k][j][i].y;


                        } else  if (rr-r[kk-1]>-1.e-9 && rr-r[kk]<1.e-9) {
                            fac1=(r[kk]-rr)/(r[kk]-r[kk-1]);
                            fac2=(-r[kk-1]+rr)/(r[kk]-r[kk-1]);

                            ucont[k][j][i].y = (utheta[kk-1]*fac1+utheta[kk]*
                                                 fac2)*tx * jeta[k][j][i].x +
                                               (utheta[kk-1]*fac1+utheta[kk]*
                                                 fac2)*ty * jeta[k][j][i].y;

                       }                

                    }
                 // 2D Taylor-Green vortex
                } else if (d_inletprofile==17) {    
                    ucont[k][j][i].y = sin(xj*a) * cos(yj*a) * jeta[k][j][i].y;
                 // 3D Taylor-Green vortex
                } else if (d_inletprofile==24) {    
                    ucont[k][j][i].y = -cos(xj)*sin(yj)*cos(zj)*
                                        jeta[k][j][i].y;
                //2D Mixing layer
                } else if (d_inletprofile==25) {
                    ucont[k][j][i].y = 0.0;
                 // 2D Kovasznay flow
                } else if (d_inletprofile==18) {    
                    ucont[k][j][i].y = ( lambda/a * exp ( lambda*xj ) *
                                         sin( a*yj ) ) * jeta[k][j][i].y;
                 // Enright test
                } else if (d_inletprofile==20) {
                    ucont[k][j][i].y = - sin (2.*M_PI*xj) *
                                         pow(sin (M_PI*yj), 2.) * 
                                         sin (2.*M_PI*zj) * jeta[k][j][i].y;
                } else if( d_inletprofile==0 ) ucont[k][j][i].y = 0;

            }
    
    for (k=zs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
                double xk = (coor[k][j][i].x + coor[k][j-1][i].x + 
                             coor[k][j][i-1].x + coor[k][j-1][i-1].x) * 0.25;
                double yk = (coor[k][j][i].y + coor[k][j-1][i].y + 
                             coor[k][j][i-1].y + coor[k][j-1][i-1].y) * 0.25;
                double zk = (coor[k][j][i].z + coor[k][j-1][i].z + 
                             coor[k][j][i-1].z + coor[k][j-1][i-1].z) * 0.25;

                if (d_inletprofile==22 || d_inletprofile==23) {
                    double rr;
                    if (d_inletprofile==22) 
                        rr=sqrt(pow(xk-x_c,2)+pow(yk-y_c,2));
                    if (d_inletprofile==23) 
                        rr=sqrt(pow(zk-z_c,2)+pow(yk-y_c,2));

                    int kk;
                    double fac1, fac2;
                    for (kk=1;kk<numberofuinlet;kk++) {
                         if (rr-r[numberofuinlet-1]>-1.e-9) {
                             ucont[k][j][i].z = uk[numberofuinlet-1] * 
                                                kzet[k][j][i].z;
                         } else  if (rr-r[kk-1]>-1.e-9 && rr-r[kk]<1.e-9) {
                             fac1=(r[kk]-rr)/(r[kk]-r[kk-1]);
                             fac2=(-r[kk-1]+rr)/(r[kk]-r[kk-1]);

                             ucont[k][j][i].z = (uk[kk-1]*fac1+uk[kk]*fac2) * 
                                                 kzet[k][j][i].z;
                         }                

                    }
                 // Enright test
                } else if(d_inletprofile==20) {
                    ucont[k][j][i].z = - sin (2.*M_PI*xk) * sin (2.*M_PI*yk) *
                                     pow( sin (M_PI*zk), 2.) * kzet[k][j][i].z;
                 // 2D Taylor-Green vortex            
                } else if(d_inletprofile==17) {   
                    ucont[k][j][i].z = 0;
                    if (k) {
                       p[k][j][i] = - 0.25 * ( cos(2.*a*cent[k][j][i].x) + 
                                               cos(2.*a*cent[k][j][i].y) );
                    }
                // 3d Taylor-Green
                } else if (d_inletprofile==24) { 
                    ucont[k][j][i].z = 0;
                       p[k][j][i] = 0.0625 * ( cos(2.*cent[k][j][i].x) + 
                                               cos(2.*cent[k][j][i].y) ) *
                                             ( cos(2.*cent[k][j][i].z)+2.);
                //2D Mixing layer
                } else if (d_inletprofile==25) {
                    double gam = 3.0;
                    double U1 = 1.0;
                    double U2 = U1/gam;
                    double del = 1.0;
                    ucont[k][j][i].z = ( (U1+U2)/2.0 + 
                                         (U1-U2)/2.0*tanh((yk-10)/del) ) *
                                       kzet[k][j][i].z;
                    
                    
                 // 18: 2D Kovasznay flow
                } else if(d_inletprofile==18) {   
                    ucont[k][j][i].z = 0;
                } else if( d_inletprofile==0 ) { ucont[k][j][i].z = 0; 
                } else if(nvert[k][j][i]+nvert[k+1][j][i]<0.1) {
                    double w;
              
                    double area = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                        zet[k][j][i].y*zet[k][j][i].y + 
                                        zet[k][j][i].z*zet[k][j][i].z );
                    double xc = (coor[k][j][i].x + coor[k][j-1][i].x + 
                                 coor[k][j][i-1].x + coor[k][j-1][i-1].x)*0.25;
                    double yc = (coor[k][j][i].y + coor[k][j-1][i].y + 
                                 coor[k][j][i-1].y + coor[k][j-1][i-1].y)*0.25;
                    double zc = (coor[k][j][i].z + coor[k][j-1][i].z + 
                                 coor[k][j][i-1].z + coor[k][j-1][i-1].z)*0.25;
                    if (d_inletprofile==10) {
                        double delta = 0.45263, a=5.99;
                        if ( yc>=delta ) w=1.0;
                        else if (yc<=0) w=0.0;
                        else w = pow( yc/delta, 1./a );
                     // pipe shear stress test
                    } else if (d_inletprofile==12) {    
                        double r = sqrt(xc * xc + yc * yc);
                        w = 2*(1 - pow(r/0.5,2.0) );
                     // periodic channel flow
                    } else if (d_inletprofile==13) {   
                        double w_bulk=d_inlet_flux/SumArea;
                        w = 1.5 * w_bulk * zc * ( 2 - zc );
                     // jet
                    } else if (d_inletprofile==15){ w = 0;
                     // periodic pipe flow   
                    } else if (d_inletprofile==16) {      
                        double r = sqrt(xc * xc + yc * yc);
                        double w_bulk=d_inlet_flux/SumArea;
                        w = w_bulk*2.*(1. - pow(r/0.5,2.0) );
                    } else if (d_inletprofile==-9) { w=0;
                    } else if (d_inlet_flux>0) {
                        w=d_inlet_flux/d_inletArea;
                    } else w=1;
                        
                    ucont[k][j][i].z = w * area;
               
                    if (d_inletprofile==19) {
                        double u=0, v=0, w=1;
                        ucont[k][j][i].x = u * icsi[k][j][i].x  + 
                                           v * icsi[k][j][i].y + 
                                           w * icsi[k][j][i].z;
                        ucont[k][j][i].y = u * jeta[k][j][i].x  + 
                                           v * jeta[k][j][i].y +  
                                           w * jeta[k][j][i].z;
                        ucont[k][j][i].z = u * kzet[k][j][i].x  + 
                                           v * kzet[k][j][i].y + 
                                           w * kzet[k][j][i].z;
                    }
                }
            }
    
    srand( time(NULL)) ; 
    for (i = 0; i < (rand() % 3000); i++) (rand() % 3000); 

    if (d_initial_perturbation) {
        PetscPrintf(PETSC_COMM_WORLD, "\nGenerating initial perturbation\n");
        for (k=lzs; k<lze; k++) 
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {
                    if (nvert[k][j][i]+nvert[k+1][j][i] < d_threshold) {
                        int n1, n2, n3;
                        double F;
                    
                        F  = 1.00; // 100%
                        n1 = rand() % 20000 - 10000;
                        n2 = rand() % 20000 - 10000;
                        n3 = rand() % 20000 - 10000;
                        ucont[k][j][i].x = ((double)n3)/10000. * 0.1 * 
                                            ucont[k][j][i].z;
                        ucont[k][j][i].y = ((double)n2)/10000. * 0.1 * 
                                            ucont[k][j][i].z;
                        ucont[k][j][i].z *= (1 + ((double)n1)/10000.*F);
                    }
                }
    } else if ( d_initial_gaussian_perturbation) {
        PetscPrintf(PETSC_COMM_WORLD, 
                            "\nGenerating initial Gaussian perturbation\n");
        for (k=lzs; k<lze; k++) 
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {
                    if (nvert[k][j][i]+nvert[k+1][j][i] < d_threshold) {
                        double n1, n2, n3;
    
                        double areai = sqrt( csi[k][j][i].x*csi[k][j][i].x + 
                                             csi[k][j][i].y*csi[k][j][i].y + 
                                             csi[k][j][i].z*csi[k][j][i].z );
                        double areaj = sqrt( eta[k][j][i].x*eta[k][j][i].x +
                                             eta[k][j][i].y*eta[k][j][i].y + 
                                             eta[k][j][i].z*eta[k][j][i].z );
                        double areak = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                             zet[k][j][i].y*zet[k][j][i].y + 
                                             zet[k][j][i].z*zet[k][j][i].z );
                        n1 = randn_notrig();
                        n2 = randn_notrig();
                        n3 = randn_notrig();
                        ucont[k][j][i].x += 
                           n1 * d_magnitude_gaussian_perturbation * areai;
                        ucont[k][j][i].y += 
                           n2 * d_magnitude_gaussian_perturbation * areaj;
                        ucont[k][j][i].z += 
                           n3 *d_magnitude_gaussian_perturbation * areak;
                    }
                }
    }
    
    
    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(da, P, &p);
    DMDAVecRestoreArray(fda, Coor, &coor);
    DMDAVecRestoreArray(fda, Ucont, &ucont);
    DMDAVecRestoreArray(fda, ICsi,  &icsi);
    DMDAVecRestoreArray(fda, JEta,  &jeta);
    DMDAVecRestoreArray(fda, KZet,  &kzet);
    DMDAVecRestoreArray(fda, Zet,  &zet);
    DMDAVecRestoreArray(fda, Eta,  &eta);
    DMDAVecRestoreArray(fda, Csi,  &csi);
    DMDAVecRestoreArray(fda, Cent,  &cent);
   
    Vec lUcont = d_data->getlUcont(); 
    DMGlobalToLocalBegin(fda, Ucont, INSERT_VALUES, lUcont);
    DMGlobalToLocalEnd(fda, Ucont, INSERT_VALUES, lUcont);
   
    Vec lP = d_data->getlP();
 
    DMGlobalToLocalBegin(da, P, INSERT_VALUES, lP);
    DMGlobalToLocalEnd(da, P, INSERT_VALUES, lP);
    
    d_data->Contra2Cart();
        
    Vec Ucont_o = d_data->getUcont_o();
    Vec lUcont_o = d_data->getlUcont_o();
    VecCopy(Ucont, Ucont_o);
    DMGlobalToLocalBegin(fda, Ucont_o, INSERT_VALUES, lUcont_o);
    DMGlobalToLocalEnd(fda, Ucont_o, INSERT_VALUES, lUcont_o);

    DMGlobalToLocalBegin(fda, Ucat, INSERT_VALUES, lUcat);
    DMGlobalToLocalEnd(fda, Ucat, INSERT_VALUES, lUcat);

    Vec lUcat_old = d_data->getlUcat_old();
    DMGlobalToLocalBegin(fda, Ucat, INSERT_VALUES, lUcat_old);
    DMGlobalToLocalEnd(fda, Ucat, INSERT_VALUES, lUcat_old);


    return 0;
}


double BcsUtility::randn_notrig() 
{

    static bool deviateAvailable=false;   
    static float storedDeviate;
    double polar, rsquared, var1, var2;
    double mu=0.0; double sigma=1.0;
    if (!deviateAvailable) 
    {
        do {
            var1=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
            var2=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
            rsquared=var1*var1+var2*var2;
        } while ( rsquared>=1.0 || rsquared == 0.0);

        polar=sqrt(-2.0*log(rsquared)/rsquared);
        storedDeviate=var1*polar;
        deviateAvailable=true;
        return var2*polar*sigma + mu;
    } else {
        deviateAvailable=false;
        return storedDeviate*sigma + mu;
    }

}

PetscErrorCode BcsUtility::IbBC()
{
    int      i, j, k;

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    int xs, xe, ys, ye, zs, ze;
    int mx,my,mz;    
    int lxs, lxe, lys, lye, lzs, lze;
  
    xs = info.xs; xe = info.xs + info.xm;
    ys = info.ys; ye = info.ys + info.ym;
    zs = info.zs; ze = info.zs + info.zm;
    mx = info.mx; my = info.my; mz = info.mz;
  
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;
  
    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;
  
    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;
  
    Cmpnts ***ucont;
    PetscReal ***nvert;
    Cmpnts ***ucat; 
       
    Vec lUcont = d_data->getlUcont();
    Vec lUcat = d_data->getlUcat();
    Vec lNvert = d_data->getlNvert();
 
    DMDAVecGetArray(fda, lUcat, &ucat);
    DMDAVecGetArray(fda, lUcont, &ucont);
    DMDAVecGetArray(da, lNvert, &nvert);
    
    PetscInt i_periodic = d_grid->isIPeriodic();
    PetscInt j_periodic = d_grid->isJPeriodic();
    PetscInt k_periodic = d_grid->isKPeriodic();
    PetscInt ii_periodic = d_grid->isIIPeriodic();
    PetscInt jj_periodic = d_grid->isJJPeriodic();
    PetscInt kk_periodic = d_grid->isKKPeriodic();


    if (d_grid->isPeriodic())
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) {    
      
                    int flag=0, a=i, b=j, c=k;
        
                    if (i_periodic && i==0) a=mx-2, flag=1;
                    else if (i_periodic && i==mx-1) a=1, flag=1;
        
                    if (j_periodic && j==0) b=my-2, flag=1;
                    else if (j_periodic && j==my-1) b=1, flag=1;
        
                    if (k_periodic && k==0) c=mz-2, flag=1;
                    else if (k_periodic && k==mz-1) c=1, flag=1;
        
                    if (ii_periodic && i==0) a=-2, flag=1;
                    else if (ii_periodic && i==mx-1) a=mx+1, flag=1;
        
                    if (jj_periodic && j==0) b=-2, flag=1;
                    else if (jj_periodic && j==my-1) b=my+1, flag=1;
        
                    if (kk_periodic && k==0) c=-2, flag=1;
                    else if (kk_periodic && k==mz-1) c=mz+1, flag=1;
        
                    if (flag) {
                        ucat[k][j][i] = ucat[c][b][a];
                    }
                }

    /*
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
      
                if (d_immersed) {
                    if (!movefsi && !rotatefsi && immersed==3) {
                        if ((nvert[k][j][i+1]+nvert[k][j][i])>1.1) 
                            ucont[k][j][i].x = 0;
                        if ((nvert[k][j+1][i]+nvert[k][j][i])>1.1) 
                            ucont[k][j][i].y = 0;
                        if ((nvert[k+1][j][i]+nvert[k][j][i])>1.1) 
                            ucont[k][j][i].z = 0;
                    }
                }
            }
    */    

    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                if ((d_grid->getBC(0)==10) && i==1) ucont[k][j][0].x = 0;
                if ((d_grid->getBC(1)==10) && i==mx-2) ucont[k][j][mx-2].x = 0;
                if ((d_grid->getBC(2)==10) && j==1) ucont[k][0][i].y = 0;
                if ((d_grid->getBC(3)==10) && j==my-2) ucont[k][my-2][i].y = 0;
                if ((d_grid->getBC(4)==10) && k==1) ucont[0][j][i].z = 0;
                if ((d_grid->getBC(5)==10) && k==mz-2) ucont[mz-2][j][i].z = 0;
        
                if (i_periodic && i==0) ucont[k][j][0].x=ucont[k][j][mx-2].x;
                if (i_periodic && i==mx-1) ucont[k][j][mx-1].x=ucont[k][j][1].x;
                if (j_periodic && j==0) ucont[k][0][i].y=ucont[k][my-2][i].y;
                if (j_periodic && j==my-1) ucont[k][my-1][i].y=ucont[k][1][i].y;
                if (k_periodic && k==0) ucont[0][j][i].z=ucont[mz-2][j][i].z;
                if (k_periodic && k==mz-1) ucont[mz-1][j][i].z=ucont[1][j][i].z;
        
                if (ii_periodic && i==0) ucont[k][j][0].x=ucont[k][j][-2].x;
                if (ii_periodic && i==mx-1)
                    ucont[k][j][mx-1].x=ucont[k][j][mx+1].x;

                if (jj_periodic && j==0) ucont[k][0][i].y=ucont[k][-2][i].y;
                if (jj_periodic && j==my-1) 
                    ucont[k][my-1][i].y=ucont[k][my+1][i].y;

                if (kk_periodic && k==0) ucont[0][j][i].z=ucont[-2][j][i].z;
                if (kk_periodic && k==mz-1) 
                    ucont[mz-1][j][i].z=ucont[mz+1][j][i].z;
            }

    DMDAVecRestoreArray(fda, lUcat, &ucat);
    DMDAVecRestoreArray(fda, lUcont, &ucont);
    DMDAVecRestoreArray(da, lNvert, &nvert);
    
    DMDALocalToLocalBegin(fda, lUcont, INSERT_VALUES, lUcont);
    DMDALocalToLocalEnd(fda, lUcont, INSERT_VALUES, lUcont);
    
}

PetscErrorCode BcsUtility::CalculateInflowFlux()
{
    int i, j, k;

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
  
    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;
  
    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;
    
    PetscReal    ***nvert, ***level, ***aj;
    Cmpnts ***ucont, ***ucat, ***zet, ***kzet, ***eta, ***csi;

    Vec lUcont = d_data->getlUcont();
    Vec lUcat = d_data->getlUcat();
    Vec lNvert = d_data->getlNvert();

    Vec Zet = d_grid->getlZet();

    DMDAVecGetArray(fda, lUcont, &ucont);
    DMDAVecGetArray(fda, lUcont, &ucat);
    DMDAVecGetArray(da, lNvert, &nvert);

    DMDAVecGetArray(fda, Zet, &zet);
    
    double lFlux=0, lFlux_ibm=0;
    double Flux, Flux_ibm;
    
    for (k=lzs; k<lze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                if (j>=1 && j<=my-2 && i>=1 && i<=mx-2) {

                    if (nvert[k+1][j][i]+nvert[k][j][i] < 0.1) 
                        lFlux += ucont[k][j][i].z;

                    else if (nvert[k+1][j][i]+nvert[k][j][i] < 2.1) {
                        //lFlux_ibm += ucont[k][j][i].z;
                        double ucx = (ucat[k+1][j][i].x+ucat[k][j][i].x)*0.5;
                        double ucy = (ucat[k+1][j][i].y+ucat[k][j][i].y)*0.5;
                        double ucz = (ucat[k+1][j][i].z+ucat[k][j][i].z)*0.5;

                        lFlux_ibm += (ucx * zet[k][j][i].x + 
                                      ucy * zet[k][j][i].y + 
                                      ucz * zet[k][j][i].z);
                    }
                }
            }

    GlobalSum_All(&lFlux, &Flux, PETSC_COMM_WORLD);
    GlobalSum_All(&lFlux_ibm, &Flux_ibm, PETSC_COMM_WORLD);
    
    Flux /= (mz-2);
    Flux_ibm /= (mz-2);
    
    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(fda, lUcont, &ucont);
    DMDAVecRestoreArray(fda, lUcont, &ucat);
    DMDAVecRestoreArray(fda, Zet, &zet);
    
    PetscPrintf(PETSC_COMM_WORLD, 
                "\n...Mean k Flux:%f, k=1 area: %f\n", 
                Flux, d_k_area[1]);

    PetscPrintf(PETSC_COMM_WORLD, 
                "...Flux ibnode:%f, k=1 area ibnode: %f\n\n", 
                Flux_ibm, d_k_area_ibnode[1]);
    
    d_data->setMeanFlux(Flux);
    d_data->setMeanFluxIb(Flux_ibm);

    return 0;
};

PetscErrorCode BcsUtility::ScaleInitialFlow()
{
    PetscInt i, j, k;

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    PetscInt xs = info.xs, xe = info.xs + info.xm;
    PetscInt ys = info.ys, ye = info.ys + info.ym;
    PetscInt zs = info.zs, ze = info.zs + info.zm;
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt lxs, lxe, lys, lye, lzs, lze;
 
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;
  
    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;
  
    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    Cmpnts    ***csi, ***eta, ***zet;
    PetscReal ***nvert, ***aj;

    Vec Ucont = d_data->getUcont();
    Vec lNvert = d_data->getlNvert();

    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();

    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, lNvert, &nvert);  
    DMDAVecGetArray(da, Aj, &aj);

    Cmpnts ***ucont;
    DMDAVecGetArray(fda, Ucont, &ucont);

    double *areak = new double [mz];
    double *fluxk = new double [mz];

    // Area         
    std::vector<double> lArea(mz);
    std::vector<double> Sum_lArea(mz);
    
    std::fill( lArea.begin(), lArea.end(), 0 );
    std::fill( Sum_lArea.begin(), Sum_lArea.end(), 0 );

    for (k=lzs; k<lze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                if (j>=1 && j<=my-2 && i>=1 && i<=mx-2) {

                    double k_area = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                          zet[k][j][i].y*zet[k][j][i].y + 
                                          zet[k][j][i].z*zet[k][j][i].z );
                    if (nvert[k+1][j][i]+nvert[k][j][i] < 0.1) 
                       lArea[k] += k_area;
                }   
            }

    MPI_Allreduce(&lArea[0], &Sum_lArea[0], mz, 
                  MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    
    Sum_lArea[0] = Sum_lArea[1];

    for (k=0;k<mz-1;k++) {
         areak[k]=Sum_lArea[k];
    }

    // Flux
    std::vector<double> lFlux(mz);
    std::vector<double> Sum_lFlux(mz);
    
    std::fill (lFlux.begin(), lFlux.end(), 0 );
    std::fill (Sum_lFlux.begin(), Sum_lFlux.end(), 0 );

    for (k=lzs; k<lze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                if (j>=1 && j<=my-2 && i>=1 && i<=mx-2) {
                    if (nvert[k+1][j][i]+nvert[k][j][i] < 0.1) 
                        lFlux[k] += ucont[k][j][i].z;
                }
            }

    MPI_Allreduce(&lFlux[0], &Sum_lFlux[0], mz, 
                  MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    
    Sum_lFlux[0] = Sum_lFlux[1];

    for (k=0;k<mz-1;k++) {
        fluxk[k]=Sum_lFlux[k];
    }

    
    for (k=lzs; k<lze; k++) 
        for (j=lys; j<lye; j++) 
            for (i=lxs; i<lxe; i++) {            

                double Area = sqrt(zet[k-1][j][i].x*zet[k-1][j][i].x + 
                                   zet[k-1][j][i].y*zet[k-1][j][i].y + 
                                   zet[k-1][j][i].z*zet[k-1][j][i].z );
                 double AreaSum=areak[k];
                 double FluxInlet=fluxk[0];
                 double Flux=fluxk[k];
                 if (nvert[k+1][j][i]+nvert[k][j][i] < 0.1) 
                     ucont[k][j][i].z += (FluxInlet - Flux) * Area / AreaSum;


            }        
        

    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(da, lNvert, &nvert);    //seokkoo 
    DMDAVecRestoreArray(da, Aj, &aj);
    DMDAVecRestoreArray(fda, Ucont, &ucont);
    
    d_data->Contra2Cart();

    return 0;
};


PetscErrorCode BcsUtility::ReadPlane(PetscInt ti)
{
    if (d_inletprofile == 100)
    {
        d_iplane->Read(ti);

        //Set ucat plane when allocated
        if (ti==d_data->get_tistart())
            setUcatPlane(d_iplane->getUcatPlane());
    }

    return 0;
} 


PetscErrorCode BcsUtility::ReadFromInput()
{
    PetscOptionsGetReal(PETSC_NULL, "-flux", &d_inlet_flux, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-inlet", &d_inletprofile, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-pseudo", &d_pseudo_periodic, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-fluct_rms", &d_fluct_rms, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-perturb", &d_initial_perturbation, 
                       PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-initial_gaussian_perturb", 
                       &d_initial_gaussian_perturbation, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-magnitude_gaussian_perturb", 
                        &d_magnitude_gaussian_perturbation, PETSC_NULL);
}

