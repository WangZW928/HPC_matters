#include "RHSSolver.h"

RHSSolver::RHSSolver(
    const std::string& object_name,
    CurvGrid *grid,
    UData *data,
    LESModel *les):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data),
    d_les(les)
{

    d_inlet_flux = -1;
    d_second_order = 1;
    d_immersed = 0;
    d_mean_pressure_gradient = 0;

    ReadFromInput();
}

RHSSolver::~RHSSolver()
{
    VecDestroy(&d_Div1);
    VecDestroy(&d_Div2);
    VecDestroy(&d_Div3);
    VecDestroy(&d_Visc1);
    VecDestroy(&d_Visc2);
    VecDestroy(&d_Visc3);
    VecDestroy(&d_Fp);
}

PetscErrorCode RHSSolver::Initialize()
{
       
    Vec Csi = d_grid->getlCsi();

    VecDuplicate(Csi, &d_Div1);
    VecDuplicate(Csi, &d_Div2);
    VecDuplicate(Csi, &d_Div3);
    VecDuplicate(Csi, &d_Visc1);
    VecDuplicate(Csi, &d_Visc2);
    VecDuplicate(Csi, &d_Visc3);
    VecDuplicate(Csi, &d_Fp);

    if (d_les->useLES()) { 
        d_les->Initialize();
        if (d_data->isRestart()) {
            d_les->ReadCs();
        }
    }
        

    return 0;
}


void RHSSolver::CalculatePressureGradient()
{
    int i, j, k;

    Cmpnts ***csi, ***eta, ***zet;
    Cmpnts ***icsi, ***ieta, ***izet;
    Cmpnts ***jcsi, ***jeta, ***jzet;
    Cmpnts ***kcsi, ***keta, ***kzet, ***dp;
    PetscReal ***p, ***level, ***rho;
    PetscReal    ***nvert;

    DMDALocalInfo info;
    int xs, xe, ys, ye, zs, ze; 
    int mx, my, mz; 
    PetscReal ***iaj, ***jaj, ***kaj,  ***aj;

    int lxs, lxe, lys, lye, lzs, lze;

    PetscScalar solid =0.5;

    //Lets get the Vecs we need
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();

    Vec ICsi = d_grid->getlICsi();
    Vec IEta = d_grid->getlIEta();
    Vec IZet = d_grid->getlIZet();
    Vec IAj = d_grid->getlIAj();

    Vec JCsi = d_grid->getlJCsi();
    Vec JEta = d_grid->getlJEta();
    Vec JZet = d_grid->getlJZet();
    Vec JAj = d_grid->getlJAj();

    Vec KCsi = d_grid->getlKCsi();
    Vec KEta = d_grid->getlKEta();
    Vec KZet = d_grid->getlKZet();
    Vec KAj = d_grid->getlKAj();

    Vec lP = d_data->getlP();
    Vec P = d_data->getP();
    Vec Nvert = d_data->getlNvert();
    Vec Dp = d_data->getDp();

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDAGetLocalInfo(da, &info);
    mx = info.mx; my = info.my; mz = info.mz;
    xs = info.xs; xe = xs + info.xm;
    ys = info.ys; ye = ys + info.ym;
    zs = info.zs; ze = zs + info.zm;

  /* First we calculate the flux on cell surfaces. Stored on the upper integer
     node. For example, along i direction, the flux are stored at node 0:mx-2*/
    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;
    
    
    DMGlobalToLocalBegin(da, P, INSERT_VALUES, lP);
    DMGlobalToLocalEnd(da, P, INSERT_VALUES, lP);
    
    PetscInt i_periodic = d_grid->isIPeriodic();
    PetscInt j_periodic = d_grid->isJPeriodic();
    PetscInt k_periodic = d_grid->isKPeriodic();
    PetscInt ii_periodic = d_grid->isIIPeriodic();
    PetscInt jj_periodic = d_grid->isJJPeriodic();
    PetscInt kk_periodic = d_grid->isKKPeriodic();

    if (d_grid->isPeriodic()) 
    {
        DMDAVecGetArray(da, lP, &p);
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) 
                {
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
                            
                    if (flag) 
                        p[k][j][i] = p[c][b][a];
        }
        DMDAVecRestoreArray(da, lP, &p);
        
        DMDALocalToLocalBegin(da, lP, INSERT_VALUES, lP);
        DMDALocalToLocalEnd(da, lP, INSERT_VALUES, lP);
        DMLocalToGlobalBegin(da, lP, INSERT_VALUES, P);
        DMLocalToGlobalEnd(da, P, INSERT_VALUES, P);
    }
    
    DMDAVecGetArray(fda, ICsi, &icsi);
    DMDAVecGetArray(fda, IEta, &ieta);
    DMDAVecGetArray(fda, IZet, &izet);

    DMDAVecGetArray(fda, JCsi, &jcsi);
    DMDAVecGetArray(fda, JEta, &jeta);
    DMDAVecGetArray(fda, JZet, &jzet);

    DMDAVecGetArray(fda, KCsi, &kcsi);
    DMDAVecGetArray(fda, KEta, &keta);
    DMDAVecGetArray(fda, KZet, &kzet);

    DMDAVecGetArray(da, Nvert, &nvert);
    DMDAVecGetArray(da, lP, &p);

    DMDAVecGetArray(da, IAj, &iaj);
    DMDAVecGetArray(da, JAj, &jaj);
    DMDAVecGetArray(da, KAj, &kaj);


    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, Aj, &aj);


    VecSet(Dp, 0.);
    
    DMDAVecGetArray(fda, Dp,  &dp);
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) 
            {
                double g11_i = (icsi[k][j][i].x*icsi[k][j][i].x + 
                                icsi[k][j][i].y*icsi[k][j][i].y + 
                                icsi[k][j][i].z*icsi[k][j][i].z);
                double g12_i = (ieta[k][j][i].x*icsi[k][j][i].x + 
                                ieta[k][j][i].y*icsi[k][j][i].y + 
                                ieta[k][j][i].z*icsi[k][j][i].z);
                double g13_i = (izet[k][j][i].x*icsi[k][j][i].x + 
                                izet[k][j][i].y*icsi[k][j][i].y + 
                                izet[k][j][i].z*icsi[k][j][i].z);
                double g21_j = (jcsi[k][j][i].x*jeta[k][j][i].x + 
                                jcsi[k][j][i].y*jeta[k][j][i].y + 
                                jcsi[k][j][i].z*jeta[k][j][i].z);
                double g22_j = (jeta[k][j][i].x*jeta[k][j][i].x + 
                                jeta[k][j][i].y*jeta[k][j][i].y + 
                                jeta[k][j][i].z*jeta[k][j][i].z);
                double g23_j = (jzet[k][j][i].x*jeta[k][j][i].x + 
                                jzet[k][j][i].y*jeta[k][j][i].y + 
                                jzet[k][j][i].z*jeta[k][j][i].z);
                double g31_k = (kcsi[k][j][i].x*kzet[k][j][i].x + 
                                kcsi[k][j][i].y*kzet[k][j][i].y +  
                                kcsi[k][j][i].z*kzet[k][j][i].z);
                double g32_k = (keta[k][j][i].x*kzet[k][j][i].x + 
                                keta[k][j][i].y*kzet[k][j][i].y + 
                                keta[k][j][i].z*kzet[k][j][i].z);
                double g33_k = (kzet[k][j][i].x*kzet[k][j][i].x + 
                                kzet[k][j][i].y*kzet[k][j][i].y + 
                                kzet[k][j][i].z*kzet[k][j][i].z);
                double r1=1.0, r2=1.0, r3=1.0;
        
        
                double dpdc, dpde, dpdz;
                //i direction
                if ( (i==0 || i==mx-2) && i_periodic) 
                    dpdc = p[k][j][1] - p[k][j][i];
                else if( i==mx-2 && ii_periodic) 
                    dpdc = p[k][j][mx+1] - p[k][j][i];
                else dpdc = p[k][j][i+1] - p[k][j][i];
        
                if ((int)(nvert[k][j+1][i]+0.5)==1 || 
                    (int)(nvert[k][j+1][i+1]+0.5)==1 || 
                    (j==my-2 && !j_periodic && !jj_periodic)) 
                { 
                     dpde = (p[k][j][i] - p[k][j-1][i] + 
                             p[k][j][i+1] - p[k][j-1][i+1])*0.5;
                }
                else if ((int)(nvert[k][j-1][i]+0.5)==1 || 
                         (int)(nvert[k][j-1][i+1]+0.5)==1 || 
                         (j==1 && !j_periodic && !jj_periodic)) 
                {
                    dpde = (p[k][j+1][i] - p[k][j][i] + 
                            p[k][j+1][i+1] - p[k][j][i+1])*0.5;
                }
                else dpde = (p[k][j+1][i] - p[k][j-1][i] + 
                             p[k][j+1][i+1] - p[k][j-1][i+1])*0.25;

                if ((int)(nvert[k+1][j][i]+0.5)==1 || 
                    (int)(nvert[k+1][j][i+1]+0.5)==1 || 
                    (k==mz-2 && !k_periodic && !kk_periodic)) 
                {
                    dpdz = (p[k][j][i] - p[k-1][j][i] + 
                            p[k][j][i+1] - p[k-1][j][i+1])*0.5;
                }
                else if ((int)(nvert[k-1][j][i]+0.5)==1 || 
                         (int)(nvert[k-1][j][i+1]+0.5)==1 || 
                         (k==1 && !k_periodic && !kk_periodic)) 
                {
                    dpdz = (p[k+1][j][i] - p[k][j][i] + 
                            p[k+1][j][i+1] - p[k][j][i+1])*0.5;
                }
                else dpdz = (p[k+1][j][i] - p[k-1][j][i] + 
                             p[k+1][j][i+1] - p[k-1][j][i+1])*0.25;
            
                //i direction dp
                dp[k][j][i].x = (dpdc*g11_i + dpde*g12_i + dpdz*g13_i ) * 
                                 iaj[k][j][i] / r1;
         
                //j direction  
                if ((int)(nvert[k][j][i+1]+0.5)==1 || 
                    (int)(nvert[k][j+1][i+1]+0.5)==1 || 
                    (i==mx-2&& !i_periodic && !ii_periodic)) 
                {
                    dpdc = (p[k][j][i] - p[k][j][i-1] + 
                            p[k][j+1][i] - p[k][j+1][i-1])*0.5;
                }  
                else if ((int)(nvert[k][j][i-1]+0.5)==1 || 
                         (int)(nvert[k][j+1][i-1]+0.5)==1 || 
                         (i==1&& !i_periodic && !ii_periodic)) 
                {
                    dpdc = (p[k][j][i+1] - p[k][j][i] + 
                            p[k][j+1][i+1] - p[k][j+1][i])*0.5;
                } 
                else dpdc = (p[k][j][i+1] - p[k][j][i-1] + 
                             p[k][j+1][i+1] - p[k][j+1][i-1])*0.25;
        
                if ( (j==0 || j==my-2) && j_periodic) 
                    dpde = p[k][1][i] - p[k][j][i];
                else if ( j==my-2 && jj_periodic) 
                    dpde = p[k][my+1][i] - p[k][j][i];
                else dpde = p[k][j+1][i] - p[k][j][i];
        
                if ((int)(nvert[k+1][j][i]+0.5)==1  || 
                    (int)(nvert[k+1][j+1][i]+0.5)==1 || 
                    (k==mz-2 && !k_periodic && !kk_periodic))
                { 
                    dpdz = (p[k][j][i] - p[k-1][j][i] + 
                            p[k][j+1][i] - p[k-1][j+1][i])*0.5;
                } 
                else if ((int)(nvert[k-1][j][i]+0.5)==1  || 
                         (int)(nvert[k-1][j+1][i]+0.5)==1 || 
                         (k==1 && !k_periodic && !kk_periodic)) 
                {
                    dpdz = (p[k+1][j][i] - p[k][j][i] + 
                            p[k+1][j+1][i] - p[k][j+1][i])*0.5;
                }
                else dpdz = (p[k+1][j][i] - p[k-1][j][i] + 
                             p[k+1][j+1][i] - p[k-1][j+1][i])*0.25;
        
                //j-direction dp
                dp[k][j][i].y = (dpdc*g21_j + dpde*g22_j + dpdz*g23_j ) * 
                                 jaj[k][j][i] / r2;
        
                //k-direction  
                if ((int)(nvert[k][j][i+1]+0.5)==1 || 
                    (int)(nvert[k+1][j][i+1]+0.5)==1 || 
                    (i==mx-2 && !i_periodic && !ii_periodic)) 
                {
                    dpdc = (p[k][j][i] - p[k][j][i-1] + 
                            p[k+1][j][i] - p[k+1][j][i-1])*0.5;
                } 
                else if ((int)(nvert[k][j][i-1]+0.5)==1 || 
                         (int)(nvert[k+1][j][i-1]+0.5)==1 || 
                         (i==1 && !i_periodic && !ii_periodic)) 
                {
                    dpdc = (p[k][j][i+1] - p[k][j][i] + 
                            p[k+1][j][i+1] - p[k+1][j][i])*0.5;
                }  
                else dpdc = (p[k][j][i+1] - p[k][j][i-1] + 
                             p[k+1][j][i+1] - p[k+1][j][i-1])*0.25;

                if ((int)(nvert[k][j+1][i]+0.5) ==1 || 
                    (int)(nvert[k+1][j+1][i]+0.5)==1 || 
                    (j==my-2 && !j_periodic && !jj_periodic)) 
                {
                    dpde = (p[k][j][i] - p[k][j-1][i] + 
                            p[k+1][j][i] - p[k+1][j-1][i])*0.5;
                } 
                else if ((int)(nvert[k][j-1][i]+0.5) ==1 || 
                         (int)(nvert[k+1][j-1][i]+0.5)==1 || 
                         (j==1 && !j_periodic && !jj_periodic)) 
                {
                   dpde = (p[k][j+1][i] - p[k][j][i] + 
                           p[k+1][j+1][i] - p[k+1][j][i])*0.5;
                }
                else dpde = (p[k][j+1][i] - p[k][j-1][i] + 
                             p[k+1][j+1][i] - p[k+1][j-1][i])*0.25;
        
                if ( (k==0 || k==mz-2) && k_periodic) 
                    dpdz = p[1][j][i] - p[k][j][i];
                else if ( k==mz-2 && kk_periodic) 
                    dpdz = p[mz+1][j][i] - p[k][j][i];
                else dpdz = (p[k+1][j][i] - p[k][j][i]);
        
                //k-direction dp
                dp[k][j][i].z = (dpdc*g31_k + dpde*g32_k + dpdz*g33_k ) * 
                                 kaj[k][j][i] / r3;
       
                double vf = 1.0;

                if ( d_dpdz_set ) 
                {
                    double dp_dx = 0, dp_dy = 0;
                    double dp_dz = d_mean_pressure_gradient;
                    double dpdc_add=0, dpde_add=0, dpdz_add=0;
            
                    Calculate_dP_dc_de_dz(dp_dx, dp_dy, dp_dz, 
                                          icsi[k][j][i], ieta[k][j][i], 
                                          izet[k][j][i], iaj[k][j][i], 
                                          &dpdc_add, &dpde_add, &dpdz_add);
                    dp[k][j][i].x += vf*(dpdc_add*g11_i + 
                                         dpde_add*g12_i + 
                                         dpdz_add*g13_i) * iaj[k][j][i] / r1;
                
                    Calculate_dP_dc_de_dz(dp_dx, dp_dy, dp_dz, 
                                          jcsi[k][j][i], jeta[k][j][i], 
                                          jzet[k][j][i], jaj[k][j][i], 
                                          &dpdc_add, &dpde_add, &dpdz_add);

                    dp[k][j][i].y += vf*(dpdc_add*g21_j + 
                                         dpde_add*g22_j + 
                                         dpdz_add*g23_j) * jaj[k][j][i] / r2;
                
                    Calculate_dP_dc_de_dz(dp_dx, dp_dy, dp_dz, 
                                          kcsi[k][j][i], keta[k][j][i], 
                                          kzet[k][j][i], kaj[k][j][i], 
                                          &dpdc_add, &dpde_add, &dpdz_add);
                    dp[k][j][i].z += vf*(dpdc_add*g31_k + 
                                         dpde_add*g32_k + 
                                         dpdz_add*g33_k) * kaj[k][j][i] / r3;

                } else if(d_inlet_flux>0) {   
                
                    double dpdc_add=0, dpde_add=0, dpdz_add=0;
                    double dt = d_data->getDt();
                    double mean_flux = d_data->getMeanFlux();
                    double mean_area = d_data->getMeanArea();
                    double dz = 1./ kaj[k][j][i] / 
                               sqrt(kzet[k][j][i].x*kzet[k][j][i].x + 
                               kzet[k][j][i].y*kzet[k][j][i].y + 
                               kzet[k][j][i].z*kzet[k][j][i].z);
                    dpdz_add = dz * (mean_flux-d_inlet_flux) / dt / mean_area;
                
                    dp[k][j][i].z += vf*(dpdc_add*g31_k + 
                                         dpde_add*g32_k + 
                                         dpdz_add*g33_k) * kaj[k][j][i] / r3;
                }
        
                if ( i==0 || nvert[k][j][i]+nvert[k][j][i+1]>0.1 || 
                   (!i_periodic && !ii_periodic && i==mx-2) ) {
                    dp[k][j][i].x = 0;
                }
                if ( j==0 || nvert[k][j][i]+nvert[k][j+1][i]>0.1 || 
                   (!j_periodic && !jj_periodic && j==my-2) ) {
                    dp[k][j][i].y = 0;
                }
                if ( k==0 || nvert[k][j][i]+nvert[k+1][j][i]>0.1 || 
                   (!k_periodic && !kk_periodic && k==mz-2) ) {
                    dp[k][j][i].z = 0;
                }
            }

    DMDAVecRestoreArray(fda, ICsi, &icsi);
    DMDAVecRestoreArray(fda, IEta, &ieta);
    DMDAVecRestoreArray(fda, IZet, &izet);

    DMDAVecRestoreArray(fda, JCsi, &jcsi);
    DMDAVecRestoreArray(fda, JEta, &jeta);
    DMDAVecRestoreArray(fda, JZet, &jzet);

    DMDAVecRestoreArray(fda, KCsi, &kcsi);
    DMDAVecRestoreArray(fda, KEta, &keta);
    DMDAVecRestoreArray(fda, KZet, &kzet);

    DMDAVecRestoreArray(da, Nvert, &nvert);
    DMDAVecRestoreArray(da, lP, &p);

    DMDAVecRestoreArray(da, IAj, &iaj);
    DMDAVecRestoreArray(da, JAj, &jaj);
    DMDAVecRestoreArray(da, KAj, &kaj);
    

    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(da, Aj, &aj);

    DMDAVecRestoreArray(fda, Dp,  &dp);

};



PetscErrorCode RHSSolver::Solve(Vec Rhs, double scale)
{
    Cmpnts ***ucont, ***ucont_o, ***ucat;

    Cmpnts ***csi, ***eta, ***zet;
    Cmpnts ***icsi, ***ieta, ***izet;
    Cmpnts ***jcsi, ***jeta, ***jzet;
    Cmpnts ***kcsi, ***keta, ***kzet;
    PetscScalar ***p;

    PetscReal ***nvert;


    DMDALocalInfo info;
    int xs, xe, ys, ye, zs, ze; // Local grid information
    int mx, my, mz; // Dimensions in three directions
    int i, j, k;
    Vec Fp;
    Vec Visc1, Visc2, Visc3;
    
    Cmpnts ***div1, ***div2, ***div3, ***fp;
    Cmpnts ***visc1, ***visc2, ***visc3;
    Cmpnts ***rhs, ***stension;
    PetscReal ***aj, ***iaj, ***jaj, ***kaj; 
    PetscReal ***level, ***rho, ***mu;
    PetscReal ***lnu_t;

    int lxs, lxe, lys, lye, lzs, lze;

    PetscReal dudc, dude, dudz, dvdc, dvde, dvdz, dwdc, dwde, dwdz;
    PetscReal csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2;
    PetscReal g11, g21, g31;
    PetscReal r11, r21, r31, r12, r22, r32, r13, r23, r33;

    PetscScalar solid = 0.5;

    //Lets get the Vecs we need
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();

    Vec ICsi = d_grid->getlICsi();
    Vec IEta = d_grid->getlIEta();
    Vec IZet = d_grid->getlIZet();
    Vec IAj = d_grid->getlIAj();

    Vec JCsi = d_grid->getlJCsi();
    Vec JEta = d_grid->getlJEta();
    Vec JZet = d_grid->getlJZet();
    Vec JAj = d_grid->getlJAj();

    Vec KCsi = d_grid->getlKCsi();
    Vec KEta = d_grid->getlKEta();
    Vec KZet = d_grid->getlKZet();
    Vec KAj = d_grid->getlKAj();
    
    Vec Nu_t;
    Vec Ucont = d_data->getlUcont();
    Vec lUcont_o = d_data->getlUcont_o();
    Vec Ucat = d_data->getlUcat();
    Vec Nvert = d_data->getlNvert();

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDAGetLocalInfo(da, &info);
    mx = info.mx; my = info.my; mz = info.mz;
    xs = info.xs; xe = xs + info.xm;
    ys = info.ys; ye = ys + info.ym;
    zs = info.zs; ze = zs + info.zm;

    /* First we calculate the flux on cell surfaces. 
       Stored on the upper integer node. For example, 
       along i direction, the flux are stored at node 0:mx-2 */

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    
    DMDAVecGetArray(fda, Ucat,  &ucat);
    DMDAVecGetArray(fda, Ucont,  &ucont);
    DMDAVecGetArray(fda, lUcont_o, &ucont_o);
    DMDAVecGetArray(fda, Rhs,  &rhs);

    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, Aj, &aj);

    DMDAVecGetArray(fda, ICsi, &icsi);
    DMDAVecGetArray(fda, IEta, &ieta);
    DMDAVecGetArray(fda, IZet, &izet);
    DMDAVecGetArray(da, IAj, &iaj);

    DMDAVecGetArray(fda, JCsi, &jcsi);
    DMDAVecGetArray(fda, JEta, &jeta);
    DMDAVecGetArray(fda, JZet, &jzet);
    DMDAVecGetArray(da, JAj, &jaj);

    DMDAVecGetArray(fda, KCsi, &kcsi);
    DMDAVecGetArray(fda, KEta, &keta);
    DMDAVecGetArray(fda, KZet, &kzet);
    DMDAVecGetArray(da, KAj, &kaj);
    

    DMDAVecGetArray(da, Nvert, &nvert);
    

    
    //if(user->bctype[0]==11) 
    // {
    //    user->lA_cyl=0;
    //    user->lA_cyl_x=0;
    //    user->lA_cyl_z=0;
    //    user->lFvx_cyl=0;
    //    user->lFvz_cyl=0;
    //    user->lFpx_cyl=0;
    //    user->lFpz_cyl=0;
    //}
    
    
    VecSet(d_Div1, 0);
    VecSet(d_Div2, 0);
    VecSet(d_Div3, 0);
    VecSet(d_Visc1, 0); 
    VecSet(d_Visc2, 0);
    VecSet(d_Visc3, 0);
    
    DMDAVecGetArray(fda, d_Div1, &div1);
    DMDAVecGetArray(fda, d_Div2, &div2);
    DMDAVecGetArray(fda, d_Div3, &div3);
    
    DMDAVecGetArray(fda, d_Visc1, &visc1);
    DMDAVecGetArray(fda, d_Visc2, &visc2);
    DMDAVecGetArray(fda, d_Visc3, &visc3);
    
    
    if (d_les->useLES()) 
    {
        Nu_t = d_les->getlNu_t();
        DMDAVecGetArray(da, Nu_t, &lnu_t);
    }
      

    PetscInt i_periodic = d_grid->isIPeriodic();
    PetscInt j_periodic = d_grid->isJPeriodic();
    PetscInt k_periodic = d_grid->isKPeriodic();
    PetscInt ii_periodic = d_grid->isIIPeriodic();
    PetscInt jj_periodic = d_grid->isJJPeriodic();
    PetscInt kk_periodic = d_grid->isKKPeriodic();

    if (d_grid->isPeriodic())
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) 
                {
                     int flag=0, a=i, b=j, c=k;
            
                     if(i_periodic && i==0) a=mx-2, flag=1;
                     else if(i_periodic && i==mx-1) a=1, flag=1;
        
                     if(j_periodic && j==0) b=my-2, flag=1;
                     else if(j_periodic && j==my-1) b=1, flag=1;
        
                     if(k_periodic && k==0) c=mz-2, flag=1;
                     else if(k_periodic && k==mz-1) c=1, flag=1;
        
                     if(ii_periodic && i==0) a=-2, flag=1;
                     else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
        
                     if(jj_periodic && j==0) b=-2, flag=1;
                     else if(jj_periodic && j==my-1) b=my+1, flag=1;
        
                     if(kk_periodic && k==0) c=-2, flag=1;
                     else if(kk_periodic && k==mz-1) c=mz+1, flag=1;

                        
                     if (flag) 
                         ucont[k][j][i] = ucont[c][b][a];
                }
    


      
    // Thi is the i direction fluxes
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) 
            {
                if(i==mx-1 || j==my-1 || k==mz-1) continue;
                if(j==0 || k==0) continue;
        
                PetscReal ajc = iaj[k][j][i];
                csi0 = icsi[k][j][i].x; 
                csi1 = icsi[k][j][i].y; 
                csi2 = icsi[k][j][i].z;
                eta0 = ieta[k][j][i].x; 
                eta1 = ieta[k][j][i].y; 
                eta2 = ieta[k][j][i].z;
                zet0 = izet[k][j][i].x; 
                zet1 = izet[k][j][i].y; 
                zet2 = izet[k][j][i].z;

                Compute_du_i(i, j, k, 
                             mx, my, mz, 
                             ucat, nvert, 
                             &dudc, &dvdc, &dwdc, 
                             &dude, &dvde, &dwde, 
                             &dudz, &dvdz, &dwdz);
        
                g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
                g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
                g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

                r11 = dudc * csi0 + dude * eta0 + dudz * zet0;    //du_dx * J
                r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;    //dv_dx * J
                r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;    //dw_dx * J

                r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
                r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
                r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

                r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
                r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
                r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;

                /* 
                 This is used for the clark model only so dont need
                 Leaving it so I don't have to write it again

                double du_dx, du_dy, du_dz, 
                       dv_dx, dv_dy, dv_dz, 
                       dw_dx, dw_dy, dw_dz;

                Compute_du_dxyz(csi0, csi1, csi2, 
                                eta0, eta1, eta2, 
                                zet0, zet1, zet2, ajc, 
                                dudc, dvdc, dwdc, 
                                dude, dvde, dwde, 
                                dudz, dvdz, dwdz, 
                                &du_dx, &dv_dx, &dw_dx, 
                                &du_dy, &dv_dy, &dw_dy, 
                                &du_dz, &dv_dz, &dw_dz );
                */ 
        
                int iL=i-1, iR=i+2;
        
                if (i==0 || i==mx-2) {
                    if(i_periodic) iL = mx-3, iR = 2;
                    else if(ii_periodic && i==mx-2) iR=mx+2;
                    else if(ii_periodic && i==0) iL=-3;
                    else iL = i, iR=i+1;
                }
                else if (nvert[k][j][iL]+nvert[k][j][iR] > 0.1) iL = i, iR=i+1;
        
                if (d_second_order) {
                    iL = i, iR=i+1;
                }
        
                if (d_immersed && i!=mx-2 && nvert[k][j][i]>0.1) 
                {
                    double ucon = ucont[k][j][i].x;
                    if (i_periodic && i==0) ucon = ucont[k][j][mx-2].x;
                    else if (ii_periodic && i==0) ucon = ucont[k][j][-2].x;
                    double up = -0.5 * ( ucon + fabs(ucon) );
                    double um = -0.5 * ( ucon - fabs(ucon) );
                    div1[k][j][i].x = 
                       um * (0.125 * ( -ucat[k][j][i+2].x - 
                                       2.*ucat[k][j][i+1].x + 
                                       3.*ucat[k][j][i].x ) + 
                             ucat[k][j][i+1].x ) +
                       up * (0.125 * ( -ucat[k][j][i].x -  
                                      2.*ucat[k][j][i].x +  
                                      3.*ucat[k][j][i+1].x ) +  
                             ucat[k][j][i  ].x);
                    div1[k][j][i].y = 
                       um * (0.125 * ( -ucat[k][j][i+2].y - 
                                       2.*ucat[k][j][i+1].y +  
                                       3.*ucat[k][j][i].y ) +  
                             ucat[k][j][i+1].y) +
                       up * (0.125 * ( -ucat[k][j][i].y - 
                                       2.*ucat[k][j][i].y +
                                       3.*ucat[k][j][i+1].y) +
                             ucat[k][j][i  ].y);
                    div1[k][j][i].z = 
                       um * (0.125 * ( -ucat[k][j][i+2].z -
                                       2.*ucat[k][j][i+1].z +
                                       3.*ucat[k][j][i].z) + 
                             ucat[k][j][i+1].z) +
                       up * (0.125 * ( -ucat[k][j][i].z -
                                       2.*ucat[k][j][i].z +
                                       3.*ucat[k][j][i+1].z) + 
                             ucat[k][j][i].z);
                } else if (d_immersed &&  i!=0 && nvert[k][j][i+1]>0.1) {
                    double ucon = ucont[k][j][i].x;
                    if (i_periodic && i==mx-2) ucon = ucont[k][j][mx-2].x;
                    double up = -0.5 * ( ucon + fabs(ucon) );
                    double um = -0.5 * ( ucon - fabs(ucon) );
                    div1[k][j][i].x = 
                       um * (0.125 * ( -ucat[k][j][i+1].x -
                                       2.*ucat[k][j][i+1].x +
                                       3.*ucat[k][j][i].x) + 
                             ucat[k][j][i+1].x) +
                       up * (0.125 * ( -ucat[k][j][i-1].x -
                                       2.*ucat[k][j][i].x + 
                                       3.*ucat[k][j][i+1].x) + 
                                       ucat[k][j][i].x);
                    div1[k][j][i].y = 
                       um * (0.125 * ( -ucat[k][j][i+1].y -
                                       2.*ucat[k][j][i+1].y +
                                       3. * ucat[k][j][i].y) +
                             ucat[k][j][i+1].y) +
                       up * (0.125 * ( -ucat[k][j][i-1].y -
                                       2.*ucat[k][j][i].y +
                                       3.*ucat[k][j][i+1].y) +  
                             ucat[k][j][i].y);
                    div1[k][j][i].z = 
                       um * (0.125 * ( -ucat[k][j][i+1].z -
                                       2.*ucat[k][j][i+1].z +
                                       3.*ucat[k][j][i].z) + 
                             ucat[k][j][i+1].z) +
                       up * (0.125 * ( -ucat[k][j][i-1].z -
                                       2.*ucat[k][j][i].z +
                                       3.*ucat[k][j][i+1].z) + 
                             ucat[k][j][i].z);
                } else {
                    //This is the second order Inviscid Flux
                    if (d_second_order) 
                    {
                        double ucon = ucont[k][j][i].x;
                        div1[k][j][i].x = -ucon * 0.5 * 
                                         (ucat[k][j][i].x + ucat[k][j][i+1].x);
                        div1[k][j][i].y = -ucon * 0.5 * 
                                         (ucat[k][j][i].y + ucat[k][j][i+1].y);
                        div1[k][j][i].z = -ucon * 0.5 * 
                                         (ucat[k][j][i].z + ucat[k][j][i+1].z);


                    } else {
                        //fourth order
                        double ucon = ucont[k][j][i].x;
                        div1[k][j][i].x = -ucon * 0.0625 * ( 
                           -ucat[k][j][iL].x + 9.*ucat[k][j][i].x + 
                            9.*ucat[k][j][i+1].x - ucat[k][j][iR].x );
                        div1[k][j][i].y = -ucon * 0.0625 * ( 
                           -ucat[k][j][iL].y + 9.*ucat[k][j][i].y + 
                            9.*ucat[k][j][i+1].y - ucat[k][j][iR].y );
                        div1[k][j][i].z = -ucon * 0.0625 * ( 
                           -ucat[k][j][iL].z + 9.*ucat[k][j][i].z + 
                            9.*ucat[k][j][i+1].z - ucat[k][j][iR].z );
                    }
                }
        
                if (nvert[k][j][i]+nvert[k][j][i+1]>0.1) {
                    //if(d_immersed==3/* || !d_immersed*/) {
                    //    div1[k][j][i].x = 0;
                    //    div1[k][j][i].y = 0;
                    //    div1[k][j][i].z = 0;
                    //}
                 } 
       
                 //Viscosity Nu = 1/Re 
                 PetscReal nu = 1./d_data->getRe(), nu_t=0;
        
            
                 //SGS Diffusion
                 if (d_les->useLES()) 
                 {
                     if ( (i==0 && !i_periodic && !ii_periodic) || 
                             nvert[k][j][i]>0.1 )  nu_t = lnu_t[k][j][i+1];
                     else if ( (i==mx-2 && !i_periodic && !ii_periodic) || 
                              nvert[k][j][i+1]>0.1 )  nu_t = lnu_t[k][j][i];
                     else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j][i+1]);

                     if (i==0 && (d_grid->getBC(0)==-1) ) nu_t=0; 
                     if (i==mx-2 && (d_grid->getBC(1)==-1) ) nu_t=0;
                     if (j==1 && (d_grid->getBC(2)==-1||d_grid->getBC(2)==10)) 
                        nu_t=0;
                     if (j==my-2 &&(d_grid->getBC(3)==-1||d_grid->getBC(3)==10))
                        nu_t=0;
                     if (k==1 && (d_grid->getBC(4)==-1||d_grid->getBC(4)==10)) 
                        nu_t=0;
                     if (k==mz-2 &&(d_grid->getBC(5)==-1||d_grid->getBC(5)==10))
                        nu_t=0;
                     if (nvert[k][j][i]+nvert[k][j][i+1]>0.1) nu_t=0; 

                     visc1[k][j][i].x = ajc*nu_t*
                                       ( g11*dudc + g21*dude + g31*dudz + 
                                         r11*csi0 + r21*csi1 + r31*csi2 );
                     visc1[k][j][i].y = ajc*nu_t*
                                       ( g11*dvdc + g21*dvde + g31*dvdz + 
                                         r12*csi0 + r22*csi1 + r32*csi2);
                     visc1[k][j][i].z = ajc*nu_t*
                                       ( g11*dwdc + g21*dwde + g31*dwdz + 
                                         r13*csi0 + r23*csi1 + r33*csi2);
                 } else {
                     visc1[k][j][i].x = 0;
                     visc1[k][j][i].y = 0;
                     visc1[k][j][i].z = 0;
                 }
        
                 //Viscous Diffusion 
                 visc1[k][j][i].x += ajc*nu*
                                     ( g11*dudc + g21*dude + g31*dudz + 
                                       r11*csi0 + r21*csi1 + r31*csi2);
                 visc1[k][j][i].y += ajc*nu*
                                     ( g11*dvdc + g21*dvde + g31*dvdz + 
                                       r12*csi0 + r22*csi1 + r32*csi2);
                 visc1[k][j][i].z += ajc*nu*
                                     ( g11*dwdc + g21*dwde + g31*dwdz + 
                                       r13*csi0 + r23*csi1 + r33*csi2);
        
            }
  
    // j direction
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) 
            {
                if (i==mx-1 || j==my-1 || k==mz-1) continue;
                if (i==0 || k==0) continue;
        
                PetscReal ajc = jaj[k][j][i];
                csi0 = jcsi[k][j][i].x;
                csi1 = jcsi[k][j][i].y; 
                csi2 = jcsi[k][j][i].z;
                eta0 = jeta[k][j][i].x; 
                eta1 = jeta[k][j][i].y; 
                eta2 = jeta[k][j][i].z;
                zet0 = jzet[k][j][i].x; 
                zet1 = jzet[k][j][i].y; 
                zet2 = jzet[k][j][i].z;

                Compute_du_j(i, j, k, 
                             mx, my, mz, 
                             ucat, nvert, 
                             &dudc, &dvdc, &dwdc, 
                             &dude, &dvde, &dwde, 
                             &dudz, &dvdz, &dwdz);
        
                g11 = csi0 * eta0 + csi1 * eta1 + csi2 * eta2;
                g21 = eta0 * eta0 + eta1 * eta1 + eta2 * eta2;
                g31 = zet0 * eta0 + zet1 * eta1 + zet2 * eta2;

                r11 = dudc * csi0 + dude * eta0 + dudz * zet0;
                r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;
                r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;

                r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
                r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
                r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

                r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
                r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
                r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;


                /*
                 see i direction
                double du_dx, du_dy, du_dz, 
                       dv_dx, dv_dy, dv_dz,  
                       dw_dx, dw_dy, dw_dz;

                Compute_du_dxyz(csi0, csi1, csi2, 
                                eta0, eta1, eta2, 
                                zet0, zet1, zet2, ajc, 
                                dudc, dvdc, dwdc, 
                                dude, dvde, dwde, 
                                dudz, dvdz, dwdz, 
                                &du_dx, &dv_dx, &dw_dx, 
                                &du_dy, &dv_dy, &dw_dy, 
                                &du_dz, &dv_dz, &dw_dz );
        
                */

                int jL=j-1, jR=j+2;
        
                if (j==0 || j==my-2) 
                {
                    if(j_periodic) jL = my-3, jR = 2;
                    else if(jj_periodic && j==my-2) jR=my+2;
                    else if(jj_periodic && j==0) jL=-3;
                    else jL = j, jR=j+1;
                }
                else if (nvert[k][jL][i]+nvert[k][jR][i] > 0.1) jL = j, jR=j+1;
        
                if (d_second_order) {
                    jL = j, jR=j+1;
                }
        
        
                if (d_immersed &&  j!=my-2 && nvert[k][j][i]>0.1 ) 
                {
                    double ucon = ucont[k][j][i].y;

                    if (j_periodic && j==0 ) ucon = ucont[k][my-2][i].y;
                    else if (jj_periodic && j==0) ucon = ucont[k][-2][i].y;

                    double up = - 0.5 * ( ucon + fabs(ucon) );
                    double um = - 0.5 * ( ucon - fabs(ucon) );

                    div2[k][j][i].x = 
                       um * (0.125 * ( -ucat[k][j+2][i].x -
                                       2.*ucat[k][j+1][i].x + 
                                       3.*ucat[k][j][i].x ) +
                             ucat[k][j+1][i].x) +             
                       up * (0.125 * ( -ucat[k][j][i].x -
                                       2.*ucat[k][j][i].x +
                                       3.*ucat[k][j+1][i].x) +
                             ucat[k][j][i].x);             
                    div2[k][j][i].y =                      
                       um * (0.125 * ( -ucat[k][j+2][i].y -
                                       2.*ucat[k][j+1][i].y +
                                       3.*ucat[k][j][i].y) + 
                             ucat[k][j+1][i].y) +             
                       up * (0.125 * ( -ucat[k][j][i].y -
                                       2.*ucat[k][j][i].y +
                                       3.*ucat[k][j+1][i].y) +
                             ucat[k][j][i].y);             
                    div2[k][j][i].z =                      
                       um * (0.125 * ( -ucat[k][j+2][i].z -
                                       2.*ucat[k][j+1][i].z + 
                                       3.*ucat[k][j][i].z) +
                             ucat[k][j+1][i].z) +             
                       up * (0.125 * ( -ucat[k][j][i].z -
                                       2.*ucat[k][j][i].z +
                                       3.*ucat[k][j+1][i].z) +
                             ucat[k][j][i].z);
                } else if ( d_immersed &&  j!=0 && nvert[k][j+1][i]>0.1 ) {
                    double ucon = ucont[k][j][i].y;

                    if (j_periodic && j==my-2) ucon = ucont[k][my-2][i].y;

                    double up = - 0.5 * ( ucon + fabs(ucon) );
                    double um = - 0.5 * ( ucon - fabs(ucon) );

                    div2[k][j][i].x = 
                       um * (0.125 * ( -ucat[k][j+1][i].x -
                                       2.*ucat[k][j+1][i].x +
                                       3.*ucat[k][j][i].x) +
                             ucat[k][j+1][i].x) +             
                       up * (0.125 * ( -ucat[k][j-1][i].x -
                                       2.*ucat[k][j][i].x +
                                       3.*ucat[k][j+1][i].x) +
                             ucat[k][j][i].x);             
                    div2[k][j][i].y =                      
                       um * (0.125 * ( -ucat[k][j+1][i].y -
                                       2.*ucat[k][j+1][i].y +
                                       3.*ucat[k][j  ][i].y) + 
                             ucat[k][j+1][i].y) +             
                       up * (0.125 * ( -ucat[k][j-1][i].y -
                                       2.*ucat[k][j][i].y +
                                       3.*ucat[k][j+1][i].y) +
                             ucat[k][j][i].y);             
                    div2[k][j][i].z =                      
                       um * (0.125 * ( -ucat[k][j+1][i].z -
                                       2.*ucat[k][j+1][i].z +
                                       3.*ucat[k][j][i].z) + 
                             ucat[k][j+1][i].z) +             
                       up * (0.125 * ( -ucat[k][j-1][i].z -
                                       2.*ucat[k][j][i].z +
                                       3.*ucat[k][j+1][i].z) +
                             ucat[k][j][i  ].z);
                } else {
                    if (d_second_order) 
                    {
                        double ucon = ucont[k][j][i].y;

                        div2[k][j][i].x = -ucon * 0.5 * 
                                         (ucat[k][j][i].x + ucat[k][j+1][i].x);
                        div2[k][j][i].y = -ucon * 0.5 *
                                         (ucat[k][j][i].y + ucat[k][j+1][i].y);
                        div2[k][j][i].z = -ucon * 0.5 * 
                                         (ucat[k][j][i].z + ucat[k][j+1][i].z);


                    } else {
                        div2[k][j][i].x = -ucont[k][j][i].y * 0.0625 * 
                                   ( -ucat[k][jL][i].x + 9.*ucat[k][j][i].x +
                                     9.*ucat[k][j+1][i].x - ucat[k][jR][i].x );
                        div2[k][j][i].y = -ucont[k][j][i].y * 0.0625 * 
                                   ( -ucat[k][jL][i].y + 9.*ucat[k][j][i].y +
                                     9.*ucat[k][j+1][i].y - ucat[k][jR][i].y );
                        div2[k][j][i].z = -ucont[k][j][i].y * 0.0625 *  
                                   ( -ucat[k][jL][i].z + 9.*ucat[k][j][i].z + 
                                     9.*ucat[k][j+1][i].z - ucat[k][jR][i].z );
                    }
                }
        
        
                if ( nvert[k][j][i]+nvert[k][j+1][i]>0.1) {
                   //if(d_immersed==3/* || !d_immersed*/) {
                   //    div2[k][j][i].x = 0;
                   //    div2[k][j][i].y = 0;
                   //    div2[k][j][i].z = 0;
                   // }
                }
        
                double nu = 1./d_data->getRe(), nu_t = 0;
    
        
                if (d_les->useLES()) {
                    if ((j==0 && !j_periodic && !jj_periodic) || 
                         nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j+1][i];
                    else if ((j==my-2 && !j_periodic && !jj_periodic) || 
                         nvert[k][j+1][i]>0.1 ) nu_t = lnu_t[k][j][i];
                    else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j+1][i]);
        
                    if (j==0 && d_grid->getBC(2)==-1 ) nu_t=0; // xiaolei add
                    if (j==my-2 && d_grid->getBC(3)==-1) nu_t=0; // xiaolei add 
                    if (i==1 && (d_grid->getBC(0)==-1||d_grid->getBC(0)==10)) 
                        nu_t=0;
                    if (i==mx-2 && (d_grid->getBC(1)==-1||d_grid->getBC(1)==10))
                        nu_t=0;
                    if (k==1 && (d_grid->getBC(4)==-1 || d_grid->getBC(4)==10)) 
                        nu_t=0;
                    if (k==mz-2 && (d_grid->getBC(5)==-1||d_grid->getBC(5)==10))
                        nu_t=0;
            
                    if (nvert[k][j][i]+nvert[k][j+1][i]>0.1) nu_t=0;
            
                    visc2[k][j][i].x = ajc*nu_t*
                                       ( g11*dudc + g21*dude + g31*dudz +
                                         r11*eta0 + r21*eta1 + r31*eta2);
                    visc2[k][j][i].y = ajc*nu_t*
                                       ( g11*dvdc + g21*dvde + g31*dvdz + 
                                         r12*eta0 + r22*eta1 + r32*eta2);
                    visc2[k][j][i].z = ajc*nu_t*
                                       ( g11*dwdc + g21*dwde + g31*dwdz + 
                                         r13*eta0 + r23*eta1 + r33*eta2);
                } else {
                    visc2[k][j][i].x = 0;
                    visc2[k][j][i].y = 0;
                    visc2[k][j][i].z = 0;
                }
        
        
                visc2[k][j][i].x += ajc*nu*
                                    ( g11*dudc + g21*dude + g31*dudz + 
                                      r11*eta0 + r21*eta1 + r31*eta2);
                visc2[k][j][i].y += ajc*nu*
                                    ( g11*dvdc + g21*dvde + g31*dvdz +
                                      r12*eta0 + r22*eta1 + r32*eta2);
                visc2[k][j][i].z += ajc*nu*
                                    ( g11*dwdc + g21*dwde + g31*dwdz + 
                                      r13*eta0 + r23*eta1 + r33*eta2);
        
            }
  
    // k direction
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) 
            {
                if (i==mx-1 || j==my-1 || k==mz-1) continue;
                if (i==0 || j==0) continue;
        
                PetscReal ajc = kaj[k][j][i];
                csi0 = kcsi[k][j][i].x;
                csi1 = kcsi[k][j][i].y;
                csi2 = kcsi[k][j][i].z;
                eta0 = keta[k][j][i].x; 
                eta1 = keta[k][j][i].y; 
                eta2 = keta[k][j][i].z;
                zet0 = kzet[k][j][i].x;
                zet1 = kzet[k][j][i].y;
                zet2 = kzet[k][j][i].z;
        
                Compute_du_k(i, j, k, 
                             mx, my, mz, 
                             ucat, nvert, 
                             &dudc, &dvdc, &dwdc, 
                             &dude, &dvde, &dwde, 
                             &dudz, &dvdz, &dwdz);
        
                g11 = csi0 * zet0 + csi1 * zet1 + csi2 * zet2;
                g21 = eta0 * zet0 + eta1 * zet1 + eta2 * zet2;
                g31 = zet0 * zet0 + zet1 * zet1 + zet2 * zet2;

                r11 = dudc * csi0 + dude * eta0 + dudz * zet0;
                r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;
                r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;

                r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
                r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
                r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

                r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
                r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
                r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;

                /*
                double du_dx, du_dy, du_dz, 
                       dv_dx, dv_dy, dv_dz, 
                       dw_dx, dw_dy, dw_dz;

                Compute_du_dxyz(csi0, csi1, csi2, 
                                eta0, eta1, eta2, 
                                zet0, zet1, zet2, ajc, 
                                dudc, dvdc, dwdc, 
                                dude, dvde, dwde, 
                                dudz, dvdz, dwdz, 
                                &du_dx, &dv_dx, &dw_dx, 
                                &du_dy, &dv_dy, &dw_dy, 
                                &du_dz, &dv_dz, &dw_dz );
                */
 
                int kL=k-1, kR=k+2;
        
                if (k==0 || k==mz-2) 
                {
                    if (k_periodic) kL = mz-3, kR = 2;
                    else if (kk_periodic && k==mz-2) kR=mz+2;
                    else if (kk_periodic && k==0) kL=-3;
                    else kL = k, kR=k+1;
                }
                else if(nvert[kL][j][i]+nvert[kR][j][i] > 0.1) kL = k, kR=k+1;
        
                if (d_second_order) {
                    kL = k, kR=k+1;
                }
        
        

                if (d_immersed && k!=mz-2 && nvert[k][j][i]>0.1 ) 
                {
                    double ucon = ucont[k][j][i].z;
        
                    if (k_periodic && k==0) ucon = ucont[mz-2][j][i].z;
                    else if(kk_periodic && k==0) ucon = ucont[-2][j][i].z;
                    double up = - 0.5 * ( ucon + fabs(ucon) );
                    double um = - 0.5 * ( ucon - fabs(ucon) );

                    div3[k][j][i].x = 
                       um * (0.125 * ( -ucat[k+2][j][i].x -
                                       2.*ucat[k+1][j][i].x +
                                       3.*ucat[k][j][i].x) + 
                             ucat[k+1][j][i].x)   +             
                       up * (0.125 * ( -ucat[k][j][i].x -
                                       2.*ucat[k][j][i].x +
                                       3.*ucat[k+1][j][i].x) +
                             ucat[k][j][i].x);               
                    div3[k][j][i].y =                             
                       um * (0.125 * ( -ucat[k+2][j][i].y -
                                       2.*ucat[k+1][j][i].y +
                                       3.*ucat[k][j][i].y) + 
                             ucat[k+1][j][i].y)   +             
                       up * (0.125 * ( -ucat[k][j][i].y -
                                       2.*ucat[k][j][i].y +
                                       3.*ucat[k+1][j][i].y) +
                             ucat[k][j][i].y);               
                    div3[k][j][i].z =                             
                       um * (0.125 * ( -ucat[k+2][j][i].z -
                                       2.*ucat[k+1][j][i].z +
                                       3.*ucat[k][j][i].z) +
                             ucat[k+1][j][i].z)   +             
                       up * (0.125 * ( -ucat[k][j][i].z -
                                       2.*ucat[k][j][i].z +
                                       3.*ucat[k+1][j][i].z) +
                             ucat[k][j][i].z);
                } else if ( d_immersed &&  k!=0 && nvert[k+1][j][i]>0.1 ) {

                    double ucon = ucont[k][j][i].z;
        
                    if (k_periodic && k==mz-2)  ucon = ucont[mz-2][j][i].z;

                    double up = - 0.5 * ( ucon + fabs(ucon) );
                    double um = - 0.5 * ( ucon - fabs(ucon) );
                    div3[k][j][i].x = 
                       um * (0.125 * ( -ucat[k+1][j][i].x -
                                       2.*ucat[k+1][j][i].x +
                                       3.*ucat[k][j][i].x) + 
                             ucat[k+1][j][i].x)   +             
                       up * (0.125 * ( -ucat[k-1][j][i].x -
                                       2.*ucat[k][j][i].x +
                                       3.*ucat[k+1][j][i].x) +
                             ucat[k][j][i].x);               
                    div3[k][j][i].y =                             
                       um * (0.125 * ( -ucat[k+1][j][i].y -
                                       2.*ucat[k+1][j][i].y +
                                       3.*ucat[k][j][i].y) +
                             ucat[k+1][j][i].y)   +             
                       up * (0.125 * ( -ucat[k-1][j][i].y -
                                       2.*ucat[k][j][i].y +
                                       3.*ucat[k+1][j][i].y) +
                             ucat[k][j][i].y);               
                     div3[k][j][i].z =                             
                        um * (0.125 * ( -ucat[k+1][j][i].z -
                                        2.*ucat[k+1][j][i].z +
                                        3.*ucat[k][j][i].z) +
                              ucat[k+1][j][i].z)   +             
                        up * (0.125 * ( -ucat[k-1][j][i].z -
                                        2.*ucat[k][j][i].z + 
                                        3.*ucat[k+1][j][i].z) +
                              ucat[k][j][i].z);
                } else {
                    if (d_second_order) 
                    {

                        double ucon = ucont[k][j][i].z;

                        div3[k][j][i].x = -ucon * 0.5 * 
                                          (ucat[k][j][i].x + ucat[k+1][j][i].x);
                        div3[k][j][i].y = -ucon * 0.5 * 
                                          (ucat[k][j][i].y + ucat[k+1][j][i].y);
                        div3[k][j][i].z = -ucon * 0.5 * 
                                          (ucat[k][j][i].z + ucat[k+1][j][i].z);


                    } else {
                        div3[k][j][i].x = -ucont[k][j][i].z * 0.0625 * 
                                    ( -ucat[kL][j][i].x + 9.*ucat[k][j][i].x +
                                      9.*ucat[k+1][j][i].x - ucat[kR][j][i].x );
                        div3[k][j][i].y = -ucont[k][j][i].z * 0.0625 * 
                                    ( -ucat[kL][j][i].y + 9.*ucat[k][j][i].y + 
                                      9.*ucat[k+1][j][i].y - ucat[kR][j][i].y );
                        div3[k][j][i].z = -ucont[k][j][i].z * 0.0625 * 
                                    ( -ucat[kL][j][i].z  + 9.*ucat[k][j][i].z +
                                      9.*ucat[k+1][j][i].z - ucat[kR][j][i].z );
                    }
                }        
        
        
                if ( nvert[k][j][i]+nvert[k+1][j][i]>0.1) {
                    //if(d_immersed==3/* || !d_immersed*/) {
                    //    div3[k][j][i].x = 0;
                    //    div3[k][j][i].y = 0;
                    //    div3[k][j][i].z = 0;
                    //    }
                }
        
                double nu = 1./d_data->getRe(), nu_t =0;
    
                if (d_les->useLES()) 
                {

                    if ((k==0 && !k_periodic && !kk_periodic) || 
                        nvert[k][j][i]>0.1 )  nu_t = lnu_t[k+1][j][i];
                    else if ((k==mz-2 && !k_periodic && !kk_periodic) || 
                        nvert[k+1][j][i]>0.1 )  nu_t = lnu_t[k][j][i];
                    else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k+1][j][i]);
            
                    if (k==0 && d_grid->getBC(4)==-1) nu_t=0;
                    if (k==mz-2 && d_grid->getBC(5)==-1) nu_t=0;

                    if (i==1 && (d_grid->getBC(0)==-1 || d_grid->getBC(0)==10))
                        nu_t=0;
                    if (i==mx-2 && (d_grid->getBC(1)==-1||d_grid->getBC(1)==10))
                        nu_t=0;
                    if (j==1 && (d_grid->getBC(2)==-1 || d_grid->getBC(2)==10))
                        nu_t=0;
                    if (j==my-2 && (d_grid->getBC(3)==-1||d_grid->getBC(3)==10))
                        nu_t=0;
                    if(nvert[k][j][i]+nvert[k+1][j][i]>0.1) nu_t=0;

                    visc3[k][j][i].x = ajc*nu_t*
                                       (g11*dudc + g21*dude + g31*dudz + 
                                        r11*zet0 + r21*zet1 + r31*zet2);
                    visc3[k][j][i].y = ajc*nu_t*
                                       (g11*dvdc + g21*dvde + g31*dvdz + 
                                        r12*zet0 + r22*zet1 + r32*zet2);
                    visc3[k][j][i].z = ajc*nu_t*
                                       (g11*dwdc + g21*dwde + g31*dwdz +
                                        r13*zet0 + r23*zet1 + r33*zet2);
                 } else {
                    visc3[k][j][i].x = 0;
                    visc3[k][j][i].y = 0;
                    visc3[k][j][i].z = 0;
                 }
        
                 visc3[k][j][i].x += ajc*nu*
                                     (g11*dudc + g21*dude + g31*dudz + 
                                      r11*zet0 + r21*zet1 + r31*zet2);
                 visc3[k][j][i].y += ajc*nu*
                                     (g11*dvdc + g21*dvde + g31*dvdz +
                                      r12*zet0 + r22*zet1 + r32*zet2);
                 visc3[k][j][i].z += ajc*nu*
                                     (g11*dwdc + g21*dwde + g31*dwdz + 
                                      r13*zet0 + r23*zet1 + r33*zet2);
        
            }
    
    
    DMDAVecRestoreArray(fda, d_Div1, &div1);
    DMDAVecRestoreArray(fda, d_Div2, &div2);
    DMDAVecRestoreArray(fda, d_Div3, &div3);
        
    DMDALocalToLocalBegin(fda, d_Div1, INSERT_VALUES, d_Div1);
    DMDALocalToLocalEnd(fda, d_Div1, INSERT_VALUES, d_Div1);
    DMDALocalToLocalBegin(fda, d_Div2, INSERT_VALUES, d_Div2);
    DMDALocalToLocalEnd(fda, d_Div2, INSERT_VALUES, d_Div2);
    DMDALocalToLocalBegin(fda, d_Div3, INSERT_VALUES, d_Div3);
    DMDALocalToLocalEnd(fda, d_Div3, INSERT_VALUES, d_Div3);
        
    DMDAVecGetArray(fda, d_Div1, &div1);
    DMDAVecGetArray(fda, d_Div2, &div2);
    DMDAVecGetArray(fda, d_Div3, &div3);
    
    DMDAVecRestoreArray(fda, d_Visc1, &visc1);
    DMDAVecRestoreArray(fda, d_Visc2, &visc2);
    DMDAVecRestoreArray(fda, d_Visc3, &visc3);
    
    DMDALocalToLocalBegin(fda, d_Visc1, INSERT_VALUES, d_Visc1);
    DMDALocalToLocalEnd(fda, d_Visc1, INSERT_VALUES, d_Visc1);
    DMDALocalToLocalBegin(fda, d_Visc2, INSERT_VALUES, d_Visc2);
    DMDALocalToLocalEnd(fda, d_Visc2, INSERT_VALUES, d_Visc2);
    DMDALocalToLocalBegin(fda, d_Visc3, INSERT_VALUES, d_Visc3);
    DMDALocalToLocalEnd(fda, d_Visc3, INSERT_VALUES, d_Visc3);
    
    DMDAVecGetArray(fda, d_Visc1, &visc1);
    DMDAVecGetArray(fda, d_Visc2, &visc2);
    DMDAVecGetArray(fda, d_Visc3, &visc3);

    DMDAVecGetArray(fda, d_Fp, &fp);
    
    if (d_grid->isPeriodic())
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) 
                {
                    int a=i, b=j, c=k;

                    int flag=0;
        
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
                        div1[k][j][i] = div1[c][b][a];
                        div2[k][j][i] = div2[c][b][a];
                        div3[k][j][i] = div3[c][b][a];
                        visc1[k][j][i] = visc1[c][b][a];
                        visc2[k][j][i] = visc2[c][b][a];
                        visc3[k][j][i] = visc3[c][b][a];
            
                    }
                }
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) 
            {
                Cmpnts div, div_4th;
        
                div.x = (div1[k][j][i].x - div1[k][j][i-1].x + 
                         div2[k][j][i].x - div2[k][j-1][i].x + 
                         div3[k][j][i].x - div3[k-1][j][i].x);
                div.y = (div1[k][j][i].y - div1[k][j][i-1].y + 
                         div2[k][j][i].y - div2[k][j-1][i].y + 
                         div3[k][j][i].y - div3[k-1][j][i].y);
                div.z = (div1[k][j][i].z - div1[k][j][i-1].z + 
                         div2[k][j][i].z - div2[k][j-1][i].z + 
                         div3[k][j][i].z - div3[k-1][j][i].z);
        
                div_4th.x = div_4th.y = div_4th.z = 0;
         
                int iR=i+1, iL=i-2;
                int jR=j+1, jL=j-2;
                int kR=k+1, kL=k-2;
                double denom_i=3.;
                double denom_j=3.;
                double denom_k=3.;
        
                if (i==1) 
                {
                    if ( i_periodic ) iL=mx-3;
                    else if ( ii_periodic ) iL=-3;
                    else iR=i, iL=i-1, denom_i=1.;
                } else if (i==2 || i==mx-3) {
                    if ( i_periodic ) {}
                    else if ( ii_periodic ) {}
                    else iR=i, iL=i-1, denom_i=1.;
                } else if (i==mx-2) {
                    if ( i_periodic) iR=1;
                    else if ( ii_periodic) iR=mx+1;
                    else iR=i, iL=i-1, denom_i=1.;
                }
        
                if (j==1) 
                {
                    if ( j_periodic ) jL=my-3;
                    else if ( jj_periodic ) jL=-3;
                    else jR=j, jL=j-1, denom_j=1.;
                } else if (j==2 || j==my-3) {
                    if( j_periodic ) {}
                    else if( jj_periodic ) {}
                    else jR=j, jL=j-1, denom_j=1.;
                } else if(j==my-2) {
                    if ( j_periodic) jR=1;
                    else if ( jj_periodic) jR=my+1;
                    else jR=j, jL=j-1, denom_j=1.;
                }
        
                if (k==1) 
                {
                   if ( k_periodic ) kL=mz-3;
                   else if ( kk_periodic ) kL=-3;
                   else kR=k, kL=k-1, denom_k=1.;
                } else if(k==2 || k==mz-3) {
                   if ( k_periodic ) {}
                   else if ( kk_periodic ) {}
                   else kR=k, kL=k-1, denom_k=1.;
                } else if(k==mz-2) {
                   if ( k_periodic) kR=1;
                   else if ( kk_periodic) kR=mz+1;
                   else kR=k, kL=k-1, denom_k=1.;
                }
        
               if (nvert[k][j][i-1]+nvert[k][j][i]+nvert[k][j][i+1]>0.1) 
               {
                   iR=i, iL=i-1, denom_i=1.;
               }
               if (nvert[k][j-1][i]+nvert[k][j][i]+nvert[k][j+1][i]>0.1) 
               {
                   jR=j, jL=j-1, denom_j=1.;
               }
               if (nvert[k-1][j][i]+nvert[k][j][i]+nvert[k+1][j][i]>0.1)
               {   
                   kR=k, kL=k-1, denom_k=1.;
               }
            
               if (d_second_order) 
               {
                   fp[k][j][i] = div;
                   iR=i, iL=i-1;
                   jR=j, jL=j-1;
                   kR=k, kL=k-1;
               } else {
                   Subtract_Scale_AddTo(div1[k][j][iR], 
                                        div1[k][j][iL], 1./denom_i, &div_4th); 
                   Subtract_Scale_AddTo(div2[k][jR][i], 
                                        div2[k][jL][i], 1./denom_j, &div_4th); 
                   Subtract_Scale_AddTo(div3[kR][j][i], 
                                        div3[kL][j][i], 1./denom_k, &div_4th); 
                   AxByC ( 9./8., div, -1./8., div_4th, &fp[k][j][i]);
               }

               double r=1.0;
        
        
               fp[k][j][i].x += (visc1[k][j][i].x - visc1[k][j][i-1].x + 
                                 visc2[k][j][i].x - visc2[k][j-1][i].x + 
                                 visc3[k][j][i].x - visc3[k-1][j][i].x) / r;
               fp[k][j][i].y += (visc1[k][j][i].y - visc1[k][j][i-1].y + 
                                 visc2[k][j][i].y - visc2[k][j-1][i].y + 
                                 visc3[k][j][i].y - visc3[k-1][j][i].y) / r;
               fp[k][j][i].z += (visc1[k][j][i].z - visc1[k][j][i-1].z + 
                                 visc2[k][j][i].z - visc2[k][j-1][i].z + 
                                 visc3[k][j][i].z - visc3[k-1][j][i].z) / r;
        }
        
        
    DMDAVecRestoreArray(fda, d_Fp, &fp);
    
    DMDALocalToLocalBegin(fda, d_Fp, INSERT_VALUES, d_Fp);
    DMDALocalToLocalEnd(fda, d_Fp, INSERT_VALUES, d_Fp);
    
    DMDAVecGetArray(fda, d_Fp, &fp);
    
    if (d_grid->isPeriodic())
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) 
                {
                    int flag=0, a=i, b=j, c=k;
                
                    if(i_periodic && i==0) a=mx-2, flag=1;
                    else if(i_periodic && i==mx-1) a=1, flag=1;
        
                    if(j_periodic && j==0) b=my-2, flag=1;
                    else if(j_periodic && j==my-1) b=1, flag=1;
        
                    if(k_periodic && k==0) c=mz-2, flag=1;
                    else if(k_periodic && k==mz-1) c=1, flag=1;
        
                    if(ii_periodic && i==0) a=-2, flag=1;
                    else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
        
                    if(jj_periodic && j==0) b=-2, flag=1;
                    else if(jj_periodic && j==my-1) b=my+1, flag=1;
        
                    if(kk_periodic && k==0) c=-2, flag=1;
                    else if(kk_periodic && k==mz-1) c=mz+1, flag=1;
        
                     if(flag) fp[k][j][i] = fp[c][b][a];

                }         
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) 
            {
                rhs[k][j][i].x += scale*iaj[k][j][i]*
                                   (0.5*(csi[k][j][i].x*fp[k][j][i].x + 
                                         csi[k][j][i].y*fp[k][j][i].y + 
                                         csi[k][j][i].z*fp[k][j][i].z) + 
                                    0.5*(csi[k][j][i+1].x*fp[k][j][i+1].x +
                                         csi[k][j][i+1].y*fp[k][j][i+1].y +
                                         csi[k][j][i+1].z*fp[k][j][i+1].z));
                rhs[k][j][i].y += scale*jaj[k][j][i]*
                                   (0.5*(eta[k][j][i].x*fp[k][j][i].x + 
                                         eta[k][j][i].y*fp[k][j][i].y + 
                                         eta[k][j][i].z*fp[k][j][i].z) + 
                                    0.5*(eta[k][j+1][i].x*fp[k][j+1][i].x + 
                                         eta[k][j+1][i].y*fp[k][j+1][i].y + 
                                         eta[k][j+1][i].z*fp[k][j+1][i].z));
                rhs[k][j][i].z += scale*kaj[k][j][i]*
                                   (0.5*(zet[k][j][i].x*fp[k][j][i].x +
                                         zet[k][j][i].y*fp[k][j][i].y + 
                                         zet[k][j][i].z*fp[k][j][i].z) + 
                                    0.5*(zet[k+1][j][i].x*fp[k+1][j][i].x + 
                                         zet[k+1][j][i].y*fp[k+1][j][i].y + 
                                         zet[k+1][j][i].z*fp[k+1][j][i].z));

        
        
                if (nvert[k][j][i]+nvert[k][j][i+1]>0.1 || 
                   (!i_periodic && !ii_periodic && i==mx-2) ) 
                   rhs[k][j][i].x = 0;
                if (nvert[k][j][i]+nvert[k][j+1][i]>0.1 || 
                   (!j_periodic && !jj_periodic && j==my-2) ) 
                    rhs[k][j][i].y = 0;
                if (nvert[k][j][i]+nvert[k+1][j][i]>0.1 || 
                   (!k_periodic && !kk_periodic && k==mz-2) ) 
                    rhs[k][j][i].z = 0;
            }

    if (d_les->useLES())  
        DMDAVecRestoreArray(da, Nu_t, &lnu_t);
  
    DMDAVecRestoreArray(da, IAj, &iaj);
    DMDAVecRestoreArray(da, JAj, &jaj);
    DMDAVecRestoreArray(da, KAj, &kaj);

    if (xs ==0) 
    {
        i = 0;
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++) 
            {
                rhs[k][j][i].x = 0;
                rhs[k][j][i].y = 0;
                rhs[k][j][i].z = 0;
            }

    }

    if (xe == mx) 
    {
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++) 
            {
                if (!i_periodic && !ii_periodic) {
                    i = mx-2;
                    rhs[k][j][i].x = 0;
                }
                i = mx-1;
                rhs[k][j][i].x = 0;
                rhs[k][j][i].y = 0;
                rhs[k][j][i].z = 0;
            }

    }


    if (ys == 0) 
    {
        for (k=zs; k<ze; k++)
            for (i=xs; i<xe; i++) 
            {
                j=0;
                rhs[k][j][i].x = 0;
                rhs[k][j][i].y = 0;
                rhs[k][j][i].z = 0;
            }

    }
  
    if (ye == my) 
    {
        for (k=zs; k<ze; k++) 
            for (i=xs; i<xe; i++) 
            {
                if (!j_periodic && !jj_periodic) {
                    j=my-2;
                    rhs[k][j][i].y = 0;
                }
                j=my-1;
                rhs[k][j][i].x = 0;
                rhs[k][j][i].y = 0;
                rhs[k][j][i].z = 0;
            }

    }
    
    
    if (zs == 0) 
    {
        k=0;
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) 
            {
                rhs[k][j][i].x = 0;
                rhs[k][j][i].y = 0;
                rhs[k][j][i].z = 0;
            }
    }
  
    if (ze == mz) 
    {
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                if (!k_periodic && !kk_periodic) {
                    k=mz-2;
                    rhs[k][j][i].z = 0;
                }
                k=mz-1;
                rhs[k][j][i].x = 0;
                rhs[k][j][i].y = 0;
                rhs[k][j][i].z = 0;
            }
    }
    
    DMDAVecRestoreArray(fda, Ucont, &ucont);
    DMDAVecRestoreArray(fda, lUcont_o, &ucont_o);
    DMDAVecRestoreArray(fda, Ucat,  &ucat);
    DMDAVecRestoreArray(fda, Rhs,  &rhs);

    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
      
    DMDAVecRestoreArray(fda, d_Fp, &fp);
    DMDAVecRestoreArray(fda, d_Div1, &div1);
    DMDAVecRestoreArray(fda, d_Div2, &div2);
    DMDAVecRestoreArray(fda, d_Div3, &div3);
    
    DMDAVecRestoreArray(fda, d_Visc1, &visc1);
    DMDAVecRestoreArray(fda, d_Visc2, &visc2);
    DMDAVecRestoreArray(fda, d_Visc3, &visc3);
    
    DMDAVecRestoreArray(da, Aj, &aj);
  
    DMDAVecRestoreArray(fda, ICsi, &icsi);
    DMDAVecRestoreArray(fda, IEta, &ieta);
    DMDAVecRestoreArray(fda, IZet, &izet);

    DMDAVecRestoreArray(fda, JCsi, &jcsi);
    DMDAVecRestoreArray(fda, JEta, &jeta);
    DMDAVecRestoreArray(fda, JZet, &jzet);

    DMDAVecRestoreArray(fda, KCsi, &kcsi);
    DMDAVecRestoreArray(fda, KEta, &keta);
    DMDAVecRestoreArray(fda, KZet, &kzet);
    
    DMDAVecRestoreArray(da, Nvert, &nvert);
    
    return 0;
};


void RHSSolver::Calculate_dP_dc_de_dz(
    double dp_dx, double dp_dy, double dp_dz, 
    Cmpnts csi, Cmpnts eta, Cmpnts zet, 
    double aj, 
    double *dpdc, double *dpde, double *dpdz)
{
    const double a11=csi.x, a12=eta.x, a13=zet.x;
    const double a21=csi.y, a22=eta.y, a23=zet.y;
    const double a31=csi.z, a32=eta.z, a33=zet.z;
    double invA[3][3];
    
    double det = a11*(a33*a22-a32*a23)-
                 a21*(a33*a12-a32*a13)+
                 a31*(a23*a12-a22*a13);

    invA[0][0] = (a33*a22-a32*a23)/det;   
    invA[0][1] = - (a33*a12-a32*a13)/det;        
    invA[0][2] = (a23*a12-a22*a13)/det;

    invA[1][0] = -(a33*a21-a31*a23)/det;    
    invA[1][1] = (a33*a11-a31*a13)/det; 
    invA[1][2] = - (a23*a11-a21*a13)/det;
       
    invA[2][0] = (a32*a21-a31*a22)/det;
    invA[2][1] = - (a32*a11-a31*a12)/det;
    invA[2][2] = (a22*a11-a21*a12)/det;
    
    *dpdc = (invA[0][0]*dp_dx + invA[0][1]*dp_dy + invA[0][2]*dp_dz) / aj;
    *dpde = (invA[1][0]*dp_dx + invA[1][1]*dp_dy + invA[1][2]*dp_dz) / aj;
    *dpdz = (invA[2][0]*dp_dx + invA[2][1]*dp_dy + invA[2][2]*dp_dz) / aj;
}


PetscErrorCode RHSSolver::ReadFromInput()
{
    PetscOptionsGetReal(PETSC_NULL, "-dpdz", &d_mean_pressure_gradient, 
                                             &d_dpdz_set);
    PetscOptionsGetReal(PETSC_NULL, "-flux", &d_inlet_flux, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-second_order",&d_second_order,PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-imm", &d_immersed, PETSC_NULL);
}
