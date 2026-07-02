#include "WallModel.h"

WallModel::WallModel(const std::string& object_name,
    CurvGrid *grid,
    UData *data,
    LESModel *les,
    ImmersedBoundary *ib):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data),
    d_les(les),
    d_ib(ib)
{

    d_imin_wm = 0;
    d_imax_wm = 0;
    d_jmin_wm = 0;
    d_jmax_wm = 0;
    d_kmin_wm = 0;
    d_kmax_wm = 0;
    d_ib_wm = 0;

    d_use_wall = PETSC_FALSE;
    d_roughness_size = 0.0;
    d_alfa_wm = 0;
    d_les_eps = 1.e-7;
    d_powerlawwallmodel = 0; 
    d_dhratio_wm = 1.05;
    d_dh1_wm = 0.001;
    d_infRe = 0;

    ReadFromInput();
}

WallModel::~WallModel()
{}

void WallModel::Initialize()
{

   if (!d_use_wall) return;

   Vec lUcat = d_data->getlUcat();
   Vec lP = d_data->getlP();

   VecDuplicate(lUcat, &d_lVisc1_wm);
   VecDuplicate(lUcat, &d_lVisc2_wm);
   VecDuplicate(lUcat, &d_lVisc3_wm);
   VecDuplicate(lP, &d_lTau);

}

void WallModel::CalculateVisc()
{

   if (!d_use_wall) return;

    int i, j, k;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    int xs, xe, ys, ye, zs, ze; // Local grid information
    int mx, my, mz; // Dimensions in three directions
    int lxs, lxe, lys, lye, lzs, lze;
    DMDAGetLocalInfo(da, &info);
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

    Cmpnts ***ucat, ***lucat_o;

    Cmpnts ***csi, ***eta, ***zet;
    Cmpnts ***icsi, ***ieta, ***izet;
    Cmpnts ***jcsi, ***jeta, ***jzet;
    Cmpnts ***kcsi, ***keta, ***kzet;
    Cmpnts ***visc1_wm, ***visc2_wm, ***visc3_wm;
    Cmpnts ***coor;

    PetscReal ***ustar;
    PetscReal ***tau;
    PetscReal ***lnu_t;
    PetscReal ***p;
    PetscReal ***nvert;
    PetscReal ***aj, ***iaj, ***jaj, ***kaj;

    int ibi;
    double area, nu_t;
    double sd, sb, sc, st;
    double nx, ny, nz;
    double ren, nu_t_b,  nu_t_c, nu_t_d;
    double tau_w;
    PetscInt bctype;
    Cmpnts Ua, Ub, Uc, Ud;
    Cmpnts Ug;

    PetscReal dudc, dude, dudz, dvdc, dvde, dvdz, dwdc, dwde, dwdz;
    PetscReal csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2;
    PetscReal g11, g21, g31;
    PetscReal r11, r21, r31, r12, r22, r32, r13, r23, r33;

    PetscReal dudc_wm, dvdc_wm, dwdc_wm;
    PetscReal dude_wm, dvde_wm, dwde_wm; 
    PetscReal dudz_wm, dvdz_wm, dwdz_wm;
    PetscReal r11_wm, r21_wm, r31_wm, r12_wm;
    PetscReal r22_wm, r32_wm, r13_wm, r23_wm, r33_wm;

    Vec Coor;
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
 
    Vec lUcat = d_data->getlUcat();
    Vec lUstar = d_data->getlUstar();
    Vec lP = d_data->getlP();
    Vec lNvert = d_data->getlNvert();

    Vec lNu_t = d_les->getlNu_t();


  /* First we calculate the flux on cell surfaces. Stored on the upper integer
     node. For example, along i direction, the flux are stored at node 0:mx-2*/

    VecSet(d_lVisc1_wm, 0.0);
    VecSet(d_lVisc2_wm, 0.0);
    VecSet(d_lVisc3_wm, 0.0);

    DMDAVecGetArray(fda, d_lVisc1_wm, &visc1_wm);
    DMDAVecGetArray(fda, d_lVisc2_wm, &visc2_wm);
    DMDAVecGetArray(fda, d_lVisc3_wm, &visc3_wm);
    DMDAVecGetArray(da, d_lTau, &tau);
    
    DMDAVecGetArray(da, lNu_t, &lnu_t);

    DMGetCoordinatesLocal(da, &Coor);
    DMDAVecGetArray(fda, Coor, &coor);

    DMDAVecGetArray(fda, lUcat,  &ucat);
    DMDAVecGetArray(da, lP, &p);

    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);

    DMDAVecGetArray(fda, ICsi, &icsi);
    DMDAVecGetArray(fda, IEta, &ieta);
    DMDAVecGetArray(fda, IZet, &izet);

    DMDAVecGetArray(fda, JCsi, &jcsi);
    DMDAVecGetArray(fda, JEta, &jeta);
    DMDAVecGetArray(fda, JZet, &jzet);

    DMDAVecGetArray(fda, KCsi, &kcsi);
    DMDAVecGetArray(fda, KEta, &keta);
    DMDAVecGetArray(fda, KZet, &kzet);

    DMDAVecGetArray(da, lNvert, &nvert);
    
    DMDAVecGetArray(da, Aj, &aj);
    DMDAVecGetArray(da, IAj, &iaj);
    DMDAVecGetArray(da, JAj, &jaj);
    DMDAVecGetArray(da, KAj, &kaj);
  
    DMDAVecGetArray(da, lUstar, &ustar);
    
    PetscReal ts, te;

    PetscInt i_periodic = d_grid->isIPeriodic();
    PetscInt j_periodic = d_grid->isJPeriodic();
    PetscInt k_periodic = d_grid->isKPeriodic();
    PetscInt ii_periodic = d_grid->isIIPeriodic();
    PetscInt jj_periodic = d_grid->isJJPeriodic();
    PetscInt kk_periodic = d_grid->isKKPeriodic();

    PetscTime(&ts);
    if (d_immersed && d_ib_wm) { 
    
        IBMListNode *current;
        IBMList *ibmlist = d_ib->getIBMList();
        for (int ibi=0; ibi<d_ib->getNumberOfIBMBodies(); ibi++) {
            current = ibmlist[ibi].head;
            
            IBMNodes *ibm = d_ib->getIBMNodes();
            ibm += ibi;

            std::vector<double> count;

            while (current) {

                double sk1, sk2, sk3;
                double cs1, cs2, cs3;
                double nx, ny, nz;
                const double ren = d_data->getRe();
                
                IBMInfo *ibminfo = &current->ibm_intp;
                current = current->next;

                int ni = ibminfo->cell;
                int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
                int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
                int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
                i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;

                sb = ibminfo->d_s; sc = sb + ibminfo->d_i;
                sk1  = ibminfo->cr1, sk2 = ibminfo->cr2, sk3 = ibminfo->cr3;
                cs1 = ibminfo->cs1, cs2 = ibminfo->cs2, cs3 = ibminfo->cs3;
                nx = ibm->nf_x[ni], ny = ibm->nf_y[ni], nz = ibm->nf_z[ni];
                            
                if (ni>=0) {
                    Ua.x = ibm->u[ibm->nv1[ni]].x * cs1 + 
                           ibm->u[ibm->nv2[ni]].x * cs2 + 
                           ibm->u[ibm->nv3[ni]].x * cs3;
                    Ua.y = ibm->u[ibm->nv1[ni]].y * cs1 + 
                           ibm->u[ibm->nv2[ni]].y * cs2 + 
                           ibm->u[ibm->nv3[ni]].y * cs3;
                    Ua.z = ibm->u[ibm->nv1[ni]].z * cs1 + 
                           ibm->u[ibm->nv2[ni]].z * cs2 + 
                           ibm->u[ibm->nv3[ni]].z * cs3;
                } else {
                    Ua.x = Ua.y = Ua.z = 0;
                }
                
                Uc.x = (ucat[kp1][jp1][ip1].x * sk1 + 
                        ucat[kp2][jp2][ip2].x * sk2 + 
                        ucat[kp3][jp3][ip3].x * sk3);
                Uc.y = (ucat[kp1][jp1][ip1].y * sk1 + 
                        ucat[kp2][jp2][ip2].y * sk2 + 
                        ucat[kp3][jp3][ip3].y * sk3);
                Uc.z = (ucat[kp1][jp1][ip1].z * sk1 + 
                        ucat[kp2][jp2][ip2].z * sk2 + 
                        ucat[kp3][jp3][ip3].z * sk3);

                int i1, j1, k1;

                double ajc;
                double csi0, csi1, csi2;
                double eta0, eta1, eta2;
                double zet0, zet1, zet2;
    
                i1=ip1; j1=jp1; k1=kp1;

                ajc = aj[k1][j1][i1];
                csi0 = csi[k1][j1][i1].x;
                csi1 = csi[k1][j1][i1].y;
                csi2 = csi[k1][j1][i1].z;
                eta0 = eta[k1][j1][i1].x;
                eta1 = eta[k1][j1][i1].y;
                eta2 = eta[k1][j1][i1].z;
                zet0 = zet[k1][j1][i1].x;
                zet1 = zet[k1][j1][i1].y;
                zet2 = zet[k1][j1][i1].z;

                double dpdc, dpde, dpdz;
                double dp_dx1, dp_dy1, dp_dz1;

                Compute_dscalar_center(i1, j1, k1, 
                                       mx, my, mz, 
                                       p, nvert, 
                                       &dpdc, &dpde, &dpdz );

                Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                     eta0, eta1, eta2, 
                                     zet0, zet1, zet2, ajc, 
                                     dpdc, dpde, dpdz, 
                                     &dp_dx1, &dp_dy1, &dp_dz1);

                i1=ip2; j1=jp2; k1=kp2;

                ajc = aj[k1][j1][i1];
                csi0 = csi[k1][j1][i1].x; 
                csi1 = csi[k1][j1][i1].y;
                csi2 = csi[k1][j1][i1].z;
                eta0 = eta[k1][j1][i1].x;
                eta1 = eta[k1][j1][i1].y;
                eta2 = eta[k1][j1][i1].z;
                zet0 = zet[k1][j1][i1].x;
                zet1 = zet[k1][j1][i1].y;
                zet2 = zet[k1][j1][i1].z;

                double dp_dx2, dp_dy2, dp_dz2;

                Compute_dscalar_center(i1, j1, k1, 
                                       mx, my, mz, 
                                       p, nvert, 
                                       &dpdc, &dpde, &dpdz );

                Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                     eta0, eta1, eta2, 
                                     zet0, zet1, zet2, ajc, 
                                     dpdc, dpde, dpdz, 
                                     &dp_dx2, &dp_dy2, &dp_dz2);

                i1=ip3; j1=jp3; k1=kp3;

                ajc = aj[k1][j1][i1];
                csi0 = csi[k1][j1][i1].x;
                csi1 = csi[k1][j1][i1].y;
                csi2 = csi[k1][j1][i1].z;
                eta0 = eta[k1][j1][i1].x;
                eta1 = eta[k1][j1][i1].y;
                eta2 = eta[k1][j1][i1].z;
                zet0 = zet[k1][j1][i1].x;
                zet1 = zet[k1][j1][i1].y;
                zet2 = zet[k1][j1][i1].z;

                double dp_dx3, dp_dy3, dp_dz3;

                Compute_dscalar_center(i1, j1, k1, 
                                       mx, my, mz, 
                                       p, nvert, 
                                       &dpdc, &dpde, &dpdz);

                Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                     eta0, eta1, eta2, 
                                     zet0, zet1, zet2, ajc, 
                                     dpdc, dpde, dpdz, 
                                     &dp_dx3, &dp_dy3, &dp_dz3);
 
                double dp_dx = (dp_dx1*sk1 + dp_dx2*sk2 + dp_dx3*sk3 );
                double dp_dy = (dp_dy1*sk1 + dp_dy2*sk2 + dp_dy3*sk3 );
                double dp_dz = (dp_dz1*sk1 + dp_dz2*sk2 + dp_dz3*sk3 );

                nu_t_b = lnu_t[k][j][i];
                nu_t_c = lnu_t[kp1][jp1][ip1]*sk1 + 
                         lnu_t[kp2][jp2][ip2]*sk2 + 
                         lnu_t[kp3][jp3][ip3]*sk3;

                Ub.x = ucat[k][j][i].x;
                Ub.y = ucat[k][j][i].y;
                Ub.z = ucat[k][j][i].z;

                double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
                double un = u_c * nx + v_c * ny + w_c * nz;
                double ut = u_c - un*nx, vt = v_c - un*ny, wt = w_c - un*nz;
                double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

                double t1x = ut / (ut_mag_c+1.e-11);
                double t1y = vt / (ut_mag_c+1.e-11);
                double t1z = wt / (ut_mag_c+1.e-11);
 
                double t2x = ny*t1z - nz*t1y;
                double t2y = nz*t1x - nx*t1z;
                double t2z = nx*t1y - ny*t1x;

                bctype = d_ib_wm; 
                
                double nu = 1./ren;
        
                double nut_b, nut_c, tau_w, nut_2sb;

                if (d_powerlawwallmodel) 
                    wallmodel_s(nu,sb,sc,
                                Uc,&ucat[k][j][i],Ua,
                                bctype,d_roughness_size,
                                nx,ny,nz,&tau_w,&ustar[k][j][i],
                                dp_dx,dp_dy,dp_dz,
                                &nut_2sb,nu_t_c);
                else 
                    wallmodel_0424(d_roughness_size, 
                                   &(ustar[k][j][i]), 
                                   dp_dx, dp_dy, dp_dz, 
                                   nu, sb, sc, &ucat[k][j][i], 
                                   Uc, Ua, nx, ny, nz, d_alfa_wm);



                tau_w = ustar[k][j][i]*ustar[k][j][i];
                tau[k][j][i]=tau_w;

                double nx_1=nx, ny_1=ny, nz_1=nz;
                double t1x_1=t1x, t1y_1=t1y, t1z_1=t1z;
                double t2x_1=t2x, t2y_1=t2y, t2z_1=t2z;


                double du_dx, du_dy, du_dz;
                double dv_dx, dv_dy, dv_dz;
                double dw_dx, dw_dy, dw_dz;

                double dut1dn, dut2dn, dundn; 
                double dut1dt1, dut2dt1, dundt1, dut1dt2, dut2dt2, dundt2;        
                double dut1dn_wm;
                double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;

                i1=i; j1=j; k1=k;        

                int is=i-1;
                int ie=i+1;

                for (i1=is;i1<ie;i1++){
                    if ( (nvert[k1][j1][i1]<0.1 && nvert[k1][j1][i1+1]>0.1) || 
                          (nvert[k1][j1][i1]>0.1 && nvert[k1][j1][i1+1]<0.1)) {
                        ajc = iaj[k1][j1][i1];
                        csi0 = icsi[k1][j1][i1].x;
                        csi1 = icsi[k1][j1][i1].y;
                        csi2 = icsi[k1][j1][i1].z;
                        eta0 = ieta[k1][j1][i1].x;
                        eta1 = ieta[k1][j1][i1].y;
                        eta2 = ieta[k1][j1][i1].z;
                        zet0 = izet[k1][j1][i1].x;
                        zet1 = izet[k1][j1][i1].y;
                        zet2 = izet[k1][j1][i1].z;

                        Compute1_du_i(i1, j1, k1, 
                                      mx, my, mz, 
                                      ucat, nvert, 
                                      &dudc, &dvdc, &dwdc, 
                                      &dude, &dvde, &dwde, 
                                      &dudz, &dvdz, &dwdz);

                        Compute_du_dxyz(csi0, csi1, csi2, 
                                        eta0, eta1, eta2, 
                                        zet0, zet1, zet2, ajc,
                                        dudc, dvdc, dwdc, 
                                        dude, dvde, dwde, 
                                        dudz, dvdz, dwdz, 
                                        &du_dx, &dv_dx, &dw_dx, 
                                        &du_dy, &dv_dy, &dw_dy, 
                                        &du_dz, &dv_dz, &dw_dz );

                        Comput_du_wmlocal(nx_1, ny_1, nz_1, 
                                          t1x_1, t1y_1, t1z_1, 
                                          t2x_1, t2y_1, t2z_1, 
                                          du_dx, dv_dx, dw_dx, 
                                          du_dy, dv_dy, dw_dy, 
                                          du_dz, dv_dz, dw_dz, 
                                          &dut1dn, &dut2dn, &dundn, 
                                          &dut1dt1, &dut2dt1, &dundt1, 
                                          &dut1dt2, &dut2dt2, &dundt2); 

                        if ( i1==0 || nvert[k1][j1][i1]>0.1 ) {
                            nu_t = lnu_t[k1][j1][i1+1];
                        }
                        else if( i1==mx-2 || nvert[k1][j1][i1+1]>0.1 ) {
                            nu_t = lnu_t[k1][j1][i1];

                        }
                        else nu_t = 0.5*(lnu_t[k1][j1][i1]+lnu_t[k1][j1][i1+1]);

                        dut1dn_wm = (tau_w);

                        if (!d_infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
                        if (d_infRe) dut1dn_wm = dut1dn_wm/(d_les_eps+nu_t);

                        dut1dn = dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;

                        Comput_JacobTensor_i(i1, j1, k1, 
                                             mx, my, mz, 
                                             coor, 
                                             &dxdc, &dxde, &dxdz, 
                                             &dydc, &dyde, &dydz, 
                                             &dzdc, &dzde, &dzdz);

                        Comput_du_Compgrid(dxdc, dxde, dxdz, 
                                           dydc, dyde, dydz, 
                                           dzdc, dzde, dzdz, 
                                           nx_1, ny_1, nz_1, 
                                           t1x_1, t1y_1, t1z_1, 
                                           t2x_1, t2y_1, t2z_1, 
                                           dut1dn, dut2dn, dundn, 
                                           dut1dt1, dut2dt1, dundt1, 
                                           dut1dt2, dut2dt2, dundt2, 
                                           &dudc_wm, &dvdc_wm, &dwdc_wm, 
                                           &dude_wm, &dvde_wm, &dwde_wm, 
                                           &dudz_wm, &dvdz_wm, &dwdz_wm);


                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

                        g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
                        g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
                        g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;
            
                        r11 = dudc * csi0 + dude * eta0 + dudz * zet0;//du_dx*J
                        r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;//dv_dx*J
                        r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;//dw_dx*J
                        r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
                        r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
                        r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

                        r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
                        r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
                        r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;


                        if (d_infRe) nu=0.0;
                        visc1_wm[k1][j1][i1].x = 
                             (g11*dudc + g21*dude + g31*dudz + 
                              r11*csi0 + r21*csi1 + r31*csi2)*ajc*(nu_t+nu);
                        visc1_wm[k1][j1][i1].y = 
                             (g11*dvdc + g21*dvde + g31*dvdz + 
                              r12*csi0 + r22*csi1 + r32*csi2)*ajc*(nu_t+nu);
                        visc1_wm[k1][j1][i1].z = 
                             (g11*dwdc + g21*dwde + g31*dwdz + 
                              r13*csi0 + r23*csi1 + r33*csi2)*ajc*(nu_t+nu);
        
                    }
                }
                

                i1=i, k1=k;
                j1=j;
                int js=j-1;
                int je=j+1;

                for (j1=js;j1<je;j1++) {
                    if ( (nvert[k1][j1][i1]<0.1 && nvert[k1][j1+1][i1]>0.1) || 
                         (nvert[k1][j1][i1]>0.1 && nvert[k1][j1+1][i1]<0.1)) {

                        ajc = jaj[k1][j1][i1];
                        csi0 = jcsi[k1][j1][i1].x;
                        csi1 = jcsi[k1][j1][i1].y; 
                        csi2 = jcsi[k1][j1][i1].z;
                        eta0 = jeta[k1][j1][i1].x;
                        eta1 = jeta[k1][j1][i1].y;
                        eta2 = jeta[k1][j1][i1].z;
                        zet0 = jzet[k1][j1][i1].x;
                        zet1 = jzet[k1][j1][i1].y;
                        zet2 = jzet[k1][j1][i1].z;

                        Compute1_du_j(i1, j1, k1, 
                                      mx, my, mz, 
                                      ucat, nvert, 
                                      &dudc, &dvdc, &dwdc, 
                                      &dude, &dvde, &dwde, 
                                      &dudz, &dvdz, &dwdz);

                        Compute_du_dxyz(csi0, csi1, csi2, 
                                        eta0, eta1, eta2, 
                                        zet0, zet1, zet2, ajc, 
                                        dudc, dvdc, dwdc, 
                                        dude, dvde, dwde, 
                                        dudz, dvdz, dwdz, 
                                        &du_dx, &dv_dx, &dw_dx, 
                                        &du_dy, &dv_dy, &dw_dy, 
                                        &du_dz, &dv_dz, &dw_dz );

                        Comput_du_wmlocal(nx_1, ny_1, nz_1, 
                                          t1x_1, t1y_1, t1z_1, 
                                          t2x_1, t2y_1, t2z_1, 
                                          du_dx, dv_dx, dw_dx, 
                                          du_dy, dv_dy, dw_dy, 
                                          du_dz, dv_dz, dw_dz, 
                                          &dut1dn, &dut2dn, &dundn, 
                                          &dut1dt1, &dut2dt1, &dundt1, 
                                          &dut1dt2, &dut2dt2, &dundt2);

                        if ( j1==0 || nvert[k1][j1][i1]>0.1 ) {
                            nu_t = lnu_t[k1][j1+1][i1];
                        }
                        else if( j1==my-2 || nvert[k1][j1+1][i1]>0.1 ) {
                            nu_t = lnu_t[k1][j1][i1];
                        }
                        else nu_t = 0.5*(lnu_t[k1][j1][i1]+lnu_t[k1][j1+1][i1]);


                        dut1dn_wm = (tau_w);

                        if (!d_infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
                        if (d_infRe) dut1dn_wm = dut1dn_wm/(d_les_eps+nu_t);

                        dut1dn = dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;

                        Comput_JacobTensor_j(i, j, k, 
                                             mx, my, mz, 
                                             coor, 
                                             &dxdc, &dxde, &dxdz, 
                                             &dydc, &dyde, &dydz, 
                                             &dzdc, &dzde, &dzdz);

                        Comput_du_Compgrid(dxdc, dxde, dxdz, 
                                           dydc, dyde, dydz, 
                                           dzdc, dzde, dzdz, 
                                           nx_1, ny_1, nz_1, 
                                           t1x_1, t1y_1, t1z_1, 
                                           t2x_1, t2y_1, t2z_1, 
                                           dut1dn, dut2dn, dundn, 
                                           dut1dt1, dut2dt1, dundt1, 
                                           dut1dt2, dut2dt2, dundt2, 
                                           &dudc_wm, &dvdc_wm, &dwdc_wm, 
                                           &dude_wm, &dvde_wm, &dwde_wm, 
                                           &dudz_wm, &dvdz_wm, &dwdz_wm);


                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

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

                        if (d_infRe) nu=0.0;
                        visc2_wm[k1][j1][i1].x = 
                            (g11*dudc + g21*dude + g31*dudz + 
                             r11*eta0 + r21*eta1 + r31*eta2)*ajc*(nu_t+nu);
                        visc2_wm[k1][j1][i1].y = 
                            (g11*dvdc + g21*dvde + g31*dvdz + 
                             r12*eta0 + r22*eta1 + r32*eta2)*ajc*(nu_t+nu);
                        visc2_wm[k1][j1][i1].z = 
                            (g11*dwdc + g21*dwde + g31*dwdz + 
                             r13*eta0 + r23*eta1 + r33*eta2)*ajc*(nu_t+nu);
                    }
                }

                i1=i, j1=j;
                k1=k;
                int ks=k-1;
                int ke=k+1;
                
                for (k1=ks;k1<ke;k1++ ) {
                    if ( (nvert[k1][j1][i1]<0.1 && nvert[k1+1][j1][i1]>0.1) || 
                         (nvert[k1][j1][i1]>0.1 && nvert[k1+1][j1][i1]<0.1)) {

                        ajc = kaj[k1][j1][i1];
                        csi0 = kcsi[k1][j1][i1].x;
                        csi1 = kcsi[k1][j1][i1].y;
                        csi2 = kcsi[k1][j1][i1].z;
                        eta0 = keta[k1][j1][i1].x;
                        eta1 = keta[k1][j1][i1].y;
                        eta2 = keta[k1][j1][i1].z;
                        zet0 = kzet[k1][j1][i1].x;
                        zet1 = kzet[k1][j1][i1].y;
                        zet2 = kzet[k1][j1][i1].z;

                        Compute1_du_k(i1, j1, k1, 
                                      mx, my, mz, 
                                      ucat, nvert, 
                                      &dudc, &dvdc, &dwdc, 
                                      &dude, &dvde, &dwde, 
                                      &dudz, &dvdz, &dwdz);

                        Compute_du_dxyz(csi0, csi1, csi2, 
                                        eta0, eta1, eta2, 
                                        zet0, zet1, zet2, ajc, 
                                        dudc, dvdc, dwdc, 
                                        dude, dvde, dwde, 
                                        dudz, dvdz, dwdz, 
                                        &du_dx, &dv_dx, &dw_dx, 
                                        &du_dy, &dv_dy, &dw_dy, 
                                        &du_dz, &dv_dz, &dw_dz );

                        Comput_du_wmlocal(nx_1, ny_1, nz_1, 
                                          t1x_1, t1y_1, t1z_1, 
                                          t2x_1, t2y_1, t2z_1, 
                                          du_dx, dv_dx, dw_dx, 
                                          du_dy, dv_dy, dw_dy, 
                                          du_dz, dv_dz, dw_dz, 
                                          &dut1dn, &dut2dn, &dundn, 
                                          &dut1dt1, &dut2dt1, &dundt1, 
                                          &dut1dt2, &dut2dt2, &dundt2);

                        if ( k1==0 || nvert[k1][j1][i1]>0.1 ) {
                            nu_t = lnu_t[k1+1][j1][i1];
                        }
                        else if( k1==mz-2 || nvert[k1+1][j1][i1]>0.1 )     {
                            nu_t = lnu_t[k1][j1][i1];
                        }
                        else nu_t = 0.5*(lnu_t[k1][j1][i1]+lnu_t[k1+1][j1][i1]);


                        dut1dn_wm = (tau_w);


                        if (!d_infRe) dut1dn_wm = dut1dn_wm/(nu+nu_t);
                        if (d_infRe) dut1dn_wm = dut1dn_wm/(d_les_eps+nu_t);
                        dut1dn = dut1dn_wm; // + (1.0-ratio_wm)*dut1dn;


                        Comput_JacobTensor_k(i, j, k, 
                                             mx, my, mz, 
                                             coor, 
                                             &dxdc, &dxde, &dxdz, 
                                             &dydc, &dyde, &dydz, 
                                             &dzdc, &dzde, &dzdz);

                        Comput_du_Compgrid(dxdc, dxde, dxdz, 
                                           dydc, dyde, dydz, 
                                           dzdc, dzde, dzdz, 
                                           nx_1, ny_1, nz_1, 
                                           t1x_1, t1y_1, t1z_1, 
                                           t2x_1, t2y_1, t2z_1, 
                                           dut1dn, dut2dn, dundn, 
                                           dut1dt1, dut2dt1, dundt1, 
                                           dut1dt2, dut2dt2, dundt2, 
                                           &dudc_wm, &dvdc_wm, &dwdc_wm, 
                                           &dude_wm, &dvde_wm, &dwde_wm, 
                                           &dudz_wm, &dvdz_wm, &dwdz_wm);

                        dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                        dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                        dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

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

                        if (d_infRe) nu=0.0;
            
                        visc3_wm[k1][j1][i1].x = 
                            (g11*dudc + g21*dude + g31*dudz + 
                             r11*zet0 + r21*zet1 + r31*zet2)*ajc*(nu+nu_t);
                        visc3_wm[k1][j1][i1].y = 
                            (g11*dvdc + g21*dvde + g31*dvdz + 
                             r12*zet0 + r22*zet1 + r32*zet2)*ajc*(nu+nu_t);
                        visc3_wm[k1][j1][i1].z = 
                            (g11*dwdc + g21*dwde + g31*dwdz + 
                             r13*zet0 + r23*zet1 + r33*zet2)*ajc*(nu+nu_t);
                    }
                }

            }
        }
    }

    double ni[3], nj[3], nk[3];
   
    // i direction
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {

                if(i==mx-1 || j==my-1 || k==mz-1) continue;
                if(j==0 || k==0) continue;

                if ( d_imin_wm != 0 && i == 0 ) {
                    area = sqrt( csi[k][j][i+1].x*csi[k][j][i+1].x + 
                                 csi[k][j][i+1].y*csi[k][j][i+1].y + 
                                 csi[k][j][i+1].z*csi[k][j][i+1].z );

                    st = sqrt(area);

                    Ua.x = Ua.y = Ua.z = 0;

                    sc = 2* 0.5/aj[k][j][i+1]/area + 0.5/aj[k][j][i+2]/area;
                    Uc = ucat[k][j][i+2];


                    sb = 0.5/aj[k][j][i+1]/area;
                    Ub = ucat[k][j][i+1];

                    bctype = d_imin_wm; //user->bctype[0];
                    Calculate_normal(csi[k][j][i+1], eta[k][j][i+1], 
                                     zet[k][j][i+1], ni, nj, nk);
                    nx =  ni[0], ny =  ni[1], nz =  ni[2];

                    nu_t_c = lnu_t[k][j][i+2];
                    nu_t_b = lnu_t[k][j][i+1];

                    int i1 = i+2, j1 = j, k1 = k;
                    double ajc = aj[k1][j1][i1];
                    double csi0 = csi[k1][j1][i1].x;
                    double csi1 = csi[k1][j1][i1].y;
                    double csi2 = csi[k1][j1][i1].z;
                    double eta0 = eta[k1][j1][i1].x;
                    double eta1 = eta[k1][j1][i1].y;
                    double eta2 = eta[k1][j1][i1].z;
                    double zet0 = zet[k1][j1][i1].x;
                    double zet1 = zet[k1][j1][i1].y;
                    double zet2 = zet[k1][j1][i1].z;

                    double dpdc, dpde, dpdz, dp_dx, dp_dy, dp_dz;

                    Compute_dscalar_center(i1, j1, k1, 
                                           mx, my, mz, 
                                           p, nvert, 
                                           &dpdc, &dpde, &dpdz );

                    Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                         eta0, eta1, eta2, 
                                         zet0, zet1, zet2, ajc, 
                                         dpdc, dpde, dpdz, 
                                         &dp_dx, &dp_dy, &dp_dz);


                    double u_c = Uc.x-Ua.x, v_c = Uc.y-Ua.y, w_c = Uc.z-Ua.z;
                    double un = u_c * nx + v_c * ny + w_c * nz;
                    double ut = u_c - un*nx, vt = v_c - un*ny, wt = w_c - un*nz;
                    double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

                    double t1x = ut / (ut_mag_c+1.e-11); 
                    double t1y = vt / (ut_mag_c+1.e-11);
                    double t1z = wt / (ut_mag_c+1.e-11);

                    double t2x = ny*t1z - nz*t1y;
                    double t2y = nz*t1x - nx*t1z;
                    double t2z = nx*t1y - ny*t1x;

                    i1 = i+2, j1 = j, k1 = k;                 

                    Compute_du_center(i1, j1, k1, 
                                      mx, my, mz, 
                                      ucat, nvert,
                                      i_periodic, ii_periodic, 
                                      j_periodic, jj_periodic,
                                      k_periodic, kk_periodic, 
                                      &dudc, &dvdc, &dwdc, 
                                      &dude, &dvde, &dwde, 
                                      &dudz, &dvdz, &dwdz);

                    double du_dx, du_dy, du_dz; 
                    double dv_dx, dv_dy, dv_dz;
                    double dw_dx, dw_dy, dw_dz;

                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx, 
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz );

                    double dut1dn, dut2dn, dundn; 
                    double dut1dt1, dut2dt1, dundt1;
                    double dut1dt2, dut2dt2, dundt2;

                    Comput_du_wmlocal(nx, ny, nz, 
                                      t1x, t1y, t1z, 
                                      t2x, t2y, t2z, 
                                      du_dx, dv_dx, dw_dx, 
                                      du_dy, dv_dy, dw_dy, 
                                      du_dz, dv_dz, dw_dz, 
                                      &dut1dn, &dut2dn, &dundn, 
                                      &dut1dt1, &dut2dt1, &dundt1, 
                                      &dut1dt2, &dut2dt2, &dundt2);

                    double dundn_out = dundn, dut1dn_out = dut1dn;
                    double nu = 1./d_data->getRe(), nu_t=0;
                    double nut_b, nut_c, tau_w, nut_2sb;


                    Cmpnts Utmp;

                    if (d_powerlawwallmodel) 
                        wallmodel_s(nu,sb,sc,
                                    Uc,&Ub,Ua,
                                    bctype,d_roughness_size,
                                    nx,ny,nz,
                                    &tau_w,&ustar[k][j][i+1],
                                    dp_dx,dp_dy,dp_dz,&nut_2sb,nu_t_c);
                    else wallmodel_0424(d_roughness_size, &(ustar[k][j][i+1]), 
                                        dp_dx, dp_dy, dp_dz, 
                                        nu, sb, sc, 
                                        &Ub, Uc, Ua, nx, ny, nz, d_alfa_wm);



                    ajc = iaj[k][j][i];
                    csi0 = icsi[k][j][i].x;
                    csi1 = icsi[k][j][i].y;
                    csi2 = icsi[k][j][i].z;
                    eta0 = ieta[k][j][i].x;
                    eta1 = ieta[k][j][i].y;
                    eta2 = ieta[k][j][i].z;
                    zet0 = izet[k][j][i].x;
                    zet1 = izet[k][j][i].y;
                    zet2 = izet[k][j][i].z;

                    Compute1_du_i(i, j, k,  
                                  mx, my, mz, 
                                  ucat, nvert, 
                                  &dudc, &dvdc, &dwdc, 
                                  &dude, &dvde, &dwde, 
                                  &dudz, &dvdz, &dwdz);

                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx, 
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz );

                    Comput_du_wmlocal(nx, ny, nz, 
                                      t1x, t1y, t1z, 
                                      t2x, t2y, t2z, 
                                      du_dx, dv_dx, dw_dx, 
                                      du_dy, dv_dy, dw_dy, 
                                      du_dz, dv_dz, dw_dz, 
                                      &dut1dn, &dut2dn, &dundn, 
                                      &dut1dt1, &dut2dt1, &dundt1, 
                                      &dut1dt2, &dut2dt2, &dundt2);

                    if ( i==0 || nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j][i+1];
                    else if( i==mx-2 || nvert[k][j][i+1]>0.1 ) 
                        nu_t = lnu_t[k][j][i];
                    else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j][i+1]);
    
                    if (!d_powerlawwallmodel) 
                         tau_w = ustar[k][j][i+1]*ustar[k][j][i+1];

                    if (!d_infRe) dut1dn = tau_w/(nu /*+nu_t*/);
                    if (d_infRe) dut1dn = tau_w/(d_les_eps+nu_t);

                    double dxdc, dxde, dxdz; 
                    double dydc, dyde, dydz;
                    double dzdc, dzde, dzdz;

                    Comput_JacobTensor_i(i, j, k, 
                                         mx, my, mz, 
                                         coor, 
                                         &dxdc, &dxde, &dxdz, 
                                         &dydc, &dyde, &dydz, 
                                         &dzdc, &dzde, &dzdz);

                    Comput_du_Compgrid(dxdc, dxde, dxdz, 
                                       dydc, dyde, dydz, 
                                       dzdc, dzde, dzdz, 
                                       nx, ny, nz, 
                                       t1x, t1y, t1z, 
                                       t2x, t2y, t2z, 
                                       dut1dn, dut2dn, dundn, 
                                       dut1dt1, dut2dt1, dundt1, 
                                       dut1dt2, dut2dt2, dundt2, 
                                       &dudc_wm, &dvdc_wm, &dwdc_wm, 
                                       &dude_wm, &dvde_wm, &dwde_wm, 
                                       &dudz_wm, &dvdz_wm, &dwdz_wm);

                    dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                    dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                    dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;
        
                    g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
                    g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
                    g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

                    r11 = dudc * csi0 + dude * eta0 + dudz * zet0;//du_dx*J
                    r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;//dv_dx*J
                    r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;//dw_dx*J

                    r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
                    r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
                    r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

                    r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
                    r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
                    r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;
        
              
                    if (d_infRe) nu=0.0;
                    nu_t = 0.0;
                    visc1_wm[k][j][i].x = 
                       (g11*dudc + g21*dude + g31*dudz + 
                        r11*csi0 + r21*csi1 + r31*csi2)*ajc*(nu_t+nu);
                    visc1_wm[k][j][i].y = 
                       (g11*dvdc + g21*dvde + g31*dvdz + 
                        r12*csi0 + r22*csi1 + r32*csi2)*ajc*(nu_t+nu);
                    visc1_wm[k][j][i].z = 
                       (g11*dwdc + g21*dwde + g31*dwdz + 
                        r13*csi0 + r23*csi1 + r33*csi2)*ajc*(nu_t+nu);

                }

                if ( d_imax_wm != 0 && i == mx-2 ) {
                    area = sqrt( csi[k][j][i].x*csi[k][j][i].x + 
                                 csi[k][j][i].y*csi[k][j][i].y + 
                                 csi[k][j][i].z*csi[k][j][i].z );

                    st = sqrt(area);
                    Ua.x = Ua.y = Ua.z = 0;
            
                    sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j][i-1]/area;
                    Uc = ucat[k][j][i-1];

                    sb = 0.5/aj[k][j][i]/area;
                    Ub = ucat[k][j][i];

                    bctype = d_imax_wm; //user->bctype[1];
                    Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], 
                                     ni, nj, nk);
                    nx =  -ni[0], ny =  -ni[1], nz =  -ni[2];

                    nu_t_c = lnu_t[k][j][i-1];
                    nu_t_b = lnu_t[k][j][i];

                    int i1 = i-1, j1 = j, k1 = k;
                    double ajc = aj[k1][j1][i1];
                    double csi0 = csi[k1][j1][i1].x;
                    double csi1 = csi[k1][j1][i1].y;
                    double csi2 = csi[k1][j1][i1].z;
                    double eta0 = eta[k1][j1][i1].x;
                    double eta1 = eta[k1][j1][i1].y;
                    double eta2 = eta[k1][j1][i1].z;
                    double zet0 = zet[k1][j1][i1].x;
                    double zet1 = zet[k1][j1][i1].y;
                    double zet2 = zet[k1][j1][i1].z;

                    double dpdc, dpde, dpdz, dp_dx, dp_dy, dp_dz;

                    Compute_dscalar_center(i1, j1, k1, 
                                           mx, my, mz, 
                                           p, nvert, 
                                           &dpdc, &dpde, &dpdz );

                    Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                         eta0, eta1, eta2, 
                                         zet0, zet1, zet2, ajc, 
                                         dpdc, dpde, dpdz, 
                                         &dp_dx, &dp_dy, &dp_dz);


                    double u_c = Uc.x-Ua.x, v_c = Uc.y-Ua.y, w_c = Uc.z-Ua.z;
                    double un = u_c * nx + v_c * ny + w_c * nz;
                    double ut = u_c - un*nx, vt = v_c - un*ny, wt = w_c - un*nz;
                    double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

                    double t1x = ut / (ut_mag_c+1.e-11);
                    double t1y = vt / (ut_mag_c+1.e-11);
                    double t1z = wt / (ut_mag_c+1.e-11);

                    double t2x = ny*t1z - nz*t1y;
                    double t2y = nz*t1x - nx*t1z;
                    double t2z = nx*t1y - ny*t1x;

                    i1 = i-1, j1 = j, k1 = k;

                    Compute_du_center(i1, j1, k1, 
                                      mx, my, mz, 
                                      ucat, nvert, 
                                      i_periodic, ii_periodic, 
                                      j_periodic, jj_periodic,
                                      k_periodic, kk_periodic, 
                                      &dudc, &dvdc, &dwdc, 
                                      &dude, &dvde, &dwde, 
                                      &dudz, &dvdz, &dwdz);

                    double du_dx, du_dy, du_dz;
                    double dv_dx, dv_dy, dv_dz;
                    double dw_dx, dw_dy, dw_dz;
                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx, 
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz );

                    double dut1dn, dut2dn, dundn;
                    double dut1dt1, dut2dt1, dundt1;
                    double dut1dt2, dut2dt2, dundt2;
                    Comput_du_wmlocal(nx, ny, nz, 
                                      t1x, t1y, t1z, 
                                      t2x, t2y, t2z, 
                                      du_dx, dv_dx, dw_dx, 
                                      du_dy, dv_dy, dw_dy, 
                                      du_dz, dv_dz, dw_dz, 
                                      &dut1dn, &dut2dn, &dundn, 
                                      &dut1dt1, &dut2dt1, &dundt1, 
                                      &dut1dt2, &dut2dt2, &dundt2);

                    double dundn_out = dundn, dut1dn_out = dut1dn;
                    double nu = 1./d_data->getRe(), nu_t=0;        
                    double nut_b, nut_c, tau_w, nut_2sb;

                    Cmpnts Utmp;
                    if (d_powerlawwallmodel) 
                        wallmodel_s(nu,sb,sc,
                                    Uc,&Ub,Ua,
                                    bctype,d_roughness_size,
                                    nx,ny,nz,
                                    &tau_w,&ustar[k][j][i],
                                    dp_dx,dp_dy,dp_dz,
                                    &nut_2sb,nu_t_c);
                    else wallmodel_0424(d_roughness_size, &(ustar[k][j][i]), 
                                        dp_dx, dp_dy, dp_dz, 
                                        nu, sb, sc, 
                                        &Ub, Uc, Ua, 
                                        nx, ny, nz, d_alfa_wm);


                    ajc = iaj[k][j][i];
                    csi0 = icsi[k][j][i].x;
                    csi1 = icsi[k][j][i].y;
                    csi2 = icsi[k][j][i].z;
                    eta0 = ieta[k][j][i].x;
                    eta1 = ieta[k][j][i].y;
                    eta2 = ieta[k][j][i].z;
                    zet0 = izet[k][j][i].x;
                    zet1 = izet[k][j][i].y;
                    zet2 = izet[k][j][i].z;

                    Compute1_du_i(i, j, k, 
                                  mx, my, mz, 
                                  ucat, nvert, 
                                  &dudc, &dvdc, &dwdc, 
                                  &dude, &dvde, &dwde, 
                                  &dudz, &dvdz, &dwdz);

                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx, 
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz );


                    Comput_du_wmlocal(nx, ny, nz, 
                                      t1x, t1y, t1z, 
                                      t2x, t2y, t2z, 
                                      du_dx, dv_dx, dw_dx, 
                                      du_dy, dv_dy, dw_dy,  
                                      du_dz, dv_dz, dw_dz, 
                                      &dut1dn, &dut2dn, &dundn, 
                                      &dut1dt1, &dut2dt1, &dundt1, 
                                      &dut1dt2, &dut2dt2, &dundt2);

                    if ( i==0 || nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j][i+1];
                    else if( i==mx-2 || nvert[k][j][i+1]>0.1 ) 
                        nu_t = lnu_t[k][j][i];
                    else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j][i+1]);

                    if (!d_powerlawwallmodel) 
                        tau_w = ustar[k][j][i]*ustar[k][j][i];

                    if (!d_infRe) dut1dn = tau_w/(nu /*+nu_t*/);
                    if (d_infRe) dut1dn = tau_w/(d_les_eps+nu_t);

                    double dxdc, dxde, dxdz, dydc, dyde, dydz, dzdc, dzde, dzdz;
                    Comput_JacobTensor_i(i, j, k, 
                                         mx, my, mz, 
                                         coor, 
                                         &dxdc, &dxde, &dxdz, 
                                         &dydc, &dyde, &dydz, 
                                         &dzdc, &dzde, &dzdz);

                    Comput_du_Compgrid(dxdc, dxde, dxdz, 
                                       dydc, dyde, dydz, 
                                       dzdc, dzde, dzdz, 
                                       nx, ny, nz, 
                                       t1x, t1y, t1z, 
                                       t2x, t2y, t2z, 
                                       dut1dn, dut2dn, dundn,  
                                       dut1dt1, dut2dt1, dundt1, 
                                       dut1dt2, dut2dt2, dundt2, 
                                       &dudc_wm, &dvdc_wm, &dwdc_wm, 
                                       &dude_wm, &dvde_wm, &dwde_wm, 
                                       &dudz_wm, &dvdz_wm, &dwdz_wm);

                    dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                    dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                    dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;
        
                     g11 = csi0 * csi0 + csi1 * csi1 + csi2 * csi2;
                     g21 = eta0 * csi0 + eta1 * csi1 + eta2 * csi2;
                     g31 = zet0 * csi0 + zet1 * csi1 + zet2 * csi2;

                     r11 = dudc * csi0 + dude * eta0 + dudz * zet0;//du_dx*J
                     r21 = dvdc * csi0 + dvde * eta0 + dvdz * zet0;//dv_dx*J
                     r31 = dwdc * csi0 + dwde * eta0 + dwdz * zet0;//dw_dx*J

                     r12 = dudc * csi1 + dude * eta1 + dudz * zet1;
                     r22 = dvdc * csi1 + dvde * eta1 + dvdz * zet1;
                     r32 = dwdc * csi1 + dwde * eta1 + dwdz * zet1;

                     r13 = dudc * csi2 + dude * eta2 + dudz * zet2;
                     r23 = dvdc * csi2 + dvde * eta2 + dvdz * zet2;
                     r33 = dwdc * csi2 + dwde * eta2 + dwdz * zet2;
        

                     if (d_infRe) nu=0.0;
                     nu_t=0.0;
                     visc1_wm[k][j][i].x = 
                         (g11*dudc + g21*dude + g31*dudz +
                          r11*csi0 + r21*csi1 + r31*csi2)*ajc*(nu_t+nu);
                     visc1_wm[k][j][i].y = 
                         (g11*dvdc + g21*dvde + g31*dvdz + 
                          r12*csi0 + r22*csi1 + r32*csi2)*ajc*(nu_t+nu);
                     visc1_wm[k][j][i].z = 
                         (g11*dwdc + g21*dwde + g31*dwdz + 
                          r13*csi0 + r23*csi1 + r33*csi2)*ajc*(nu_t+nu);


                }
            }
  
    // j direction
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                if (i==mx-1 || j==my-1 || k==mz-1) continue;
                if (i==0 || k==0) continue;

                if ( d_jmin_wm != 0 && j == 0 ) {
                    area = sqrt( eta[k][j+1][i].x*eta[k][j+1][i].x + 
                                 eta[k][j+1][i].y*eta[k][j+1][i].y + 
                                 eta[k][j+1][i].z*eta[k][j+1][i].z );

                    st = sqrt(area);
                    Ua.x = Ua.y = Ua.z = 0;

                    sc = 2* 0.5/aj[k][j+1][i]/area + 0.5/aj[k][j+2][i]/area;
                    Uc = ucat[k][j+2][i];

                    sd = 1.0/aj[k][j+1][i]/area + 
                         1.0/aj[k][j+2][i]/area + 0.5/aj[k][j+3][i]/area;
                    Ud = ucat[k][j+3][i];

                    sb = 0.5/aj[k][j+1][i]/area;
                    Ub = ucat[k][j+1][i];

                    bctype = d_jmin_wm; //user->bctype[2];
                    Calculate_normal(csi[k][j+1][i], eta[k][j+1][i], 
                                     zet[k][j+1][i], ni, nj, nk);
                    nx =  nj[0], ny =  nj[1], nz =  nj[2];

                    nu_t_c = lnu_t[k][j+2][i];
                    nu_t_b = lnu_t[k][j+1][i];
                    nu_t_d = lnu_t[k][j+3][i];

                    int i1 = i, j1 = j+1, k1 = k;
                    double ajc = aj[k1][j1][i1];
                    double csi0 = csi[k1][j1][i1].x;
                    double csi1 = csi[k1][j1][i1].y;
                    double csi2 = csi[k1][j1][i1].z;
                    double eta0 = eta[k1][j1][i1].x;
                    double eta1 = eta[k1][j1][i1].y;
                    double eta2 = eta[k1][j1][i1].z;
                    double zet0 = zet[k1][j1][i1].x;
                    double zet1 = zet[k1][j1][i1].y;
                    double zet2 = zet[k1][j1][i1].z;

                    double dpdc, dpde, dpdz, dp_dx=0.0, dp_dy=0.0, dp_dz=0.0;

                    Compute_dscalar_center(i1, j1, k1, 
                                           mx, my, mz, 
                                           p, nvert, 
                                           &dpdc, &dpde, &dpdz );

                    Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                         eta0, eta1, eta2, 
                                         zet0, zet1, zet2, ajc, 
                                         dpdc, dpde, dpdz, 
                                         &dp_dx, &dp_dy, &dp_dz);
                    double dpdx_b=dp_dx, dpdy_b=dp_dy, dpdz_b=dp_dz;

                    i1 = i; j1 = j+2; k1 = k;
                    ajc = aj[k1][j1][i1];
                    csi0 = csi[k1][j1][i1].x; 
                    csi1 = csi[k1][j1][i1].y; 
                    csi2 = csi[k1][j1][i1].z;
                    eta0 = eta[k1][j1][i1].x; 
                    eta1 = eta[k1][j1][i1].y; 
                    eta2 = eta[k1][j1][i1].z;
                    zet0 = zet[k1][j1][i1].x; 
                    zet1 = zet[k1][j1][i1].y; 
                    zet2 = zet[k1][j1][i1].z;

                    dpdc=0.0; dpde=0.0; dpdz=0.0; 
                    dp_dx=0.0; dp_dy=0.0; dp_dz=0.0;

                    Compute_dscalar_center(i1, j1, k1, 
                                           mx, my, mz, 
                                           p, nvert,
                                           &dpdc, &dpde, &dpdz );

                    Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                         eta0, eta1, eta2, 
                                         zet0, zet1, zet2, ajc, 
                                         dpdc, dpde, dpdz, 
                                         &dp_dx, &dp_dy, &dp_dz);
                    double dpdx_c=dp_dx, dpdy_c=dp_dy, dpdz_c=dp_dz;

                    double u_c = Uc.x-Ua.x, v_c = Uc.y-Ua.y, w_c = Uc.z-Ua.z;
                    double un = u_c * nx + v_c * ny + w_c * nz;
                    double ut = u_c - un*nx, vt = v_c - un*ny, wt = w_c - un*nz;
                    double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

                    double t1x = ut / (ut_mag_c+1.e-11);
                    double t1y = vt / (ut_mag_c+1.e-11);
                    double t1z = wt / (ut_mag_c+1.e-11);

                    double t2x = ny*t1z - nz*t1y;
                    double t2y = nz*t1x - nx*t1z;
                    double t2z = nx*t1y - ny*t1x;

                    i1 = i, j1 = j+2, k1 = k;

                    Compute_du_center(i1, j1, k1, 
                                      mx, my, mz, 
                                      ucat, nvert, 
                                      i_periodic, ii_periodic, 
                                      j_periodic, jj_periodic,
                                      k_periodic, kk_periodic, 
                                      &dudc, &dvdc, &dwdc,
                                      &dude, &dvde, &dwde,
                                      &dudz, &dvdz, &dwdz);

                    double du_dx, du_dy, du_dz;
                    double dv_dx, dv_dy, dv_dz;
                    double dw_dx, dw_dy, dw_dz;

                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx, 
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz );

                    double dut1dn, dut2dn, dundn; 
                    double dut1dt1, dut2dt1, dundt1;
                    double dut1dt2, dut2dt2, dundt2;
                        
                    Comput_du_wmlocal(nx, ny, nz, 
                                      t1x, t1y, t1z,  
                                      t2x, t2y, t2z, 
                                      du_dx, dv_dx, dw_dx, 
                                      du_dy, dv_dy, dw_dy, 
                                      du_dz, dv_dz, dw_dz, 
                                      &dut1dn, &dut2dn, &dundn, 
                                      &dut1dt1, &dut2dt1, &dundt1, 
                                      &dut1dt2, &dut2dt2, &dundt2);

                    double dundn_out = dundn, dut1dn_out = dut1dn;

            
                    double nu = 1./d_data->getRe(), nu_t = 0;
                
                    double nut_b, nut_c, tau_w, nut_2sb;

                    Cmpnts Utmp;
                    if (d_powerlawwallmodel) 
                        wallmodel_s(nu,sb,sc,
                                    Uc,&Ub,Ua,
                                    bctype,d_roughness_size,
                                    nx,ny,nz,
                                    &tau_w,&ustar[k][j+1][i],
                                    dp_dx,dp_dy,dp_dz,&nut_2sb,nu_t_c);
                    else wallmodel_0424(d_roughness_size, &(ustar[k][j+1][i]),
                                        dp_dx, dp_dy, dp_dz, 
                                        nu, sb, sc, 
                                        &Ub, Uc, Ua, nx, ny, nz, d_alfa_wm);


                    ajc = jaj[k][j][i];
                    csi0 = jcsi[k][j][i].x;
                    csi1 = jcsi[k][j][i].y;
                    csi2 = jcsi[k][j][i].z;
                    eta0 = jeta[k][j][i].x;
                    eta1 = jeta[k][j][i].y;
                    eta2 = jeta[k][j][i].z;
                    zet0 = jzet[k][j][i].x;
                    zet1 = jzet[k][j][i].y;
                    zet2 = jzet[k][j][i].z;

                    Compute1_du_j(i, j, k, 
                                  mx, my, mz, 
                                  ucat, nvert, 
                                  &dudc, &dvdc, &dwdc, 
                                  &dude, &dvde, &dwde, 
                                  &dudz, &dvdz, &dwdz);


                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx,
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz );

                    Comput_du_wmlocal(nx, ny, nz, 
                                      t1x, t1y, t1z, 
                                      t2x, t2y, t2z, 
                                      du_dx, dv_dx, dw_dx, 
                                      du_dy, dv_dy, dw_dy, 
                                      du_dz, dv_dz, dw_dz, 
                                      &dut1dn, &dut2dn, &dundn, 
                                      &dut1dt1, &dut2dt1, &dundt1,
                                      &dut1dt2, &dut2dt2, &dundt2);

                    if ( j==0 || nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j+1][i];
                    else if( j==my-2 || nvert[k][j+1][i]>0.1 ) 
                        nu_t = lnu_t[k][j][i];
                    else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j+1][i]);

                    if (!d_powerlawwallmodel) 
                         tau_w = ustar[k][j+1][i]*ustar[k][j+1][i];

                    if (!d_infRe) dut1dn = tau_w/(nu /*+nu_t*/);
                    if (d_infRe) dut1dn = tau_w/(d_les_eps+nu_t);
                    double dxdc, dxde, dxdz;
                    double dydc, dyde, dydz;
                    double dzdc, dzde, dzdz;

                    Comput_JacobTensor_j(i, j, k, 
                                         mx, my, mz, 
                                         coor,
                                         &dxdc, &dxde, &dxdz, 
                                         &dydc, &dyde, &dydz,
                                         &dzdc, &dzde, &dzdz);

                    Comput_du_Compgrid(dxdc, dxde, dxdz, 
                                       dydc, dyde, dydz, 
                                       dzdc, dzde, dzdz, 
                                       nx, ny, nz, 
                                       t1x, t1y, t1z, 
                                       t2x, t2y, t2z, 
                                       dut1dn, dut2dn, dundn, 
                                       dut1dt1, dut2dt1, dundt1, 
                                       dut1dt2, dut2dt2, dundt2, 
                                       &dudc_wm, &dvdc_wm, &dwdc_wm, 
                                       &dude_wm, &dvde_wm, &dwde_wm, 
                                       &dudz_wm, &dvdz_wm, &dwdz_wm);



                    dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                    dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                    dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;

    
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

                    if (d_infRe) nu = 0.0;
                    nu_t = 0.0;
                    visc2_wm[k][j][i].x = 
                       (g11*dudc + g21*dude + g31*dudz + 
                        r11*eta0 + r21*eta1 + r31*eta2)*ajc*(nu_t+nu);
                    visc2_wm[k][j][i].y = 
                       (g11*dvdc + g21*dvde + g31*dvdz + 
                        r12*eta0 + r22*eta1 + r32*eta2)*ajc*(nu_t+nu);
                    visc2_wm[k][j][i].z = 
                       (g11*dwdc + g21*dwde + g31*dwdz + 
                        r13*eta0 + r23*eta1 + r33*eta2)*ajc*(nu_t+nu);

    
                }

                if ( d_jmax_wm != 0 && j == my-2 ) {

                    area = sqrt( eta[k][j][i].x*eta[k][j][i].x + 
                                 eta[k][j][i].y*eta[k][j][i].y + 
                                 eta[k][j][i].z*eta[k][j][i].z );


                    st = sqrt(area);
                    Ua.x = Ua.y = Ua.z = 0;

                    sc = 2* 0.5/aj[k][j][i]/area + 0.5/aj[k][j-1][i]/area;
                    Uc = ucat[k][j-1][i];
        

                    sb = 0.5/aj[k][j][i]/area;
                    Ub = ucat[k][j][i];

                    bctype = d_jmax_wm; //user->bctype[3];
                    Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i],
                                     ni, nj, nk);
                    nx =  -nj[0], ny =  -nj[1], nz =  -nj[2];

                    nu_t_c = lnu_t[k][j-1][i];
                    nu_t_b = lnu_t[k][j][i];

                    int i1 = i, j1 = j-1, k1 = k;
                    double ajc = aj[k1][j1][i1];
                    double csi0 = csi[k1][j1][i1].x;
                    double csi1 = csi[k1][j1][i1].y;
                    double csi2 = csi[k1][j1][i1].z;
                    double eta0 = eta[k1][j1][i1].x;
                    double eta1 = eta[k1][j1][i1].y;
                    double eta2 = eta[k1][j1][i1].z;
                    double zet0 = zet[k1][j1][i1].x;
                    double zet1 = zet[k1][j1][i1].y;
                    double zet2 = zet[k1][j1][i1].z;

                    double dpdc, dpde, dpdz, dp_dx=0.0, dp_dy=0.0, dp_dz=0.0;

                    Compute_dscalar_center(i1, j1, k1, 
                                           mx, my, mz, 
                                           p, nvert, 
                                           &dpdc, &dpde, &dpdz );

                    Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                         eta0, eta1, eta2, 
                                         zet0, zet1, zet2, ajc, 
                                         dpdc, dpde, dpdz, 
                                         &dp_dx, &dp_dy, &dp_dz);

                   double dpdx_b=dp_dx, dpdy_b=dp_dy, dpdz_b=dp_dz;


                   i1 = i, j1 = j-2, k1 = k;
                   ajc = aj[k1][j1][i1];
                   csi0 = csi[k1][j1][i1].x;
                   csi1 = csi[k1][j1][i1].y;
                   csi2 = csi[k1][j1][i1].z;
                   eta0 = eta[k1][j1][i1].x;
                   eta1 = eta[k1][j1][i1].y;
                   eta2 = eta[k1][j1][i1].z;
                   zet0 = zet[k1][j1][i1].x;
                   zet1 = zet[k1][j1][i1].y;
                   zet2 = zet[k1][j1][i1].z;

                   Compute_dscalar_center(i1, j1, k1, 
                                          mx, my, mz, 
                                          p, nvert, 
                                          &dpdc, &dpde, &dpdz );

                   Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                        eta0, eta1, eta2, 
                                        zet0, zet1, zet2, ajc, 
                                        dpdc, dpde, dpdz, 
                                        &dp_dx, &dp_dy, &dp_dz);

                    double dpdx_c=dp_dx, dpdy_c=dp_dy, dpdz_c=dp_dz;

                    double u_c = Uc.x-Ua.x, v_c = Uc.y-Ua.y, w_c = Uc.z-Ua.z;
                    double un = u_c * nx + v_c * ny + w_c * nz;
                    double ut = u_c - un*nx, vt = v_c - un*ny, wt = w_c - un*nz;
                    double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

                    double t1x = ut / (ut_mag_c+1.e-11);
                    double t1y = vt / (ut_mag_c+1.e-11);
                    double t1z = wt / (ut_mag_c+1.e-11);

                    double t2x = ny*t1z - nz*t1y;
                    double t2y = nz*t1x - nx*t1z;
                    double t2z = nx*t1y - ny*t1x;

                    i1 = i, j1 = j-1, k1 = k;

                    Compute_du_center(i1, j1, k1, 
                                      mx, my, mz, 
                                      ucat, nvert, 
                                      i_periodic, ii_periodic, 
                                      j_periodic, jj_periodic,
                                      k_periodic, kk_periodic, 
                                      &dudc, &dvdc, &dwdc, 
                                      &dude, &dvde, &dwde, 
                                      &dudz, &dvdz, &dwdz);

                    double du_dx, du_dy, du_dz;
                    double dv_dx, dv_dy, dv_dz;
                    double dw_dx, dw_dy, dw_dz;

                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx, 
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz );

                    double dut1dn, dut2dn, dundn;
                    double dut1dt1, dut2dt1, dundt1;
                    double dut1dt2, dut2dt2, dundt2;

                    Comput_du_wmlocal(nx, ny, nz, 
                                      t1x, t1y, t1z, 
                                      t2x, t2y, t2z, 
                                      du_dx, dv_dx, dw_dx, 
                                      du_dy, dv_dy, dw_dy, 
                                      du_dz, dv_dz, dw_dz, 
                                      &dut1dn, &dut2dn, &dundn, 
                                      &dut1dt1, &dut2dt1, &dundt1, 
                                      &dut1dt2, &dut2dt2, &dundt2);

                    double dundn_out = dundn, dut1dn_out = dut1dn;


                    double nu = 1./d_data->getRe(), nu_t = 0;        

                    double nut_b, nut_c, tau_w, nut_2sb;

                    Cmpnts Utmp;

                    if (d_powerlawwallmodel) 
                        wallmodel_s(nu,sb,sc,
                                    Uc,&Ub,Ua,
                                    bctype,d_roughness_size,
                                    nx,ny,nz,
                                    &tau_w,&ustar[k][j][i],
                                    dp_dx,dp_dy,dp_dz,
                                    &nut_2sb,nu_t_c);
                    else wallmodel_0424(d_roughness_size, &(ustar[k][j][i]), 
                                        dp_dx, dp_dy, dp_dz, 
                                        nu, sb, sc, 
                                        &Ub, Uc, Ua, nx, ny, nz, d_alfa_wm);


                    ajc = jaj[k][j][i];
                    csi0 = jcsi[k][j][i].x;
                    csi1 = jcsi[k][j][i].y;
                    csi2 = jcsi[k][j][i].z;
                    eta0 = jeta[k][j][i].x; 
                    eta1 = jeta[k][j][i].y;
                    eta2 = jeta[k][j][i].z;
                    zet0 = jzet[k][j][i].x;
                    zet1 = jzet[k][j][i].y;
                    zet2 = jzet[k][j][i].z;

                    Compute1_du_j(i, j, k, 
                                  mx, my, mz, 
                                  ucat, nvert, 
                                  &dudc, &dvdc, &dwdc, 
                                  &dude, &dvde, &dwde, 
                                  &dudz, &dvdz, &dwdz);

                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx, 
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz );

                    Comput_du_wmlocal(nx, ny, nz, 
                                      t1x, t1y, t1z, 
                                      t2x, t2y, t2z, 
                                      du_dx, dv_dx, dw_dx, 
                                      du_dy, dv_dy, dw_dy, 
                                      du_dz, dv_dz, dw_dz, 
                                      &dut1dn, &dut2dn, &dundn, 
                                      &dut1dt1, &dut2dt1, &dundt1, 
                                      &dut1dt2, &dut2dt2, &dundt2);

                    if ( j==0 || nvert[k][j][i]>0.1 ) nu_t = lnu_t[k][j+1][i];
                    else if( j==my-2 || nvert[k][j+1][i]>0.1 ) 
                        nu_t = lnu_t[k][j][i];
                    else nu_t = 0.5 * (lnu_t[k][j][i] + lnu_t[k][j+1][i]);

                    if (!d_powerlawwallmodel) 
                        tau_w = ustar[k][j][i]*ustar[k][j][i];
                    if (!d_infRe) dut1dn = tau_w/(nu /*+nu_t*/);

                    if (d_infRe) dut1dn = tau_w/(d_les_eps+nu_t);

                    double dxdc, dxde, dxdz;
                    double dydc, dyde, dydz;
                    double dzdc, dzde, dzdz;

                    Comput_JacobTensor_j(i, j, k,
                                         mx, my, mz, 
                                         coor, 
                                         &dxdc, &dxde, &dxdz, 
                                         &dydc, &dyde, &dydz, 
                                         &dzdc, &dzde, &dzdz);

                    Comput_du_Compgrid(dxdc, dxde, dxdz, 
                                       dydc, dyde, dydz, 
                                       dzdc, dzde, dzdz, 
                                       nx, ny, nz, 
                                       t1x, t1y, t1z, 
                                       t2x, t2y, t2z, 
                                       dut1dn, dut2dn, dundn, 
                                       dut1dt1, dut2dt1, dundt1, 
                                       dut1dt2, dut2dt2, dundt2, 
                                       &dudc_wm, &dvdc_wm, &dwdc_wm,
                                       &dude_wm, &dvde_wm, &dwde_wm,
                                       &dudz_wm, &dvdz_wm, &dwdz_wm);

                    dudc = dudc_wm, dvdc = dvdc_wm, dwdc = dwdc_wm;
                    dude = dude_wm, dvde = dvde_wm, dwde = dwde_wm;
                    dudz = dudz_wm, dvdz = dvdz_wm, dwdz = dwdz_wm;
    
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

                    if (d_infRe) nu=0.0;
                    nu_t=0.0;
                    visc2_wm[k][j][i].x = 
                        (g11*dudc + g21*dude + g31*dudz + 
                         r11*eta0 + r21*eta1 + r31*eta2)*ajc*(nu_t+nu);
                    visc2_wm[k][j][i].y = 
                        (g11*dvdc + g21*dvde + g31*dvdz + 
                         r12*eta0 + r22*eta1 + r32*eta2)*ajc*(nu_t+nu);
                    visc2_wm[k][j][i].z = 
                        (g11*dwdc + g21*dwde + g31*dwdz + 
                         r13*eta0 + r23*eta1 + r33*eta2)*ajc*(nu_t+nu);


                }

            }


    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {

                if (d_jmin_wm && d_grid->getBC(0)==1 && j==0 && i==1) {
                    Set(&visc2_wm[k][j][i],0);
                }

                if (d_jmin_wm && d_grid->getBC(1)==1 && j==0 && i==mx-2) {
                    Set(&visc2_wm[k][j][i],0);
                }

                if (d_jmax_wm && d_grid->getBC(0)==1 && j==my-2 && i==1) {
                    Set(&visc2_wm[k][j][i],0);
                }

                if (d_jmax_wm && d_grid->getBC(1)==1 && j==my-2 && i==mx-2) {
                    Set(&visc2_wm[k][j][i],0);
                }

                if (d_imin_wm && d_grid->getBC(2)==1 && i==0 && j==1) {
                    Set(&visc1_wm[k][j][i],0);
                }

                if (d_imin_wm && d_grid->getBC(3)==1 && i==0 && j==my-2) {
                    Set(&visc1_wm[k][j][i],0);
                }

                if (d_imax_wm && d_grid->getBC(2)==1 && i==mx-2 && j==1) {
                    Set(&visc1_wm[k][j][i],0);
                }

                if (d_imax_wm && d_grid->getBC(3)==1 && i==mx-2 && j==my-2) {
                    Set(&visc1_wm[k][j][i],0);
                }

           }

    PetscTime(&te);
        
    PetscPrintf(PETSC_COMM_WORLD, "Time: wall model  %le\n", te-ts);
    DMDAVecRestoreArray(fda, d_lVisc1_wm, &visc1_wm);
    DMDAVecRestoreArray(fda, d_lVisc2_wm, &visc2_wm);
    DMDAVecRestoreArray(fda, d_lVisc3_wm, &visc3_wm);
    DMDAVecRestoreArray(da, lNu_t, &lnu_t);

    PetscTime(&ts);
    DMDALocalToLocalBegin(fda, d_lVisc1_wm, INSERT_VALUES, d_lVisc1_wm);
    DMDALocalToLocalEnd(fda, d_lVisc1_wm, INSERT_VALUES, d_lVisc1_wm);

    DMDALocalToLocalBegin(fda, d_lVisc2_wm, INSERT_VALUES, d_lVisc2_wm);
    DMDALocalToLocalEnd(fda, d_lVisc2_wm, INSERT_VALUES, d_lVisc2_wm);

    DMDALocalToLocalBegin(fda, d_lVisc3_wm, INSERT_VALUES, d_lVisc3_wm);
    DMDALocalToLocalEnd(fda, d_lVisc3_wm, INSERT_VALUES, d_lVisc3_wm);

    DMDALocalToLocalBegin(da, lNu_t, INSERT_VALUES, lNu_t);
    DMDALocalToLocalEnd(da, lNu_t, INSERT_VALUES, lNu_t);

    PetscTime(&te);

    PetscPrintf(PETSC_COMM_WORLD, "Time for local to local  %le\n", te-ts);


    DMDAVecGetArray(fda, d_lVisc1_wm, &visc1_wm);
    DMDAVecGetArray(fda, d_lVisc2_wm, &visc2_wm);
    DMDAVecGetArray(fda, d_lVisc3_wm, &visc3_wm);
    DMDAVecGetArray(da, lNu_t, &lnu_t);



    if (d_grid->isPeriodic())
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) {
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
                        visc1_wm[k][j][i] = visc1_wm[c][b][a];
                        visc2_wm[k][j][i] = visc2_wm[c][b][a];
                        visc3_wm[k][j][i] = visc3_wm[c][b][a];
                        lnu_t[k][j][i] = lnu_t[c][b][a];
                    }
                }

    DMDAVecRestoreArray(fda, d_lVisc1_wm, &visc1_wm);
    DMDAVecRestoreArray(fda, d_lVisc2_wm, &visc2_wm);
    DMDAVecRestoreArray(fda, d_lVisc3_wm, &visc3_wm);
    DMDAVecRestoreArray(da, lNu_t, &lnu_t);

    DMDAVecRestoreArray(fda, Coor, &coor);

    DMDAVecRestoreArray(fda, lUcat,  &ucat);
    DMDAVecRestoreArray(da, lP, &p);

    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);

    DMDAVecRestoreArray(fda, ICsi, &icsi);
    DMDAVecRestoreArray(fda, IEta, &ieta);
    DMDAVecRestoreArray(fda, IZet, &izet);

    DMDAVecRestoreArray(fda, JCsi, &jcsi);
    DMDAVecRestoreArray(fda, JEta, &jeta);
    DMDAVecRestoreArray(fda, JZet, &jzet);

    DMDAVecRestoreArray(fda, KCsi, &kcsi);
    DMDAVecRestoreArray(fda, KEta, &keta);
    DMDAVecRestoreArray(fda, KZet, &kzet);

    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(da, Aj, &aj);
    DMDAVecRestoreArray(da, lUstar, &ustar);
    DMDAVecRestoreArray(da, d_lTau, &tau);
    
    DMDAVecRestoreArray(da, IAj, &iaj);
    DMDAVecRestoreArray(da, JAj, &jaj);
    DMDAVecRestoreArray(da, KAj, &kaj);
    
}

void WallModel::Solve(Vec Rhs, double coeff)
{
    if (!d_use_wall) return;

    int i, j, k;

    Cmpnts ***ucat;
    Cmpnts ***csi, ***eta, ***zet;
    Cmpnts ***icsi, ***ieta, ***izet;
    Cmpnts ***jcsi, ***jeta, ***jzet;
    Cmpnts ***kcsi, ***keta, ***kzet;
    PetscReal ***p;
    PetscReal ***nvert, ***rho, ***mu;

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();
    DMDALocalInfo info;
    int xs, xe, ys, ye, zs, ze; // Local grid information
    int mx, my, mz; // Dimensions in three directions
    int lxs, lxe, lys, lye, lzs, lze;
    DMDAGetLocalInfo(da, &info);
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
    
    Cmpnts ***visc1, ***visc2, ***visc3, ***fp;
    Cmpnts ***rhs;
    PetscReal ***aj, ***iaj, ***jaj, ***kaj;//, ***vol;
    Cmpnts ***visc1_wm, ***visc2_wm, ***visc3_wm, ***fp_wm;

    PetscReal dudc, dude, dudz, dvdc, dvde, dvdz, dwdc, dwde, dwdz;
    PetscReal csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2;
    PetscReal g11, g21, g31;
    PetscReal r11, r21, r31, r12, r22, r32, r13, r23, r33;

    PetscReal dudc_wm, dvdc_wm, dwdc_wm, dude_wm, dvde_wm;
    PetscReal dwde_wm, dudz_wm, dvdz_wm, dwdz_wm;
    PetscReal r11_wm, r21_wm, r31_wm, r12_wm, r22_wm, r32_wm;
    PetscReal r13_wm, r23_wm, r33_wm;

    Vec Fp_wm;
    Vec Coor;
    Cmpnts ***coor;

    DMGetCoordinates(da, &Coor);
    DMDAVecGetArray(fda, Coor, &coor);

    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();
    Vec IAj = d_grid->getlIAj();
    Vec JAj = d_grid->getlJAj();
    Vec KAj = d_grid->getlKAj();

    Vec lNvert = d_data->getlNvert();

    DMDAVecGetArray(da, lNvert, &nvert);

    DMDAVecGetArray(fda, Rhs,  &rhs);

    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);

    
    
    DMDAVecGetArray(fda, d_lVisc1, &visc1);
    DMDAVecGetArray(fda, d_lVisc2, &visc2);
    DMDAVecGetArray(fda, d_lVisc3, &visc3);
    
    DMDAVecGetArray(da, Aj, &aj);
      
    DMDAVecGetArray(da, IAj, &iaj);
    DMDAVecGetArray(da, JAj, &jaj);
    DMDAVecGetArray(da, KAj, &kaj);

    VecDuplicate(Csi, &Fp_wm);

    VecSet(Fp_wm, 0.0);

    DMDAVecGetArray(fda, d_lVisc1_wm, &visc1_wm);
    DMDAVecGetArray(fda, d_lVisc2_wm, &visc2_wm);
    DMDAVecGetArray(fda, d_lVisc3_wm, &visc3_wm);

    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {

                if (fabs(visc1_wm[k][j][i].x)<1.0e-9) 
                    visc1_wm[k][j][i].x = visc1[k][j][i].x;
                if (fabs(visc1_wm[k][j][i].y)<1.0e-9) 
                    visc1_wm[k][j][i].y = visc1[k][j][i].y;
                if (fabs(visc1_wm[k][j][i].z)<1.0e-9) 
                    visc1_wm[k][j][i].z = visc1[k][j][i].z;

                if (fabs(visc2_wm[k][j][i].x)<1.0e-9) 
                    visc2_wm[k][j][i].x = visc2[k][j][i].x;
                if (fabs(visc2_wm[k][j][i].y)<1.0e-9) 
                    visc2_wm[k][j][i].y = visc2[k][j][i].y;
                if (fabs(visc2_wm[k][j][i].z)<1.0e-9) 
                    visc2_wm[k][j][i].z = visc2[k][j][i].z;

                if (fabs(visc3_wm[k][j][i].x)<1.0e-9) 
                    visc3_wm[k][j][i].x = visc3[k][j][i].x;
                if (fabs(visc3_wm[k][j][i].y)<1.0e-9) 
                    visc3_wm[k][j][i].y = visc3[k][j][i].y;
                if (fabs(visc3_wm[k][j][i].z)<1.0e-9) 
                    visc3_wm[k][j][i].z = visc3[k][j][i].z;

            }

    
    DMDAVecGetArray(fda, d_Fp, &fp);
    DMDAVecGetArray(fda, Fp_wm, &fp_wm);
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
         
                double r=1.0;
            
                fp[k][j][i].x = (visc1[k][j][i].x - visc1[k][j][i-1].x + 
                                 visc2[k][j][i].x - visc2[k][j-1][i].x + 
                                 visc3[k][j][i].x - visc3[k-1][j][i].x) / r;
                fp[k][j][i].y = (visc1[k][j][i].y - visc1[k][j][i-1].y + 
                                 visc2[k][j][i].y - visc2[k][j-1][i].y + 
                                 visc3[k][j][i].y - visc3[k-1][j][i].y) / r;
                fp[k][j][i].z = (visc1[k][j][i].z - visc1[k][j][i-1].z + 
                                 visc2[k][j][i].z - visc2[k][j-1][i].z + 
                                 visc3[k][j][i].z - visc3[k-1][j][i].z) / r;


                fp_wm[k][j][i].x=(visc1_wm[k][j][i].x-visc1_wm[k][j][i-1].x +
                                  visc2_wm[k][j][i].x-visc2_wm[k][j-1][i].x +
                                  visc3_wm[k][j][i].x-visc3_wm[k-1][j][i].x)/r;
                fp_wm[k][j][i].y=(visc1_wm[k][j][i].y-visc1_wm[k][j][i-1].y + 
                                  visc2_wm[k][j][i].y-visc2_wm[k][j-1][i].y +
                                  visc3_wm[k][j][i].y-visc3_wm[k-1][j][i].y)/r;
                fp_wm[k][j][i].z=(visc1_wm[k][j][i].z-visc1_wm[k][j][i-1].z + 
                                  visc2_wm[k][j][i].z-visc2_wm[k][j-1][i].z + 
                                  visc3_wm[k][j][i].z-visc3_wm[k-1][j][i].z)/r;
            }
    
    
    DMDAVecRestoreArray(fda, d_Fp, &fp);
    
    DMDALocalToLocalBegin(fda, d_Fp, INSERT_VALUES, d_Fp);
    DMDALocalToLocalEnd(fda, d_Fp, INSERT_VALUES, d_Fp);
    
    DMDAVecRestoreArray(fda, Fp_wm, &fp_wm);

    DMDALocalToLocalBegin(fda, Fp_wm, INSERT_VALUES, Fp_wm);
    DMDALocalToLocalEnd(fda, Fp_wm, INSERT_VALUES, Fp_wm);


    DMDAVecGetArray(fda, d_Fp, &fp);
    DMDAVecGetArray(fda, Fp_wm, &fp_wm);
   
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
        
                   if (flag) fp[k][j][i] = fp[c][b][a];
                   if(flag) fp_wm[k][j][i] = fp_wm[c][b][a];

               }

    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
        
                rhs[k][j][i].x = 
                    -coeff * ( 0.5 * ( csi[k][j][i].x * fp[k][j][i].x + 
                                       csi[k][j][i].y * fp[k][j][i].y + 
                                       csi[k][j][i].z * fp[k][j][i].z) + 
                               0.5 * ( csi[k][j][i+1].x * fp[k][j][i+1].x + 
                                       csi[k][j][i+1].y * fp[k][j][i+1].y + 
                                       csi[k][j][i+1].z * fp[k][j][i+1].z) ) * 
                             iaj[k][j][i];
                rhs[k][j][i].y = 
                    -coeff * ( 0.5 * ( eta[k][j][i].x * fp[k][j][i].x + 
                                       eta[k][j][i].y * fp[k][j][i].y + 
                                       eta[k][j][i].z * fp[k][j][i].z) + 
                               0.5 * ( eta[k][j+1][i].x * fp[k][j+1][i].x + 
                                       eta[k][j+1][i].y * fp[k][j+1][i].y + 
                                       eta[k][j+1][i].z * fp[k][j+1][i].z) ) *
                               jaj[k][j][i];
                rhs[k][j][i].z = 
                    -coeff * ( 0.5 * ( zet[k][j][i].x * fp[k][j][i].x + 
                                       zet[k][j][i].y * fp[k][j][i].y +
                                       zet[k][j][i].z * fp[k][j][i].z) +
                               0.5 * ( zet[k+1][j][i].x * fp[k+1][j][i].x +
                                       zet[k+1][j][i].y * fp[k+1][j][i].y +
                                       zet[k+1][j][i].z * fp[k+1][j][i].z) ) *
                               kaj[k][j][i];
        

                rhs[k][j][i].x += 
                     coeff * ( 0.5 * ( csi[k][j][i].x * fp_wm[k][j][i].x + 
                                       csi[k][j][i].y * fp_wm[k][j][i].y + 
                                       csi[k][j][i].z * fp_wm[k][j][i].z) +
                               0.5 * ( csi[k][j][i+1].x * fp_wm[k][j][i+1].x +
                                       csi[k][j][i+1].y * fp_wm[k][j][i+1].y +
                                       csi[k][j][i+1].z * fp_wm[k][j][i+1].z))*
                               iaj[k][j][i];
                rhs[k][j][i].y += 
                     coeff * ( 0.5 * ( eta[k][j][i].x * fp_wm[k][j][i].x + 
                                       eta[k][j][i].y * fp_wm[k][j][i].y + 
                                       eta[k][j][i].z * fp_wm[k][j][i].z) +
                               0.5 * ( eta[k][j+1][i].x * fp_wm[k][j+1][i].x +
                                       eta[k][j+1][i].y * fp_wm[k][j+1][i].y + 
                                       eta[k][j+1][i].z * fp_wm[k][j+1][i].z))*
                               jaj[k][j][i];
                rhs[k][j][i].z += 
                     coeff * ( 0.5 * ( zet[k][j][i].x * fp_wm[k][j][i].x + 
                                       zet[k][j][i].y * fp_wm[k][j][i].y + 
                                       zet[k][j][i].z * fp_wm[k][j][i].z) + 
                               0.5 * ( zet[k+1][j][i].x * fp_wm[k+1][j][i].x + 
                                       zet[k+1][j][i].y * fp_wm[k+1][j][i].y +
                                       zet[k+1][j][i].z * fp_wm[k+1][j][i].z))*
                               kaj[k][j][i];

        
                if (nvert[k][j][i]+nvert[k][j][i+1]>0.1 || 
                    (!i_periodic && !ii_periodic && i==mx-2) ) {

                   rhs[k][j][i].x = 0;
                }
                if (nvert[k][j][i]+nvert[k][j+1][i]>0.1 || 
                    (!j_periodic && !jj_periodic && j==my-2) ) {
                   rhs[k][j][i].y = 0;
                }
                if (nvert[k][j][i]+nvert[k+1][j][i]>0.1 || 
                    (!k_periodic && !kk_periodic && k==mz-2) ) {
                   rhs[k][j][i].z = 0;
                }

            }

    
    DMDAVecRestoreArray(da, IAj, &iaj);
    DMDAVecRestoreArray(da, JAj, &jaj);
    DMDAVecRestoreArray(da, KAj, &kaj);

    if (xs ==0) {
        i = 0;
        for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++) {
            rhs[k][j][i].x = 0;
            rhs[k][j][i].y = 0;
            rhs[k][j][i].z = 0;

        }
    }

    if (xe == mx) {
        for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++) {
            if(!i_periodic && !ii_periodic) {
                i = mx-2;
                rhs[k][j][i].x = 0;
            }
            i = mx-1;
            rhs[k][j][i].x = 0;
            rhs[k][j][i].y = 0;
            rhs[k][j][i].z = 0;



        }
    }


    if (ys == 0) {
        for (k=zs; k<ze; k++)
        for (i=xs; i<xe; i++) {
            j=0;
            rhs[k][j][i].x = 0;
            rhs[k][j][i].y = 0;
            rhs[k][j][i].z = 0;
        }
    }
  
    if (ye == my) {
        for (k=zs; k<ze; k++) 
        for (i=xs; i<xe; i++) {
            if(!j_periodic && !jj_periodic) {
                j=my-2;
                rhs[k][j][i].y = 0;
            }
            j=my-1;
            rhs[k][j][i].x = 0;
            rhs[k][j][i].y = 0;
            rhs[k][j][i].z = 0;
        }
    }
    
    
    if (zs == 0) {
        k=0;
        for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++) {
            rhs[k][j][i].x = 0;
            rhs[k][j][i].y = 0;
            rhs[k][j][i].z = 0;


        }
    }
  
    if (ze == mz) {
        for (j=ys; j<ye; j++)
        for (i=xs; i<xe; i++) {
            if(!k_periodic && !kk_periodic) {
                k=mz-2;
                rhs[k][j][i].z = 0;
            }
            k=mz-1;
            rhs[k][j][i].x = 0;
            rhs[k][j][i].y = 0;
            rhs[k][j][i].z = 0;


        }
    }

    DMDAVecRestoreArray(fda, Rhs,  &rhs);

    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    
    DMDAVecRestoreArray(fda, d_Fp, &fp);
    
    DMDAVecRestoreArray(fda, d_lVisc1, &visc1);
    DMDAVecRestoreArray(fda, d_lVisc2, &visc2);
    DMDAVecRestoreArray(fda, d_lVisc3, &visc3);
    
    DMDAVecRestoreArray(da, Aj, &aj);
  
    DMDAVecRestoreArray(da, lNvert, &nvert);
    
    DMDAVecRestoreArray(fda, d_lVisc1_wm, &visc1_wm);
    DMDAVecRestoreArray(fda, d_lVisc2_wm, &visc2_wm);
    DMDAVecRestoreArray(fda, d_lVisc3_wm, &visc3_wm);
    DMDAVecRestoreArray(fda, Fp_wm, &fp_wm);

    DMDAVecRestoreArray(fda, Coor, &coor);

    VecDestroy(&Fp_wm);

}


void WallModel::Compute1_du_i(int i, int j, int k, 
                   int mx, int my, int mz, 
                   Cmpnts ***ucat, PetscReal ***nvert, 
                   double *dudc, double *dvdc, double *dwdc, 
                   double *dude, double *dvde, double *dwde,
                   double *dudz, double *dvdz, double *dwdz)
{
    
    if ((nvert[k][j][i])> 1.1 || (nvert[k][j][i+1])> 1.1) {
        *dudc = 0.0;
        *dvdc = 0.0;
        *dwdc = 0.0;

        *dude = 0.0;
        *dvde = 0.0;
        *dwde = 0.0;

        *dudz = 0.0;
        *dvdz = 0.0;
        *dwdz = 0.0;

    } else {
        *dudc = ucat[k][j][i+1].x - ucat[k][j][i].x;
        *dvdc = ucat[k][j][i+1].y - ucat[k][j][i].y;
        *dwdc = ucat[k][j][i+1].z - ucat[k][j][i].z;

        double dude1, dude2, dvde1, dvde2, dwde1, dwde2;

        if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])< 1.1) {
            dude1 =  ucat[k][j][i].x - ucat[k][j-1][i].x;
            dvde1 =  ucat[k][j][i].y - ucat[k][j-1][i].y;
            dwde1 =  ucat[k][j][i].z - ucat[k][j-1][i].z;
        } else if ((nvert[k][j+1][i])< 1.1 && (nvert[k][j-1][i])> 1.1)     {
            dude1 =  ucat[k][j+1][i].x - ucat[k][j][i].x;
            dvde1 =  ucat[k][j+1][i].y - ucat[k][j][i].y;
            dwde1 =  ucat[k][j+1][i].z - ucat[k][j][i].z;
        } else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
            dude1 =  0.0;
            dvde1 =  0.0;
            dwde1 =  0.0;
        }else {
            dude1 =  0.5*(ucat[k][j+1][i].x - ucat[k][j-1][i].x);
            dvde1 =  0.5*(ucat[k][j+1][i].y - ucat[k][j-1][i].y);
            dwde1 =  0.5*(ucat[k][j+1][i].z - ucat[k][j-1][i].z);
        }

        if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])< 1.1) {
            dude2 =  ucat[k][j][i+1].x - ucat[k][j-1][i+1].x;
            dvde2 =  ucat[k][j][i+1].y - ucat[k][j-1][i+1].y;
            dwde2 =  ucat[k][j][i+1].z - ucat[k][j-1][i+1].z;
        } else if ((nvert[k][j+1][i+1])< 1.1 && (nvert[k][j-1][i+1])> 1.1)  {
            dude2 =  ucat[k][j+1][i+1].x - ucat[k][j][i+1].x;
            dvde2 =  ucat[k][j+1][i+1].y - ucat[k][j][i+1].y;
            dwde2 =  ucat[k][j+1][i+1].z - ucat[k][j][i+1].z;
        } else if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])> 1.1) {
            dude2 =  0.0;
            dvde2 =  0.0;
            dwde2 =  0.0;
        }else {
            dude2 =  0.5*(ucat[k][j+1][i+1].x - ucat[k][j-1][i+1].x);
            dvde2 =  0.5*(ucat[k][j+1][i+1].y - ucat[k][j-1][i+1].y);
            dwde2 =  0.5*(ucat[k][j+1][i+1].z - ucat[k][j-1][i+1].z);
        }

        
        if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j-1][i+1])> 1.1) {
            *dude = dude1;
            *dvde = dvde1;
            *dwde = dwde1;
        } else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
            *dude = dude2;
            *dvde = dvde2;
            *dwde = dwde2;
        } else {
            *dude = 0.5*(dude1+dude2);
            *dvde = 0.5*(dvde1+dvde2);
            *dwde = 0.5*(dwde1+dwde2);
        }


        double dudz1, dudz2, dvdz1, dvdz2, dwdz1, dwdz2;

        if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])< 1.1) {
            dudz1 = ucat[k][j][i].x - ucat[k-1][j][i].x;
            dvdz1 = ucat[k][j][i].y - ucat[k-1][j][i].y;
            dwdz1 = ucat[k][j][i].z - ucat[k-1][j][i].z;
        } else if ((nvert[k+1][j][i])< 1.1 && (nvert[k-1][j][i])> 1.1)     {
            dudz1 =  ucat[k+1][j][i].x - ucat[k][j][i].x;
            dvdz1 =  ucat[k+1][j][i].y - ucat[k][j][i].y;
            dwdz1 =  ucat[k+1][j][i].z - ucat[k][j][i].z;
        } else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
            dudz1 =  0.0;
            dvdz1 =  0.0;
            dwdz1 =  0.0;
        } else {
            dudz1 =  0.5*(ucat[k+1][j][i].x - ucat[k-1][j][i].x);
            dvdz1 =  0.5*(ucat[k+1][j][i].y - ucat[k-1][j][i].y);
            dwdz1 =  0.5*(ucat[k+1][j][i].z - ucat[k-1][j][i].z);
        } 

        if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])< 1.1) {
            dudz2 =  ucat[k][j][i+1].x - ucat[k-1][j][i+1].x;
            dvdz2 =  ucat[k][j][i+1].y - ucat[k-1][j][i+1].y;
            dwdz2 =  ucat[k][j][i+1].z - ucat[k-1][j][i+1].z;
        } else if ((nvert[k+1][j][i+1])< 1.1 && (nvert[k-1][j][i+1])> 1.1)  {
            dudz2 =  ucat[k+1][j][i+1].x - ucat[k][j][i+1].x;
            dvdz2 =  ucat[k+1][j][i+1].y - ucat[k][j][i+1].y;
            dwdz2 =  ucat[k+1][j][i+1].z - ucat[k][j][i+1].z;
        } else if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])> 1.1) {
            dudz2 =  0.0;
            dvdz2 =  0.0;
            dwdz2 =  0.0;
        }else {
            dudz2 =  0.5*(ucat[k+1][j][i+1].x - ucat[k-1][j][i+1].x);
            dvdz2 =  0.5*(ucat[k+1][j][i+1].y - ucat[k-1][j][i+1].y);
            dwdz2 =  0.5*(ucat[k+1][j][i+1].z - ucat[k-1][j][i+1].z);
        }
        
        if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k-1][j][i+1])> 1.1) {
           *dudz = dudz1;
           *dvdz = dvdz1;
           *dwdz = dwdz1;
        } else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
           *dudz = dudz2;
           *dvdz = dvdz2;
           *dwdz = dwdz2;
        } else {
           *dudz = 0.5*(dudz1+dudz2);
           *dvdz = 0.5*(dvdz1+dvdz2);
           *dwdz = 0.5*(dwdz1+dwdz2);
        } 

    }

}


void WallModel::Compute1_du_j(int i, int j, int k, 
                   int mx, int my, int mz, 
                   Cmpnts ***ucat, PetscReal ***nvert, 
                   double *dudc, double *dvdc, double *dwdc, 
                   double *dude, double *dvde, double *dwde,
                   double *dudz, double *dvdz, double *dwdz)
{
    
    if ((nvert[k][j][i])> 1.1 || (nvert[k][j+1][i])> 1.1) {
        *dudc = 0.0;
        *dvdc = 0.0;
        *dwdc = 0.0;

        *dude = 0.0;
        *dvde = 0.0;
        *dwde = 0.0;

        *dudz = 0.0;
        *dvdz = 0.0;
        *dwdz = 0.0;

    } else {

        double dudc1, dudc2, dvdc1, dvdc2, dwdc1, dwdc2;

        if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])< 1.1) {
            dudc1 =  ucat[k][j][i].x - ucat[k][j][i-1].x;
            dvdc1 =  ucat[k][j][i].y - ucat[k][j][i-1].y;
            dwdc1 =  ucat[k][j][i].z - ucat[k][j][i-1].z;
        } else if ((nvert[k][j][i+1])< 1.1 && (nvert[k][j][i-1])> 1.1)     {
            dudc1 =  ucat[k][j][i+1].x - ucat[k][j][i].x;
            dvdc1 =  ucat[k][j][i+1].y - ucat[k][j][i].y;
            dwdc1 =  ucat[k][j][i+1].z - ucat[k][j][i].z;
        } else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
            dudc1 =  0.0;
            dvdc1 =  0.0;
            dwdc1 =  0.0;
        }else {
            dudc1 =  0.5*(ucat[k][j][i+1].x - ucat[k][j][i-1].x);
            dvdc1 =  0.5*(ucat[k][j][i+1].y - ucat[k][j][i-1].y);
            dwdc1 =  0.5*(ucat[k][j][i+1].z - ucat[k][j][i-1].z);
        }

        if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])< 1.1) {
            dudc2 =  ucat[k][j+1][i].x - ucat[k][j+1][i-1].x;
            dvdc2 =  ucat[k][j+1][i].y - ucat[k][j+1][i-1].y;
            dwdc2 =  ucat[k][j+1][i].z - ucat[k][j+1][i-1].z;
        } else if ((nvert[k][j+1][i+1])< 1.1 && (nvert[k][j+1][i-1])> 1.1)  {
            dudc2 =  ucat[k][j+1][i+1].x - ucat[k][j+1][i].x;
            dvdc2 =  ucat[k][j+1][i+1].y - ucat[k][j+1][i].y;
            dwdc2 =  ucat[k][j+1][i+1].z - ucat[k][j+1][i].z;
        } else if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])> 1.1) {
            dudc2 =  0.0;
            dvdc2 =  0.0;
            dwdc2 =  0.0;
        }else {
            dudc2 =  0.5*(ucat[k][j+1][i+1].x - ucat[k][j+1][i-1].x);
            dvdc2 =  0.5*(ucat[k][j+1][i+1].y - ucat[k][j+1][i-1].y);
            dwdc2 =  0.5*(ucat[k][j+1][i+1].z - ucat[k][j+1][i-1].z);
        }

        if ((nvert[k][j+1][i+1])> 1.1 && (nvert[k][j+1][i-1])> 1.1) {
            *dudc = dudc1;
            *dvdc = dvdc1;
            *dwdc = dwdc1;
        } else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
            *dudc = dudc2;
            *dvdc = dvdc2;
            *dwdc = dwdc2;
        } else {
            *dudc = 0.5*(dudc1+dudc2);
            *dvdc = 0.5*(dvdc1+dvdc2);
            *dwdc = 0.5*(dwdc1+dwdc2);
        }



        *dude = ucat[k][j+1][i].x - ucat[k][j][i].x;
        *dvde = ucat[k][j+1][i].y - ucat[k][j][i].y;
        *dwde = ucat[k][j+1][i].z - ucat[k][j][i].z;


        double dudz1, dudz2, dvdz1, dvdz2, dwdz1, dwdz2;

        if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])< 1.1) {
            dudz1 =  ucat[k][j][i].x - ucat[k-1][j][i].x;
            dvdz1 =  ucat[k][j][i].y - ucat[k-1][j][i].y;
            dwdz1 =  ucat[k][j][i].z - ucat[k-1][j][i].z;
        } else if ((nvert[k+1][j][i])< 1.1 && (nvert[k-1][j][i])> 1.1)     {
            dudz1 =  ucat[k+1][j][i].x - ucat[k][j][i].x;
            dvdz1 =  ucat[k+1][j][i].y - ucat[k][j][i].y;
            dwdz1 =  ucat[k+1][j][i].z - ucat[k][j][i].z;
        } else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
            dudz1 =  0.0;
            dvdz1 =  0.0;
            dwdz1 =  0.0;
        }else {
            dudz1 =  0.5*(ucat[k+1][j][i].x - ucat[k-1][j][i].x);
            dvdz1 =  0.5*(ucat[k+1][j][i].y - ucat[k-1][j][i].y);
            dwdz1 =  0.5*(ucat[k+1][j][i].z - ucat[k-1][j][i].z);
        }

        if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])< 1.1) {
            dudz2 =  ucat[k][j+1][i].x - ucat[k-1][j+1][i].x;
            dvdz2 =  ucat[k][j+1][i].y - ucat[k-1][j+1][i].y;
            dwdz2 =  ucat[k][j+1][i].z - ucat[k-1][j+1][i].z;
        } else if ((nvert[k+1][j+1][i])< 1.1 && (nvert[k-1][j+1][i])> 1.1)  {
            dudz2 =  ucat[k+1][j+1][i].x - ucat[k][j+1][i].x;
            dvdz2 =  ucat[k+1][j+1][i].y - ucat[k][j+1][i].y;
            dwdz2 =  ucat[k+1][j+1][i].z - ucat[k][j+1][i].z;
        } else if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])> 1.1) {
            dudz2 =  0.0;
            dvdz2 =  0.0;
            dwdz2 =  0.0;
        }else {
            dudz2 =  0.5*(ucat[k+1][j+1][i].x - ucat[k-1][j+1][i].x);
            dvdz2 =  0.5*(ucat[k+1][j+1][i].y - ucat[k-1][j+1][i].y);
            dwdz2 =  0.5*(ucat[k+1][j+1][i].z - ucat[k-1][j+1][i].z);
        }
        
        if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k-1][j+1][i])> 1.1) {
            *dudz = dudz1;
            *dvdz = dvdz1;
            *dwdz = dwdz1;
        } else if ((nvert[k+1][j][i])> 1.1 && (nvert[k-1][j][i])> 1.1) {
            *dudz = dudz2;
            *dvdz = dvdz2;
            *dwdz = dwdz2;
        } else {
            *dudz = 0.5*(dudz1+dudz2);
            *dvdz = 0.5*(dvdz1+dvdz2);
            *dwdz = 0.5*(dwdz1+dwdz2);
        }
    }

}


void WallModel::Compute1_du_k(int i, int j, int k, 
                   int mx, int my, int mz, 
                   Cmpnts ***ucat, PetscReal ***nvert, 
                   double *dudc, double *dvdc, double *dwdc, 
                   double *dude, double *dvde, double *dwde,
                   double *dudz, double *dvdz, double *dwdz)
{
    
    if ((nvert[k][j][i])> 1.1 || (nvert[k+1][j][i])> 1.1) {
        *dudc = 0.0;
        *dvdc = 0.0;
        *dwdc = 0.0;

        *dude = 0.0;
        *dvde = 0.0;
        *dwde = 0.0;

        *dudz = 0.0;
        *dvdz = 0.0;
        *dwdz = 0.0;

    } else {

        double dudc1, dudc2, dvdc1, dvdc2, dwdc1, dwdc2;

        if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])< 1.1) {
            dudc1 =  ucat[k][j][i].x - ucat[k][j][i-1].x;
            dvdc1 =  ucat[k][j][i].y - ucat[k][j][i-1].y;
            dwdc1 =  ucat[k][j][i].z - ucat[k][j][i-1].z;
        } else if ((nvert[k][j][i+1])< 1.1 && (nvert[k][j][i-1])> 1.1)     {
            dudc1 =  ucat[k][j][i+1].x - ucat[k][j][i].x;
            dvdc1 =  ucat[k][j][i+1].y - ucat[k][j][i].y;
            dwdc1 =  ucat[k][j][i+1].z - ucat[k][j][i].z;
        } else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
            dudc1 =  0.0;
            dvdc1 =  0.0;
            dwdc1 =  0.0;
        }else {
            dudc1 =  0.5*(ucat[k][j][i+1].x - ucat[k][j][i-1].x);
            dvdc1 =  0.5*(ucat[k][j][i+1].y - ucat[k][j][i-1].y);
            dwdc1 =  0.5*(ucat[k][j][i+1].z - ucat[k][j][i-1].z);
        }

        if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])< 1.1) {
            dudc2 =  ucat[k+1][j][i].x - ucat[k+1][j][i-1].x;
            dvdc2 =  ucat[k+1][j][i].y - ucat[k+1][j][i-1].y;
            dwdc2 =  ucat[k+1][j][i].z - ucat[k+1][j][i-1].z;
        } else if ((nvert[k+1][j][i+1])< 1.1 && (nvert[k+1][j][i-1])> 1.1)  {
            dudc2 =  ucat[k+1][j][i+1].x - ucat[k+1][j][i].x;
            dvdc2 =  ucat[k+1][j][i+1].y - ucat[k+1][j][i].y;
            dwdc2 =  ucat[k+1][j][i+1].z - ucat[k+1][j][i].z;
        } else if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])> 1.1) {
            dudc2 =  0.0;
            dvdc2 =  0.0;
            dwdc2 =  0.0;
        }else {
            dudc2 =  0.5*(ucat[k+1][j][i+1].x - ucat[k+1][j][i-1].x);
            dvdc2 =  0.5*(ucat[k+1][j][i+1].y - ucat[k+1][j][i-1].y);
            dwdc2 =  0.5*(ucat[k+1][j][i+1].z - ucat[k+1][j][i-1].z);
        }

        if ((nvert[k+1][j][i+1])> 1.1 && (nvert[k+1][j][i-1])> 1.1) {
            *dudc = dudc1;
            *dvdc = dvdc1;
            *dwdc = dwdc1;
        } else if ((nvert[k][j][i+1])> 1.1 && (nvert[k][j][i-1])> 1.1) {
            *dudc = dudc2;
            *dvdc = dvdc2;
            *dwdc = dwdc2;
        } else {
            *dudc = 0.5*(dudc1+dudc2);
            *dvdc = 0.5*(dvdc1+dvdc2);
            *dwdc = 0.5*(dwdc1+dwdc2);
        } 


        double dude1, dude2, dvde1, dvde2, dwde1, dwde2;

        if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])< 1.1) {
            dude1 =  ucat[k][j][i].x - ucat[k][j-1][i].x;
            dvde1 =  ucat[k][j][i].y - ucat[k][j-1][i].y;
            dwde1 =  ucat[k][j][i].z - ucat[k][j-1][i].z;
        } else if ((nvert[k][j+1][i])< 1.1 && (nvert[k][j-1][i])> 1.1)     {
            dude1 =  ucat[k][j+1][i].x - ucat[k][j][i].x;
            dvde1 =  ucat[k][j+1][i].y - ucat[k][j][i].y;
            dwde1 =  ucat[k][j+1][i].z - ucat[k][j][i].z;
        } else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
            dude1 =  0.0;
            dvde1 =  0.0;
            dwde1 =  0.0;
        }else {
            dude1 =  0.5*(ucat[k][j+1][i].x - ucat[k][j-1][i].x);
            dvde1 =  0.5*(ucat[k][j+1][i].y - ucat[k][j-1][i].y);
            dwde1 =  0.5*(ucat[k][j+1][i].z - ucat[k][j-1][i].z);
        }

        if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])< 1.1) {
            dude2 =  ucat[k+1][j][i].x - ucat[k+1][j-1][i].x;
            dvde2 =  ucat[k+1][j][i].y - ucat[k+1][j-1][i].y;
            dwde2 =  ucat[k+1][j][i].z - ucat[k+1][j-1][i].z;
        } else if ((nvert[k+1][j+1][i])< 1.1 && (nvert[k+1][j-1][i])> 1.1)  {
            dude2 =  ucat[k+1][j+1][i].x - ucat[k+1][j][i].x;
            dvde2 =  ucat[k+1][j+1][i].y - ucat[k+1][j][i].y;
            dwde2 =  ucat[k+1][j+1][i].z - ucat[k+1][j][i].z;
        } else if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])> 1.1) {
            dude2 =  0.0;
            dvde2 =  0.0;
            dwde2 =  0.0;
        }else {
            dude2 =  0.5*(ucat[k+1][j+1][i].x - ucat[k+1][j-1][i].x);
            dvde2 =  0.5*(ucat[k+1][j+1][i].y - ucat[k+1][j-1][i].y);
            dwde2 =  0.5*(ucat[k+1][j+1][i].z - ucat[k+1][j-1][i].z);
        }

        
        if ((nvert[k+1][j+1][i])> 1.1 && (nvert[k+1][j-1][i])> 1.1) {
            *dude = dude1;
            *dvde = dvde1;
            *dwde = dwde1;
        } else if ((nvert[k][j+1][i])> 1.1 && (nvert[k][j-1][i])> 1.1) {
            *dude = dude2;
            *dvde = dvde2;
            *dwde = dwde2;
        } else {
            *dude = 0.5*(dude1+dude2);
            *dvde = 0.5*(dvde1+dvde2);
            *dwde = 0.5*(dwde1+dwde2);
        }


        *dudz = ucat[k+1][j][i].x - ucat[k][j][i].x;
        *dvdz = ucat[k+1][j][i].y - ucat[k][j][i].y;
        *dwdz = ucat[k+1][j][i].z - ucat[k][j][i].z;

    }

}

void WallModel::Comput_du_wmlocal(
     double nx, double ny, double nz, 
     double t1x, double t1y, double t1z, 
     double t2x, double t2y, double t2z, 
     double du_dx,double dv_dx,double dw_dx,
     double du_dy,double dv_dy,double dw_dy,
     double du_dz,double dv_dz,double dw_dz, 
     double *dut1dn, double *dut2dn, double *dundn, 
     double *dut1dt1, double *dut2dt1, double *dundt1,
     double *dut1dt2, double *dut2dt2, double *dundt2) {

    double dudn = du_dx*nx+du_dy*ny+du_dz*nz;    
    double dvdn = dv_dx*nx+dv_dy*ny+dv_dz*nz;    
    double dwdn = dw_dx*nx+dw_dy*ny+dw_dz*nz;    

    double dudt1 = du_dx*t1x+du_dy*t1y+du_dz*t1z;    
    double dvdt1 = dv_dx*t1x+dv_dy*t1y+dv_dz*t1z;    
    double dwdt1 = dw_dx*t1x+dw_dy*t1y+dw_dz*t1z;    

    double dudt2 = du_dx*t2x+du_dy*t2y+du_dz*t2z;
    double dvdt2 = dv_dx*t2x+dv_dy*t2y+dv_dz*t2z;
    double dwdt2 = dw_dx*t2x+dw_dy*t2y+dw_dz*t2z;

    *dut1dn=dudn*t1x+dvdn*t1y+dwdn*t1z;    
    *dut2dn=dudn*t2x+dvdn*t2y+dwdn*t2z;    
    *dundn=dudn*nx+dvdn*ny+dwdn*nz;    

    *dut1dt1=dudt1*t1x+dvdt1*t1y+dwdt1*t1z;
    *dut2dt1=dudt1*t2x+dvdt1*t2y+dwdt1*t2z;
    *dundt1=dudt1*nx+dvdt1*ny+dwdt1*nz;

    *dut1dt2=dudt2*t1x+dvdt2*t1y+dwdt2*t1z;
    *dut2dt2=dudt2*t2x+dvdt2*t2y+dwdt2*t2z;
    *dundt2=dudt2*nx+dvdt2*ny+dwdt2*nz;

}


void WallModel::Comput_JacobTensor_i(
    int i, int j, int k, 
    int mx, int my, int mz, 
    Cmpnts ***coor, 
    double *dxdc, double *dxde, double *dxdz, 
    double *dydc, double *dyde, double *dydz, 
    double *dzdc, double *dzde, double *dzdz) {

    double centx, centy, centz;
    double centx_ip1, centy_ip1, centz_ip1;
    double centx_im1, centy_im1, centz_im1;
    double centx_jp1, centy_jp1, centz_jp1;
    double centx_jm1, centy_jm1, centz_jm1;
    double centx_kp1, centy_kp1, centz_kp1;
    double centx_km1, centy_km1, centz_km1;


    int i1=i,j1=j,k1=k;    

    centx = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
             coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
    centy = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
             coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
    centz = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
             coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;

    if (i!=mx-2) {
        i1=i+1,j1=j,k1=k;

        centx_ip1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
        centy_ip1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
        centz_ip1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
    }

    if (i!=0) {
        i1=i-1,j1=j,k1=k;

        centx_im1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
        centy_im1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
        centz_im1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
    }

    if (j!=my-2) {
        i1=i,j1=j+1,k1=k;

        centx_jp1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
        centy_jp1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
        centz_jp1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
    }

    if (j!=1) {
        i1=i,j1=j-1,k1=k;

        centx_jm1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
        centy_jm1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
        centz_jm1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
    }

    if (k!=mz-2) {
        i1=i,j1=j,k1=k+1;

        centx_kp1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
        centy_kp1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
        centz_kp1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
    }

    if (k!=1) {
        i1=i,j1=j,k1=k-1;

        centx_km1 = (coor[k1  ][j1  ][i1].x + coor[k1-1][j1  ][i1].x +
                     coor[k1  ][j1-1][i1].x + coor[k1-1][j1-1][i1].x) * 0.25;
        centy_km1 = (coor[k1  ][j1  ][i1].y + coor[k1-1][j1  ][i1].y +
                     coor[k1  ][j1-1][i1].y + coor[k1-1][j1-1][i1].y) * 0.25;
        centz_km1 = (coor[k1  ][j1  ][i1].z + coor[k1-1][j1  ][i1].z +
                     coor[k1  ][j1-1][i1].z + coor[k1-1][j1-1][i1].z) * 0.25;
    }
    
    if (i==0) {
      *dxdc = centx_ip1 - centx;
      *dydc = centy_ip1 - centy;
      *dzdc = centz_ip1 - centz;
    }
    else if (i==mx-2) {
      *dxdc = centx - centx_im1;
      *dydc = centy - centy_im1;
      *dzdc = centz - centz_im1;
    }
    else {
      *dxdc = (centx_ip1 - centx_im1) * 0.5;
      *dydc = (centy_ip1 - centy_im1) * 0.5;
      *dzdc = (centz_ip1 - centz_im1) * 0.5;
    }

    
    if (j==1) {
      *dxde = centx_jp1 - centx;
      *dyde = centy_jp1 - centy;
      *dzde = centz_jp1 - centz;
    }
    else if (j==my-2) {
      *dxde = centx - centx_jm1;
      *dyde = centy - centy_jm1;
      *dzde = centz - centz_jm1;
    }
    else {
      *dxde = (centx_jp1 - centx_jm1) * 0.5;
      *dyde = (centy_jp1 - centy_jm1) * 0.5;
      *dzde = (centz_jp1 - centz_jm1) * 0.5;
    }
    
    if (k==1) {
      *dxdz = (centx_kp1 - centx);
      *dydz = (centy_kp1 - centy);
      *dzdz = (centz_kp1 - centz);
    }
    else if (k==mz-2) {
      *dxdz = (centx - centx_km1);
      *dydz = (centy - centy_km1);
      *dzdz = (centz - centz_km1);
    }
    else {
      *dxdz = (centx_kp1 - centx_km1) * 0.5;
      *dydz = (centy_kp1 - centy_km1) * 0.5;
      *dzdz = (centz_kp1 - centz_km1) * 0.5;
    }
    
}



void WallModel::Comput_JacobTensor_j(
    int i, int j, int k, 
    int mx, int my, int mz, 
    Cmpnts ***coor, 
    double *dxdc, double *dxde, double *dxdz, 
    double *dydc, double *dyde, double *dydz, 
    double *dzdc, double *dzde, double *dzdz) {
    
    double centx, centy, centz;
    double centx_ip1, centy_ip1, centz_ip1;
    double centx_im1, centy_im1, centz_im1;
    double centx_jp1, centy_jp1, centz_jp1;
    double centx_jm1, centy_jm1, centz_jm1;
    double centx_kp1, centy_kp1, centz_kp1;
    double centx_km1, centy_km1, centz_km1;

    int i1=i,j1=j,k1=k;    
    centx = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
             coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
    centy = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
             coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
    centz = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
             coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

    if (i!=mx-2) {
        i1=i+1,j1=j,k1=k;

        centx_ip1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                    coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
        centy_ip1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                    coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
        centz_ip1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                    coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

    }

    if (i!=1) {
        i1=i-1,j1=j,k1=k;

        centx_im1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
        centy_im1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
        centz_im1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

    }

    if (j!=my-2) {
        i1=i,j1=j+1,k1=k;

        centx_jp1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                    coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
        centy_jp1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                    coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
        centz_jp1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                    coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

    }

    if (j!=0) {
        i1=i,j1=j-1,k1=k;

        centx_jm1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                     coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
        centy_jm1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
        centz_jm1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                    coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

}

    if (k!=mz-2) {
        i1=i,j1=j,k1=k+1;

        centx_kp1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
        centy_kp1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                    coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
        centz_kp1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                    coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

    }

    if (k!=1) {
        i1=i,j1=j,k1=k-1;

        centx_km1 = (coor[k1  ][j1][i1  ].x + coor[k1-1][j1][i1  ].x +
                    coor[k1  ][j1][i1-1].x + coor[k1-1][j1][i1-1].x) * 0.25;
        centy_km1 = (coor[k1  ][j1][i1  ].y + coor[k1-1][j1][i1  ].y +
                    coor[k1  ][j1][i1-1].y + coor[k1-1][j1][i1-1].y) * 0.25;
        centz_km1 = (coor[k1  ][j1][i1  ].z + coor[k1-1][j1][i1  ].z +
                    coor[k1  ][j1][i1-1].z + coor[k1-1][j1][i1-1].z) * 0.25;

    }
    
    if (i==1) {
      *dxdc = centx_ip1 - centx;
      *dydc = centy_ip1 - centy;
      *dzdc = centz_ip1 - centz;
    }
    else if (i==mx-2) {
      *dxdc = centx - centx_im1;
      *dydc = centy - centy_im1;
      *dzdc = centz - centz_im1;
    }
    else {
      *dxdc = (centx_ip1 - centx_im1) * 0.5;
      *dydc = (centy_ip1 - centy_im1) * 0.5;
      *dzdc = (centz_ip1 - centz_im1) * 0.5;
    }

    
    if (j==0) {
      *dxde = centx_jp1 - centx;
      *dyde = centy_jp1 - centy;
      *dzde = centz_jp1 - centz;
    }
    else if (j==my-2) {
      *dxde = centx - centx_jm1;
      *dyde = centy - centy_jm1;
      *dzde = centz - centz_jm1;
    }
    else {
      *dxde = (centx_jp1 - centx_jm1) * 0.5;
      *dyde = (centy_jp1 - centy_jm1) * 0.5;
      *dzde = (centz_jp1 - centz_jm1) * 0.5;
    }
    
    if (k==1) {
      *dxdz = (centx_kp1 - centx);
      *dydz = (centy_kp1 - centy);
      *dzdz = (centz_kp1 - centz);
    }
    else if (k==mz-2) {
      *dxdz = (centx - centx_km1);
      *dydz = (centy - centy_km1);
      *dzdz = (centz - centz_km1);
    }
    else {
      *dxdz = (centx_kp1 - centx_km1) * 0.5;
      *dydz = (centy_kp1 - centy_km1) * 0.5;
      *dzdz = (centz_kp1 - centz_km1) * 0.5;
    }
    
}



void WallModel::Comput_JacobTensor_k(
    int i, int j, int k, 
    int mx, int my, int mz, 
    Cmpnts ***coor, 
    double *dxdc, double *dxde, double *dxdz, 
    double *dydc, double *dyde, double *dydz, 
    double *dzdc, double *dzde, double *dzdz) {
    
    double centx, centy, centz;
    double centx_ip1, centy_ip1, centz_ip1;
    double centx_im1, centy_im1, centz_im1;
    double centx_jp1, centy_jp1, centz_jp1;
    double centx_jm1, centy_jm1, centz_jm1;
    double centx_kp1, centy_kp1, centz_kp1;
    double centx_km1, centy_km1, centz_km1;

    int i1=i,j1=j,k1=k;

    centx = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
             coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
    centy = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
             coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
    centz = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
             coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

    if (i!=mx-2) {
        i1=i+1,j1=j,k1=k;

        centx_ip1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                      coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
        centy_ip1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                      coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
        centz_ip1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                          coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

    }

    if (i!=1) {
        i1=i-1,j1=j,k1=k;

            centx_im1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                         coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
            centy_im1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                         coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
            centz_im1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                         coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

    }

    if (j!=my-2) {
        i1=i,j1=j+1,k1=k;

            centx_jp1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                         coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
            centy_jp1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                         coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
            centz_jp1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                         coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

    }

    if (j!=1) {
        i1=i,j1=j-1,k1=k;

            centx_jm1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                         coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
            centy_jm1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                         coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
            centz_jm1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                         coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

    }

    if (k!=mz-2) {
        i1=i,j1=j,k1=k+1;

            centx_kp1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                         coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
            centy_kp1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                         coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
            centz_kp1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                         coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

    }

    if (k!=0) {
        i1=i,j1=j,k1=k-1;

            centx_km1 = (coor[k1  ][j1][i1  ].x + coor[k1][j1-1][i1  ].x +
                         coor[k1  ][j1][i1-1].x + coor[k1][j1-1][i1-1].x) * 0.25;
            centy_km1 = (coor[k1  ][j1][i1  ].y + coor[k1][j1-1][i1  ].y +
                         coor[k1  ][j1][i1-1].y + coor[k1][j1-1][i1-1].y) * 0.25;
            centz_km1 = (coor[k1  ][j1][i1  ].z + coor[k1][j1-1][i1  ].z +
                         coor[k1  ][j1][i1-1].z + coor[k1][j1-1][i1-1].z) * 0.25;

    }
    
    if (i==1) {
      *dxdc = centx_ip1 - centx;
      *dydc = centy_ip1 - centy;
      *dzdc = centz_ip1 - centz;
    }
    else if (i==mx-2) {
      *dxdc = centx - centx_im1;
      *dydc = centy - centy_im1;
      *dzdc = centz - centz_im1;
    }
    else {
      *dxdc = (centx_ip1 - centx_im1) * 0.5;
      *dydc = (centy_ip1 - centy_im1) * 0.5;
      *dzdc = (centz_ip1 - centz_im1) * 0.5;
    }

    
    if (j==1) {
      *dxde = centx_jp1 - centx;
      *dyde = centy_jp1 - centy;
      *dzde = centz_jp1 - centz;
    }
    else if (j==my-2) {
      *dxde = centx - centx_jm1;
      *dyde = centy - centy_jm1;
      *dzde = centz - centz_jm1;
    }
    else {
      *dxde = (centx_jp1 - centx_jm1) * 0.5;
      *dyde = (centy_jp1 - centy_jm1) * 0.5;
      *dzde = (centz_jp1 - centz_jm1) * 0.5;
    }
    
    if (k==0) {
      *dxdz = (centx_kp1 - centx);
      *dydz = (centy_kp1 - centy);
      *dzdz = (centz_kp1 - centz);
    }
    else if (k==mz-2) {
      *dxdz = (centx - centx_km1);
      *dydz = (centy - centy_km1);
      *dzdz = (centz - centz_km1);
    }
    else {
      *dxdz = (centx_kp1 - centx_km1) * 0.5;
      *dydz = (centy_kp1 - centy_km1) * 0.5;
      *dzdz = (centz_kp1 - centz_km1) * 0.5;
    }
    
}

void WallModel::Comput_du_Compgrid(
    double dxdc, double dxde, double dxdz, 
    double dydc, double dyde, double dydz, 
    double dzdc, double dzde, double dzdz, 
    double nx, double ny, double nz, 
    double t1x, double t1y, double t1z, 
    double t2x, double t2y, double t2z, 
    double dut1dn, double dut2dn, double dundn, 
    double dut1dt1, double dut2dt1, double dundt1, 
    double dut1dt2, double dut2dt2, double dundt2, 
    double *dudc, double *dvdc, double *dwdc, 
    double *dude, double *dvde, double *dwde, 
    double *dudz, double *dvdz, double *dwdz) {

    double dxdn=nx, dydn=ny, dzdn=nz;
    double dxdt1=t1x, dydt1=t1y, dzdt1=t1z;
    double dxdt2=t2x, dydt2=t2y, dzdt2=t2z;

    double dndx = dydt1*dzdt2-dydt2*dzdt1;
    double dt1dx = dydt2*dzdn-dydn*dzdt2;
    double dt2dx = dydn*dzdt1-dydt1*dzdn;

    double dndy = dzdt1*dxdt2-dzdt2*dxdt1;
    double dt1dy = dzdt2*dxdn-dzdn*dxdt2;
    double dt2dy = dzdn*dxdt1-dzdt1*dxdn;

    double dndz = dxdt1*dydt2-dxdt2*dydt1;
    double dt1dz = dxdt2*dydn-dxdn*dydt2;
    double dt2dz = dxdn*dydt1-dxdt1*dydn;


    double dundx = dundn*dndx+dundt1*dt1dx+dundt2*dt2dx;
    double dundy = dundn*dndy+dundt1*dt1dy+dundt2*dt2dy;
    double dundz = dundn*dndz+dundt1*dt1dz+dundt2*dt2dz;

    double dut1dx = dut1dn*dndx+dut1dt1*dt1dx+dut1dt2*dt2dx;
    double dut1dy = dut1dn*dndy+dut1dt1*dt1dy+dut1dt2*dt2dy;
    double dut1dz = dut1dn*dndz+dut1dt1*dt1dz+dut1dt2*dt2dz;

    double dut2dx = dut2dn*dndx+dut2dt1*dt1dx+dut2dt2*dt2dx;
    double dut2dy = dut2dn*dndy+dut2dt1*dt1dy+dut2dt2*dt2dy;
    double dut2dz = dut2dn*dndz+dut2dt1*dt1dz+dut2dt2*dt2dz;


    double du_dx = dundx*nx+dut1dx*t1x+dut2dx*t2x;
    double du_dy = dundy*nx+dut1dy*t1x+dut2dy*t2x;
    double du_dz = dundz*nx+dut1dz*t1x+dut2dz*t2x;

    double dv_dx = dundx*ny+dut1dx*t1y+dut2dx*t2y;
    double dv_dy = dundy*ny+dut1dy*t1y+dut2dy*t2y;
    double dv_dz = dundz*ny+dut1dz*t1y+dut2dz*t2y;

    double dw_dx = dundx*nz+dut1dx*t1z+dut2dx*t2z;
    double dw_dy = dundy*nz+dut1dy*t1z+dut2dy*t2z;
    double dw_dz = dundz*nz+dut1dz*t1z+dut2dz*t2z;

    *dudc = du_dx*dxdc+du_dy*dydc+du_dz*dzdc;
    *dude = du_dx*dxde+du_dy*dyde+du_dz*dzde;
    *dudz = du_dx*dxdz+du_dy*dydz+du_dz*dzdz;

    *dvdc = dv_dx*dxdc+dv_dy*dydc+dv_dz*dzdc;
    *dvde = dv_dx*dxde+dv_dy*dyde+dv_dz*dzde;
    *dvdz = dv_dx*dxdz+dv_dy*dydz+dv_dz*dzdz;

    *dwdc = dw_dx*dxdc+dw_dy*dydc+dw_dz*dzdc;
    *dwde = dw_dx*dxde+dw_dy*dyde+dw_dz*dzde;
    *dwdz = dw_dx*dxdz+dw_dy*dydz+dw_dz*dzdz;



}


void WallModel::wallmodel_s(
    double nu, double sb, double sc, 
    Cmpnts Uc, Cmpnts *Ub,  Cmpnts Ua, 
    PetscInt bctype, 
    double ks, 
    double nx, double ny, double nz, 
    double *tau_w, PetscReal *ustar, 
    double dpdx, double dpdy, double dpdz,
    double *nut_2sb, double nut_c)
{

    double kappa_rans = 0.4;
    double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
    double un = u_c * nx + v_c * ny + w_c * nz;
    double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
    double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );
    double eps=1.e-11;

    double ustar_noslip=sqrt(nu*ut_mag_c/sc);

    double A=8.3;
    double B=1.0/7.0; 
    
    *ustar = pow( ut_mag_c * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));  

    double _ks=ks*0.033;
    if (bctype==6)  *ustar = ut_mag_c*kappa_rans/log(sc/_ks);
    
    
    double ybp = sb*max((*ustar),ustar_noslip)/nu;
    double ycp = sc*max((*ustar),ustar_noslip)/nu;
    double ut_mag_modeled;
    if (ycp>12.0) { 
        ut_mag_modeled = A*(*ustar)*pow(ybp,B);
        if (bctype==6) ut_mag_modeled = (*ustar)*log(sb/_ks)/kappa_rans;
    } 
    else {
        ut_mag_modeled = ut_mag_c*sb/sc;    
        *ustar = sqrt(fabs(nu*ut_mag_c/sc));
    }

    double sign=ut_mag_c/(fabs(ut_mag_c)+eps);
    *tau_w=pow(*ustar,2)*sign;

    double t1_x, t1_y, t1_z, t2_x, t2_y, t2_z;

    t1_x = ut / (ut_mag_c+eps); 
    t1_y = vt / (ut_mag_c+eps); 
    t1_z = wt / (ut_mag_c+eps);

    double dpdt = dpdx * t1_x + dpdy * t1_y + dpdz * t1_z;

    double up = pow( fabs(nu*dpdt), 1.0/3.0 );
    double uall = sqrt(pow(*ustar,2)+up*up);  
    double yp=sc*uall/nu;

    double damping = (1.0-exp(-yp/A))*(1.0-exp(-yp/A));
    double kappa = nut_c/(nu*yp*damping+eps);

    yp=0.5*(sb+sc)*uall/nu;
    damping =(1.0-exp(-yp/A))*(1.0-exp(-yp/A));
    *nut_2sb=nu*kappa*yp*damping;

    if(ut_mag_c>eps) {
        ut *= ut_mag_modeled/ut_mag_c;
        vt *= ut_mag_modeled/ut_mag_c;
        wt *= ut_mag_modeled/ut_mag_c;
    }
    else ut=vt=wt=0;
                
    double dpdn_sc=dpdx*nx+dpdy*ny+dpdy*nz;
    double dpdn_sb=dpdn_sc*sb/sc;
    double sign_sc=-dpdn_sc/(fabs(dpdn_sc)+eps);
    double coeff=sign_sc*un/sqrt(fabs(dpdn_sc)*sc);
    double sign_sb=-dpdn_sb/(fabs(dpdn_sb)+eps);
    double un_sb=fabs(coeff)*sqrt(fabs(dpdn_sb)*sb)*sign_sb;

    (*Ub).x = ut + pow(sb/sc,1) * un * nx;
    (*Ub).y = vt + pow(sb/sc,1) * un * ny;
    (*Ub).z = wt + pow(sb/sc,1) * un * nz;    
    
    (*Ub).x += Ua.x;
    (*Ub).y += Ua.y;
    (*Ub).z += Ua.z;

}


void WallModel::wallmodel_0424(
    double ks, PetscReal *ustar,
    double dpdx, double dpdy, double dpdz, 
    double nu, double sb, double sc, 
    Cmpnts *Ub, Cmpnts Uc, Cmpnts Ua, 
    double nx, double ny, double nz, 
    PetscReal alfa)
{
    double kappa_rans = 0.41, A = 19;

    double t1_x, t1_y, t1_z, t2_x, t2_y, t2_z;

    double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
    double un = u_c * nx + v_c * ny + w_c * nz;
    double ut = u_c - un * nx, vt = v_c - un * ny, wt = w_c - un * nz;
    double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );

    t1_x = ut / (ut_mag_c+1.e-19); 
    t1_y = vt / (ut_mag_c+1.e-19); 
    t1_z = wt / (ut_mag_c+1.e-19);

    double u_b = (*Ub).x - Ua.x, v_b = (*Ub).y - Ua.y, w_b = (*Ub).z - Ua.z;
    double ut_mag_b1 = u_b * t1_x + v_b * t1_y + w_b * t1_z;

    double u_a = Ua.x, v_a = Ua.y, w_a = Ua.z;
    double ut_mag_a = u_a * t1_x + v_a * t1_y + w_a * t1_z;

    double dpdt1 = dpdx * t1_x + dpdy * t1_y + dpdz * t1_z;

    double tau_w;
    int i, j, k;
    double yp, damping;

    double coeff_c1, f0_sum_c, f1_sum_c,coeff_c0;


    double ut_mag_b;    
    double _ks=ks*0.033;

    //if ( ti == tistart) 
    if (*ustar==0) (*ustar) = utau_powerlaw(nu, ut_mag_c, sc); 
    
    double ycp = sc*(*ustar)/nu;

    double Tau_noslip = nu*ut_mag_c/sc;

    double coef_modify=alfa;
    if (ycp>12.0) {
        if (_ks>1.e-11) {
            *ustar = ut_mag_b1*kappa_rans/log(sb/_ks);

            ut_mag_b = (*ustar)*log(sb/_ks)/kappa_rans;
        } else {

            d_num_innergrid = (int)(log(sc*(d_dhratio_wm-1)/d_dh1_wm+1.0)/
                log(d_dhratio_wm))+1;
            double z_in[d_num_innergrid], f0[d_num_innergrid];
            double  f1[d_num_innergrid];
            double f2[d_num_innergrid], nu_t_rans[d_num_innergrid];
            double  u_inner[d_num_innergrid];
            innergrid( z_in, sc );

            double ustarp = pow( fabs(nu*dpdt1), 1.0/3.0 );
            double ustartp = sqrt( (*ustar)*(*ustar) + ustarp * ustarp);  

            nu_t_rans[0] = 0.0;
            for (k = 1; k < d_num_innergrid; k++) {
                 yp = z_in[k] * ustartp / nu;
                 damping = (1.0 - exp(-yp/A)) * (1.0 - exp(-yp/A));
                 nu_t_rans[k] = nu * kappa_rans * yp * damping;
            }

            double sum0, sum1, sum2, dz;
            sum0 = 0.0; sum1 = 0.0; 
            f0[0] = 0.0; f1[0] = 0.0; 
            for (k = 1; k < d_num_innergrid; k++) {
                dz = z_in[k] - z_in[k-1];
                sum0 += dz / ( nu + 0.5*(nu_t_rans[k]+nu_t_rans[k-1]) );
                sum1 += dz * 0.5 * ( z_in[k] + z_in[k-1] ) / 
                        ( nu + 0.5*(nu_t_rans[k]+nu_t_rans[k-1]) );
                f0[k] = sum0; f1[k] = sum1; 
            }

            
            double r1 = ut_mag_c - dpdt1 * f1[d_num_innergrid-1] - ut_mag_a;
            coeff_c1 = r1 / f0[d_num_innergrid-1];

            for (k = 0; k < d_num_innergrid; k++) {
                u_inner[k] = ut_mag_a + dpdt1 * f1[k] +  coeff_c1 * f0[k];
            }

            tau_w = nu * (u_inner[1]-u_inner[0]) / (z_in[1]-z_in[0]);

            *ustar = sqrt( fabs(tau_w) );

            int k_sb;
            for (k = 1; k < d_num_innergrid; k++) {
                if (sb>=z_in[k-1] && sb<=z_in[k]) {
                    k_sb=k-1;
                }
            }
            
            double fac=1/(z_in[k_sb+1]-z_in[k_sb]);
            double fac1=(z_in[k_sb+1]-sb)*fac;
            double fac2=(sb-z_in[k_sb])*fac;

            double u1 = ut_mag_a + dpdt1 * f1[k_sb  ] +  coeff_c1 * f0[k_sb  ];
            double u2 = ut_mag_a + dpdt1 * f1[k_sb+1] +  coeff_c1 * f0[k_sb+1];

            ut_mag_b = u1*fac1+u2*fac2;
        }
    } else {
        ut_mag_b = ut_mag_c * sb/sc; 
        *ustar = sqrt(Tau_noslip);
    }

            
    (*Ub).x = ut_mag_b*t1_x + sb/sc * un * nx;
    (*Ub).y = ut_mag_b*t1_y + sb/sc * un * ny;
    (*Ub).z = ut_mag_b*t1_z + sb/sc * un * nz;
    
    (*Ub).x += Ua.x;
    (*Ub).y += Ua.y;
    (*Ub).z += Ua.z;

}

double WallModel::utau_powerlaw(double nu, double ut_mag, double sc)
{
    double A=8.3;
    double B=1.0/7.0; 
    double ustar;
        ustar = pow( ut_mag * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));
    double ycp = sc*ustar/nu;
    if (ycp>12.0) {
        ustar = pow( ut_mag * pow(nu, B) / (A * pow(sc, B)),  1.0/(1.0+B));
    } 
    else {
        ustar = sqrt(fabs(nu*ut_mag/sc));
    }

    return ustar;

}

void WallModel::innergrid( double *z_in, double h)
{
    int k;
    double f;

    z_in[0] = 0.0;
    for (k = 1; k < d_num_innergrid; k++) {
        z_in[k] = z_in[k-1] + d_dh1_wm*pow(d_dhratio_wm,(double)k);
    }
}

PetscErrorCode WallModel::ReadFromInput()
{
    PetscOptionsGetInt(PETSC_NULL, "-imin_wm", &d_imin_wm, PETSC_NULL); 
    PetscOptionsGetInt(PETSC_NULL, "-imax_wm", &d_imax_wm, PETSC_NULL); 

    PetscOptionsGetInt(PETSC_NULL, "-jmin_wm", &d_jmin_wm, PETSC_NULL); 
    PetscOptionsGetInt(PETSC_NULL, "-jmax_wm", &d_jmax_wm, PETSC_NULL); 

    PetscOptionsGetInt(PETSC_NULL, "-kmin_wm", &d_kmin_wm, PETSC_NULL); 
    PetscOptionsGetInt(PETSC_NULL, "-kmax_wm", &d_kmax_wm, PETSC_NULL); 
    PetscOptionsGetInt(PETSC_NULL, "-imm", &d_immersed, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-IB_wm", &d_ib_wm, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-roughness", &d_roughness_size,
                        PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-alfa_wm", &d_alfa_wm, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-infRe", &d_infRe, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL,"-les_eps", &d_les_eps, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-powerlawwallmodel", 
                        &d_powerlawwallmodel, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-dhratio_wm", &d_dhratio_wm, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-dh1_wm", &d_dh1_wm, PETSC_NULL);

    if (d_imin_wm != 0 || d_imax_wm != 0 || 
        d_jmin_wm != 0 || d_jmax_wm != 0 ||
        d_kmin_wm != 0 || d_kmax_wm != 0 || (d_ib_wm != 0 && d_immersed))
       
        d_use_wall = PETSC_TRUE;

}

