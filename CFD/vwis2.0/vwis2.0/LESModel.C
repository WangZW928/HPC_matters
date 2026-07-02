#include "LESModel.h"

LESModel::LESModel(const std::string& object_name,
    CurvGrid *grid,
    UData *data):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data)
{

    d_i_homo_filter = 0;
    d_j_homo_filter = 0;
    d_k_homo_filter = 0;
    d_testfilter_ik = 0;
    d_les = 2;
    d_les_eps = 1.e-7;
    d_max_cs = 0.5;
    d_wall_cs = 0.001;
    d_max_norm = 1e10;
    d_filter_size = 0;
    d_wallfunction=0;

    d_restart = PETSC_FALSE;
    d_use_les = PETSC_FALSE;
 
    d_tistart = 0; 
    d_tiout = 1;
    d_hdf5 = 0;
    sprintf(d_fieldpath, ".") ;

    ReadFromInput();
}

LESModel::~LESModel()
{
    VecDestroy(&d_lCs);
    VecDestroy(&d_lNu_t);
    if (d_les and d_les==4) {
        VecDestroy(&d_lLM_old);
        VecDestroy(&d_lMM_old);
    }
}

void LESModel::Initialize()
{
    Vec lP = d_data->getlP();
    Vec P = d_data->getP();

    VecDuplicate(lP, &d_lCs);
    VecDuplicate(P, &d_Cs);
    PetscObjectSetName((PetscObject) d_lCs, "cs");
    PetscObjectSetName((PetscObject) d_Cs, "cs");

    VecDuplicate(lP, &d_lNu_t);

    if (d_les==4) {
        VecDuplicate(lP, &d_lLM_old);
        VecDuplicate(lP, &d_lMM_old);
    }
}

void LESModel::ReadCs()
{
   if (!d_use_les) return;

    char filen[90];
    
    DM da = d_grid->getDA();
    PetscInt ti = d_data->get_tistart();
   
    sprintf(filen, "%s/cs%06d_%1d.%s", d_fieldpath, ti, 0, d_rext);
    d_data->ReadFile(filen, d_Cs);

    DMGlobalToLocalBegin(da, d_Cs, INSERT_VALUES, d_lCs);
    DMGlobalToLocalEnd(da, d_Cs, INSERT_VALUES, d_lCs);
 
}

void LESModel::WriteCs(PetscInt ti)
{
    if (!d_use_les) return;

    char filen[90];
    //We only everything output at tiout intervals
    if (ti == (ti/d_tiout) * d_tiout) {

        //write Data here
        sprintf(filen, "%s/cs%06d_%1d.%s", d_fieldpath, ti, 0, d_wext);

        DM da = d_grid->getDA();
        DMLocalToGlobalBegin(da, d_lCs, INSERT_VALUES, d_Cs);
        DMLocalToGlobalEnd(da, d_lCs, INSERT_VALUES, d_Cs);
     
        d_data->WriteFile(filen, d_Cs);
    }
}

 

void LESModel::ComputeSmagorinksyConstant(PetscInt ti)
{
    if (!d_use_les) return;

    //Constant Smag
    if (d_les==1) {
        VecSet(d_lCs, 0.01);
        return;
    }
    

    int i, j, k;

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    int lxs, lxe, lys, lye, lzs, lze;
    int xs, xe, ys, ye, zs, ze; // Local grid information
    int mx, my, mz; // Dimensions in three directions
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
  
    double ***LM, ***MM;
    double ***LM_tmp, ***MM_tmp;
    PetscReal ajc;
    PetscReal dudc, dude, dudz, dvdc, dvde, dvdz, dwdc, dwde, dwdz;
    PetscReal ***nvert, ***Cs;
    PetscReal ***aj; 
    PetscReal ***Sabs;
    PetscReal ***iaj, ***jaj, ***kaj;

    PetscInt i_periodic = d_grid->isIPeriodic();
    PetscInt j_periodic = d_grid->isJPeriodic();
    PetscInt k_periodic = d_grid->isKPeriodic();
    PetscInt ii_periodic = d_grid->isIIPeriodic();
    PetscInt jj_periodic = d_grid->isJJPeriodic();
    PetscInt kk_periodic = d_grid->isKKPeriodic();

    Cmpnts ***icsi, ***ieta, ***izet;
    Cmpnts ***jcsi, ***jeta, ***jzet;
    Cmpnts ***kcsi, ***keta, ***kzet;
    Cmpnts ***ucont, ***ucat;
    Cmpnts ***csi, ***eta, ***zet;
    Cmpnts ***Ax, ***Ay, ***Az, ***cent;
    
    Vec Cent = d_grid->getlCent();
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

    Vec lUcont = d_data->getlUcont();
    Vec lUcat = d_data->getlUcat();
    Vec lNvert = d_data->getlNvert();
    Vec lP = d_data->getlP();

    Vec lSx, lSy, lSz, lS;
    Vec lLM_tmp, lMM_tmp;
    
    DMDAVecGetArray(fda, lUcont, &ucont);
    DMDAVecGetArray(fda, lUcat,  &ucat);
    DMDAVecGetArray(da, lNvert, &nvert);
    
    DMDAVecGetArray(da, d_lCs, &Cs);
    
    DMDAVecGetArray(fda, Cent, &cent);

    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, Aj, &aj); 

    DMDAVecGetArray(da, IAj, &iaj);  
    DMDAVecGetArray(da, JAj, &jaj);  
    DMDAVecGetArray(da, KAj, &kaj);  
    
    DMDAVecGetArray(fda, ICsi, &icsi);
    DMDAVecGetArray(fda, IEta, &ieta);
    DMDAVecGetArray(fda, IZet, &izet);

    DMDAVecGetArray(fda, JCsi, &jcsi);
    DMDAVecGetArray(fda, JEta, &jeta);
    DMDAVecGetArray(fda, JZet, &jzet);

    DMDAVecGetArray(fda, KCsi, &kcsi);
    DMDAVecGetArray(fda, KEta, &keta);
    DMDAVecGetArray(fda, KZet, &kzet);
    //
  
    VecDuplicate(lP, &d_lLM);
    VecDuplicate(lP, &d_lMM);

    VecDuplicate(lP, &lLM_tmp); 
    VecDuplicate(lP, &lMM_tmp); 
    DMDAVecGetArray(da, lLM_tmp, &LM_tmp);
    DMDAVecGetArray(da, lMM_tmp, &MM_tmp);
    VecSet(lLM_tmp, 0); 
    VecSet(lMM_tmp, 0);

    VecSet(d_lLM, 0);
    VecSet(d_lMM, 0);
    
    VecDuplicate(lUcont, &lSx);
    VecDuplicate(lUcont, &lSy);
    VecDuplicate(lUcont, &lSz);
    VecDuplicate(lNvert, &lS);
    
    VecSet(lSx, 0);  
    VecSet(lSy, 0);  
    VecSet(lSz, 0);    
    VecSet(lS, 0);

    DMDAVecGetArray(da, d_lLM, &LM);
    DMDAVecGetArray(da, d_lMM, &MM);

    DMDAVecGetArray(fda, lSx, &Ax);
    DMDAVecGetArray(fda, lSy, &Ay);
    DMDAVecGetArray(fda, lSz, &Az);
    DMDAVecGetArray(da, lS, &Sabs);
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
                if (nvert[k][j][i]>1.1) continue;
        
                ajc = aj[k][j][i];
                double csi0 = csi[k][j][i].x;
                double csi1 = csi[k][j][i].y;
                double csi2 = csi[k][j][i].z;
                double eta0 = eta[k][j][i].x;
                double eta1 = eta[k][j][i].y;
                double eta2 = eta[k][j][i].z;
                double zet0 = zet[k][j][i].x;
                double zet1 = zet[k][j][i].y;
                double zet2 = zet[k][j][i].z;

                
                double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
                double du_dx, du_dy, du_dz;
                double dv_dx, dv_dy, dv_dz; 
                double dw_dx, dw_dy, dw_dz;
        
                Compute_du_center(i, j, k, 
                                  mx, my, mz, 
                                  ucat, nvert, 
                                  i_periodic, ii_periodic, j_periodic, 
                                  jj_periodic, k_periodic, kk_periodic, 
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
                                &du_dz, &dv_dz, &dw_dz);
        
                double Sxx = 0.5*(du_dx + du_dx);
                double Sxy = 0.5*(du_dy + dv_dx);
                double Sxz = 0.5*(du_dz + dw_dx);
                double Syx = Sxy;
                double Syy = 0.5*(dv_dy + dv_dy);
                double Syz = 0.5*(dv_dz + dw_dy);
                double Szx = Sxz;
                double Szy = Syz; 
                double Szz = 0.5*(dw_dz + dw_dz);
    
                Sabs[k][j][i] = sqrt( 2.0*( Sxx*Sxx + Sxy*Sxy + Sxz*Sxz + 
                                            Syx*Syx + Syy*Syy + Syz*Syz + 
                                            Szx*Szx + Szy*Szy + Szz*Szz ) );
        
                Ax[k][j][i].x = du_dx;    
                Ax[k][j][i].y = du_dy;    
                Ax[k][j][i].z = du_dz;
                Ay[k][j][i].x = dv_dx;    
                Ay[k][j][i].y = dv_dy;    
                Ay[k][j][i].z = dv_dz;
                Az[k][j][i].x = dw_dx;    
                Az[k][j][i].y = dw_dy;    
                Az[k][j][i].z = dw_dz;
            }
    
    DMDAVecRestoreArray(fda, lSx, &Ax);
    DMDAVecRestoreArray(fda, lSy, &Ay);
    DMDAVecRestoreArray(fda, lSz, &Az);
    DMDAVecRestoreArray(da, lS, &Sabs);
        
    DMDALocalToLocalBegin(fda, lSx, INSERT_VALUES, lSx);
    DMDALocalToLocalEnd(fda, lSx, INSERT_VALUES, lSx);
 
    DMDALocalToLocalBegin(fda, lSy, INSERT_VALUES, lSy);
    DMDALocalToLocalEnd(fda, lSy, INSERT_VALUES, lSy);
 
    DMDALocalToLocalBegin(fda, lSz, INSERT_VALUES, lSz);
    DMDALocalToLocalEnd(fda, lSz, INSERT_VALUES, lSz);
 
    DMDALocalToLocalBegin(da, lS, INSERT_VALUES, lS);
    DMDALocalToLocalEnd(da, lS, INSERT_VALUES, lS);


    DMDAVecGetArray(fda, lSx, &Ax);
    DMDAVecGetArray(fda, lSy, &Ay);
    DMDAVecGetArray(fda, lSz, &Az);
    DMDAVecGetArray(da, lS, &Sabs);
    
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                int c=k, b=j, a=i, flag=0;
        
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
                    Sabs[k][j][i] = Sabs[c][b][a];
                    Ax[k][j][i] = Ax[c][b][a];
                    Ay[k][j][i] = Ay[c][b][a];
                    Az[k][j][i] = Az[c][b][a];
                }
            }
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
                if (nvert[k][j][i]>1.1) {
                   LM[k][j][i]=MM[k][j][i]=0;
                   continue;
                }

                ajc = aj[k][j][i];
                double csi0 = csi[k][j][i].x;
                double csi1 = csi[k][j][i].y;
                double csi2 = csi[k][j][i].z;
                double eta0 = eta[k][j][i].x;
                double eta1 = eta[k][j][i].y;
                double eta2 = eta[k][j][i].z;
                double zet0 = zet[k][j][i].x;
                double zet1 = zet[k][j][i].y;
                double zet2 = zet[k][j][i].z;
    
                int a, b;
                double Lij[3][3], Sij_hat[3][3], SSij_hat[3][3];
                double Mij[3][3], Nij[3][3], Nij_cat[3][3];
                double Lij_cat[3][3], Mij_cat[3][3];
        
                double filter, test_filter;
                double S[3][3][3], S_hat;
        
                double u[3][3][3], v[3][3][3], w[3][3][3], d[3][3][3];
                double U[3][3][3], V[3][3][3], W[3][3][3];
                double Uu[3][3][3], Uv[3][3][3], Uw[3][3][3];
                double Vu[3][3][3], Vv[3][3][3], Vw[3][3][3];
                double Wu[3][3][3], Wv[3][3][3], Ww[3][3][3];
        
                double S11[3][3][3], S12[3][3][3], S13[3][3][3]; 
                double S21[3][3][3], S22[3][3][3], S23[3][3][3];
                double S31[3][3][3], S32[3][3][3], S33[3][3][3];
                double SS11[3][3][3], SS12[3][3][3], SS13[3][3][3];
                double SS21[3][3][3], SS22[3][3][3], SS23[3][3][3];
                double SS31[3][3][3], SS32[3][3][3], SS33[3][3][3];
        
        
                int p,q,r;
                for (p=-1; p<=1; p++)
                    for (q=-1; q<=1; q++)
                        for (r=-1; r<=1; r++) {
                            int R=r+1, Q=q+1, P=p+1;
                            int K=k+r, J=j+q, I=i+p;
            
                            u[R][Q][P] = ucat[K][J][I].x;
                            v[R][Q][P] = ucat[K][J][I].y;
                            w[R][Q][P] = ucat[K][J][I].z;
            
                            // metric tensors are also test-filtered
            
                            U[R][Q][P] = u[R][Q][P]*csi[K][J][I].x + 
                                         v[R][Q][P]*csi[K][J][I].y + 
                                         w[R][Q][P]*csi[K][J][I].z;
                            V[R][Q][P] = u[R][Q][P]*eta[K][J][I].x + 
                                         v[R][Q][P]*eta[K][J][I].y + 
                                         w[R][Q][P]*eta[K][J][I].z;
                            W[R][Q][P] = u[R][Q][P]*zet[K][J][I].x + 
                                         v[R][Q][P]*zet[K][J][I].y + 
                                         w[R][Q][P]*zet[K][J][I].z;
        
    
                            Uu[R][Q][P] = U[R][Q][P] * u[R][Q][P];
                            Uv[R][Q][P] = U[R][Q][P] * v[R][Q][P];
                            Uw[R][Q][P] = U[R][Q][P] * w[R][Q][P];
            
                            Vu[R][Q][P] = V[R][Q][P] * u[R][Q][P];
                            Vv[R][Q][P] = V[R][Q][P] * v[R][Q][P];
                            Vw[R][Q][P] = V[R][Q][P] * w[R][Q][P];
            
                            Wu[R][Q][P] = W[R][Q][P] * u[R][Q][P];
                            Wv[R][Q][P] = W[R][Q][P] * v[R][Q][P];
                            Ww[R][Q][P] = W[R][Q][P] * w[R][Q][P];
            
                            const double du_dx = Ax[K][J][I].x;
                            const double du_dy = Ax[K][J][I].y; 
                            const double du_dz = Ax[K][J][I].z;
                            const double dv_dx = Ay[K][J][I].x;
                            const double dv_dy = Ay[K][J][I].y; 
                            const double dv_dz = Ay[K][J][I].z;
                            const double dw_dx = Az[K][J][I].x;
                            const double dw_dy = Az[K][J][I].y;
                            const double dw_dz = Az[K][J][I].z;
        
                            const double Sxx = 0.5*(du_dx + du_dx);
                            const double Sxy = 0.5*(du_dy + dv_dx);
                            const double Sxz = 0.5*(du_dz + dw_dx);
                            const double Syx = Sxy;
                            const double Syy = 0.5*(dv_dy + dv_dy); 
                            const double Syz = 0.5*(dv_dz + dw_dy);
                            const double Szx = Sxz;
                            const double Szy = Syz; 
                            const double Szz = 0.5*(dw_dz + dw_dz);
                    
                            S11[R][Q][P] = Sxx;
                            S12[R][Q][P] = Sxy;
                            S13[R][Q][P] = Sxz;
                            S21[R][Q][P] = Syx;
                            S22[R][Q][P] = Syy;
                            S23[R][Q][P] = Syz;
                            S31[R][Q][P] = Szx;
                            S32[R][Q][P] = Szy;
                            S33[R][Q][P] = Szz;
            
                            S[R][Q][P] = Sabs[K][J][I];
            
                            SS11[R][Q][P] = S11[R][Q][P]*S[R][Q][P];
                            SS12[R][Q][P] = S12[R][Q][P]*S[R][Q][P];
                            SS13[R][Q][P] = S13[R][Q][P]*S[R][Q][P];
                            SS21[R][Q][P] = S21[R][Q][P]*S[R][Q][P];
                            SS22[R][Q][P] = S22[R][Q][P]*S[R][Q][P];
                            SS23[R][Q][P] = S23[R][Q][P]*S[R][Q][P];
                            SS31[R][Q][P] = S31[R][Q][P]*S[R][Q][P];
                            SS32[R][Q][P] = S32[R][Q][P]*S[R][Q][P];
                            SS33[R][Q][P] = S33[R][Q][P]*S[R][Q][P];
            
                        }
        
                double sum_weight=0;
                double coef[3][3][3]={
                                      0.125, 0.250, 0.125, 
                                      0.250, 0.500, 0.250, 
                                      0.125, 0.250, 0.125, 
                
                                      0.250, 0.500, 0.250,
                                      0.500, 1.000, 0.500,
                                      0.250, 0.500, 0.250,
                
                                      0.125, 0.250, 0.125, 
                                      0.250, 0.500, 0.250,
                                      0.125, 0.250, 0.125
                                     };
        
                double weight[3][3][3];
                double sum_vol=0;
        
        
                for (p=-1; p<=1; p++)
                    for (q=-1; q<=1; q++)
                        for (r=-1; r<=1; r++) {
                            int R=r+1, Q=q+1, P=p+1;
                            int K=k+r, J=j+q, I=i+p;
            
                            sum_weight += weight[R][Q][P] * coef[R][Q][P];
                            if (nvert[K][J][I]<0.1) {
                                sum_vol += 1./aj[K][J][I] * coef[R][Q][P];
                                weight[R][Q][P] = 1;
                            } else weight[R][Q][P] = 0;
                        }


                // xiaolei add filter_size==1
                double dhi_test, dhj_test, dhk_test;
                dhi_test=0.0;
                dhj_test=0.0;
                dhk_test=0.0;

                d_filter_size = 0;

                if (d_filter_size==1) {
                    for (p=-1; p<=1; p++)
                        for (q=-1; q<=1; q++)
                            for (r=-1; r<=1; r++) {
                                int R=r+1, Q=q+1, P=p+1;
                                int K=k+r, J=j+q, I=i+p;
            
                                if (nvert[K][J][I]<0.1) {
                                    double area;
                                    area = sqrt(csi[K][J][I].x*csi[K][J][I].x +
                                                csi[K][J][I].y*csi[K][J][I].y +
                                                csi[K][J][I].z*csi[K][J][I].z );
                                    dhi_test += 1.0/aj[K][J][I]/area;
                                    area = sqrt(eta[K][J][I].x*eta[K][J][I].x +
                                                eta[K][J][I].y*eta[K][J][I].y +
                                                eta[K][J][I].z*eta[K][J][I].z );
                                    dhj_test += 1.0/aj[K][J][I]/area;
                                    area = sqrt(zet[K][J][I].x*zet[K][J][I].x +
                                                zet[K][J][I].y*zet[K][J][I].y +
                                                zet[K][J][I].z*zet[K][J][I].z );
                                    dhk_test += 1.0/aj[K][J][I]/area;
                                }
                            }
                }

                // xiaolei add filter_size==1
                if (d_filter_size==1) {
                    double area = sqrt(csi[k][j][i].x*csi[k][j][i].x + 
                                       csi[k][j][i].y*csi[k][j][i].y + 
                                       csi[k][j][i].z*csi[k][j][i].z );
                    double dhi = 1.0/aj[k][j][i]/area;
                    area = sqrt(eta[k][j][i].x*eta[k][j][i].x + 
                                eta[k][j][i].y*eta[k][j][i].y + 
                                eta[k][j][i].z*eta[k][j][i].z );
                    double dhj = 1.0/aj[k][j][i]/area;
                    area = sqrt(zet[k][j][i].x*zet[k][j][i].x + 
                                zet[k][j][i].y*zet[k][j][i].y + 
                                zet[k][j][i].z*zet[k][j][i].z );
                    double dhk = 1.0/aj[k][j][i]/area;
                    filter = sqrt(dhi*dhi+dhj*dhj+dhk*dhk);
                }     
                else filter = pow( 1./aj[k][j][i], 1./3. );
        
                // xiaolei add filter_size==1
                if (d_testfilter_ik) test_filter = pow(5.0, 1./3.) * filter;
                else {
                    if (d_filter_size==1) 
                        test_filter = sqrt(dhi_test*dhi_test+
                                           dhj_test*dhj_test+
                                           dhk_test*dhk_test);
                    else test_filter = pow( sum_vol, 1./3. );
                }
                
                double _U=integrate_testfilter_simpson(U, weight);
                double _V=integrate_testfilter_simpson(V, weight);
                double _W=integrate_testfilter_simpson(W, weight);
        
                double _u=integrate_testfilter_simpson(u, weight);
                double _v=integrate_testfilter_simpson(v, weight);
                double _w=integrate_testfilter_simpson(w, weight);
                double _d=1;
                Lij[0][0] = integrate_testfilter_simpson(Uu, weight) - _U*_u;
                Lij[0][1] = integrate_testfilter_simpson(Uv, weight) - _U*_v;
                Lij[0][2] = integrate_testfilter_simpson(Uw, weight) - _U*_w;
                Lij[1][0] = integrate_testfilter_simpson(Vu, weight) - _V*_u;
                Lij[1][1] = integrate_testfilter_simpson(Vv, weight) - _V*_v;
                Lij[1][2] = integrate_testfilter_simpson(Vw, weight) - _V*_w;
                Lij[2][0] = integrate_testfilter_simpson(Wu, weight) - _W*_u;
                Lij[2][1] = integrate_testfilter_simpson(Wv, weight) - _W*_v;
                Lij[2][2] = integrate_testfilter_simpson(Ww, weight) - _W*_w;
                
                Sij_hat[0][0] = integrate_testfilter_simpson(S11, weight);    
                Sij_hat[0][1] = integrate_testfilter_simpson(S12, weight);    
                Sij_hat[0][2] = integrate_testfilter_simpson(S13, weight);
                Sij_hat[1][0] = integrate_testfilter_simpson(S21, weight);    
                Sij_hat[1][1] = integrate_testfilter_simpson(S22, weight);    
                Sij_hat[1][2] = integrate_testfilter_simpson(S23, weight);
                Sij_hat[2][0] = integrate_testfilter_simpson(S31, weight);    
                Sij_hat[2][1] = integrate_testfilter_simpson(S32, weight);    
                Sij_hat[2][2] = integrate_testfilter_simpson(S33, weight);
        
                S_hat=0;
                for (a=0; a<3; a++)
                    for (b=0; b<3; b++) {
                        S_hat += pow( Sij_hat[a][b], 2. );
                    }
                S_hat = sqrt ( 2 * S_hat );
        
                SSij_hat[0][0] = integrate_testfilter_simpson(SS11, weight);    
                SSij_hat[0][1] = integrate_testfilter_simpson(SS12, weight);    
                SSij_hat[0][2] = integrate_testfilter_simpson(SS13, weight);
                SSij_hat[1][0] = integrate_testfilter_simpson(SS21, weight);
                SSij_hat[1][1] = integrate_testfilter_simpson(SS22, weight);    
                SSij_hat[1][2] = integrate_testfilter_simpson(SS23, weight);
                SSij_hat[2][0] = integrate_testfilter_simpson(SS31, weight);    
                SSij_hat[2][1] = integrate_testfilter_simpson(SS32, weight);    
                SSij_hat[2][2] = integrate_testfilter_simpson(SS33, weight);
        
        
                double gg[3][3], ggc[3][3], G[3][3];
                double xcsi, xeta, xzet, ycsi, yeta, yzet, zcsi, zeta, zzet;

                gg[0][0]=csi0, gg[0][1]=csi1, gg[0][2]=csi2;
                gg[1][0]=eta0, gg[1][1]=eta1, gg[1][2]=eta2;
                gg[2][0]=zet0, gg[2][1]=zet1, gg[2][2]=zet2;
                Calculate_Covariant_metrics(gg, ggc);
                xcsi=ggc[0][0], xeta=ggc[0][1], xzet=ggc[0][2];
                ycsi=ggc[1][0], yeta=ggc[1][1], yzet=ggc[1][2];
                zcsi=ggc[2][0], zeta=ggc[2][1], zzet=ggc[2][2];
                G[0][0] = xcsi * xcsi + ycsi * ycsi + zcsi * zcsi;
                G[1][1] = xeta * xeta + yeta * yeta + zeta * zeta;
                G[2][2] = xzet * xzet + yzet * yzet + zzet * zzet;
                G[0][1] = G[1][0] = xeta * xcsi + yeta * ycsi + zeta * zcsi;
                G[0][2] = G[2][0] = xzet * xcsi + yzet * ycsi + zzet * zcsi;
                G[1][2] = G[2][1] = xeta * xzet + yeta * yzet + zeta * zzet;
        
                for (a=0; a<3; a++)
                    for (b=0; b<3; b++) {
                        Mij_cat[a][b] =-pow(test_filter,2.)*S_hat*Sij_hat[a][b]
                                       +pow(filter,2.)*SSij_hat[a][b];
                    }
        
                Mij[0][0] = Mij_cat[0][0] * csi0 + 
                            Mij_cat[0][1] * csi1 + 
                            Mij_cat[0][2] * csi2;
                Mij[0][1] = Mij_cat[0][0] * eta0 + 
                            Mij_cat[0][1] * eta1 + 
                            Mij_cat[0][2] * eta2;
                Mij[0][2] = Mij_cat[0][0] * zet0 + 
                            Mij_cat[0][1] * zet1 + 
                            Mij_cat[0][2] * zet2;
                Mij[1][0] = Mij_cat[1][0] * csi0 + 
                            Mij_cat[1][1] * csi1 + 
                            Mij_cat[1][2] * csi2;
                Mij[1][1] = Mij_cat[1][0] * eta0 + 
                            Mij_cat[1][1] * eta1 + 
                            Mij_cat[1][2] * eta2;
                Mij[1][2] = Mij_cat[1][0] * zet0 + 
                            Mij_cat[1][1] * zet1 + 
                            Mij_cat[1][2] * zet2;
                Mij[2][0] = Mij_cat[2][0] * csi0 + 
                            Mij_cat[2][1] * csi1 + 
                            Mij_cat[2][2] * csi2;
                Mij[2][1] = Mij_cat[2][0] * eta0 + 
                            Mij_cat[2][1] * eta1 + 
                            Mij_cat[2][2] * eta2;
                Mij[2][2] = Mij_cat[2][0] * zet0 + 
                            Mij_cat[2][1] * zet1 + 
                            Mij_cat[2][2] * zet2;
                    
                double num=0, num1=0, denom=0;
                int m, n, l;
        
    
                for (q=0; q<3; q++)
                    for (a=0; a<3; a++)
                        for (b=0; b<3; b++) {
                            num += Lij[b][a] * Mij[a][q] * G[b][q];
                        }
        
                for (m=0; m<3; m++)
                    for (n=0; n<3; n++)
                        for (l=0; l<3; l++) {
                            denom += Mij[n][m] * Mij[n][l] * G[m][l];
                        }
    
         
                LM[k][j][i] = num;
                MM[k][j][i] = denom;
            }
    
    DMDAVecRestoreArray(da, d_lLM, &LM);
    DMDAVecRestoreArray(da, d_lMM, &MM);


    DMDALocalToLocalBegin(da, d_lLM, INSERT_VALUES, d_lLM);
    DMDALocalToLocalEnd(da, d_lLM, INSERT_VALUES, d_lLM);
    DMDALocalToLocalBegin(da, d_lMM, INSERT_VALUES, d_lMM);
    DMDALocalToLocalEnd(da, d_lMM, INSERT_VALUES, d_lMM);

    // xiaolei add
    double ***LM_old, ***MM_old;
    PetscReal dt = d_data->getDt();
    PetscInt tistart = d_data->get_tistart(); 

    if (d_les==4) {
        if (ti<tistart+2) {
        
            DMDAVecGetArray(da, d_lLM, &LM);
            DMDAVecGetArray(da, d_lMM, &MM);
    
            DMDAVecGetArray(da, d_lLM_old, &LM_old);
            DMDAVecGetArray(da, d_lMM_old, &MM_old);
    
            for (k=lzs; k<lze; k++)
                for (j=lys; j<lye; j++)
                    for (i=lxs; i<lxe; i++) {
                        LM_old[k][j][i]=LM[k][j][i];
                        MM_old[k][j][i]=MM[k][j][i];
                    }

            DMDAVecRestoreArray(da, d_lLM, &LM);
            DMDAVecRestoreArray(da, d_lMM, &MM);

            DMDAVecRestoreArray(da, d_lLM_old, &LM_old);
            DMDAVecRestoreArray(da, d_lMM_old, &MM_old);

            DMDALocalToLocalBegin(da, d_lLM_old, INSERT_VALUES, d_lLM_old);
            DMDALocalToLocalEnd(da, d_lLM_old, INSERT_VALUES, d_lLM_old);
 
            DMDALocalToLocalBegin(da, d_lMM_old, INSERT_VALUES, d_lMM_old);
            DMDALocalToLocalEnd(da, d_lMM_old, INSERT_VALUES, d_lMM_old);

            VecSet(d_lCs, 0.01);

            return;

        } else {
    
            DMDAVecGetArray(da, d_lLM, &LM);
            DMDAVecGetArray(da, d_lMM, &MM);
    
            DMDAVecGetArray(da, d_lLM_old, &LM_old);
            DMDAVecGetArray(da, d_lMM_old, &MM_old);

            for (k=lzs; k<lze; k++)
                for (j=lys; j<lye; j++)
                    for (i=lxs; i<lxe; i++) {
                        LM_tmp[k][j][i]=LM[k][j][i];
                        MM_tmp[k][j][i]=MM[k][j][i];
                    }

    
            for (k=lzs; k<lze; k++)
                for (j=lys; j<lye; j++)
                    for (i=lxs; i<lxe; i++) {
                        double filter, T_scale;
                        filter = pow( 1./aj[k][j][i], 1./3. );
                        T_scale=1.5*filter*pow(fabs(LM_old[k][j][i]*
                                                    MM_old[k][j][i])+1.e-19,
                                                -1.0/8.0)+1.e-19;

                        double LM_new, MM_new;
                        LM_new=LM[k][j][i];
                        MM_new=MM[k][j][i];

                        Cmpnts X_new, X_old;

                        X_new=cent[k][j][i];

                        X_old.x=X_new.x-ucat[k][j][i].x*dt;
                        X_old.y=X_new.y-ucat[k][j][i].y*dt;
                        X_old.z=X_new.z-ucat[k][j][i].z*dt;

                        int i1, j1, k1;
                        double dmin=10.0e6, d;
                        int i_old, j_old, k_old;
        
                        i_old=i; j_old=j; k_old=k;
                        for (k1=k-2; k1<k+2; k1++)
                            for (j1=j-2; j1<j+2; j1++)
                                for (i1=i-2; i1<i+2; i1++) {
                                    if (k1>=lzs && k1<lze && j1>=lys && 
                                        j1<lye && i1>=lxs && i1<lxe) {

                                        d=pow((X_old.x-cent[k1][j1][i1].x),2)+
                                          pow((X_old.y-cent[k1][j1][i1].y),2)+
                                          pow((X_old.z-cent[k1][j1][i1].z),2);
                                        if (d<dmin) {
                                            dmin=d;
                                            i_old=i1; j_old=j1; k_old=k1;
                                        }    
                                    }
                                }
    
                        double _LM_old, _MM_old;

                        _LM_old=LM_old[k_old][j_old][i_old];
                        _MM_old=MM_old[k_old][j_old][i_old];


                        double fac1=1.0/(1.0+dt/T_scale);
                        double fac2=1.0-fac1;

                        LM[k][j][i]=fac1*_LM_old+fac2*LM_new;    
                        MM[k][j][i]=fac1*_MM_old+fac2*MM_new+1.e-19;    
                    }

    
    
            for (k=lzs; k<lze; k++)
                for (j=lys; j<lye; j++)
                    for (i=lxs; i<lxe; i++) {
                        LM_old[k][j][i]=LM_tmp[k][j][i];
                        MM_old[k][j][i]=MM_tmp[k][j][i];
                    }

            DMDAVecRestoreArray(da, d_lLM, &LM);
            DMDAVecRestoreArray(da, d_lMM, &MM);

            DMDAVecRestoreArray(da, d_lLM_old, &LM_old);
            DMDAVecRestoreArray(da, d_lMM_old, &MM_old);

            DMDALocalToLocalBegin(da, d_lLM_old, INSERT_VALUES, d_lLM_old);
            DMDALocalToLocalEnd(da, d_lLM_old, INSERT_VALUES, d_lLM_old);
 
            DMDALocalToLocalBegin(da, d_lMM_old, INSERT_VALUES, d_lMM_old);
            DMDALocalToLocalEnd(da, d_lMM_old, INSERT_VALUES, d_lMM_old);

        }    
    }

    
    DMDALocalToLocalBegin(da, d_lLM, INSERT_VALUES, d_lLM);
    DMDALocalToLocalEnd(da, d_lLM, INSERT_VALUES, d_lLM);
    DMDALocalToLocalBegin(da, d_lMM, INSERT_VALUES, d_lMM);
    DMDALocalToLocalEnd(da, d_lMM, INSERT_VALUES, d_lMM);
      
    DMDAVecGetArray(da, d_lLM, &LM);
    DMDAVecGetArray(da, d_lMM, &MM);
    
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                int c=k, b=j, a=i, flag=0;
        
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
                    LM[k][j][i] = LM[c][b][a];
                    MM[k][j][i] = MM[c][b][a];
                }
            }
        
    DMDAVecRestoreArray(da, d_lLM, &LM);
    DMDAVecRestoreArray(da, d_lMM, &MM);

    DMDAVecGetArray(da, d_lLM, &LM);
    DMDAVecGetArray(da, d_lMM, &MM);

    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
                if (nvert[k][j][i]>1.1) {
                    Cs[k][j][i] = 0;
                    continue;
                }
                
                double weight[3][3][3];
                double LM0[3][3][3], MM0[3][3][3];
                int a, b, c;
            
                for (a=-1; a<=1; a++)
                    for (b=-1; b<=1; b++)
                        for (c=-1; c<=1; c++) {
                            int R=c+1, Q=b+1, P=a+1;
                            int K=k+c, J=j+b, I=i+a;
            
                            weight[R][Q][P] = 1./aj[K][J][I];
            
                            if ( nvert[K][J][I]>1.1 ) weight[R][Q][P]=0;
            
                            if ( i_periodic ) {
                                if ( I==0 ) I=mx-2;
                                else if( I==mx-1 ) I=1;
                            }
                            else if( ii_periodic ) {
                                if( I==0 ) I=-2;
                                else if( I==mx-1 ) I=mx+1;
                            }
                            else if( I==0 || I==mx-1) weight[R][Q][P]=0;

                            if ( j_periodic ) {
                                if( J==0 ) J=my-2;
                                else if( J==my-1 ) J=1;
                            }
                            else if( jj_periodic ) {
                                if( J==0 ) J=-2;
                                else if( J==my-1 ) J=my+1;
                            }
                            else if( J==0 || j==my-1) weight[R][Q][P]=0;
            
                            if ( k_periodic ) {
                                if( K==0 ) K=mz-2;
                                else if( K==mz-1 ) K=1;
                            }
                            else if( kk_periodic ) {
                                if( K==0 ) K=-2;
                                else if( K==mz-1 ) K=mz+1;
                            }
                            else if( K==0 || K==mz-1) weight[R][Q][P]=0;
            
                            LM0[R][Q][P] = LM[K][J][I];
                            MM0[R][Q][P] = MM[K][J][I];
                        }
            
                double C=0;
                double LM_avg, MM_avg;

                if ( d_i_homo_filter || d_j_homo_filter || d_k_homo_filter || 
                     d_les==3 || d_les==4) { 

                    LM_avg = LM[k][j][i];
                    MM_avg = MM[k][j][i];
                } else {

                    if (d_grid->getBC(5)==4) {
                        LM_avg = (1.0*LM0[0][1][1] + 4.0*LM0[1][1][1] + 
                                  1.0*LM0[2][1][1]) / 6.;
                        MM_avg = (1.0*MM0[0][1][1] + 4.0*MM0[1][1][1] + 
                                  1.0*MM0[2][1][1]) / 6.;
                    } else {
                        LM_avg = integrate_testfilter_simpson(LM0,weight);
                        MM_avg = integrate_testfilter_simpson(MM0,weight);
                    }
                }
        
                C = 0.5 * LM_avg / (MM_avg + d_les_eps );
        
                if ( d_les==3 ) {
                    if (ti<100 && tistart==0 && !d_restart) { }
                    else {
                       C = (1.0 - 0.001) * Cs[k][j][i] + 0.001 * C;
                    }
                }
        
                if (d_les==1) Cs[k][j][i] = 0.01;
                else Cs[k][j][i] = PetscMax(C, 0);
            }
    
    if( d_les==3 || d_les==4) {}   
    else if ( d_i_homo_filter && d_k_homo_filter ) {
        std::vector<int> count, total_count;
        std::vector<double> J_LM(my), J_MM(my), LM_tmp(my), MM_tmp(my);
        
        count.resize(my);
        total_count.resize(my);
        
        for (j=0; j<my; j++) {
             LM_tmp[j] = 0;
             MM_tmp[j] = 0;
             count[j] = total_count[j] = 0;
        }
        
        for (k=lzs; k<lze; k++)
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {
                    if ( nvert[k][j][i]<0.1 ) {
                        LM_tmp[j] += LM[k][j][i];
                        MM_tmp[j] += MM[k][j][i];
                        count[j] ++;
                    }
                }
        
        MPI_Allreduce(&LM_tmp[0], &J_LM[0], my, MPI_DOUBLE, 
                      MPI_SUM, PETSC_COMM_WORLD);
        MPI_Allreduce(&MM_tmp[0], &J_MM[0], my, MPI_DOUBLE, 
                      MPI_SUM, PETSC_COMM_WORLD);
        MPI_Allreduce(&count[0], &total_count[0], my, MPI_INT, 
                      MPI_SUM, PETSC_COMM_WORLD);    
        
        for (j=0; j<my; j++) {
            if (total_count[j]>0) {
                J_LM[j] /= (double) (total_count[j]);
                J_MM[j] /= (double) (total_count[j]);
            }
        }
        
        for (j=lys; j<lye; j++)
            for (k=lzs; k<lze; k++)
                for (i=lxs; i<lxe; i++) {
                    Cs[k][j][i] = 0.5 * J_LM[j] / ( J_MM[j]+d_les_eps);
                }
    }
    else if (d_i_homo_filter || d_j_homo_filter || d_k_homo_filter) {
        int plane_size;
        
        if (d_i_homo_filter) plane_size = my*mz;
        else if(d_j_homo_filter) plane_size = mx*mz;
        else if(d_k_homo_filter) plane_size = mx*my;
        
        std::vector<int> count(plane_size), total_count(plane_size);
        std::vector<double> J_LM(plane_size), J_MM(plane_size), 
                            LM_tmp(plane_size), MM_tmp(plane_size);
        int pos;
        
        pos=0;
        
        std::fill(LM_tmp.begin(), LM_tmp.end(), 0.);
        std::fill(MM_tmp.begin(), MM_tmp.end(), 0.);
        std::fill(count.begin(), count.end(), 0);
        std::fill(total_count.begin(), total_count.end(), 0);
        
        for (pos=0; pos<plane_size; pos++) {
             LM_tmp[pos] = 0;
             MM_tmp[pos] = 0;
             count[pos] = total_count[pos] = 0;
        }
        
        pos=0;
        
        if (d_i_homo_filter)  {
            for (k=0; k<mz; k++)
                for (j=0; j<my; j++) {
                    if ( k>=lzs && k<lze && j>=lys && j<lye) {
                        for (i=lxs; i<lxe; i++) {
                            if ( nvert[k][j][i]<0.1 ) {
                                LM_tmp[pos] += LM[k][j][i];
                                MM_tmp[pos] += MM[k][j][i];
                                count[pos] ++;
                            }
                        }
                    }
                    pos++;
               }
        }
        else if(d_j_homo_filter)  {
            for (k=0; k<mz; k++)
                for (i=0; i<mx; i++) {
                    if ( i>=lxs && i<lxe && k>=lzs && k<lze) {
                        for (j=lys; j<lye; j++) {
                            if ( nvert[k][j][i]<0.1 ) {
                                LM_tmp[pos] += LM[k][j][i];
                                MM_tmp[pos] += MM[k][j][i];
                                count[pos] ++;
                            }
                        }
                    }
                    pos++;
            }
        }
        else if(d_k_homo_filter)  {
            for (j=0; j<my; j++)
                for (i=0; i<mx; i++) {
                    if ( i>=lxs && i<lxe && j>=lys && j<lye) {
                        for (k=lzs; k<lze; k++) {
                            if ( nvert[k][j][i]<0.1 ) {
                                LM_tmp[pos] += LM[k][j][i];
                                MM_tmp[pos] += MM[k][j][i];
                                count[pos] ++;
                            }
                        }
                    }
                    pos++;
                }
        }
        
        MPI_Allreduce(&LM_tmp[0], &J_LM[0], plane_size, 
                      MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
        MPI_Allreduce(&MM_tmp[0], &J_MM[0], plane_size, 
                      MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
        MPI_Allreduce(&count[0], &total_count[0], plane_size, 
                      MPI_INT, MPI_SUM, PETSC_COMM_WORLD);    
        
        pos=0;
        
        for (pos=0; pos<plane_size; pos++) {
            if ( total_count[pos]>0) {
                double N = (double) (total_count[pos]);
                J_LM[pos] /= N;
                J_MM[pos] /= N;
            }
        }
        
        pos=0;
        if (d_i_homo_filter)  {
            for (k=0; k<mz; k++)
                for (j=0; j<my; j++) {
                    if ( k>=lzs && k<lze && j>=lys && j<lye) {
                        for (i=lxs; i<lxe; i++) {
                            if ( nvert[k][j][i]<0.1 ) {
                                Cs[k][j][i] = 0.5*J_LM[pos]/(J_MM[pos]+d_les_eps);
                            }
                        }
                    }
                    pos++;
                }
        }
        else if(d_j_homo_filter)  {
            for (k=0; k<mz; k++)
                for (i=0; i<mx; i++) {
                    if ( i>=lxs && i<lxe && k>=lzs && k<lze) {
                        for (j=lys; j<lye; j++) {
                            if ( nvert[k][j][i]<0.1 ) {
                                Cs[k][j][i] = 0.5*J_LM[pos]/(J_MM[pos]+d_les_eps);
                            }
                        }
                    }
                    pos++;
                }
        }
        else if(d_k_homo_filter)  {
            for (j=0; j<my; j++)
                for (i=0; i<mx; i++) {
                    if ( i>=lxs && i<lxe && j>=lys && j<lye) {
                        for (k=lzs; k<lze; k++) {
                            if ( nvert[k][j][i]<0.1 ) {
                                Cs[k][j][i] = 0.5*J_LM[pos]/(J_MM[pos]+d_les_eps);
                            }
                        }
                    }
                    pos++;
                }
        }
    }
    
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
        
                if (nvert[k][j][i]>1.1 || k==0 || k==mz-1 
                    || j==0 || j==my-1 || i==0 || i==mx-1) {

                    Cs[k][j][i] = 0;
                } else {
                    if (nvert[k][j][i]>0.1 && nvert[k][j][i]<1.1) {
                       // stabilize at high Re, osl 0.005
                       Cs[k][j][i] = PetscMax(d_wall_cs, Cs[k][j][i]);    
                    }
                    Cs[k][j][i] = PetscMin(PetscMax(Cs[k][j][i], 0), d_max_cs);
                }
            }
    
    DMDAVecRestoreArray(fda, lSx, &Ax);
    DMDAVecRestoreArray(fda, lSy, &Ay);
    DMDAVecRestoreArray(fda, lSz, &Az);
    DMDAVecRestoreArray(da, lS, &Sabs);
    
    DMDAVecRestoreArray(da, d_lLM, &LM);
    DMDAVecRestoreArray(da, d_lMM, &MM);

    DMDAVecRestoreArray(fda, lUcont, &ucont);
    DMDAVecRestoreArray(fda, lUcat,  &ucat);
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(da, Aj, &aj); 
    DMDAVecRestoreArray(da, d_lCs, &Cs);
    
    DMDAVecRestoreArray(fda, Cent, &cent);
    VecDestroy(&d_lLM);
    VecDestroy(&d_lMM);
  
    VecDestroy(&lSx);
    VecDestroy(&lSy);
    VecDestroy(&lSz);
    VecDestroy(&lS);

    DMDAVecRestoreArray(da, lLM_tmp, &LM_tmp);
    DMDAVecRestoreArray(da, lMM_tmp, &MM_tmp);
    VecDestroy(&lLM_tmp); 
    VecDestroy(&lMM_tmp); 

    
    DMDAVecRestoreArray(da, IAj, &iaj);  
    DMDAVecRestoreArray(da, JAj, &jaj);  
    DMDAVecRestoreArray(da, KAj, &kaj);  
    
    DMDAVecRestoreArray(fda, ICsi, &icsi);
    DMDAVecRestoreArray(fda, IEta, &ieta);
    DMDAVecRestoreArray(fda, IZet, &izet);

    DMDAVecRestoreArray(fda, JCsi, &jcsi);
    DMDAVecRestoreArray(fda, JEta, &jeta);
    DMDAVecRestoreArray(fda, JZet, &jzet);

    DMDAVecRestoreArray(fda, KCsi, &kcsi);
    DMDAVecRestoreArray(fda, KEta, &keta);
    DMDAVecRestoreArray(fda, KZet, &kzet);
    
    DMDALocalToLocalBegin(da, d_lCs, INSERT_VALUES, d_lCs);
    DMDALocalToLocalEnd(da, d_lCs, INSERT_VALUES, d_lCs);
    
    DMDAVecGetArray(da, d_lCs, &Cs);
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

                if (flag) Cs[k][j][i] = Cs[c][b][a];
            }

    DMDAVecRestoreArray(da, d_lCs, &Cs);

    double lmax_norm=0, max_norm;
    PetscInt p;
    
    if (d_testfilter_ik) 
        PetscPrintf(PETSC_COMM_WORLD, "Filter type : Box filter homogeneous\n");
    else 
        PetscPrintf(PETSC_COMM_WORLD, "Filter type : Box filter 3D\n");
    
    VecMax(d_lCs, &p, &lmax_norm);
    GlobalMax_All(&lmax_norm, &max_norm, PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Max Cs  = %e \n", max_norm);

}

void LESModel::ComputeEddyViscosity()
{
    if (!d_use_les) return;
    int i, j, k;

    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    int xs, xe, ys, ye, zs, ze; // Local grid information
    int mx, my, mz; // Dimensions in three directions
    mx = info.mx, my = info.my, mz = info.mz;
    xs = info.xs, xe = xs + info.xm;
    ys = info.ys, ye = ys + info.ym;
    zs = info.zs, ze = zs + info.zm;

    int lxs = xs, lxe = xe;
    int lys = ys, lye = ye;
    int lzs = zs, lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    PetscInt i_periodic = d_grid->isIPeriodic();
    PetscInt j_periodic = d_grid->isJPeriodic();
    PetscInt k_periodic = d_grid->isKPeriodic();
    PetscInt ii_periodic = d_grid->isIIPeriodic();
    PetscInt jj_periodic = d_grid->isJJPeriodic();
    PetscInt kk_periodic = d_grid->isKKPeriodic();

    PetscReal ***Cs, ***lnu_t, ***nvert, ***aj, ***ustar;
    Cmpnts ***csi, ***eta, ***zet, ***ucat;
    
    VecSet(d_lNu_t, 0);

    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();

    Vec lUstar = d_data->getlUstar();
    Vec lUcat = d_data->getlUcat();
    Vec lNvert = d_data->getlNvert();

    DMDAVecGetArray(fda, lUcat,  &ucat);
    DMDAVecGetArray(da, lNvert, &nvert);
    DMDAVecGetArray(da, lUstar, &ustar);

    DMDAVecGetArray(da, d_lNu_t, &lnu_t);
    DMDAVecGetArray(da, d_lCs, &Cs);

    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, Aj, &aj);
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
                if (nvert[k][j][i]>1.1) {
                    lnu_t[k][j][i]=0;
                    continue;
                }

                double ajc = aj[k][j][i];
                double csi0 = csi[k][j][i].x;
                double csi1 = csi[k][j][i].y;
                double csi2 = csi[k][j][i].z;
                double eta0 = eta[k][j][i].x;
                double eta1 = eta[k][j][i].y;
                double eta2 = eta[k][j][i].z;
                double zet0 = zet[k][j][i].x;
                double zet1 = zet[k][j][i].y;
                double zet2 = zet[k][j][i].z;
                double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
                double du_dx, du_dy, du_dz;
                double dv_dx, dv_dy, dv_dz;
                double dw_dx, dw_dy, dw_dz;

                Compute_du_center(i, j, k, 
                                  mx, my, mz, 
                                  ucat, nvert,
                                  i_periodic, ii_periodic, j_periodic, 
                                  jj_periodic, k_periodic, kk_periodic, 
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
        
                double Sxx = 0.5*(du_dx + du_dx); 
                double Sxy = 0.5*(du_dy + dv_dx);
                double Sxz = 0.5*(du_dz + dw_dx);
                double Syx = Sxy;
                double Syy = 0.5*(dv_dy + dv_dy);
                double Syz = 0.5*(dv_dz + dw_dy);
                double Szx = Sxz;
                double Szy=Syz;
                double Szz = 0.5*(dw_dz + dw_dz);
    
                double Sabs = sqrt( 2.0*(Sxx*Sxx + Sxy*Sxy + Sxz*Sxz + 
                                         Syx*Syx + Syy*Syy + Syz*Syz + 
                                         Szx*Szx + Szy*Szy + Szz*Szz ) );

                // xiaolei add filter_size==1
                double filter;
                if (d_filter_size==1) {
                    double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + 
                                        csi[k][j][i].y*csi[k][j][i].y + 
                                        csi[k][j][i].z*csi[k][j][i].z );
                    double dhi = 1.0/aj[k][j][i]/area;
                    area = sqrt( eta[k][j][i].x*eta[k][j][i].x + 
                                 eta[k][j][i].y*eta[k][j][i].y + 
                                 eta[k][j][i].z*eta[k][j][i].z );
                    double dhj = 1.0/aj[k][j][i]/area;
                    area = sqrt( zet[k][j][i].x*zet[k][j][i].x + 
                                 zet[k][j][i].y*zet[k][j][i].y + 
                                 zet[k][j][i].z*zet[k][j][i].z );
                    double dhk = 1.0/aj[k][j][i]/area;
                    filter = sqrt(dhi*dhi+dhj*dhj+dhk*dhk);
                }    
                else filter = pow( 1./aj[k][j][i], 1./3. );

        
                lnu_t[k][j][i] = Cs[k][j][i] * pow ( filter, 2.0 ) * Sabs;

                if (d_wallfunction==2 && nvert[k][j][i]+nvert[k][j][i+1] + 
                                       nvert[k][j][i-1]+nvert[k][j+1][i] + 
                                       nvert[k][j-1][i]+nvert[k+1][j][i] + 
                                       nvert[k-1][j][i]>0.1) 
                    lnu_t[k][j][i]=0;
        
            }
    
    /*
    
     This was commented out because it did nothing ..
     Everything that affected lnu_t was commented out before

    if (d_immersed && d_wallfunction==2) {
        DMDAVecRestoreArray(da, d_lNu_t, &lnu_t);
        
        DMDALocalToLocalBegin(da, d_lNu_t, INSERT_VALUES, d_lNu_t);
        DMDALocalToLocalEnd(da, d_lNu_t, INSERT_VALUES, d_lNu_t);
        
        DMDAVecGetArray(da, d_lNu_t, &lnu_t);
        
        for(int ibi=0; ibi<d_NumberOfBodies; ibi++)
        {
            IBMNodes *ibm = d_ibm_ptr+ibi;
            
            IBMListNode *current;
            current = d_ibmlist[ibi].head;
            while (current) {

                IBMInfo *ibminfo = &current->ibm_intp;
                current = current->next;

                double sb = ibminfo->d_s, sc = sb + ibminfo->d_i;
                int ni = ibminfo->cell;
                int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
                int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
                int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
                i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;
                double sk1=ibminfo->cr1, sk2=ibminfo->cr2, sk3=ibminfo->cr3;
                double cs1=ibminfo->cs1, cs2=ibminfo->cs2, cs3=ibminfo->cs3;
                double nx=ibm->nf_x[ni], ny=ibm->nf_y[ni], nz=ibm->nf_z[ni];
                
                Cmpnts Ua, Uc;
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
                
                double nu = 1./d_data->getRe();
                double nu_t_c = (lnu_t[kp1][jp1][ip1] * sk1 + 
                                 lnu_t[kp2][jp2][ip2] * sk2 + 
                                 lnu_t[kp3][jp3][ip3] * sk3);
                double eps=1.e-5;
                double dUc_ds = u_Cabot(nu, sc+eps, ustar[k][j][i], 0) - 
                                u_Cabot(nu, sc-eps, ustar[k][j][i], 0);
                dUc_ds /= (2.0 * eps);
                double f1 = ustar[k][j][i] * ustar[k][j][i];
                double f2 = (nu+nu_t_c) * dUc_ds;
                                
                double u_c = Uc.x - Ua.x, v_c = Uc.y - Ua.y, w_c = Uc.z - Ua.z;
                double un = u_c * nx + v_c * ny + w_c * nz;
                double ut=u_c - un * nx, vt=v_c - un * ny, wt=w_c - un * nz;
                double ut_mag_c = sqrt( ut*ut + vt*vt + wt*wt );
                double ut_mag_b  = u_Cabot(nu, sb, ustar[k][j][i], 0);
  
                //lnu_t[k][j][i] = nu_t_c * sb / sc;
                //lnu_t[k][j][i] = - nu + (f1*(sc-sb) + f2*sb) * 
                                           (sc-sb) / sc / (ut_mag_c - ut_mag_b);
                
                
                //ut_mag_b =  ut_mag_c - (f1*(sc-sb) + f2*sb) * 
                                          (sc-sb) / sc / (lnu_t[k][j][i] + nu);
                //lnu_t[k][j][i] = near_wall_eddy_viscosity(yplus) / user->ren;
                
            }
        }
    }
    */ 
    
    DMDAVecRestoreArray(fda, lUcat,  &ucat);
    DMDAVecRestoreArray(da, lNvert, &nvert);
    DMDAVecRestoreArray(da, lUstar, &ustar);
    
    DMDAVecRestoreArray(da, d_lNu_t, &lnu_t);
    DMDAVecRestoreArray(da, d_lCs, &Cs);
   
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(da, Aj, &aj);
    
    DMDALocalToLocalBegin(da, d_lNu_t, INSERT_VALUES, d_lNu_t);
    DMDALocalToLocalEnd(da, d_lNu_t, INSERT_VALUES, d_lNu_t);
    
    if (d_grid->isPeriodic()) {
        DMDAVecGetArray(da, d_lNu_t, &lnu_t);
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) {
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

                    if (flag) {
                        lnu_t[k][j][i] = lnu_t[c][b][a];
                    }
                }
        DMDAVecRestoreArray(da, d_lNu_t, &lnu_t);
    }
    
    double lmax_norm=0;
    PetscInt p;
    
    VecMax(d_lNu_t, &p, &lmax_norm);
    GlobalMax_All(&lmax_norm, &d_max_norm, PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Max Nu_t = %e\n", d_max_norm);
    
}

PetscReal LESModel::integrate_testfilter_simpson(
    double val[3][3][3], 
    double w[3][3][3])
{
    double v1, v2, v3, v4, v5, v6, v7, v8;
    double w1, w2, w3, w4, w5, w6, w7, w8;

    if (d_testfilter_ik) {
            return integrate_testfilter_ik(val, w);
    }

    double wsum=0, valsum=0;

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            for (int k=0; k<3; k++) {
                double simpson_w = 1.0;
                if(i==1) simpson_w *= 4.;
                if(j==1) simpson_w *= 4.;
                if(k==1) simpson_w *= 4.;

                wsum += simpson_w * w[i][j][k];
                valsum += simpson_w * w[i][j][k] * val[i][j][k];
            }

    return valsum / wsum;
} 

PetscReal LESModel::integrate_testfilter_ik(
    double val[3][3][3], 
    double vol[3][3][3])
{
    // Simpson rule 
    // See Morinish and Vasilyev (2001) Phys Fluids
    // 2d homogeneous test filter :  pow(5.0, 1./3.)
    return  ( (val[0][1][0]+val[2][1][0]+val[0][1][2]+val[2][1][2]) + 
             4.*(val[0][1][1]+val[1][1][0]+val[2][1][1]+val[1][1][2]) + 
              16.*val[1][1][1] ) / 36.;

};




PetscErrorCode LESModel::ReadFromInput()
{
   PetscOptionsGetBool(PETSC_NULL, "-use_les", &d_use_les, PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL, "-les", &d_les, PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL,"-i_homo_filter",&d_i_homo_filter,PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL,"-j_homo_filter",&d_j_homo_filter,PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL,"-k_homo_filter",&d_k_homo_filter,PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL,"-testfilter_ik",&d_testfilter_ik,PETSC_NULL);
   PetscOptionsGetReal(PETSC_NULL,"-les_eps", &d_les_eps, PETSC_NULL);
   PetscOptionsGetReal(PETSC_NULL,"-max_cs", &d_max_cs, PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL, "-wallfunction", &d_wallfunction, 
                       PETSC_NULL);

   PetscOptionsGetInt(PETSC_NULL, "-rstart", &d_tistart, &d_restart);
   PetscOptionsGetString(PETSC_NULL,"-field_path",
                          d_fieldpath, 256, PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL, "-hdf5", &d_hdf5, PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL, "-write_hdf5", &d_write_hdf5, PETSC_NULL);
   PetscOptionsGetInt(PETSC_NULL, "-read_hdf5", &d_read_hdf5, PETSC_NULL);

   if (d_hdf5) {
       d_write_hdf5 = 1;
       d_read_hdf5 = 1;
    }
    if (d_read_hdf5) sprintf(d_rext, "h5");
    else sprintf(d_rext, "dat");
    if (d_write_hdf5) sprintf(d_wext, "h5");
    else sprintf(d_wext, "dat");
    PetscOptionsGetInt(PETSC_NULL, "-tio", &d_tiout, PETSC_NULL);
}
