#include "ImmersedBoundary.h"
 

ImmersedBoundary::ImmersedBoundary(
    const std::string& object_name,
    CurvGrid *grid,
    UData *data):
    d_object_name(object_name),
    d_grid(grid),
    d_data(data)
{

    sprintf(d_path, ".");

    d_IB_wm = 0;
    d_movefsi = 0;
    d_rotatefsi=0;
    d_rotatefsi_noIBsearch=0;
    d_changefsi = 0;
    d_thin = 0;
    d_immersed = 0;

    d_NumberOfBodies=0;
    d_NumberOfRotatingBodies=0;   
    d_cl = 1.0;
    d_CMx_c = 0.0;  
    d_CMy_c = 0.0;  
    d_CMz_c = 0.0;  

    d_averaging = 0;
    d_wallfunction = 0;
    d_roughness_size = 0; 
    ReadFromInput();

    if (d_immersed) 
        PetscMalloc(d_NumberOfBodies*sizeof(IBMNodes), &d_ibm);
        PetscMalloc(d_NumberOfBodies*sizeof(IBMList), &d_ibmlist);
        for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++) {
             InitIBMList(&d_ibmlist[ibi]);
        }


}

ImmersedBoundary::~ImmersedBoundary()
{
    if (d_immersed) {

        for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++) {
            IBMNodes *ibm = d_ibm+ibi;

            PetscFree(ibm->x_bp);
            PetscFree(ibm->y_bp);
            PetscFree(ibm->z_bp);

            PetscFree(ibm->x_bp0);
            PetscFree(ibm->y_bp0);
            PetscFree(ibm->z_bp0);

            PetscFree(ibm->x_bp_o);
            PetscFree(ibm->y_bp_o);
            PetscFree(ibm->z_bp_o);

            PetscFree(ibm->u);
            PetscFree(ibm->uold);
            PetscFree(ibm->urm1);

            PetscFree(ibm->cent_x);
            PetscFree(ibm->cent_y);
            PetscFree(ibm->cent_z);

            PetscFree(ibm->count);
            PetscFree(ibm->shear);
            PetscFree(ibm->mean_shear);
            PetscFree(ibm->reynolds_stress1);
            PetscFree(ibm->reynolds_stress2);
            PetscFree(ibm->reynolds_stress3);
            PetscFree(ibm->pressure);
            PetscFree(ibm->rel_velocity);
        }             

        PetscFree(d_ibm);
        PetscFree(d_ibmlist);
    }
}

PetscErrorCode ImmersedBoundary::IBMRead()
{
    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++) {
         ReadUCD(d_ibm+ibi, ibi);
    }

    return 0;
}

PetscErrorCode ImmersedBoundary::IBMWrite(PetscInt ti)
{
    if (!d_changefsi) return 0;

    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++) {
         WriteOutput1(d_ibm+ibi, ibi, ti);
    }

    return 0;
}


PetscErrorCode ImmersedBoundary::CopyLastStep()
{

    if (!d_changefsi) return 0;

    for (PetscInt ibi=0; ibi<d_NumberOfBodies;ibi++) {
        for (int i=0; i<d_ibm[ibi].n_v; i++) {
            d_ibm[ibi].x_bp_o[i] = d_ibm[ibi].x_bp[i];
            d_ibm[ibi].y_bp_o[i] = d_ibm[ibi].y_bp[i];
            d_ibm[ibi].z_bp_o[i] = d_ibm[ibi].z_bp[i];

            d_ibm[ibi].urm1[i].x = d_ibm[ibi].uold[i].x;
            d_ibm[ibi].urm1[i].y = d_ibm[ibi].uold[i].y;
            d_ibm[ibi].urm1[i].z = d_ibm[ibi].uold[i].z;

            d_ibm[ibi].uold[i].x = d_ibm[ibi].u[i].x;
            d_ibm[ibi].uold[i].y = d_ibm[ibi].u[i].y;
            d_ibm[ibi].uold[i].z = d_ibm[ibi].u[i].z;
        }
   }

   return 0;
}


PetscErrorCode ImmersedBoundary::IBMSearchAdvanced(PetscInt ti)
{
    for (PetscInt ibi=0; ibi<d_NumberOfBodies; ibi++) {
        IBMSearchAdvanced1(d_ibm+ibi, ibi, ti);
    }
}
   

PetscErrorCode ImmersedBoundary::IBMSearchAdvanced1(
    IBMNodes *ibm, PetscInt ibi, PetscInt ti)

/*   Note : Always go from ibi (immersed body number) 0 -> NumberOfBodies  */
/*        Nvert should be set to zero before any new search and this */
/*        happens if ibi==0--not anymore! set nvert=0 manually!*/
{
    PetscReal ts,te,cput;
    PetscTime(&ts);

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
    PetscInt ncx = 40, ncy = 40, ncz = 40;
    PetscInt ln_v, n_v = ibm->n_v;
    PetscInt n1e, n2e, n3e;
    PetscInt iv_min, iv_max, jv_min, jv_max, kv_min, kv_max;
    PetscInt ic, jc, kc;

    PetscReal xbp_min, ybp_min, zbp_min, xbp_max, ybp_max, zbp_max;
    PetscReal dcx, dcy, dcz;
    PetscReal xv_min, yv_min, zv_min, xv_max, yv_max, zv_max;
    PetscReal *x_bp = ibm->x_bp, *y_bp = ibm->y_bp, *z_bp = ibm->z_bp;
    PetscReal ***nvert, ***nvert_o, ***nvert_tmp;

    LIST *cell_trg;

    Vec Cent = d_grid->getlCent();

    Vec Nvert = d_data->getNvert();
    Vec lNvert = d_data->getlNvert();
    Vec lNvert_o = d_data->getlNvert_o();
    Vec lNvert_o_fixed = d_data->getlNvert_o_fixed(); //???

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;


    //Find max/min vertices
    xbp_min = 1.e23; xbp_max = -1.e23;
    ybp_min = 1.e23; ybp_max = -1.e23;
    zbp_min = 1.e23; zbp_max = -1.e23;

    for (i=0; i<n_v; i++) {
        xbp_min = PetscMin(xbp_min, x_bp[i]);
        xbp_max = PetscMax(xbp_max, x_bp[i]);

        ybp_min = PetscMin(ybp_min, y_bp[i]);
        ybp_max = PetscMax(ybp_max, y_bp[i]);

        zbp_min = PetscMin(zbp_min, z_bp[i]);
        zbp_max = PetscMax(zbp_max, z_bp[i]);
    }
 
    //Expand box ...Not best way to do this
    xbp_min -= 0.05; xbp_max += 0.05;
    ybp_min -= 0.05; ybp_max += 0.05;
    zbp_min -= 0.05; zbp_max += 0.05;

    //Discretized cell size basec on ncx = 40
    dcx = (xbp_max - xbp_min) / (ncx - 1.);
    dcy = (ybp_max - ybp_min) / (ncy - 1.);
    dcz = (zbp_max - zbp_min) / (ncz - 1.);

    //Make a list of size ncx**3 -> set at 40**3 (why?)
    //cell_trg holds the possible points to search
    PetscMalloc(ncz * ncy * ncx * sizeof(LIST), &cell_trg);
    for (k=0; k<ncz; k++) {
        for (j=0; j<ncy; j++) {
            for (i=0; i<ncx; i++) {
                initlist(&cell_trg[k*ncx*ncy + j*ncx + i]);
            }
        }
    }

    //Sort elements into a list based on location
    for (ln_v=0; ln_v < ibm->n_elmt; ln_v++) {

        n1e = ibm->nv1[ln_v]; n2e = ibm->nv2[ln_v]; n3e = ibm->nv3[ln_v];

        xv_min = PetscMin(PetscMin(x_bp[n1e], x_bp[n2e]), x_bp[n3e]);
        xv_max = PetscMax(PetscMax(x_bp[n1e], x_bp[n2e]), x_bp[n3e]);

        yv_min = PetscMin(PetscMin(y_bp[n1e], y_bp[n2e]), y_bp[n3e]);
        yv_max = PetscMax(PetscMax(y_bp[n1e], y_bp[n2e]), y_bp[n3e]);

        zv_min = PetscMin(PetscMin(z_bp[n1e], z_bp[n2e]), z_bp[n3e]);
        zv_max = PetscMax(PetscMax(z_bp[n1e], z_bp[n2e]), z_bp[n3e]);
    
        iv_min = floor((xv_min - xbp_min) / dcx); //  +1???
        iv_max = floor((xv_max - xbp_min) / dcx) +1;

        jv_min = floor((yv_min - ybp_min) / dcy); //  +1???
        jv_max = floor((yv_max - ybp_min) / dcy) +1;

        kv_min = floor((zv_min - zbp_min) / dcz); //  +1???
        kv_max = floor((zv_max - zbp_min) / dcz) +1;

        iv_min = (iv_min<0) ? 0:iv_min;
        iv_max = (iv_max>ncx) ? ncx:iv_max;

        jv_min = (jv_min<0) ? 0:jv_min;
        jv_max = (jv_max>ncx) ? ncy:jv_max;

        kv_min = (kv_min<0) ? 0:kv_min;
        kv_max = (kv_max>ncz) ? ncz:kv_max;

        // Insert IBM node information into a list
        for (k=kv_min; k<kv_max; k++) {
            for (j=jv_min; j<jv_max; j++) {
                for (i=iv_min; i<iv_max; i++) {
                     insertnode(&(cell_trg[k *ncx*ncy + j*ncx +i]), ln_v);
                }
            }
        }
    }

    int rank, flg=0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    Cmpnts ***coor;
    PetscReal ***nvert_o_fixed;
    DMDAVecGetArray(fda, Cent, &coor);
   
    PetscInt tistart = d_data->get_tistart(); 
    
    // for this body nvert 4 is inside, 2 is near bndry
    // for previous bodies nvert 3 inside, 1 near bndry
    // This is the beginning of ray-casting alg
    if (d_rotatefsi && ti>tistart && ibi >= d_NumberOfRotatingBodies) {
        DMDAVecGetArray(da, Nvert, &nvert);
        DMDAVecGetArray(da, lNvert_o_fixed, &nvert_o_fixed);
        for (k=lzs; k<lze; k++)
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {
                    nvert[k][j][i] = PetscMax (nvert_o_fixed[k][j][i], 
                                               nvert[k][j][i]);
                }
        DMDAVecRestoreArray(da, Nvert, &nvert);
        DMDAVecRestoreArray(da, lNvert_o_fixed, &nvert_o_fixed);
        
        DMGlobalToLocalBegin(da, Nvert, INSERT_VALUES, lNvert);
        DMGlobalToLocalEnd(da, Nvert, INSERT_VALUES, lNvert);
    } else {
        DMDAVecGetArray(da, Nvert, &nvert);
        DMDAVecGetArray(da, lNvert_o_fixed, &nvert_o_fixed);
        DMDAVecGetArray(da, lNvert_o, &nvert_o);
        for (k=lzs; k<lze; k++)
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {
            
                    double val=0;
        
                    PetscInt ip, im, jp, jm, kp, km;
                    PetscInt ii, jj, kk;    

                    ip = (i<mx-2?(i+1):(i));
                    im = (i>1   ?(i-1):(i));

                    jp = (j<my-2?(j+1):(j));
                    jm = (j>1   ?(j-1):(j));

                    kp = (k<mz-2?(k+1):(k));
                    km = (k>1   ?(k-1):(k));

                    int dosearch=0;
                    double nvert_nb;
                    nvert_nb=0.0;
                    for (kk=km; kk<=kp; kk++)
                        for (jj=jm; jj<=jp; jj++)
                            for (ii=im; ii<=ip; ii++) {
                                double sign = nvert_o[k][j][i]-
                                              nvert_o[kk][jj][ii];
                                if (fabs(sign)>1.e-6) dosearch+=1;
                            }

    
                    dosearch=1;

                    if (coor[k][j][i].x > xbp_min &&  
                        coor[k][j][i].x < xbp_max && 
                        coor[k][j][i].y > ybp_min && 
                        coor[k][j][i].y < ybp_max && 
                        coor[k][j][i].z > zbp_min && 
                        coor[k][j][i].z < zbp_max) {
                          
                        //Searching for points
                        if (dosearch || ti<=tistart) {
                            ic = floor((coor[k][j][i].x - xbp_min )/ dcx);
                            jc = floor((coor[k][j][i].y - ybp_min )/ dcy);
                            kc = floor((coor[k][j][i].z - zbp_min )/ dcz);                        
                            val = PointCellAdvanced(coor[k][j][i], 
                                                    ic, jc, kc, 
                                                    ibm, 
                                                    ncx, ncy, ncz, 
                                                    dcx, dcy, 
                                                    xbp_min,ybp_min,zbp_max, 
                                                    cell_trg, flg);

                            nvert[k][j][i] =  PetscMax(nvert[k][j][i], val); 

                            if (d_rotatefsi && ibi >= d_NumberOfRotatingBodies )  
                                // 2 or 4
                                nvert_o_fixed[k][j][i] = 
                                   PetscMax (nvert_o_fixed[k][j][i], val); 
                        } else {
                            nvert[k][j][i] = nvert_o[k][j][i]; 
                            if (int (nvert[k][j][i]+0.5) ==3) nvert[k][j][i]=4;  
                            if (int (nvert[k][j][i]+0.5) ==1) nvert[k][j][i]=0; 
                        }
                    }
                }

        DMDAVecRestoreArray(da, lNvert_o_fixed, &nvert_o_fixed);
        DMDAVecRestoreArray(da, Nvert, &nvert);
        DMDAVecRestoreArray(da, lNvert_o, &nvert_o);

        DMGlobalToLocalBegin(da, Nvert, INSERT_VALUES, lNvert);
        DMGlobalToLocalEnd(da, Nvert, INSERT_VALUES, lNvert);

        if (d_thin) {
            PetscPrintf(PETSC_COMM_WORLD, 
                        "IBM thin  %d %d %le %le %le %le %le %le\n", 
                        ibm->n_v, ibm->n_elmt, xbp_max, xbp_min, ybp_max,  
                         ybp_min, zbp_max, zbp_min);
            PetscInt cutthrough;
            
            DMDAVecGetArray(da, lNvert, &nvert);
            for (k=lzs; k<lze; k++)
                for (j=lys; j<lye; j++)
                    for (i=lxs; i<lxe; i++) {
                        if (coor[k][j][i].x > xbp_min && 
                            coor[k][j][i].x < xbp_max &&
                            coor[k][j][i].y > ybp_min && 
                            coor[k][j][i].y < ybp_max &&
                            coor[k][j][i].z > zbp_min && 
                            coor[k][j][i].z < zbp_max && 
                            (nvert[k][j][i] < 0.5 || 
                             nvert[k][j][i+1] < 0.5 || 
                             nvert[k][j+1][i] < 0.5 || 
                             nvert[k][j+1][i+1] < 0.5 || 
                             nvert[k+1][j][i] < 0.5 ||
                             nvert[k+1][j][i+1] < 0.5 || 
                             nvert[k+1][j+1][i] < 0.5 || 
                             nvert[k+1][j+1][i+1] < 0.5)) 
                        {

                            ic = floor((coor[k][j][i].x - xbp_min )/ dcx);
                            jc = floor((coor[k][j][i].y - ybp_min )/ dcy);
                            kc = floor((coor[k][j][i].z - zbp_min )/ dcz);

                            cutthrough = PointCellThin(
                                coor[k][j][i],coor[k][j][i+1],
                                coor[k][j+1][i],coor[k+1][j][i],
                                coor[k+1][j+1][i+1], 
                                ic, jc, kc, 
                                ibm, 
                                ncx, ncy, ncz, 
                                dcx, dcy, 
                                xbp_min, ybp_min, zbp_max, 
                                cell_trg, flg);

                            if (cutthrough) {
                                if (nvert[k][j][i] < 0.5) 
                                    nvert[k][j][i] = 2.;
                                if (nvert[k][j][i+1] < 0.5) 
                                    nvert[k][j][i+1] = 2.;
                                if (nvert[k][j+1][i] < 0.5) 
                                    nvert[k][j+1][i] = 2.;
                                if (nvert[k][j+1][i+1] < 0.5) 
                                    nvert[k][j+1][i+1] = 2.;
                                if (nvert[k+1][j][i] < 0.5) 
                                    nvert[k+1][j][i] = 2.;
                                if (nvert[k+1][j][i+1] < 0.5) 
                                    nvert[k+1][j][i+1] = 2.;
                                if (nvert[k+1][j+1][i] < 0.5) 
                                    nvert[k+1][j+1][i] = 2.;
                                if (nvert[k+1][j+1][i+1] < 0.5) 
                                    nvert[k+1][j+1][i+1] = 2.;
                            }
                        }
                    }

            DMDAVecRestoreArray(da, lNvert, &nvert);
        }
    }

    /*****************************/
    
    PetscInt ip, im, jp, jm, kp, km;
    PetscInt ii, jj, kk;
    
    DMDAVecGetArray(da, lNvert, &nvert);
    
    // Near boundary?
    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                if (nvert[k][j][i] < 0 ) nvert[k][j][i] = 0;

                ip = (i<mx-1?(i+1):(i));
                im = (i>0   ?(i-1):(i));

                jp = (j<my-1?(j+1):(j));
                jm = (j>0   ?(j-1):(j));

                kp = (k<mz-1?(k+1):(k));
                km = (k>0   ?(k-1):(k));

                if ((int)(nvert[k][j][i]+0.5) != 4) {
                    for (kk=km; kk<kp+1; kk++)
                        for (jj=jm; jj<jp+1; jj++)
                            for (ii=im; ii<ip+1; ii++) {
                                if ((int)(nvert[kk][jj][ii] +0.5) == 4) {
                                    nvert[k][j][i] = 
                                       PetscMax(2, nvert[k][j][i]);
                                }
                            }
                }
            }
    
    PetscBarrier(PETSC_NULL);
    PetscPrintf(PETSC_COMM_WORLD, "IBM Search: Point Search Finished\n");
   
    //Checking to see if moved from inside to outside 
    // with no change to boundary node first
    DMDAVecGetArray(da, lNvert_o, &nvert_o);
    if (ibi==d_NumberOfBodies-1) {
        for (k=zs; k<ze; k++)
            for (j=ys; j<ye; j++)
                for (i=xs; i<xe; i++) {
                    if (nvert_o[k][j][i] >2.5 && nvert[k][j][i] < 0.5) {
                        PetscPrintf(PETSC_COMM_SELF, 
                                    "Phase Change at %d, %d, %d!\n", i, j, k);
                        nvert[k][j][i]=2;
                    }
                }
    }
    DMDAVecRestoreArray(da, lNvert_o, &nvert_o);
    DMDAVecRestoreArray(da, lNvert, &nvert);
 
    //Scatter Nvert to procs 
    DMLocalToGlobalBegin(da, lNvert, INSERT_VALUES, Nvert);
    DMLocalToGlobalEnd(da, lNvert, INSERT_VALUES, Nvert);
        
    DMGlobalToLocalBegin(da, Nvert, INSERT_VALUES, lNvert);
    DMGlobalToLocalEnd(da, Nvert, INSERT_VALUES, lNvert);


    //Now find the interception point 
    if (d_ibmlist[ibi].head) DestroyIBMList(&d_ibmlist[ibi]);

    InitIBMList(&d_ibmlist[ibi]);
    PetscInt number = 0;

    IBMInfo ibm_intp;

    DMDAVecGetArray(da, lNvert, &nvert);

    BoundingSphere(ibm); 

    PetscReal ts1,te1,cput1, cput2; 

    cput1=0.0;
    cput2=0.0;

    for (k=lzs; k<lze; k++) {
        for (j=lys; j<lye; j++) {
            for (i=lxs; i<lxe; i++) {
                if ((int)(nvert[k][j][i]+0.5) == 2) {
                    number++;
                    ic = (int)((coor[k][j][i].x - xbp_min) / dcx);
                    jc = (int)((coor[k][j][i].y - ybp_min) / dcy);
                    kc = (int)((coor[k][j][i].z - zbp_min) / dcz);

                    if (ic<0) ic=0;
                    else if (ic>=ncx) ic=ncx-1;

                    if (jc<0) jc=0;
                    else if (jc>=ncy) jc = ncy-1;

                    if (kc<0) kc=0;
                    else if (kc>=ncz) kc = ncz-1;

                    ibm_intp.ni = i;
                    ibm_intp.nj = j;
                    ibm_intp.nk = k;

                    PetscTime(&ts1);
                    NearestCell(coor[k][j][i], ibm, &ibm_intp);
                    PetscTime(&te1);
                    cput1+=te1-ts1;
    
                    PetscTime(&ts1);
                    InterceptionPoint(coor[k][j][i], i, j, k, 
                                      &ibm_intp);
                    PetscTime(&te1);
                    cput2+=te1-ts1;

                    if (ibm_intp.imode<0) {
                        PetscInt cell;
                        Cmpnts ptmp;
                        if (i==1 || i==mx-2 ||
                            j==1 || j==my-2) {

                            cell = ibm_intp.cell;
                            if (ibm->nf_z[cell] > 0) {
                                ptmp = coor[k+1][j][i];
                                ibm_intp.d_i = 
                                  sqrt((coor[k][j][i].x - ptmp.x) *
                                       (coor[k][j][i].x - ptmp.x) +
                                       (coor[k][j][i].y - ptmp.y) *
                                       (coor[k][j][i].y - ptmp.y) +
                                       (coor[k][j][i].z - ptmp.z) *
                                       (coor[k][j][i].z - ptmp.z));
                                ibm_intp.cr1 = 1.;
                                ibm_intp.cr2 = 0.;
                                ibm_intp.cr3 = 0.;
                                ibm_intp.i1 = i;
                                ibm_intp.j1 = j;
                                ibm_intp.k1 = k+1;

                                ibm_intp.i2 = i;
                                ibm_intp.j2 = j;
                                ibm_intp.k2 = k+1;
                                ibm_intp.i3 = i;
                                ibm_intp.j3 = j;
                                ibm_intp.k3 = k+1;
                            } else {
                                ptmp = coor[k-1][j][i];
                                ibm_intp.d_i = 
                                  sqrt((coor[k][j][i].x - ptmp.x) *
                                       (coor[k][j][i].x - ptmp.x) +
                                       (coor[k][j][i].y - ptmp.y) *
                                       (coor[k][j][i].y - ptmp.y) +
                                       (coor[k][j][i].z - ptmp.z) *
                                       (coor[k][j][i].z - ptmp.z));
                                ibm_intp.cr1 = 1.;
                                ibm_intp.cr2 = 0.;
                                ibm_intp.cr3 = 0.;
                                ibm_intp.i1 = i;
                                ibm_intp.j1 = j;
                                ibm_intp.k1 = k-1;

                                ibm_intp.i2 = i;
                                ibm_intp.j2 = j;
                                ibm_intp.k2 = k-1;
                                ibm_intp.i3 = i;
                                ibm_intp.j3 = j;
                                ibm_intp.k3 = k-1;
                            }
                        } else if (k==1 || k==mz-2) {
                            cell = ibm_intp.cell;
                            ptmp = coor[k][j+1][i];
                            ibm_intp.d_i = sqrt((coor[k][j][i].x - ptmp.x) *
                                                (coor[k][j][i].x - ptmp.x) +
                                                (coor[k][j][i].y - ptmp.y) *
                                                (coor[k][j][i].y - ptmp.y) +
                                                (coor[k][j][i].z - ptmp.z) *
                                                (coor[k][j][i].z - ptmp.z));
                            ibm_intp.cr1 = 1.;
                            ibm_intp.cr2 = 0.;
                            ibm_intp.cr3 = 0.;
                            ibm_intp.i1 = i;
                            ibm_intp.j1 = j+1;
                            ibm_intp.k1 = k;
          
                            ibm_intp.i2 = i;
                            ibm_intp.j2 = j+1;
                            ibm_intp.k2 = k;
                            ibm_intp.i3 = i;
                            ibm_intp.j3 = j+1;
                            ibm_intp.k3 = k;
                        } else {
                             PetscPrintf(PETSC_COMM_SELF, 
                                         "...IBM Searching Fail! %d %d %d\n",
                                         i, j, k);
                             PetscPrintf(PETSC_COMM_SELF,"%d %d %d %d %f\n", mx, my, mz, ibm_intp.imode, ibm->nf_z[cell]);
                        }
                    }

                    AddIBMNode(&d_ibmlist[ibi], ibm_intp);
                }
            }
        }
    }

    PetscPrintf(PETSC_COMM_WORLD, "IBM_search: time nearest %le\n", cput1);
    PetscPrintf(PETSC_COMM_WORLD, "IBM_search: time intercept %le\n", cput2);
    PetscBarrier(PETSC_NULL);

    // Back to the old nvert 3 and 1 
    for (k=lzs; k<lze; k++) {
        for (j=lys; j<lye; j++) {
            for (i=lxs; i<lxe; i++) {
                if ((int)(nvert[k][j][i]+0.5) == 2) nvert[k][j][i]=1;
                if ((int)(nvert[k][j][i]+0.5) == 4) nvert[k][j][i]=3;
            }
        }
    }

    for (k=lzs; k<lze; k++) {
        for (j=lys; j<lye; j++) {
            for (i=lxs; i<lxe; i++) {
                if (d_grid->getBC(0)==-1 && i==1) nvert[k][j][i]=1;
                if (d_grid->getBC(1)==-1 && i==mx-2) nvert[k][j][i]=1;
                if (d_grid->getBC(2)==-1 && j==1) nvert[k][j][i]=1;
                if (d_grid->getBC(3)==-1 && j==my-2) nvert[k][j][i]=1;
                if (d_grid->getBC(4)==-1 && k==1) nvert[k][j][i]=1;
                if (d_grid->getBC(5)==-1 && k==mz-2) nvert[k][j][i]=1;

                if (d_grid->getBC(0)==-2 && i==1) nvert[k][j][i]=1;
                if (d_grid->getBC(1)==-2 && i==mx-2) nvert[k][j][i]=1;
                if (d_grid->getBC(2)==-2 && j==1) nvert[k][j][i]=1;
                if (d_grid->getBC(3)==-2 && j==my-2) nvert[k][j][i]=1;
                if (d_grid->getBC(4)==-2 && k==1) nvert[k][j][i]=1;
                if (d_grid->getBC(5)==-2 && k==mz-2) nvert[k][j][i]=1;
            }
        }
    }


    DMDAVecRestoreArray(fda, Cent,&coor);
    DMDAVecRestoreArray(da, lNvert, &nvert);
 
    DMLocalToGlobalBegin(da, lNvert, INSERT_VALUES, Nvert);
    DMLocalToGlobalEnd(da, lNvert, INSERT_VALUES, Nvert);
        
    DMGlobalToLocalBegin(da, Nvert, INSERT_VALUES, lNvert);
    DMGlobalToLocalEnd(da, Nvert, INSERT_VALUES, lNvert);

    for (k=0; k<ncz; k++) {
        for (j=0; j<ncy; j++) {
            for (i=0; i<ncx; i++) {
                destroy(&cell_trg[k*ncx*ncy+j*ncx+i]);
            }
        }
    }

    PetscFree(cell_trg);
    PetscFree(ibm->qvec);
    PetscFree(ibm->radvec); 

    PetscTime(&te);
    cput=te-ts;

    return 0;
}


PetscInt ImmersedBoundary::PointCellAdvanced(
    Cmpnts p, 
    PetscInt ip, PetscInt jp, PetscInt kp,
    IBMNodes *ibm, 
    PetscInt ncx, PetscInt ncy, PetscInt ncz,  
    PetscReal dcx, PetscReal dcy,
    PetscReal xbp_min, PetscReal ybp_min, 
    PetscReal zbp_max, 
   LIST *cell_trg, PetscInt flg)
{
    PetscInt i, j, k, ln_v, n1e, n2e, n3e, nintp;
    PetscInt nvert_l;
    PetscInt searchtimes=0;
    int *nv1 = ibm->nv1, *nv2 = ibm->nv2, *nv3 = ibm->nv3;

    PetscReal t, u, v;
    PetscReal epsilon = 1.e-8;
    PetscReal eps_tangent=1.e-10;
    PetscReal dt[1000], ndotn, dirdotn;
    PetscReal orig[3], dir[3], vert0[3], vert1[3], vert2[3];
    PetscReal *nf_x = ibm->nf_x, *nf_y = ibm->nf_y, *nf_z = ibm->nf_z;
    PetscReal *x_bp = ibm->x_bp, *y_bp = ibm->y_bp, *z_bp = ibm->z_bp;

    Cmpnts dnn[1000], nn;

    PetscBool NotDecided = PETSC_TRUE, Singularity = PETSC_FALSE;
    PetscBool *Element_Searched;

    node *current;

    j = jp; i = ip;

    PetscMalloc(ibm->n_elmt*sizeof(PetscBool), &Element_Searched);
    if (flg) 
        PetscPrintf(PETSC_COMM_SELF, " serch itr\n");

    while (NotDecided) {

        searchtimes++;
        nintp = 0 ;

        //Random ray dirction 
        randomdirection(p, ip, jp, 
                        xbp_min, ybp_min, zbp_max, 
                        dcx, dcy, 
                        dir, searchtimes);
        Singularity = PETSC_FALSE;
        if (flg) 
            PetscPrintf(PETSC_COMM_SELF, 
                        " serch itr, dir %d %le %le %le\n", 
                        searchtimes,dir[0],dir[1],dir[2]);

        //Mark not searched
        for (ln_v=0; ln_v<ibm->n_elmt; ln_v++) {
            Element_Searched[ln_v] = PETSC_FALSE;
        }

        for (k=kp; k<ncz; k++) {
            current = cell_trg[k*ncx*ncy+j*ncx+i].head;
            while (current) {
                ln_v = current->Node;
                if (!Element_Searched[ln_v]) {
                    Element_Searched[ln_v] = PETSC_TRUE;
                    n1e = nv1[ln_v]; n2e = nv2[ln_v]; n3e = nv3[ln_v];
                    nn.x=nf_x[ln_v]; nn.y=nf_y[ln_v]; nn.z=nf_z[ln_v];

                    orig[0] = p.x; orig[1] = p.y, orig[2] = p.z;

                    vert0[0]=x_bp[n1e]; vert0[1]=y_bp[n1e]; vert0[2]=z_bp[n1e];
                    vert1[0]=x_bp[n2e]; vert1[1]=y_bp[n2e]; vert1[2]=z_bp[n2e];
                    vert2[0]=x_bp[n3e]; vert2[1]=y_bp[n3e]; vert2[2]=z_bp[n3e];
            
                    dirdotn=dir[0]*nn.x+dir[1]*nn.y+dir[2]*nn.z;

                    nvert_l = intsect_triangle(orig, dir, 
                                               vert0, vert1, vert2, 
                                               &t, &u, &v);

                    if (flg) 
                        PetscPrintf(PETSC_COMM_SELF, 
                                   "elm, %d %d %le %le %le %d %d %d %le\n",
                                   ln_v,nvert_l,t,u,v,n1e,n2e,n3e,dirdotn);
      
                    if (nvert_l > 0 && t>0) {
                        dt[nintp] = t;

                        dnn[nintp].x=nn.x;
                        dnn[nintp].y=nn.y;
                        dnn[nintp].z=nn.z;

                        nintp ++;
                        PetscInt temp;
          // Two interception points are the same, this leads to huge
          // trouble for crossing number test
          // Rather to program for all cases, we use a new line to
          // repeat the test
                        for (temp = 0; temp < nintp-1; temp++) {
                            ndotn=dnn[temp].x*nn.x+
                                  dnn[temp].y*nn.y+
                                  dnn[temp].z*nn.z;          
          
                            if ((fabs(t-dt[temp]) < epsilon && ndotn>-0.95)){
                                Singularity = PETSC_TRUE;
                            }
                        }
                        if (Singularity) break;
                    }
                }
                if (Singularity) break;
                else current = current->next;
            } // Search through the list
            if (Singularity) break;
        } // for k
        if (flg) 
            PetscPrintf(PETSC_COMM_SELF, " serch itr, %d %le \n",
                                         nintp,dirdotn);

        if (!Singularity) {
            NotDecided = PETSC_TRUE;
            if (nintp%2) { //The interception point number is odd, inside body
                PetscFree(Element_Searched);
                return 4;
            } else {
                PetscFree(Element_Searched);
                return 0;
            }
        }
    }
    PetscFree(Element_Searched);
    return 0;
}

PetscInt ImmersedBoundary::PointCellThin(
    Cmpnts p,Cmpnts p1,Cmpnts p2,Cmpnts p3,Cmpnts p4,
    PetscInt ip, PetscInt jp, PetscInt kp,
    IBMNodes *ibm, 
    PetscInt ncx, PetscInt ncy, PetscInt ncz, 
    PetscReal dcx, PetscReal dcy,
    PetscReal xbp_min, PetscReal ybp_min, PetscReal zbp_max, 
    LIST *cell_trg,
    PetscInt flg)
{
    PetscInt i, j, k, ln_v;//, n1e, n2e, n3e, nintp;
    PetscBool cut;
    PetscInt  ks, js,is;

    ks=PetscMax(kp-1,0);
    js=PetscMax(jp-1,0);
    is=PetscMax(ip-1,0);

    i = ip; j = jp; k = kp;

    node *current;
    PetscBool *Element_Searched;
    PetscMalloc(ibm->n_elmt*sizeof(PetscBool), &Element_Searched);


    for (ln_v=0; ln_v<ibm->n_elmt; ln_v++) {
        Element_Searched[ln_v] = PETSC_FALSE;
    }

    if (flg)
        PetscPrintf(PETSC_COMM_SELF, "ip,jp,kp,nc_xyz %d %d %d %d %d %d\n",
                                     i,j,k,ncx,ncy,ncz);
  
    for (k=ks; k<kp+2 && k<ncz; k++) {
        for (j=js; j<jp+2 && j<ncy; j++) {
            for (i=is; i<ip+2 && i<ncx; i++) {
                current = cell_trg[k*ncx*ncy+j*ncx+i].head;
                while (current) {
                    ln_v = current->Node;
      
                    if (flg) PetscPrintf(PETSC_COMM_SELF, 
                                         "test010 ln_v %d \n",ln_v);
                    if (!Element_Searched[ln_v]) {
                        Element_Searched[ln_v] = PETSC_TRUE;
                        cut = ISLineTriangleIntp(p,p1,ibm,ln_v);
                        if (cut) {
                            PetscFree(Element_Searched);
                            return 2;
                        }
                        cut = ISLineTriangleIntp(p,p2,ibm,ln_v);
                        if (cut) {
                            PetscFree(Element_Searched);
                            return 2;
                        }
                        cut = ISLineTriangleIntp(p,p3,ibm,ln_v);
                        if (cut) {
                            PetscFree(Element_Searched);
                            return 2;
                        }
                        cut = ISLineTriangleIntp(p,p4,ibm,ln_v);
                        if (cut) {
                            PetscFree(Element_Searched);
                            return 2;
                        }
                    } //if
                    current = current->next;
                } //while
            }
        }
    }

    PetscFree(Element_Searched);  
    return 0;
}


/* Implementing the closest triangle algorithm described as the attached
   point-pairs.pdf */
PetscErrorCode ImmersedBoundary::BoundingSphere(IBMNodes *ibm)
{
    int *nv1 = ibm->nv1, *nv2 = ibm->nv2, *nv3 = ibm->nv3;
    PetscReal *x_bp = ibm->x_bp, *y_bp = ibm->y_bp, *z_bp = ibm->z_bp;

    PetscInt n_elmt = ibm->n_elmt;
    PetscInt ln_v;
  
    Cmpnts p1, p2, p3, p0;
  
    PetscInt n1e, n2e, n3e;
  
    p0.x = 0; p0.y = 0; p0.z = 0;

    PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->qvec));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->radvec));

    Cmpnts *qvec = ibm->qvec;
    PetscReal *radvec = ibm->radvec;

    Cmpnts pa, pb, pc, pu, pv, pf, pd, pt;
    PetscReal l12, l23, l31;
    PetscReal gama, lamda;
    for (ln_v = 0; ln_v < n_elmt; ln_v ++) {
        n1e = nv1[ln_v]; n2e = nv2[ln_v]; n3e = nv3[ln_v];

        p1.x = x_bp[n1e]; p1.y = y_bp[n1e]; p1.z = z_bp[n1e];
        p2.x = x_bp[n2e]; p2.y = y_bp[n2e]; p2.z = z_bp[n2e];
        p3.x = x_bp[n3e]; p3.y = y_bp[n3e]; p3.z = z_bp[n3e];

        l12 = Dist(p1, p2); l23 = Dist(p2, p3); l31 = Dist(p3, p1);

        /* Find the longest edge and assign the corresponding two vertices
          to pa and pb */
        if (l12 > l23) {
            if (l12 > l31) {
                pa = p1; pb = p2; pc = p3;
            } else {
                pa = p3; pb = p1; pc = p2;
            }
        } else {
            if (l31 < l23) {
                pa = p2; pb = p3; pc = p1;
            } else {
                pa = p3; pb = p1; pc = p2;
            }
        }

        pf.x = 0.5 * (pa.x + pb.x);
        pf.y = 0.5 * (pa.y + pb.y);
        pf.z = 0.5 * (pa.z + pb.z);

        // u = a - f; v = c - f;
        VecAMinusB(pu, pa, pf);
        VecAMinusB(pv, pc, pf);

        // d = (u X v) X u;
        Cross(pt, pu, pv);
        Cross(pd, pt, pu);

        // gama = (v^2 - u^2) / (2 d \dot (v - u));
        gama = -(Dist(pu, p0)*Dist(pu, p0) - Dist(pv, p0) * Dist(pv, p0));

        VecAMinusB(pt, pv, pu);
        lamda = 2 * (pd.x * pt.x + pd.y * pt.y + pd.z * pt.z);

        gama /= lamda;
    
        if (gama <0) {
            lamda = 0;
        } else {
           lamda = gama;
        }
    
        qvec[ln_v].x = pf.x + lamda * pd.x;
        qvec[ln_v].y = pf.y + lamda * pd.y;
        qvec[ln_v].z = pf.z + lamda * pd.z;

        radvec[ln_v] = Dist(qvec[ln_v], pa);
    }

    return 0;
}


PetscErrorCode ImmersedBoundary::NearestCell(
    Cmpnts p, IBMNodes *ibm, IBMInfo *ibminfo)
{
    int *nv1 = ibm->nv1, *nv2 = ibm->nv2, *nv3 = ibm->nv3;
    PetscInt n_elmt = ibm->n_elmt;
    PetscInt cell_min;
    PetscInt ln_v;
    PetscInt n1e, n2e, n3e;

    PetscReal tf;
    PetscReal dmin, d;
    PetscReal d_center;
    PetscReal nfx, nfy, nfz;
    PetscReal  *nf_x = ibm->nf_x, *nf_y = ibm->nf_y, *nf_z = ibm->nf_z;
    PetscReal *x_bp = ibm->x_bp, *y_bp = ibm->y_bp, *z_bp = ibm->z_bp;

    Cmpnts p1, p2, p3;
    Cmpnts pj; // projection point
    Cmpnts pmin, po;

    dmin = 1.e20;
    cell_min = -100;

    for (ln_v=0; ln_v<n_elmt; ln_v++) {
        d_center = Dist(p, ibm->qvec[ln_v]);
        if (d_center - ibm->radvec[ln_v] < dmin) {
            n1e = nv1[ln_v]; n2e = nv2[ln_v]; n3e = nv3[ln_v];
            nfx = nf_x[ln_v];
            nfy = nf_y[ln_v];
            nfz = nf_z[ln_v];

            p1.x = x_bp[n1e]; p1.y = y_bp[n1e]; p1.z = z_bp[n1e];
            p2.x = x_bp[n2e]; p2.y = y_bp[n2e]; p2.z = z_bp[n2e];
            p3.x = x_bp[n3e]; p3.y = y_bp[n3e]; p3.z = z_bp[n3e];

            tf = ((p.x - x_bp[n1e]) * nfx +
                  (p.y - y_bp[n1e]) * nfy +
                  (p.z - z_bp[n1e]) * nfz);
            if (fabs(tf) < 1.e-10) tf = 1.e-10;
            // Point p locates on the positive side of surface triangle
            if (tf>=0) {
                pj.x = p.x - tf * nfx;
                pj.y = p.y - tf * nfy;
                pj.z = p.z - tf * nfz;
                // The projected point is inside the triangle 
                if (ISPointInTriangle(pj, p1, p2, p3, nfx, nfy, nfz) == 1) { 

                    if (tf < dmin) {
                        dmin = tf;
                        pmin.x = pj.x;
                        pmin.y = pj.y;
                        pmin.z = pj.z;
                        cell_min = ln_v;
                    }
                } else {
                    Dis_P_Line(p, p1, p2, &po, &d);
                    if (d < dmin) {
                        dmin = d;
                        pmin.x = po.x;
                        pmin.y = po.y;
                        pmin.z = po.z;

                        cell_min = ln_v;
                    }
                    Dis_P_Line(p, p2, p3, &po, &d);
                    if (d < dmin) {
                        dmin = d;
                        pmin.x = po.x;
                        pmin.y = po.y;
                        pmin.z = po.z;

                        cell_min = ln_v;
                    }
                    Dis_P_Line(p, p3, p1, &po, &d);
                    if (d < dmin) {
                        dmin = d;
                        pmin.x = po.x;
                        pmin.y = po.y;
                        pmin.z = po.z;
                        cell_min = ln_v;
                    }      
                }
            }
        }
    }

    if (cell_min == -100) {
        PetscPrintf(PETSC_COMM_SELF, "Nearest Cell Searching Error!\n");
        exit(0);
     }
  
    ibminfo->cell = cell_min;
    ibminfo->pmin = pmin;
    ibminfo->d_s = dmin;

    Cpt2D pjp, pj1, pj2, pj3;
    nfx = nf_x[cell_min]; nfy = nf_y[cell_min]; nfz=nf_z[cell_min];

    n1e = nv1[cell_min]; n2e = nv2[cell_min]; n3e = nv3[cell_min];
    p1.x = x_bp[n1e]; p1.y = y_bp[n1e]; p1.z = z_bp[n1e];
    p2.x = x_bp[n2e]; p2.y = y_bp[n2e]; p2.z = z_bp[n2e];
    p3.x = x_bp[n3e]; p3.y = y_bp[n3e]; p3.z = z_bp[n3e];

    if (fabs(nfx) >= fabs(nfy) && fabs(nfx)>= fabs(nfz)) {
        pjp.x = pmin.y; pjp.y = pmin.z;
        pj1.x = p1.y;   pj1.y = p1.z;
        pj2.x = p2.y;   pj2.y = p2.z;
        pj3.x = p3.y;   pj3.y = p3.z;
        triangle_intp2(pjp, pj1, pj2, pj3, ibminfo);
    } else if (fabs(nfy) >= fabs(nfx) && fabs(nfy)>= fabs(nfz)) {
        pjp.x = pmin.x; pjp.y = pmin.z;
        pj1.x = p1.x;   pj1.y = p1.z;
        pj2.x = p2.x;   pj2.y = p2.z;
        pj3.x = p3.x;   pj3.y = p3.z;
        triangle_intp2(pjp, pj1, pj2, pj3, ibminfo);
    } else if (fabs(nfz) >= fabs(nfy) && fabs(nfz)>= fabs(nfx)) {
        pjp.x = pmin.y; pjp.y = pmin.x;
        pj1.x = p1.y;   pj1.y = p1.x;
        pj2.x = p2.y;   pj2.y = p2.x;
        pj3.x = p3.y;   pj3.y = p3.x;
        triangle_intp2(pjp, pj1, pj2, pj3, ibminfo);
    }
    if (ibminfo->cs1 != ibminfo->cs1) {
        PetscPrintf(PETSC_COMM_SELF, "INTP2 %e %e %e %i %i %i\n", 
                    nfx, nfy, nfz, n1e, n2e, n3e);
    }
    return 0;
}


PetscErrorCode ImmersedBoundary::ICP(
    Cmpnts p, Cmpnts pc[9], 
    PetscReal nfx, PetscReal nfy, PetscReal nfz, 
    IBMInfo *ibminfo, 
    PetscInt *ip, PetscInt *jp  , PetscInt *kp)
{
    PetscInt triangles[3][8];
    Cmpnts p1, p2, p3;

    PetscReal dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, d;
    PetscReal rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3;

    Cpt2D pj1, pj2, pj3, pjp;
    PetscInt cell, flag;

    PetscInt i;
    Cmpnts pint; // Interception point
    PetscReal nfxt, nfyt, nfzt;

    ibminfo->imode = -100;

    triangles[0][0] = 0; triangles[1][0] = 1; triangles[2][0] = 4;
    triangles[0][1] = 1; triangles[1][1] = 2; triangles[2][1] = 4;
    triangles[0][2] = 2; triangles[1][2] = 4; triangles[2][2] = 5;
    triangles[0][3] = 4; triangles[1][3] = 5; triangles[2][3] = 8;
    triangles[0][4] = 4; triangles[1][4] = 7; triangles[2][4] = 8;
    triangles[0][5] = 4; triangles[1][5] = 6; triangles[2][5] = 7;
    triangles[0][6] = 3; triangles[1][6] = 4; triangles[2][6] = 6;
    triangles[0][7] = 3; triangles[1][7] = 4; triangles[2][7] = 0;

    for (i=0; i<8; i++) {
        p1 = pc[triangles[0][i]]; 
        p2 = pc[triangles[1][i]], 
        p3 = pc[triangles[2][i]];

        dx1 = p.x - p1.x; dy1 = p.y - p1.y; dz1 = p.z - p1.z;
        dx2 = p2.x - p1.x; dy2 = p2.y - p1.y; dz2 = p2.z - p1.z;
        dx3 = p3.x - p1.x; dy3 = p3.y - p1.y; dz3 = p3.z - p1.z;

        d = (nfx * (dy2 * dz3 - dz2 * dy3) - 
             nfy * (dx2 * dz3 - dz2 * dx3) + 
             nfz * (dx2 * dy3 - dy2 * dx3));
        if (fabs(d) > 1.e-10) {
            d = -(dx1 * (dy2 * dz3 - dz2 * dy3) - 
                  dy1 * (dx2 * dz3 - dz2 * dx3) + 
                  dz1 * (dx2 * dy3 - dy2 * dx3)) / d;
      

            if (d>0) {
                pint.x = p.x + d * nfx;
                pint.y = p.y + d * nfy;
                pint.z = p.z + d * nfz;

                rx1 = p2.x - p1.x; ry1 = p2.y - p1.y; rz1 = p2.z - p1.z;
                rx2 = p3.x - p1.x; ry2 = p3.y - p1.y; rz2 = p3.z - p1.z;
      
                nfxt = ry1 * rz2 - rz1 * ry2;
                nfyt = -rx1 * rz2 + rz1 * rx2;
                nfzt = rx1 * ry2 - ry1 * rx2;

                flag = ISPointInTriangle(pint, p1, p2, p3, nfxt, nfyt, nfzt);
                if (flag >= 0) {
                    cell = i;

                    if (fabs(nfxt) >= fabs(nfyt) && fabs(nfxt)>= fabs(nfzt)) {
                        pjp.x = pint.y; pjp.y = pint.z;
                        pj1.x = p1.y;   pj1.y = p1.z;
                        pj2.x = p2.y;   pj2.y = p2.z;
                        pj3.x = p3.y;   pj3.y = p3.z;
                        triangle_intp(pjp, pj1, pj2, pj3, ibminfo);
                    }
                    else if(fabs(nfyt)>=fabs(nfxt) && fabs(nfyt)>=fabs(nfzt)) {
                        pjp.x = pint.x; pjp.y = pint.z;
                        pj1.x = p1.x;   pj1.y = p1.z;
                        pj2.x = p2.x;   pj2.y = p2.z;
                        pj3.x = p3.x;   pj3.y = p3.z;
                        triangle_intp(pjp, pj1, pj2, pj3, ibminfo);
                    }
                    else if(fabs(nfzt)>=fabs(nfyt) && fabs(nfzt)>fabs(nfxt)) {
                        pjp.x = pint.y; pjp.y = pint.x;
                        pj1.x = p1.y;   pj1.y = p1.x;
                        pj2.x = p2.y;   pj2.y = p2.x;
                        pj3.x = p3.y;   pj3.y = p3.x;
                        triangle_intp(pjp, pj1, pj2, pj3, ibminfo);
                    }

                    ibminfo->d_i = sqrt((pint.x-p.x)*(pint.x - p.x) + 
                                        (pint.y-p.y) * (pint.y-p.y) + 
                                        (pint.z - p.z)* (pint.z - p.z));
                    ibminfo->imode = cell;

                    return 0;
                }
            }
        }
    }
    return 0;
}

PetscErrorCode ImmersedBoundary::InterceptionPoint(Cmpnts p, 
                                 PetscInt i, PetscInt j, PetscInt k,
                                 IBMInfo *ibminfo)
{
    //Get DMs    
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    PetscInt ip[9], jp[9], kp[9];
    Cmpnts pc[9];

    PetscInt nif;

    PetscReal nfx, nfy, nfz;
    PetscReal dr;
    Vec Coor;
    Vec Cent = d_grid->getlCent();
    Cmpnts ***coor;

    DMDAVecGetArray(fda, Cent, &coor);

    nfx = p.x - ibminfo->pmin.x;
    nfy = p.y - ibminfo->pmin.y;
    nfz = p.z - ibminfo->pmin.z;

    dr = sqrt(nfx*nfx + nfy*nfy + nfz*nfz);
    nfx /= dr; nfy /= dr; nfz /=dr;

    ip[0] = i-1; ip[1] = i-1; ip[2] = i-1;
    ip[3] = i-1; ip[4] = i-1; ip[5] = i-1;
    ip[6] = i-1; ip[7] = i-1; ip[8] = i-1;

    jp[0] = j-1; jp[3] = j-1; jp[6] = j-1;
    jp[1] = j;   jp[4] = j;   jp[7] = j;
    jp[2] = j+1; jp[5] = j+1; jp[8] = j+1;

    kp[0] = k-1; kp[1] = k-1; kp[2] = k-1;
    kp[3] = k;   kp[4] = k;   kp[5] = k;
    kp[6] = k+1; kp[7] = k+1; kp[8] = k+1;

    for (nif=0; nif<9; nif++) {
        pc[nif].x = coor[kp[nif]][jp[nif]][ip[nif]].x;
        pc[nif].y = coor[kp[nif]][jp[nif]][ip[nif]].y;
        pc[nif].z = coor[kp[nif]][jp[nif]][ip[nif]].z;
      
    }

    ICP(p, pc, nfx, nfy, nfz, ibminfo, ip, jp, kp);
    switch (ibminfo->imode) {
        case(0): {
            ibminfo->i1=ip[0]; ibminfo->j1 = jp[0]; ibminfo->k1 = kp[0];
            ibminfo->i2=ip[1]; ibminfo->j2 = jp[1]; ibminfo->k2 = kp[1];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (1): {
            ibminfo->i1=ip[1]; ibminfo->j1 = jp[1]; ibminfo->k1 = kp[1];
            ibminfo->i2=ip[2]; ibminfo->j2 = jp[2]; ibminfo->k2 = kp[2];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (2): {
            ibminfo->i1=ip[2]; ibminfo->j1 = jp[2]; ibminfo->k1 = kp[2];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[5]; ibminfo->j3 = jp[5]; ibminfo->k3 = kp[5];
            break;
        }
        case (3): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[5]; ibminfo->j2 = jp[5]; ibminfo->k2 = kp[5];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (4): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[7]; ibminfo->j2 = jp[7]; ibminfo->k2 = kp[7];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (5): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[6]; ibminfo->j2 = jp[6]; ibminfo->k2 = kp[6];
            ibminfo->i3=ip[7]; ibminfo->j3 = jp[7]; ibminfo->k3 = kp[7];
            break;
        }
        case (6): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[6]; ibminfo->j3 = jp[6]; ibminfo->k3 = kp[6];
            break;
        }
        case (7): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[0]; ibminfo->j3 = jp[0]; ibminfo->k3 = kp[0];
            break;
        }
    }

    if (ibminfo->imode >=0) {
        DMDAVecRestoreArray(fda, Cent, &coor);  
        return 0;
    }

    ip[0] = i+1; ip[1] = i+1; ip[2] = i+1;
    ip[3] = i+1; ip[4] = i+1; ip[5] = i+1;
    ip[6] = i+1; ip[7] = i+1; ip[8] = i+1;

    jp[0] = j-1; jp[3] = j-1; jp[6] = j-1;
    jp[1] = j;   jp[4] = j;   jp[7] = j;
    jp[2] = j+1; jp[5] = j+1; jp[8] = j+1;

    kp[0] = k-1; kp[1] = k-1; kp[2] = k-1;
    kp[3] = k;   kp[4] = k;   kp[5] = k;
    kp[6] = k+1; kp[7] = k+1; kp[8] = k+1;

    for (nif=0; nif<9; nif++) {
        pc[nif].x = coor[kp[nif]][jp[nif]][ip[nif]].x;
        pc[nif].y = coor[kp[nif]][jp[nif]][ip[nif]].y;
        pc[nif].z = coor[kp[nif]][jp[nif]][ip[nif]].z;
    }

    ICP(p, pc, nfx, nfy, nfz, ibminfo, ip, jp, kp);

    switch (ibminfo->imode) {
        case(0): {
            ibminfo->i1=ip[0]; ibminfo->j1 = jp[0]; ibminfo->k1 = kp[0];
            ibminfo->i2=ip[1]; ibminfo->j2 = jp[1]; ibminfo->k2 = kp[1];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (1): {
            ibminfo->i1=ip[1]; ibminfo->j1 = jp[1]; ibminfo->k1 = kp[1];
            ibminfo->i2=ip[2]; ibminfo->j2 = jp[2]; ibminfo->k2 = kp[2];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (2): {
            ibminfo->i1=ip[2]; ibminfo->j1 = jp[2]; ibminfo->k1 = kp[2];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[5]; ibminfo->j3 = jp[5]; ibminfo->k3 = kp[5];
            break;
        }
        case (3): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[5]; ibminfo->j2 = jp[5]; ibminfo->k2 = kp[5];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (4): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[7]; ibminfo->j2 = jp[7]; ibminfo->k2 = kp[7];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (5): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[6]; ibminfo->j2 = jp[6]; ibminfo->k2 = kp[6];
            ibminfo->i3=ip[7]; ibminfo->j3 = jp[7]; ibminfo->k3 = kp[7];
            break;
        }
        case (6): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[6]; ibminfo->j3 = jp[6]; ibminfo->k3 = kp[6];
            break;
        }
        case (7): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[0]; ibminfo->j3 = jp[0]; ibminfo->k3 = kp[0];
            break;
        }
    }

    if (ibminfo->imode >=0) {
        DMDAVecRestoreArray(fda, Cent, &coor);
        return 0;
    }


    ip[0] = i-1; ip[1] = i  ; ip[2] = i+1;
    ip[3] = i-1; ip[4] = i  ; ip[5] = i+1;
    ip[6] = i-1; ip[7] = i  ; ip[8] = i+1;

    jp[0] = j-1; jp[3] = j-1; jp[6] = j-1;
    jp[1] = j-1; jp[4] = j-1; jp[7] = j-1;
    jp[2] = j-1; jp[5] = j-1; jp[8] = j-1;

    kp[0] = k-1; kp[1] = k-1; kp[2] = k-1;
    kp[3] = k;   kp[4] = k;   kp[5] = k;
    kp[6] = k+1; kp[7] = k+1; kp[8] = k+1;

    for (nif=0; nif<9; nif++) {
         pc[nif].x = coor[kp[nif]][jp[nif]][ip[nif]].x;
         pc[nif].y = coor[kp[nif]][jp[nif]][ip[nif]].y;
         pc[nif].z = coor[kp[nif]][jp[nif]][ip[nif]].z;
    }

    ICP(p, pc, nfx, nfy, nfz, ibminfo, ip, jp, kp);
    switch (ibminfo->imode) {
        case(0): {
            ibminfo->i1=ip[0]; ibminfo->j1 = jp[0]; ibminfo->k1 = kp[0];
            ibminfo->i2=ip[1]; ibminfo->j2 = jp[1]; ibminfo->k2 = kp[1];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (1): {
            ibminfo->i1=ip[1]; ibminfo->j1 = jp[1]; ibminfo->k1 = kp[1];
            ibminfo->i2=ip[2]; ibminfo->j2 = jp[2]; ibminfo->k2 = kp[2];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (2): {
            ibminfo->i1=ip[2]; ibminfo->j1 = jp[2]; ibminfo->k1 = kp[2];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[5]; ibminfo->j3 = jp[5]; ibminfo->k3 = kp[5];
            break;
        }
        case (3): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[5]; ibminfo->j2 = jp[5]; ibminfo->k2 = kp[5];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (4): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[7]; ibminfo->j2 = jp[7]; ibminfo->k2 = kp[7];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
            case (5): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[6]; ibminfo->j2 = jp[6]; ibminfo->k2 = kp[6];
            ibminfo->i3=ip[7]; ibminfo->j3 = jp[7]; ibminfo->k3 = kp[7];
            break;
        }
        case (6): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[6]; ibminfo->j3 = jp[6]; ibminfo->k3 = kp[6];
            break;
        }
        case (7): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[0]; ibminfo->j3 = jp[0]; ibminfo->k3 = kp[0];
            break;
        }
    }
    if (ibminfo->imode >=0) {
        DMDAVecRestoreArray(fda, Cent, &coor);
        return 0;
    }

    ip[0] = i-1; ip[1] = i  ; ip[2] = i+1;
    ip[3] = i-1; ip[4] = i  ; ip[5] = i+1;
    ip[6] = i-1; ip[7] = i  ; ip[8] = i+1;

    jp[0] = j+1; jp[3] = j+1; jp[6] = j+1;
    jp[1] = j+1; jp[4] = j+1; jp[7] = j+1;
    jp[2] = j+1; jp[5] = j+1; jp[8] = j+1;

    kp[0] = k-1; kp[1] = k-1; kp[2] = k-1;
    kp[3] = k;   kp[4] = k;   kp[5] = k;
    kp[6] = k+1; kp[7] = k+1; kp[8] = k+1;

    for (nif=0; nif<9; nif++) {
        pc[nif].x = coor[kp[nif]][jp[nif]][ip[nif]].x;
        pc[nif].y = coor[kp[nif]][jp[nif]][ip[nif]].y;
        pc[nif].z = coor[kp[nif]][jp[nif]][ip[nif]].z;
    }

    ICP(p, pc, nfx, nfy, nfz, ibminfo, ip, jp, kp);
    switch (ibminfo->imode) {
        case(0): {
            ibminfo->i1=ip[0]; ibminfo->j1 = jp[0]; ibminfo->k1 = kp[0];
            ibminfo->i2=ip[1]; ibminfo->j2 = jp[1]; ibminfo->k2 = kp[1];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (1): {
            ibminfo->i1=ip[1]; ibminfo->j1 = jp[1]; ibminfo->k1 = kp[1];
            ibminfo->i2=ip[2]; ibminfo->j2 = jp[2]; ibminfo->k2 = kp[2];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (2): {
            ibminfo->i1=ip[2]; ibminfo->j1 = jp[2]; ibminfo->k1 = kp[2];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[5]; ibminfo->j3 = jp[5]; ibminfo->k3 = kp[5];
            break;
        }
        case (3): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[5]; ibminfo->j2 = jp[5]; ibminfo->k2 = kp[5];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (4): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[7]; ibminfo->j2 = jp[7]; ibminfo->k2 = kp[7];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (5): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[6]; ibminfo->j2 = jp[6]; ibminfo->k2 = kp[6];
            ibminfo->i3=ip[7]; ibminfo->j3 = jp[7]; ibminfo->k3 = kp[7];
            break;
        }
        case (6): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[6]; ibminfo->j3 = jp[6]; ibminfo->k3 = kp[6];
            break;
        }
        case (7): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[0]; ibminfo->j3 = jp[0]; ibminfo->k3 = kp[0];
            break;
        }
    }
    if (ibminfo->imode >=0) {
        DMDAVecRestoreArray(fda, Cent, &coor);
        return 0;
    }


    ip[0] = i-1; ip[1] = i  ; ip[2] = i+1;
    ip[3] = i-1; ip[4] = i  ; ip[5] = i+1;
    ip[6] = i-1; ip[7] = i  ; ip[8] = i+1;

    jp[0] = j-1; jp[3] = j  ; jp[6] = j+1;
    jp[1] = j-1; jp[4] = j  ; jp[7] = j+1;
    jp[2] = j-1; jp[5] = j  ; jp[8] = j+1;

    kp[0] = k-1; kp[1] = k-1; kp[2] = k-1;
    kp[3] = k-1; kp[4] = k-1; kp[5] = k-1;
    kp[6] = k-1; kp[7] = k-1; kp[8] = k-1;

    for (nif=0; nif<9; nif++) {
        pc[nif].x = coor[kp[nif]][jp[nif]][ip[nif]].x;
        pc[nif].y = coor[kp[nif]][jp[nif]][ip[nif]].y;
        pc[nif].z = coor[kp[nif]][jp[nif]][ip[nif]].z;
    }

    ICP(p, pc, nfx, nfy, nfz, ibminfo, ip, jp, kp);
    switch (ibminfo->imode) {
        case(0): {
            ibminfo->i1=ip[0]; ibminfo->j1 = jp[0]; ibminfo->k1 = kp[0];
            ibminfo->i2=ip[1]; ibminfo->j2 = jp[1]; ibminfo->k2 = kp[1];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (1): {
            ibminfo->i1=ip[1]; ibminfo->j1 = jp[1]; ibminfo->k1 = kp[1];
            ibminfo->i2=ip[2]; ibminfo->j2 = jp[2]; ibminfo->k2 = kp[2];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (2): {
            ibminfo->i1=ip[2]; ibminfo->j1 = jp[2]; ibminfo->k1 = kp[2];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[5]; ibminfo->j3 = jp[5]; ibminfo->k3 = kp[5];
            break;
        }
        case (3): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[5]; ibminfo->j2 = jp[5]; ibminfo->k2 = kp[5];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (4): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[7]; ibminfo->j2 = jp[7]; ibminfo->k2 = kp[7];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (5): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[6]; ibminfo->j2 = jp[6]; ibminfo->k2 = kp[6];
            ibminfo->i3=ip[7]; ibminfo->j3 = jp[7]; ibminfo->k3 = kp[7];
            break;
        }
        case (6): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[6]; ibminfo->j3 = jp[6]; ibminfo->k3 = kp[6];
            break;
        }
        case (7): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[0]; ibminfo->j3 = jp[0]; ibminfo->k3 = kp[0];
            break;
        }
    }
    if (ibminfo->imode >=0) {
        DMDAVecRestoreArray(fda, Cent, &coor);
        return 0;
    }


    ip[0] = i-1; ip[1] = i  ; ip[2] = i+1;
    ip[3] = i-1; ip[4] = i  ; ip[5] = i+1;
    ip[6] = i-1; ip[7] = i  ; ip[8] = i+1;

    jp[0] = j-1; jp[3] = j  ; jp[6] = j+1;
    jp[1] = j-1; jp[4] = j  ; jp[7] = j+1;
    jp[2] = j-1; jp[5] = j  ; jp[8] = j+1;

    kp[0] = k+1; kp[1] = k+1; kp[2] = k+1;
    kp[3] = k+1; kp[4] = k+1; kp[5] = k+1;
    kp[6] = k+1; kp[7] = k+1; kp[8] = k+1;

    for (nif=0; nif<9; nif++) {
         pc[nif].x = coor[kp[nif]][jp[nif]][ip[nif]].x;
         pc[nif].y = coor[kp[nif]][jp[nif]][ip[nif]].y;
         pc[nif].z = coor[kp[nif]][jp[nif]][ip[nif]].z;
    }

    ICP(p, pc, nfx, nfy, nfz, ibminfo, ip, jp, kp);
    switch (ibminfo->imode) {
        case(0): {
            ibminfo->i1=ip[0]; ibminfo->j1 = jp[0]; ibminfo->k1 = kp[0];
            ibminfo->i2=ip[1]; ibminfo->j2 = jp[1]; ibminfo->k2 = kp[1];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (1): {
            ibminfo->i1=ip[1]; ibminfo->j1 = jp[1]; ibminfo->k1 = kp[1];
            ibminfo->i2=ip[2]; ibminfo->j2 = jp[2]; ibminfo->k2 = kp[2];
            ibminfo->i3=ip[4]; ibminfo->j3 = jp[4]; ibminfo->k3 = kp[4];
            break;
        }
        case (2): {
            ibminfo->i1=ip[2]; ibminfo->j1 = jp[2]; ibminfo->k1 = kp[2];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[5]; ibminfo->j3 = jp[5]; ibminfo->k3 = kp[5];
            break;
        }
        case (3): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[5]; ibminfo->j2 = jp[5]; ibminfo->k2 = kp[5];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (4): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[7]; ibminfo->j2 = jp[7]; ibminfo->k2 = kp[7];
            ibminfo->i3=ip[8]; ibminfo->j3 = jp[8]; ibminfo->k3 = kp[8];
            break;
        }
        case (5): {
            ibminfo->i1=ip[4]; ibminfo->j1 = jp[4]; ibminfo->k1 = kp[4];
            ibminfo->i2=ip[6]; ibminfo->j2 = jp[6]; ibminfo->k2 = kp[6];
            ibminfo->i3=ip[7]; ibminfo->j3 = jp[7]; ibminfo->k3 = kp[7];
            break;
        }
        case (6): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[6]; ibminfo->j3 = jp[6]; ibminfo->k3 = kp[6];
            break;
        }
        case (7): {
            ibminfo->i1=ip[3]; ibminfo->j1 = jp[3]; ibminfo->k1 = kp[3];
            ibminfo->i2=ip[4]; ibminfo->j2 = jp[4]; ibminfo->k2 = kp[4];
            ibminfo->i3=ip[0]; ibminfo->j3 = jp[0]; ibminfo->k3 = kp[0];
            break;
        }
    }

    if (ibminfo->imode >=0) {
        DMDAVecRestoreArray(fda, Cent, &coor);
        return 0;
    }
    //  }

  DMDAVecRestoreArray(fda, Cent, &coor);
  return 0;
}

double ImmersedBoundary::ContravariantReynoldsStress(
    double uu, double uv, double uw, 
    double vv, double vw, double ww,
    double csi0, double csi1, double csi2, 
    double eta0, double eta1, double eta2)
{
    double A = uu*csi0*eta0 + vv*csi1*eta1 + ww*csi2*eta2 + 
               uv * (csi0*eta1+csi1*eta0) + 
               uw * (csi0*eta2+csi2*eta0) + 
               vw * (csi1*eta2+csi2*eta1);
    double B = sqrt(csi0*csi0+csi1*csi1+csi2*csi2)*
               sqrt(eta0*eta0+eta1*eta1+eta2*eta2);
    
    return A/B;
}



PetscErrorCode ImmersedBoundary::IBMInterpolationAdvanced(PetscInt ti)
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
    PetscInt  lxs, lxe, lys, lye, lzs, lze;

    PetscReal ucx, ucy, ucz;
    PetscReal lhs[3][3], rhs_l[3][3];
    PetscReal ***nvert, ***p, ***lp;
    PetscReal ***ustar, ***aj;

    Cmpnts ***icsi, ***jeta, ***kzet;
    Cmpnts ***csi, ***eta, ***zet;
    Cmpnts ***ucat, ***lucat;
    Cmpnts ***ucont;
    Cmpnts ***usum, ***u1sum, ***u2sum;
    
    Vec lUcat_sum, lUcat_cross_sum, lUcat_square_sum;
     
    Vec lCent = d_grid->getlCent();
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj();
    Vec ICsi = d_grid->getlICsi();
    Vec JEta = d_grid->getlJEta();
    Vec KZet = d_grid->getlKZet();

    Vec lUstar = d_data->getlUstar();
    Vec lNvert = d_data->getlNvert();
    Vec lUcat = d_data->getlUcat();
    Vec Ucat = d_data->getUcat();
    Vec P = d_data->getP();
    Vec lP = d_data->getlP();
    Vec lUcont = d_data->getlUcont();
    Vec Ucont = d_data->getUcont();
 
    double N=(double)ti-1.0;

    lxs = xs; lxe = xe;
    lys = ys; lye = ye;
    lzs = zs; lze = ze;

    if (xs==0) lxs = xs+1;
    if (ys==0) lys = ys+1;
    if (zs==0) lzs = zs+1;

    if (xe==mx) lxe = xe-1;
    if (ye==my) lye = ye-1;
    if (ze==mz) lze = ze-1;

    IBMListNode *current;
    
    if (d_averaging) {
        VecDuplicate (lUcat, &lUcat_sum);
        VecDuplicate (lUcat, &lUcat_cross_sum);
        VecDuplicate (lUcat, &lUcat_square_sum);
        
        Vec Ucat_sum = d_data->getUcat_sum();
        DMGlobalToLocalBegin(fda, Ucat_sum, INSERT_VALUES, lUcat_sum);
        DMGlobalToLocalEnd(fda, Ucat_sum, INSERT_VALUES, lUcat_sum);
        
        Vec Ucat_cross_sum = d_data->getUcat_cross_sum();
        DMGlobalToLocalBegin(fda,Ucat_cross_sum,INSERT_VALUES,lUcat_cross_sum);
        DMGlobalToLocalEnd(fda,Ucat_cross_sum,INSERT_VALUES,lUcat_cross_sum);

        Vec Ucat_sq_sum = d_data->getUcat_square_sum();
        DMGlobalToLocalBegin(fda,Ucat_sq_sum,INSERT_VALUES,lUcat_square_sum);
        DMGlobalToLocalEnd(fda,Ucat_sq_sum,INSERT_VALUES,lUcat_square_sum);
                
        DMDAVecGetArray(fda, lUcat_sum, &usum);
        DMDAVecGetArray(fda, lUcat_cross_sum, &u1sum);
        DMDAVecGetArray(fda, lUcat_square_sum, &u2sum);
    }
    
    DMDAVecGetArray(da, Aj, &aj);
    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(fda, ICsi, &icsi);
    DMDAVecGetArray(fda, JEta, &jeta);
    DMDAVecGetArray(fda, KZet, &kzet);

    DMDAVecGetArray(da, lUstar, &ustar);
    DMDAVecGetArray(da, lNvert, &nvert);

    int tmp_max=1;
    std::vector<double> count;
    for(int tmp=0; tmp<tmp_max; tmp++) { 
        
        DMDAVecGetArray(fda, Ucat, &ucat);
        DMDAVecGetArray(fda, lUcat, &lucat);
        DMDAVecGetArray(da, P, &p);
        DMDAVecGetArray(da, lP, &lp);
        
        for (int ibi=0; ibi<d_NumberOfBodies; ibi++) {
            current = d_ibmlist[ibi].head;
            IBMNodes *ibm = d_ibm+ibi;
            
            count.resize(ibm->n_elmt);

            for (int i=0; i<ibm->n_elmt; i++) {
                count[i] = 0;
                ibm->shear[i] = 0;
                ibm->mean_shear[i] = 0;
                ibm->reynolds_stress1[i] = 0;
                ibm->reynolds_stress2[i] = 0;
                ibm->reynolds_stress3[i] = 0;
                ibm->pressure[i] = 0;
                ibm->rel_velocity[i].x = 0;
                ibm->rel_velocity[i].y = 0;
                ibm->rel_velocity[i].z = 0;

            }
            
            while (current) {
                PetscInt i,j,k;
                Cmpnts Ua, Uc;
                double ustar_avg=0;
                double reynolds1=0;
                double reynolds2=0;
                double reynolds3=0;
                double pressure=0;
                
                const double ren = d_data->getRe();
                
                IBMInfo *ibminfo = &current->ibm_intp;
                current = current->next;
                    
                int ni = ibminfo->cell;
                int ip1 = ibminfo->i1, jp1 = ibminfo->j1, kp1 = ibminfo->k1;
                int ip2 = ibminfo->i2, jp2 = ibminfo->j2, kp2 = ibminfo->k2;
                int ip3 = ibminfo->i3, jp3 = ibminfo->j3, kp3 = ibminfo->k3;
                i = ibminfo->ni, j= ibminfo->nj, k = ibminfo->nk;
                    
                double sb = ibminfo->d_s, sc = sb + ibminfo->d_i;
                double sk1  = ibminfo->cr1;
                double sk2 = ibminfo->cr2;
                double sk3 = ibminfo->cr3;
                double cs1 = ibminfo->cs1;
                double cs2 = ibminfo->cs2;
                double cs3 = ibminfo->cs3;
                double nfx = ibm->nf_x[ni];
                double nfy = ibm->nf_y[ni];
                double nfz = ibm->nf_z[ni];
                
                            
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
                }
                else {
                    Ua.x = Ua.y = Ua.z = 0;
                }
                
                Uc.x = (lucat[kp1][jp1][ip1].x * sk1 + 
                        lucat[kp2][jp2][ip2].x * sk2 +
                        lucat[kp3][jp3][ip3].x * sk3);
                Uc.y = (lucat[kp1][jp1][ip1].y * sk1 + 
                        lucat[kp2][jp2][ip2].y * sk2 + 
                        lucat[kp3][jp3][ip3].y * sk3);
                Uc.z = (lucat[kp1][jp1][ip1].z * sk1 + 
                        lucat[kp2][jp2][ip2].z * sk2 + 
                        lucat[kp3][jp3][ip3].z * sk3);

                double dp_dx, dp_dy, dp_dz;

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
                                       lp, nvert, 
                                       &dpdc, &dpde, &dpdz);

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
                                       lp, nvert, 
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
                                       lp, nvert, 
                                       &dpdc, &dpde, &dpdz );

                Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                     eta0, eta1, eta2, 
                                     zet0, zet1, zet2, ajc, 
                                     dpdc, dpde, dpdz, 
                                     &dp_dx3, &dp_dy3, &dp_dz3);

            
                    
                dp_dx = dp_dx1*sk1 + dp_dx2*sk2 + dp_dx3*sk3 ;
                dp_dy = dp_dy1*sk1 + dp_dy2*sk2 + dp_dy3*sk3 ;
                dp_dz = dp_dz1*sk1 + dp_dz2*sk2 + dp_dz3*sk3 ;
                
                if (!d_movefsi && ((!d_rotatefsi ||!d_rotatefsi_noIBsearch) || 
                                  ibi>=d_NumberOfRotatingBodies) && 
                    nvert[kp1][jp1][ip1] + 
                    nvert[kp2][jp2][ip2] + 
                    nvert[kp3][jp3][ip3] > 0.1 ) {

                    Set ( &ucat[k][j][i], 0 );
                    ustar[k][j][i]=0;
                } else if ( nvert[k][j][i]>2.9 ) {
                    Set ( &ucat[k][j][i], 0 );
                    ustar[k][j][i]=0;
                } else if(d_wallfunction && ti>0) {
                      WallFunctions::wall_function_s(
                          1.0/ren,d_roughness_size,
                          sc,sb,Ua,Uc,
                          &ucat[k][j][i],&ustar[k][j][i],
                          ibm->nf_x[ni],ibm->nf_y[ni],ibm->nf_z[ni]);
                }else  {
                    if (!d_IB_wm) {
                       WallFunctions::noslip(
                          ren, sc, sb, Ua, Uc, 
                          &ucat[k][j][i], &ustar[k][j][i], 
                          ibm->nf_x[ni],ibm->nf_y[ni],ibm->nf_z[ni]);
                    } 
                }
                
                if (d_averaging) {
                    Cmpnts tmp;
                    Cmpnts Uc_avg;
                    double _sk1=sk1, _sk2=sk2, _sk3=sk3;
                    
                    Uc_avg.x = (usum[kp1][jp1][ip1].x * _sk1 + 
                                usum[kp2][jp2][ip2].x * _sk2 + 
                                usum[kp3][jp3][ip3].x * _sk3) / N;
                    Uc_avg.y = (usum[kp1][jp1][ip1].y * _sk1 + 
                                usum[kp2][jp2][ip2].y * _sk2 + 
                                usum[kp3][jp3][ip3].y * _sk3) / N;
                    Uc_avg.z = (usum[kp1][jp1][ip1].z * _sk1 + 
                                usum[kp2][jp2][ip2].z * _sk2 + 
                                usum[kp3][jp3][ip3].z * _sk3) / N;
                    
                    if (d_wallfunction) {
                        WallFunctions::wall_function_s(
                            1.0/ren,d_roughness_size,
                            sc,sb,Ua,Uc_avg,
                            &tmp,&ustar_avg,
                            ibm->nf_x[ni],ibm->nf_y[ni],ibm->nf_z[ni]);
                    } else { 
                        WallFunctions::noslip(
                            ren, sc, sb, Ua, Uc_avg, 
                            &tmp, &ustar_avg, 
                            ibm->nf_x[ni], ibm->nf_y[ni], ibm->nf_z[ni]);
                    }
                    double U;
                    double V;
                    double W;
                    double uu;
                    double vv;
                    double ww;
                    double uv;
                    double vw;
                    double uw;
                    
                    U = (usum[kp1][jp1][ip1].x * _sk1 + 
                         usum[kp2][jp2][ip2].x * _sk2 + 
                         usum[kp3][jp3][ip3].x * _sk3)/N;
                    V = (usum[kp1][jp1][ip1].y * _sk1 + 
                         usum[kp2][jp2][ip2].y * _sk2 + 
                         usum[kp3][jp3][ip3].y * _sk3)/N;
                    W = (usum[kp1][jp1][ip1].z * _sk1 + 
                         usum[kp2][jp2][ip2].z * _sk2 + 
                         usum[kp3][jp3][ip3].z * _sk3)/N;
                    uu = (u2sum[kp1][jp1][ip1].x * _sk1 + 
                          u2sum[kp2][jp2][ip2].x * _sk2 + 
                          u2sum[kp3][jp3][ip3].x * _sk3)/N - U*U;
                    vv = (u2sum[kp1][jp1][ip1].y * _sk1 + 
                          u2sum[kp2][jp2][ip2].y * _sk2 + 
                          u2sum[kp3][jp3][ip3].y * _sk3)/N - V*V;
                    ww = (u2sum[kp1][jp1][ip1].z * _sk1 + 
                          u2sum[kp2][jp2][ip2].z * _sk2 + 
                          u2sum[kp3][jp3][ip3].z * _sk3)/N - W*W;
                    uv = (u1sum[kp1][jp1][ip1].x * _sk1 + 
                          u1sum[kp2][jp2][ip2].x * _sk2 + 
                          u1sum[kp3][jp3][ip3].x * _sk3)/N - U*V;
                    vw = (u1sum[kp1][jp1][ip1].y * _sk1 + 
                          u1sum[kp2][jp2][ip2].y * _sk2 + 
                          u1sum[kp3][jp3][ip3].y * _sk3)/N - V*W;
                    uw = (u1sum[kp1][jp1][ip1].z * _sk1 + 
                          u1sum[kp2][jp2][ip2].z * _sk2 + 
                          u1sum[kp3][jp3][ip3].z * _sk3)/N - W*U;

                    double UV = ContravariantReynoldsStress(
                        uu, uv, uw, vv, vw, ww,    
                        csi[k][j][i].x, csi[k][j][i].y, csi[k][j][i].z, 
                        eta[k][j][i].x, eta[k][j][i].y, eta[k][j][i].z);
                    double VW = ContravariantReynoldsStress(
                        uu, uv, uw, vv, vw, ww,    
                        eta[k][j][i].x, eta[k][j][i].y, eta[k][j][i].z, 
                        zet[k][j][i].x, zet[k][j][i].y, zet[k][j][i].z);
                    double WU = ContravariantReynoldsStress(
                        uu, uv, uw, vv, vw, ww,    
                        csi[k][j][i].x, csi[k][j][i].y, csi[k][j][i].z, 
                        zet[k][j][i].x, zet[k][j][i].y, zet[k][j][i].z);
                    reynolds1 = UV;
                    reynolds2 = VW;
                    reynolds3 = WU;
                }
                
                double cv1 = lp[kp1][jp1][ip1];
                double cv2 = lp[kp2][jp2][ip2];
                double cv3 = lp[kp3][jp3][ip3];
        
                p[k][j][i] = (cv1 * sk1 + cv2 * sk2 + cv3 * sk3);
                
                
                PetscReal Ua_n, Ua_nold;
                if (ni>=0) {
                    Cmpnts Ua;
                    
                    Ua.x = ibm->uold[ibm->nv1[ni]].x * cs1 + 
                           ibm->uold[ibm->nv2[ni]].x * cs2 + 
                           ibm->uold[ibm->nv3[ni]].x * cs3;
                    Ua.y = ibm->uold[ibm->nv1[ni]].y * cs1 + 
                           ibm->uold[ibm->nv2[ni]].y * cs2 + 
                           ibm->uold[ibm->nv3[ni]].y * cs3;
                    Ua.z = ibm->uold[ibm->nv1[ni]].z * cs1 + 
                           ibm->uold[ibm->nv2[ni]].z * cs2 + 
                           ibm->uold[ibm->nv3[ni]].z * cs3;
                    
                    Ua_nold= Ua.x*nfx + Ua.y*nfy + Ua.z*nfz;
                    
                    Ua.x = ibm->u[ibm->nv1[ni]].x * cs1 + 
                           ibm->u[ibm->nv2[ni]].x * cs2 + 
                           ibm->u[ibm->nv3[ni]].x * cs3;
                    Ua.y = ibm->u[ibm->nv1[ni]].y * cs1 + 
                           ibm->u[ibm->nv2[ni]].y * cs2 + 
                           ibm->u[ibm->nv3[ni]].y * cs3;
                    Ua.z = ibm->u[ibm->nv1[ni]].z * cs1 + 
                           ibm->u[ibm->nv2[ni]].z * cs2 + 
                           ibm->u[ibm->nv3[ni]].z * cs3;
                    
                    Ua_n= Ua.x*nfx + Ua.y*nfy + Ua.z*nfz;
                }
                else {
                    Ua_n = 0; Ua_nold = 0.;
                }
                      
                if(tmp==tmp_max-1)
                {
                    count[ni] ++;
                    ibm->shear[ni] += ustar[k][j][i]*ustar[k][j][i];
                    ibm->mean_shear[ni] += ustar_avg*ustar_avg;
                    ibm->reynolds_stress1[ni] += reynolds1;
                    ibm->reynolds_stress2[ni] += reynolds2;
                    ibm->reynolds_stress3[ni] += reynolds3;
                    ibm->pressure[ni] += p[k][j][i];
                    ibm->rel_velocity[ni].x += Uc.x - Ua.x;
                    ibm->rel_velocity[ni].y += Uc.y - Ua.y;
                    ibm->rel_velocity[ni].z += Uc.z - Ua.z;

                }
                
            }
            
            for (int i=0; i<ibm->n_elmt; i++) {
                if( count[i] > 1.0e-9) {
                    ibm->shear[i] /= count[i];
                    ibm->mean_shear[i] /= count[i];
                    ibm->reynolds_stress1[i] /= count[i];
                    ibm->reynolds_stress2[i] /= count[i];
                    ibm->reynolds_stress3[i] /= count[i];
                    ibm->pressure[i] /= count[i];
                    ibm->rel_velocity[i].x /= count[i];
                    ibm->rel_velocity[i].y /= count[i];
                    ibm->rel_velocity[i].z /= count[i];

                }
            }
        }
        
        
        DMDAVecRestoreArray(fda, Ucat, &ucat); 
        DMDAVecRestoreArray(fda, lUcat, &lucat);
        DMDAVecRestoreArray(da, P, &p);
        DMDAVecRestoreArray(da, lP, &lp);
        
        DMGlobalToLocalBegin(fda, Ucat, INSERT_VALUES, lUcat);
        DMGlobalToLocalEnd(fda, Ucat, INSERT_VALUES, lUcat);
        
        
        DMDAVecGetArray(fda, lUcat, &lucat);
        DMDAVecGetArray(fda, Ucont, &ucont);
        for (int k=lzs; k<lze; k++)
            for (int j=lys; j<lye; j++)
                for (int i=lxs; i<lxe; i++) {
                    double f = 1.0;
                    if (d_immersed==3) f = 0;
            
                    if ((int)(nvert[k][j][i]+0.5) ==1) {
                        ucx = (lucat[k][j][i].x + lucat[k][j][i+1].x) * 0.5;
                        ucy = (lucat[k][j][i].y + lucat[k][j][i+1].y) * 0.5;
                        ucz = (lucat[k][j][i].z + lucat[k][j][i+1].z) * 0.5;
                        ucont[k][j][i].x = (ucx * icsi[k][j][i].x + 
                                            ucy * icsi[k][j][i].y + 
                                            ucz * icsi[k][j][i].z) * f;
                
                        ucx = (lucat[k][j+1][i].x + lucat[k][j][i].x) * 0.5;
                        ucy = (lucat[k][j+1][i].y + lucat[k][j][i].y) * 0.5;
                        ucz = (lucat[k][j+1][i].z + lucat[k][j][i].z) * 0.5;
                        ucont[k][j][i].y = (ucx * jeta[k][j][i].x + 
                                            ucy * jeta[k][j][i].y + 
                                            ucz * jeta[k][j][i].z) * f;
              
                        ucx = (lucat[k+1][j][i].x + lucat[k][j][i].x) * 0.5;
                        ucy = (lucat[k+1][j][i].y + lucat[k][j][i].y) * 0.5;
                        ucz = (lucat[k+1][j][i].z + lucat[k][j][i].z) * 0.5;
                        ucont[k][j][i].z = (ucx * kzet[k][j][i].x + 
                                            ucy * kzet[k][j][i].y + 
                                            ucz * kzet[k][j][i].z) * f;
                
                        if ((d_grid->getBC(0)==-1||d_grid->getBC(0)==-2)&&i==1)
                            ucont[k][j][i].x = 0;
                        if ((d_grid->getBC(2)==-1||d_grid->getBC(2)==-2)&&j==1)
                            ucont[k][j][i].y = 0;
                        if ((d_grid->getBC(4)==-1||d_grid->getBC(4)==-2)&&k==1)
                            ucont[k][j][i].z = 0;
                    }

                    if ((int)(nvert[k][j][i+1]+0.5)==1) {
                        ucx = (lucat[k][j][i].x + lucat[k][j][i+1].x) * 0.5;
                        ucy = (lucat[k][j][i].y + lucat[k][j][i+1].y) * 0.5;
                        ucz = (lucat[k][j][i].z + lucat[k][j][i+1].z) * 0.5;
                
                        ucont[k][j][i].x = (ucx * icsi[k][j][i].x + 
                                            ucy * icsi[k][j][i].y + 
                                            ucz * icsi[k][j][i].z) * f;
                        if ((d_grid->getBC(1)==-1||d_grid->getBC(1)==-2) 
                             && i==mx-3) 
                            ucont[k][j][i].x = 0;
                    }
            
                    if ((int)(nvert[k][j+1][i]+0.5)==1) {
                        ucx = (lucat[k][j+1][i].x + lucat[k][j][i].x) * 0.5;
                        ucy = (lucat[k][j+1][i].y + lucat[k][j][i].y) * 0.5;
                        ucz = (lucat[k][j+1][i].z + lucat[k][j][i].z) * 0.5;
                
                        ucont[k][j][i].y = (ucx * jeta[k][j][i].x + 
                                            ucy * jeta[k][j][i].y + 
                                            ucz * jeta[k][j][i].z) * f;
                        if ((d_grid->getBC(3)==-1||d_grid->getBC(3)==-2) 
                             && j==my-3) 
                            ucont[k][j][i].y = 0;
                    }

                    if ((int)(nvert[k+1][j][i]+0.5)==1) {
                        ucx = (lucat[k+1][j][i].x + lucat[k][j][i].x) * 0.5;
                        ucy = (lucat[k+1][j][i].y + lucat[k][j][i].y) * 0.5;
                        ucz = (lucat[k+1][j][i].z + lucat[k][j][i].z) * 0.5;
                
                        ucont[k][j][i].z = (ucx * kzet[k][j][i].x + 
                                            ucy * kzet[k][j][i].y + 
                                            ucz * kzet[k][j][i].z )* f;
                        if ((d_grid->getBC(5)==-1||d_grid->getBC(5)==-2) 
                             && k==mz-3) 
                            ucont[k][j][i].z = 0;
                    }
                }
        
        DMDAVecRestoreArray(fda, lUcat, &lucat);
        DMDAVecRestoreArray(fda, Ucont, &ucont);
        
        DMGlobalToLocalBegin(fda, Ucont, INSERT_VALUES, lUcont);
        DMGlobalToLocalEnd(fda, Ucont, INSERT_VALUES, lUcont);
        
        d_data->Contra2Cart();
    }//tmp_end

    DMGlobalToLocalBegin(da, P, INSERT_VALUES, lP);
    DMGlobalToLocalEnd(da, P, INSERT_VALUES, lP);
    
    if (d_averaging) {
        DMDAVecRestoreArray(fda, lUcat_sum, &usum);
        DMDAVecRestoreArray(fda, lUcat_cross_sum, &u1sum);
        DMDAVecRestoreArray(fda, lUcat_square_sum, &u2sum);

        VecDestroy(&lUcat_sum);
        VecDestroy(&lUcat_cross_sum);
        VecDestroy(&lUcat_square_sum);
    }
    
    DMDAVecRestoreArray(da, Aj, &aj);
    DMDAVecRestoreArray(da, lUstar, &ustar);
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    DMDAVecRestoreArray(fda, ICsi, &icsi);
    DMDAVecRestoreArray(fda, JEta, &jeta);
    DMDAVecRestoreArray(fda, KZet, &kzet);
    DMDAVecRestoreArray(da, lNvert, &nvert);

    return 0;
}



PetscErrorCode ImmersedBoundary::ReadUCD(IBMNodes *ibm, PetscInt ibi)
{
    int rank;
    int n_v , n_elmt ;
    int i,ii;
    int n1e, n2e, n3e;
    int *nv1 , *nv2 , *nv3 ;

    PetscReal *x_bp , *y_bp , *z_bp ;
    PetscReal *nf_x, *nf_y, *nf_z;
    PetscReal dx12, dy12, dz12, dx13, dy13, dz13;
    PetscReal dr;
    PetscReal *dA ;//area
    PetscReal *nt_x, *nt_y, *nt_z;
    PetscReal *ns_x, *ns_y, *ns_z;

    char ss[20];
    char string[128];

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (!rank) { // root processor read in the data
        FILE *fd;
        PetscPrintf(PETSC_COMM_SELF, "READ ibmdata\n");
        char filen[80];  
        sprintf(filen,"%s/ibmdata%2.2d" , d_path, ibi);
 
        fd = fopen(filen, "r"); 
        if (!fd) printf("Cannot open %s !!", filen),exit(0);
        else printf("Opened %s !\n", filen);

        n_v =0;

        if (fd) {
            fgets(string, 128, fd);
            fgets(string, 128, fd);
            fgets(string, 128, fd);
      
            fscanf(fd, "%i %i %i %i %i",&n_v,&n_elmt,&ii,&ii,&ii);
            PetscPrintf(PETSC_COMM_SELF, "number of nodes: %d elements: %d\n",
                      n_v, n_elmt);
      
            ibm->n_v = n_v;
            ibm->n_elmt = n_elmt;      
      
            MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
            PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
            PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
            PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
      
            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
            PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
      
            PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
            PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
            PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));
      
            for (i=0; i<n_v; i++) {
                fscanf(fd, "%i %le %le %le", 
                      &ii, &x_bp[i], &y_bp[i], &z_bp[i]);//, &t, &t, &t);
    
                x_bp[i] = x_bp[i]/d_cl + d_CMx_c;
                y_bp[i] = y_bp[i]/d_cl + d_CMy_c ;
                z_bp[i] = z_bp[i]/d_cl + d_CMz_c ;
    
                ibm->x_bp[i] = x_bp[i];
                ibm->y_bp[i] = y_bp[i];
                ibm->z_bp[i] = z_bp[i];
 
                ibm->x_bp0[i] = x_bp[i];
                ibm->y_bp0[i] = y_bp[i];
                ibm->z_bp0[i] = z_bp[i];
  
                ibm->x_bp_o[i] = x_bp[i];
                ibm->y_bp_o[i] = y_bp[i];
                ibm->z_bp_o[i] = z_bp[i];

                ibm->u[i].x = 0.;
                ibm->u[i].y = 0.;
                ibm->u[i].z = 0.;

                ibm->uold[i].x = 0.;
                ibm->uold[i].y = 0.;
                ibm->uold[i].z = 0.;

                ibm->urm1[i].x = 0.;
                ibm->urm1[i].y = 0.;
                ibm->urm1[i].z = 0.;
            }
            PetscPrintf(PETSC_COMM_WORLD, "xyz_bp %le %le %le\n", 
                        x_bp[0], y_bp[0], z_bp[0]);


            MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
            MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
            MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

            MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
            MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
            MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
            MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
            MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
            MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

            MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

            PetscMalloc(n_elmt*sizeof(int), &nv1);
            PetscMalloc(n_elmt*sizeof(int), &nv2);
            PetscMalloc(n_elmt*sizeof(int), &nv3);
      
            PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
            PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
            PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      
            PetscMalloc(n_elmt*sizeof(PetscReal), &dA); //Area

            PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
            PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
            PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

            PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
            PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
            PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
      
            PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
            PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
            PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));

            PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->count);
            PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
            PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
            PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
            PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
            PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
            PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);
            PetscMalloc(n_elmt*sizeof(Cmpnts), &ibm->rel_velocity);

            for (i=0; i<n_elmt; i++) {

                fscanf(fd, "%i %i %s %i %i %i\n", 
                       &ii,&ii, ss, nv1+i, nv2+i, nv3+i);
                 nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;

            }
            ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

            PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", 
                        nv1[0], nv2[0], nv3[0]);

            fclose(fd);
        }
      
        for (i=0; i<n_elmt; i++) {
      
            n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
            dx12 = x_bp[n2e] - x_bp[n1e];
            dy12 = y_bp[n2e] - y_bp[n1e];
            dz12 = z_bp[n2e] - z_bp[n1e];
      
            dx13 = x_bp[n3e] - x_bp[n1e];
            dy13 = y_bp[n3e] - y_bp[n1e];
            dz13 = z_bp[n3e] - z_bp[n1e];
      
            nf_x[i] = dy12 * dz13 - dz12 * dy13;
            nf_y[i] = -dx12 * dz13 + dz12 * dx13;
            nf_z[i] = dx12 * dy13 - dy12 * dx13;
      
            dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
            nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
      
            if ((((1.-nf_z[i])<=1e-6 )&((-1.+nf_z[i])<1e-6))|
               (((nf_z[i]+1.)<=1e-6 )&((-1.-nf_z[i])<1e-6))) {
                ns_x[i] = 1.;     
                ns_y[i] = 0.;     
                ns_z[i] = 0. ;
     
                nt_x[i] = 0.;
                nt_y[i] = 1.;
                nt_z[i] = 0.;
            } else {
                ns_x[i] =  nf_y[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);  
                ns_y[i] = -nf_x[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]); 
                ns_z[i] = 0. ;
    
                nt_x[i] = -nf_x[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + 
                                                 nf_y[i]*nf_y[i]);
                nt_y[i] = -nf_y[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + 
                                                 nf_y[i]*nf_y[i]);
                nt_z[i] = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
            }
      
            dA[i] = dr/2.; 
      
            ibm->cent_x[i]= (x_bp[n1e]+x_bp[n2e]+x_bp[n3e])/3.;
            ibm->cent_y[i]= (y_bp[n1e]+y_bp[n2e]+y_bp[n3e])/3.;
            ibm->cent_z[i]= (z_bp[n1e]+z_bp[n2e]+z_bp[n3e])/3.;    
        }
    
        ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;
    
        ibm->dA = dA;
        ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
        ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    
    
        MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
        MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
        MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
        MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
        MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
        MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

        //Put this here  

        int ti=0;
        FILE *f;
        sprintf(filen, "surface_nf%3.3d_%2.2d.dat",ti,ibi);
        f = fopen(filen, "w");
    
        int N_block=100*1024*1024; // 100Mb
        char str[256];
        char carriage_return = '\n';
        std::vector<char> large_buffer;
        
    
        sprintf(str, "Variables=x,y,z,n_x,n_y,n_z,nt_x,"
                     "nt_y,nt_z,ns_x,ns_y,ns_z");
        str_to_buffer(str, large_buffer);
        large_buffer.push_back(carriage_return);
        
        sprintf(str, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK,"
                     "ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,"
                     "[4-12]=CELLCENTERED)", n_v, n_elmt);

        str_to_buffer(str, large_buffer);
        large_buffer.push_back(carriage_return);
        
        for (i=0; i<n_v; i++) {
            sprintf(str, "%e ", ibm->x_bp[i]);
            str_to_buffer(str, large_buffer);
            if ((i+1)%10==0 || i==n_v-1) 
                large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_v; i++) {
            sprintf(str, "%e ", ibm->y_bp[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_v-1) 
               large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_v; i++) {    
            sprintf(str, "%e ", ibm->z_bp[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_v-1) 
                large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            sprintf(str, "%e ", ibm->nf_x[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_elmt-1) 
                large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            sprintf(str, "%e ", ibm->nf_y[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_elmt-1) 
                large_buffer.push_back(carriage_return);
            
            if(large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            sprintf(str, "%e ", ibm->nf_z[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_elmt-1) 
                large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            sprintf(str, "%e ", ibm->nt_x[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_elmt-1) 
                large_buffer.push_back(carriage_return);
            
            if(large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            sprintf(str, "%e ", ibm->nt_y[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_elmt-1) 
               large_buffer.push_back(carriage_return);
            
            if(large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            sprintf(str, "%e ", ibm->nt_z[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_elmt-1) 
                large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            sprintf(str, "%e ", ibm->ns_x[i]);
            str_to_buffer(str, large_buffer);
            if( (i+1)%10==0 || i==n_elmt-1) 
               large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            sprintf(str, "%e ", ibm->ns_y[i]);
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_elmt-1)
                large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            str_to_buffer(str, large_buffer);
            if ( (i+1)%10==0 || i==n_elmt-1) 
               large_buffer.push_back(carriage_return);
            
            if (large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        for (i=0; i<n_elmt; i++) {
            str_to_buffer(str, large_buffer);
            large_buffer.push_back(carriage_return);
            
            if(large_buffer.size()>N_block) {
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
                large_buffer.resize(0);
            }
        }
        
        if (large_buffer.size()) 
            fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
    
        fclose(f);

    } else if (rank) {
        MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
        ibm->n_v = n_v;
    
        PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
        PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
        PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
    
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
    
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
    
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));
    
        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));
    
        for (i=0; i<n_v; i++) {
            ibm->u[i].x = 0.;
            ibm->u[i].y = 0.;
            ibm->u[i].z = 0.;

            ibm->uold[i].x = 0.;
            ibm->uold[i].y = 0.;
            ibm->uold[i].z = 0.;
      
            ibm->urm1[i].x = 0.;
            ibm->urm1[i].y = 0.;
            ibm->urm1[i].z = 0.;      
        }
        
        MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);    
        MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

        MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

        MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
        MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
        ibm->n_elmt = n_elmt;

        PetscMalloc(n_elmt*sizeof(int), &nv1);
        PetscMalloc(n_elmt*sizeof(int), &nv2);
        PetscMalloc(n_elmt*sizeof(int), &nv3);

        PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
        PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
        PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

        PetscMalloc(n_elmt*sizeof(PetscReal), &dA);

        PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
        PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
        PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

        PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
        PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
        PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
    
        PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->count);
        PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
        PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
        PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
        PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
        PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
        PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);
        PetscMalloc(n_elmt*sizeof(Cmpnts), &ibm->rel_velocity);

        ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
        ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;
    
        ibm->dA = dA;
        ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
        ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    

        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));

        MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

        MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
        MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

        MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
        MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

        MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
        MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    }

    PetscFree(x_bp);
    PetscFree(y_bp);
    PetscFree(z_bp);
    return 0;
}


PetscErrorCode ImmersedBoundary::WriteOutput1(IBMNodes *ibm, 
                                              PetscInt ibi, 
                                              PetscInt ti)
{
    int n_v = ibm->n_v, n_elmt = ibm->n_elmt;
    std::vector<double> shear_tmp(ibm->n_elmt), shear(ibm->n_elmt);
    std::vector<double> mean_shear_tmp(ibm->n_elmt), mean_shear(ibm->n_elmt);
    std::vector<double> reynolds1_tmp(ibm->n_elmt), reynolds1(ibm->n_elmt);
    std::vector<double> reynolds2_tmp(ibm->n_elmt), reynolds2(ibm->n_elmt);
    std::vector<double> reynolds3_tmp(ibm->n_elmt), reynolds3(ibm->n_elmt);
    std::vector<double> pressure_tmp(ibm->n_elmt), pressure(ibm->n_elmt);
    std::vector<double> rel_velocity_x_tmp(ibm->n_elmt), 
                        rel_velocity_x(ibm->n_elmt);
    std::vector<double> rel_velocity_y_tmp(ibm->n_elmt), 
                        rel_velocity_y(ibm->n_elmt);
    std::vector<double> rel_velocity_z_tmp(ibm->n_elmt), 
                        rel_velocity_z(ibm->n_elmt);
  

    std::vector<int> count(ibm->n_elmt), count_tmp(ibm->n_elmt);

    std::fill(shear_tmp.begin(), shear_tmp.end(), 0.);
    std::fill(mean_shear_tmp.begin(), mean_shear_tmp.end(), 0.);
    std::fill(reynolds1_tmp.begin(), reynolds1_tmp.end(), 0.);
    std::fill(reynolds2_tmp.begin(), reynolds2_tmp.end(), 0.);
    std::fill(reynolds3_tmp.begin(), reynolds3_tmp.end(), 0.);
    std::fill(pressure_tmp.begin(), pressure_tmp.end(), 0.);
    std::fill(rel_velocity_x_tmp.begin(), rel_velocity_x_tmp.end(), 0.);
    std::fill(rel_velocity_y_tmp.begin(), rel_velocity_y_tmp.end(), 0.);
    std::fill(rel_velocity_z_tmp.begin(), rel_velocity_z_tmp.end(), 0.);

    std::fill(count_tmp.begin(), count_tmp.end(), 0);

    for (int i=0; i<ibm->n_elmt; i++) {
        if ( fabs(ibm->shear[i])>1.e-10 ) {
            shear_tmp[i] = ibm->shear[i];
            mean_shear_tmp[i] = ibm->mean_shear[i];
            reynolds1_tmp[i] = ibm->reynolds_stress1[i];
            reynolds2_tmp[i] = ibm->reynolds_stress2[i];
            reynolds3_tmp[i] = ibm->reynolds_stress3[i];
            pressure_tmp[i] = ibm->pressure[i];

            rel_velocity_x_tmp[i] = ibm->rel_velocity[i].x;
            rel_velocity_y_tmp[i] = ibm->rel_velocity[i].y;
            rel_velocity_z_tmp[i] = ibm->rel_velocity[i].z;

            count_tmp[i] += 1;
        }
    }

    MPI_Reduce(&shear_tmp[0], &shear[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&mean_shear_tmp[0], &mean_shear[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&reynolds1_tmp[0], &reynolds1[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&reynolds2_tmp[0], &reynolds2[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&reynolds3_tmp[0], &reynolds3[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&pressure_tmp[0], &pressure[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);

    MPI_Reduce(&rel_velocity_x_tmp[0], &rel_velocity_x[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&rel_velocity_y_tmp[0], &rel_velocity_y[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&rel_velocity_z_tmp[0], &rel_velocity_z[0], ibm->n_elmt, 
               MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);

    MPI_Reduce(&count_tmp[0], &count[0], ibm->n_elmt, 
               MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);


 

 
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  
    if (!rank) {
        if (ti == (ti/d_tiout)*d_tiout) {
            FILE *f;
            char filen[80];
            sprintf(filen, "surface%06d_%1d.dat", ti, ibi);
            f = fopen(filen, "w");
            
            int N_block=100*1024*1024; // 100Mb
            char str[256];
            char carriage_return = '\n';
            std::vector<char> large_buffer;
            
            sprintf(str, "Variables=x,y,z,u,v,w,p,shear_stress," 
                         "mean_shear_stress,UV,VW,WU,u_,v_,w_\n");
            str_to_buffer(str, large_buffer);
            large_buffer.push_back(carriage_return);
            
            sprintf(str, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK,"
               "ET=TRIANGLE, VARLOCATION=([1-6]=NODAL,[7-15]=CELLCENTERED)\n", 
                n_v, n_elmt);
            str_to_buffer(str, large_buffer);
            large_buffer.push_back(carriage_return);
            
            for (int i=0; i<n_v; i++) {
                sprintf(str, "%e ", ibm->x_bp[i]);
                str_to_buffer(str, large_buffer);
                if ((i+1)%10==0 || i==n_v-1) 
                    large_buffer.push_back(carriage_return);
                
                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char), 
                            large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_v; i++) {
                sprintf(str, "%e ", ibm->y_bp[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_v-1) 
                    large_buffer.push_back(carriage_return);
                
                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char), 
                            large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_v; i++) {
                sprintf(str, "%e ", ibm->z_bp[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_v-1) 
                    large_buffer.push_back(carriage_return);
                
                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char), 
                    large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_v; i++) {
                sprintf(str, "%e ", ibm->u[i].x);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_v-1) 
                    large_buffer.push_back(carriage_return);
                
                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char), 
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_v; i++) {
                sprintf(str, "%e ", ibm->u[i].y);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_v-1) 
                   large_buffer.push_back(carriage_return);
                
                if(large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char), 
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_v; i++) {
                sprintf(str, "%e ", ibm->u[i].z);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_v-1) 
                   large_buffer.push_back(carriage_return);
                
                if(large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char), 
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", pressure[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1) 
                    large_buffer.push_back(carriage_return);
                
                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char), 
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", shear[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1)
                    large_buffer.push_back(carriage_return);

                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char),
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", mean_shear[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1)
                    large_buffer.push_back(carriage_return);

                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char),
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", reynolds1[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1)
                    large_buffer.push_back(carriage_return);

                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char),
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", reynolds2[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1)
                    large_buffer.push_back(carriage_return);

                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char),
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", reynolds3[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1)
                    large_buffer.push_back(carriage_return);

                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char),
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", rel_velocity_x[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1)
                    large_buffer.push_back(carriage_return);

                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char),
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", rel_velocity_y[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1)
                    large_buffer.push_back(carriage_return);

                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char),
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%e ", rel_velocity_z[i]);
                str_to_buffer(str, large_buffer);
                if ( (i+1)%10==0 || i==n_elmt-1)
                    large_buffer.push_back(carriage_return);

                if (large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char),
                           large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }


            for (int i=0; i<n_elmt; i++) {
                sprintf(str, "%d %d %d\n", 
                        ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
                str_to_buffer(str, large_buffer);
                large_buffer.push_back(carriage_return);
                
                if(large_buffer.size()>N_block) {
                    fwrite(&large_buffer[0], sizeof(char), 
                            large_buffer.size(), f);
                    large_buffer.resize(0);
                }
            }
            
            if (large_buffer.size()) 
                fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
            fclose(f);
        }
    }
    return 0;
}


void ImmersedBoundary::str_to_buffer(char *str, std::vector<char> &large_buffer)
{
    char buffer[256];
    sprintf(buffer, "%s", str);
    int len = strlen(buffer), old_size = large_buffer.size();
    large_buffer.resize( old_size + len );
    for(int i=0; i<len; i++) large_buffer[old_size+i] = buffer[i];
}

PetscErrorCode ImmersedBoundary::ReadFromInput()
{
    
    PetscOptionsGetString(PETSC_NULL,"-path", d_path, 256, PETSC_NULL);

    PetscOptionsGetInt(PETSC_NULL, "-IB_wm", &d_IB_wm, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-imm", &d_immersed, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-fsi", &d_movefsi, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-rfsi", &d_rotatefsi, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-rfsi_noIBsearch", 
                       &d_rotatefsi_noIBsearch, PETSC_NULL);
    d_changefsi = d_movefsi + d_rotatefsi + d_rotatefsi_noIBsearch;
    PetscOptionsGetInt(PETSC_NULL, "-thin", &d_thin, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-body", &d_NumberOfBodies, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-rbody", &d_NumberOfRotatingBodies, 
                       PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-char_length_ibm", &d_cl, PETSC_NULL);     
    PetscOptionsGetReal(PETSC_NULL, "-CMx_c", &d_CMx_c, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-CMy_c", &d_CMy_c, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-CMz_c", &d_CMz_c, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-tio", &d_tiout, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-averaging", &d_averaging, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-wallfunction", &d_wallfunction, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-roughness", &d_roughness_size,
                        PETSC_NULL);
}

