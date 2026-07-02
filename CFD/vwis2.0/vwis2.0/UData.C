#include "UData.h"

UData::UData(
    const std::string& object_name,
    CurvGrid *grid):
    d_object_name(object_name),
    d_grid(grid)
{
    d_hdf5 = 0;
    d_tistart = 0;
    d_tisteps = 1;
    d_immersed = 0;
    d_restart = PETSC_FALSE;
 
    d_tiout = 1;
    d_tiout_ufield = 0;
    d_tiend_ufield = 10000000;
    d_ti_lastsave = 0;

    d_averaging = 0;
    d_phase_averaging = 0;
     
    d_ren = 1;
    d_dt = 1;
    d_St = 1;

    d_rough_set = PETSC_FALSE;
    d_roughness_size = 0.0;
    d_dp_wm = 0;

    d_movefsi = 0;   
    d_rotatefsi = 0;   

    sprintf(d_path, ".");
    sprintf(d_fieldpath, ".");
    sprintf(d_avepath, ".");
    sprintf(d_phpath, ".");
 
    ReadFromInput();


    //Make directories
    struct stat st = {0};

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    if (!rank) {
        if (stat(d_fieldpath, &st) == -1) {
            mkdir(d_fieldpath, 0754);
            printf("Creating Directory: %s\n", d_fieldpath);
        }
        if (stat(d_avepath, &st) == -1 && d_averaging)  {
            mkdir(d_avepath, 0754);
            printf("Creating Directory: %s\n", d_avepath);
        }
        if (stat(d_phpath, &st) == -1 && d_phase_averaging) {
            mkdir(d_phpath, 0754);
            printf("Creating Directory: %s\n", d_phpath);
        }

    }
    //Create wall function object here
    d_wallf = new WallFunctions("WallFunctions");
}

UData::~UData()
{
    delete d_wallf;

    VecDestroy(&d_Ucont);
    VecDestroy(&d_Ucont_o);
    VecDestroy(&d_Ucont_rm1);

    VecDestroy(&d_Rhs);
    VecDestroy(&d_Rhs_o);

    VecDestroy(&d_Ucat);
    VecDestroy(&d_Ucat_o);
    
    VecDestroy(&d_Nvert);
    VecDestroy(&d_Dp);
    VecDestroy(&d_P);

    VecDestroy(&d_lUcont);
    VecDestroy(&d_lUcont_o);
    VecDestroy(&d_lUcont_rm1);
    VecDestroy(&d_lUcat);
    VecDestroy(&d_lUcat_old);
  
    VecDestroy(&d_lNvert);
    VecDestroy(&d_lUstar);

    VecDestroy(&d_Ubcs); //Get rid of this in BcsUtil not needed at all

    if (d_immersed) {
        VecDestroy(&d_Nvert_o);
        VecDestroy(&d_lNvert_o);
        VecDestroy(&d_P_o);
    }

    if (d_averaging) {
        VecDestroy(&d_Ucat_sum);
        VecDestroy(&d_Ucat_cross_sum);
        VecDestroy(&d_Ucat_square_sum);
        VecDestroy(&d_P_sum);

        if (d_averaging >=2) {
            VecDestroy(&d_P_square_sum);
        }
        if (d_averaging >= 3) {
            VecDestroy(&d_Udp_sum);
            VecDestroy(&d_dU2_sum);
            VecDestroy(&d_UUU_sum);
            VecDestroy(&d_Vort_sum);
            VecDestroy(&d_Vort_square_sum);
        }
        if (d_phase_averaging) {
            VecDestroy(&d_Ucat_sum_phase);
            VecDestroy(&d_Ucat_cross_sum_phase);
            VecDestroy(&d_Ucat_square_sum_phase);
            VecDestroy(&d_P_sum_phase);
            if (d_averaging >=2) {
                VecDestroy(&d_P_square_sum_phase);
            }
            if (d_averaging >= 3) {
                VecDestroy(&d_Udp_sum_phase);
                VecDestroy(&d_dU2_sum_phase);
                VecDestroy(&d_UUU_sum_phase);
                VecDestroy(&d_Vort_sum_phase);
                VecDestroy(&d_Vort_square_sum_phase);
            }
        }
    }
}

PetscErrorCode UData::InitializeData()
{
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMCreateGlobalVector(fda, &d_Ucont);

    VecDuplicate(d_Ucont, &d_Ucont_o);
    VecDuplicate(d_Ucont, &d_Ucont_rm1);

    VecDuplicate(d_Ucont, &d_Rhs);
    VecDuplicate(d_Ucont, &d_Rhs_o);
    VecDuplicate(d_Ucont, &d_Ucat);
    VecDuplicate(d_Ucont, &d_Ucat_o);
    VecDuplicate(d_Ucont, &d_Ubcs); //Get rid of this not used
    VecDuplicate(d_Ucont, &d_Dp);

    DMCreateGlobalVector(da, &d_Nvert);
    VecDuplicate(d_Nvert, &d_P);
    
    DMCreateLocalVector(fda, &d_lUcont);
    DMCreateLocalVector(fda, &d_lUcont_o);
    DMCreateLocalVector(fda, &d_lUcont_rm1);
    VecDuplicate(d_lUcont, &d_lUcat);
    VecDuplicate(d_lUcont, &d_lUcat_old); //Note old is local and o is global

    DMCreateLocalVector(da, &d_lNvert);
    VecDuplicate(d_lNvert, &d_lP);
    VecDuplicate(d_lNvert, &d_lUstar); //Should this be owned by wallmodel?

    if (d_immersed) {
        VecDuplicate(d_Nvert, &d_Nvert_o);
        VecDuplicate(d_Nvert, &d_P_o);
        VecDuplicate(d_lNvert, &d_lNvert_o);
        VecDuplicate(d_lNvert, &d_lNvert_o_fixed);
        VecSet(d_lNvert_o_fixed, 0);
        PetscObjectSetName((PetscObject) d_Nvert_o, "nvert") ;   
    }

    //Name these because hdf5 needs a vec name to read/write properly
    PetscObjectSetName((PetscObject) d_Ucont, "ucont");    
    PetscObjectSetName((PetscObject) d_Ucat, "ucat");    
    PetscObjectSetName((PetscObject) d_P, "pressure") ;   
    PetscObjectSetName((PetscObject) d_Nvert, "nvert") ;   

    VecSet(d_lNvert, 0.);
    if (!d_restart) {
        VecSet(d_Ucont,0.);
        VecSet(d_lUcont,0.);
        VecSet(d_Ucont_o,0.);
        VecSet(d_lUcont_o,0.);
        VecSet(d_Ucat,0.);
        VecSet(d_lUcat,0.);
        VecSet(d_P,0.);
        VecSet(d_lP,0.);

    }

    //Setup averaging Data
    if (d_averaging) {
 
        VecDuplicate(d_Ucat, &d_Ucat_sum);
        VecDuplicate(d_Ucat, &d_Ucat_cross_sum);
        VecDuplicate(d_Ucat, &d_Ucat_square_sum);
        VecDuplicate(d_P, &d_P_sum);
    
        if (d_averaging >=2) {
            VecDuplicate(d_P, &d_P_square_sum); 
        }
        if (d_averaging >= 3) {
            VecDuplicate(d_P, &d_Udp_sum);
            VecDuplicate(d_P, &d_dU2_sum); 
            VecDuplicate(d_Ucont, &d_UUU_sum);
            VecDuplicate(d_Ucont, &d_Vort_sum);
            VecDuplicate(d_Ucont, &d_Vort_square_sum);
        }

        PetscObjectSetName((PetscObject) d_Ucat_sum, "usum");    
        PetscObjectSetName((PetscObject) d_Ucat_cross_sum, "ucross");    
        PetscObjectSetName((PetscObject) d_Ucat_square_sum, "usquare");    
        PetscObjectSetName((PetscObject) d_P_sum, "psum");   
 
        if (d_averaging >=2) {
            PetscObjectSetName((PetscObject) d_P_square_sum, "psquare");
        }
        if (d_averaging >= 3) {
            PetscObjectSetName((PetscObject) d_Udp_sum, "udpsum");  
            PetscObjectSetName((PetscObject) d_dU2_sum, "du2sum");  
            PetscObjectSetName((PetscObject) d_UUU_sum, "uuusum");  
            PetscObjectSetName((PetscObject) d_Vort_sum, "vortsum");  
            PetscObjectSetName((PetscObject) d_Vort_square_sum, "vortsquare");  
        } 

        //Setup phase averaging Data (averaging must be set)
        if (d_phase_averaging) {
            VecDuplicate(d_Ucat, &d_Ucat_sum_phase);
            VecDuplicate(d_Ucat, &d_Ucat_cross_sum_phase);
            VecDuplicate(d_Ucat, &d_Ucat_square_sum_phase);
            VecDuplicate(d_P, &d_P_sum_phase);
            if (d_averaging >=2) {
                VecDuplicate(d_P, &d_P_square_sum_phase);
            }
            if (d_averaging >= 3) {
                VecDuplicate(d_P, &d_Udp_sum_phase);
                VecDuplicate(d_P, &d_dU2_sum_phase); 
                VecDuplicate(d_Ucont, &d_UUU_sum_phase);
                VecDuplicate(d_Ucont, &d_Vort_sum_phase);
                VecDuplicate(d_Ucont, &d_Vort_square_sum_phase);
            }
        }

        VecSet(d_Ucat_sum, 0);
        VecSet(d_Ucat_cross_sum, 0);
        VecSet(d_Ucat_square_sum, 0);
        VecSet(d_P_sum, 0);
        if (d_averaging >= 2) {
            VecSet(d_P_square_sum, 0);
        } 
        if (d_averaging >= 3) {
            VecSet(d_Udp_sum, 0);
            VecSet(d_dU2_sum, 0);
            VecSet(d_UUU_sum, 0);
            VecSet(d_Vort_sum, 0);
            VecSet(d_Vort_square_sum, 0);
        } 

        
        if (d_phase_averaging) {
            VecSet(d_Ucat_sum_phase, 0);
            VecSet(d_Ucat_cross_sum_phase, 0);
            VecSet(d_Ucat_square_sum_phase, 0);
            VecSet(d_P_sum_phase, 0);
            if (d_averaging >= 2)
                VecSet(d_P_square_sum_phase, 0);
            if (d_averaging >= 3) {
                VecSet(d_Udp_sum_phase, 0.);
                VecSet(d_dU2_sum_phase, 0); 
                VecSet(d_UUU_sum_phase, 0);
                VecSet(d_Vort_sum_phase, 0);
                VecSet(d_Vort_square_sum_phase, 0);
            }
        } 
    }

}

void UData::ReadData()
{
    if (!d_restart) return;

    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    //Read Data here
    char filen[90];
   
    sprintf(filen, "%s/vfield%06d_%1d.%s", d_fieldpath, d_tistart, 0, d_rext);
    ReadFile(filen, d_Ucont);
    sprintf(filen, "%s/ufield%06d_%1d.%s", d_fieldpath, d_tistart, 0, d_rext);
    ReadFile(filen, d_Ucat);
    sprintf(filen, "%s/pfield%06d_%1d.%s", d_fieldpath, d_tistart, 0, d_rext);
    ReadFile(filen, d_P);
    sprintf(filen, "%s/nvfield%06d_%1d.%s", d_fieldpath, d_tistart, 0, d_rext);
    if (d_immersed)
        ReadFile(filen, d_Nvert_o);
    else
        ReadFile(filen, d_Nvert);

    DMGlobalToLocalBegin(fda, d_Ucat, INSERT_VALUES, d_lUcat);
    DMGlobalToLocalEnd(fda, d_Ucat, INSERT_VALUES, d_lUcat);

    DMGlobalToLocalBegin(fda, d_Ucont, INSERT_VALUES, d_lUcont);
    DMGlobalToLocalEnd(fda, d_Ucont, INSERT_VALUES, d_lUcont);

    DMGlobalToLocalBegin(da, d_P, INSERT_VALUES, d_lP);
    DMGlobalToLocalEnd(da, d_P, INSERT_VALUES, d_lP);

    VecCopy(d_Ucont, d_Ucont_o);
    DMGlobalToLocalBegin(fda, d_Ucont_o, INSERT_VALUES, d_lUcont_o);
    DMGlobalToLocalEnd(fda, d_Ucont_o, INSERT_VALUES, d_lUcont_o);

    DMGlobalToLocalBegin(fda, d_Ucat, INSERT_VALUES, d_lUcat_old);
    DMGlobalToLocalEnd(fda, d_Ucat, INSERT_VALUES, d_lUcat_old);

    //if (!d_immersed) VecSet(d_Nvert_o, 0);

    if (d_immersed) {
        DMGlobalToLocalBegin(da, d_Nvert_o, INSERT_VALUES, d_lNvert_o);
        DMGlobalToLocalEnd(da, d_Nvert_o, INSERT_VALUES, d_lNvert_o);
    } else
        VecSet(d_Nvert, 0);
      
    
    //Read Averaging
    if (d_averaging) {
        sprintf(filen, "%s/su0_%06d_%1d.%s", d_avepath, d_tistart, 0, d_rext);
        FILE *fp = fopen(filen, "r");
        if (fp == NULL) 
            PetscPrintf(PETSC_COMM_WORLD, "***Cannot open %s\n"
                                          "***Setting stats to zero\n", 
                                          filen);
        else {
            fclose(fp);
            MPI_Barrier(PETSC_COMM_WORLD);

            ReadFile(filen, d_Ucat_sum);
            sprintf(filen, "%s/su1_%06d_%1d.%s", d_avepath, d_tistart, 
                   0, d_rext);
            ReadFile(filen, d_Ucat_cross_sum);
            sprintf(filen, "%s/su2_%06d_%1d.%s", d_avepath, d_tistart, 
                    0, d_rext);
            ReadFile(filen, d_Ucat_square_sum);
            sprintf(filen, "%s/sp_%06d_%1d.%s", d_avepath, d_tistart, 
                    0, d_rext);
            ReadFile(filen, d_P_sum);
            if (d_averaging >= 2) {
                sprintf(filen, "%s/sp2_%06d_%1d.%s", 
                        d_avepath, d_tistart, 0, d_rext);
                ReadFile(filen, d_P_square_sum);
            }
            if (d_averaging >= 3) { 
                sprintf(filen, "%s/su3_%06d_%1d.%s", 
                        d_avepath, d_tistart, 0, d_rext);
                ReadFile(filen, d_Udp_sum);
                sprintf(filen, "%s/su4_%06d_%1d.%s", 
                        d_avepath, d_tistart, 0, d_rext);
                ReadFile(filen, d_dU2_sum);
                sprintf(filen, "%s/su5_%06d_%1d.%s", 
                        d_avepath, d_tistart, 0, d_rext);
                ReadFile(filen, d_UUU_sum);
                sprintf(filen, "%s/svo_%06d_%1d.%s", 
                        d_avepath, d_tistart, 0, d_rext);
                ReadFile(filen, d_Vort_sum);
                sprintf(filen, "%s/svo2_%06d_%1d.%s", 
                        d_avepath, d_tistart, 0, d_rext);
                ReadFile(filen, d_Vort_square_sum);
            }
        }   
    }

    if (d_averaging && d_phase_averaging) {
        PetscInt n, pti;
        PhaseNumber(d_tistart, &n, &pti);
        if (n > 0) {
            sprintf(filen, "%s/phase%03d_su0_%06d_%1d.%s", 
                    d_phpath, n, pti, 0, d_rext);
            ReadFile(filen, d_Ucat_sum_phase);
            sprintf(filen, "%s/phase%03d_su1_%06d_%1d.%s", 
                    d_phpath, n, pti, 0, d_rext);
            ReadFile(filen, d_Ucat_cross_sum_phase);
            sprintf(filen, "%s/phase%03d_su2_%06d_%1d.%s", 
                    d_phpath, n, pti, 0, d_rext);
            ReadFile(filen, d_Ucat_square_sum_phase);
            sprintf(filen, "%s/phase%03d_sp_%06d_%1d.%s", 
                    d_phpath, n, pti, 0, d_rext);
            ReadFile(filen, d_P_sum_phase);
            if (d_averaging >= 2) {
                sprintf(filen, "%s/phase%03d_sp2_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_rext);
                ReadFile(filen, d_P_square_sum_phase); 
            }
            if (d_averaging >= 3) {
                sprintf(filen, "%s/phase%03d_su3_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_rext);
                ReadFile(filen, d_Udp_sum_phase); 
                sprintf(filen, "%s/phase%03d_su4_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_rext);
                ReadFile(filen, d_dU2_sum_phase); 
                sprintf(filen, "%s/phase%03d_su4_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_rext);
                ReadFile(filen, d_UUU_sum_phase); 
                sprintf(filen, "%s/phase%03d_svo_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_rext);
                ReadFile(filen, d_Vort_sum_phase); 
                sprintf(filen, "%s/phase%03d_svo2_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_rext);
                ReadFile(filen, d_Vort_square_sum_phase); 
            }
        }
    }                
}

void UData::WriteData(PetscInt ti)
{
 
    if (d_tistart==ti && d_restart) return; 
 
    char filen[90];

    //This is to write just velocity
    if (d_tiout_ufield>0 && 
        ti == (ti/d_tiout_ufield) * d_tiout_ufield && 
        ti <= d_tiend_ufield) {

        sprintf(filen, "%s/ufield%06d_%1d.%s", d_fieldpath, ti, 0, d_wext);
        WriteFile(filen, d_Ucat);
    }

    //We only everything output at tiout intervals
    if (ti == (ti/d_tiout) * d_tiout) {

        //write Data here
        sprintf(filen, "%s/vfield%06d_%1d.%s", d_fieldpath, ti, 0, d_wext);
        WriteFile(filen, d_Ucont);
        sprintf(filen, "%s/ufield%06d_%1d.%s", d_fieldpath, ti, 0, d_wext);
        WriteFile(filen, d_Ucat);
        sprintf(filen, "%s/pfield%06d_%1d.%s", d_fieldpath, ti, 0, d_wext);
        WriteFile(filen, d_P);
        sprintf(filen, "%s/nvfield%06d_%1d.%s", d_fieldpath, ti, 0, d_wext);
        WriteFile(filen, d_Nvert);

        if (d_averaging) {
            sprintf(filen, "%s/su0_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
            WriteFile(filen, d_Ucat_sum);
            sprintf(filen, "%s/su1_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
            WriteFile(filen, d_Ucat_cross_sum);
            sprintf(filen, "%s/su2_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
            WriteFile(filen, d_Ucat_square_sum);
            sprintf(filen, "%s/sp_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
            WriteFile(filen, d_P_sum);
       
            if (d_averaging >=2) {
                sprintf(filen, "%s/sp2_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
                WriteFile(filen, d_P_square_sum);
            }

            if (d_averaging >= 3) {
                sprintf(filen, "%s/su3_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
                WriteFile(filen, d_Udp_sum);
                sprintf(filen, "%s/su4_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
                WriteFile(filen, d_dU2_sum);
                sprintf(filen, "%s/su5_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
                WriteFile(filen, d_UUU_sum);
                sprintf(filen, "%s/svo_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
                WriteFile(filen, d_Vort_sum);
                sprintf(filen, "%s/svo2_%06d_%1d.%s", d_avepath, ti, 0, d_wext);
                WriteFile(filen, d_Vort_square_sum);
            }
        }
    }

    //write phase averaging.   
    //This could occur at a different interval than output
    if (d_averaging && d_phase_averaging && ti && 
        (ti+d_ti_lastsave)%d_phase_averaging==0) {
      
        PetscInt n, pti;
        PhaseNumber(ti, &n, &pti); 
          
        if (n>0) {
            sprintf(filen, "%s/phase%03d_su0_%06d_%1d.%s", 
                    d_phpath, n, pti, 0, d_wext);
            WriteFile(filen, d_Ucat_sum_phase); 
            sprintf(filen, "%s/phase%03d_su1_%06d_%1d.%s", 
                    d_phpath, n, pti, 0, d_wext);
            WriteFile(filen, d_Ucat_cross_sum_phase); 
            sprintf(filen, "%s/phase%03d_su2_%06d_%1d.%s", 
                    d_phpath, n, pti, 0, d_wext);
            WriteFile(filen, d_Ucat_square_sum_phase); 
            sprintf(filen, "%s/phase%03d_sp_%06d_%1d.%s", 
                    d_phpath, n, pti, 0, d_wext);
            WriteFile(filen, d_P_sum_phase); 
            if (d_averaging >= 2) {
                sprintf(filen, "%s/phase%03d_sp2_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_wext);
                WriteFile(filen, d_P_square_sum_phase); 
            }
            if (d_averaging >= 3) {
                sprintf(filen, "%s/phase%03d_su3_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_wext);
                WriteFile(filen, d_Udp_sum_phase); 
                sprintf(filen, "%s/phase%03d_su4_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_wext);
                WriteFile(filen, d_dU2_sum_phase); 
                sprintf(filen, "%s/phase%03d_su4_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_wext);
                WriteFile(filen, d_UUU_sum_phase); 
                sprintf(filen, "%s/phase%03d_svo_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_wext);
                WriteFile(filen, d_Vort_sum_phase); 
                sprintf(filen, "%s/phase%03d_svo2_%06d_%1d.%s", 
                        d_phpath, n, pti, 0, d_wext);
                WriteFile(filen, d_Vort_square_sum_phase); 
            }
        }
    }
} 

void UData::CopyLastStep()
{
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    if (d_immersed) {
        VecCopy(d_Nvert, d_Nvert_o);
        DMGlobalToLocalBegin(da, d_Nvert_o, INSERT_VALUES, d_lNvert_o);
        DMGlobalToLocalEnd(da, d_Nvert_o, INSERT_VALUES, d_lNvert_o);
        VecCopy(d_P, d_P_o);
      }

      VecCopy(d_Ucont_o, d_Ucont_rm1);
      VecCopy(d_Ucont, d_Ucont_o);
      VecCopy(d_Ucat, d_Ucat_o);

      DMGlobalToLocalBegin(fda, d_Ucont_o, INSERT_VALUES, d_lUcont_o);
      DMGlobalToLocalEnd(fda, d_Ucont_o, INSERT_VALUES, d_lUcont_o);

      DMGlobalToLocalBegin(fda, d_Ucont_rm1, INSERT_VALUES, d_lUcont_rm1);
      DMGlobalToLocalEnd(fda, d_Ucont_rm1, INSERT_VALUES, d_lUcont_rm1);

      DMGlobalToLocalBegin(fda, d_Ucat, INSERT_VALUES, d_lUcat_old);
      DMGlobalToLocalEnd(fda, d_Ucat, INSERT_VALUES, d_lUcat_old);
 
}

void UData::Contra2Cart_single(Cmpnts &csi, Cmpnts &eta, Cmpnts &zet, 
                               Cmpnts &ucont, Cmpnts *ucat)
{
    double det = csi.x * (eta.y * zet.z - eta.z * zet.y) -
                 csi.y * (eta.x * zet.z - eta.z * zet.x) +
                 csi.z * (eta.x * zet.y - eta.y * zet.x);

    double det0 = ucont.x * (eta.y * zet.z - eta.z * zet.y) -
                  ucont.y * (csi.y * zet.z - csi.z * zet.y) +
                  ucont.z * (csi.y * eta.z - csi.z * eta.y);

    double det1 = -ucont.x * (eta.x * zet.z - eta.z * zet.x) +
                   ucont.y * (csi.x * zet.z - csi.z * zet.x) -
                   ucont.z * (csi.x * eta.z - csi.z * eta.x);

    double det2 = ucont.x * (eta.x * zet.y - eta.y * zet.x) -
                  ucont.y * (csi.x * zet.y - csi.y * zet.x) +
                  ucont.z * (csi.x * eta.y - csi.y * eta.x);

    (*ucat).x = det0 / det;
    (*ucat).y = det1 / det;
    (*ucat).z = det2 / det;
}


void UData::Contra2Cart()
{
    int    i, j, k;
    //Get the DMs 
    DM da = d_grid->getDA();
    DM fda = d_grid->getFDA();

    DMDALocalInfo  info;
    DMDAGetLocalInfo(da, &info);

    int xs = info.xs, xe = info.xs + info.xm;
    int ys = info.ys, ye = info.ys + info.ym;
    int zs = info.zs, ze = info.zs + info.zm;
    int mx = info.mx, my = info.my, mz = info.mz;

    int lxs, lxe, lys, lye, lzs, lze;

    lxs = xs; lxe = xe; lys = ys; lye = ye; lzs = zs; lze = ze;

    if (lxs==0) lxs++;
    if (lxe==mx) lxe--;
    if (lys==0) lys++;
    if (lye==my) lye--;
    if (lzs==0) lzs++;
    if (lze==mz) lze--;

    PetscReal mat[3][3], det, det0, det1, det2;
    
    
    PetscReal ***aj;
    Cmpnts ***ucat, ***lucat_o;
    Cmpnts ***lucat, ***lucont;
    PetscReal ***nvert;
    PetscReal q[3]; //local working array

    Cmpnts ***icsi, ***jeta, ***kzet;
    Cmpnts ***csi, ***eta, ***zet;
    Cmpnts ***coor;
   
    Vec Coor;
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec ICsi = d_grid->getlICsi();
    Vec JEta = d_grid->getlJEta();
    Vec KZet = d_grid->getlKZet();
    Vec Aj = d_grid->getlAj();

    DMGetCoordinatesLocal(da, &Coor);
    DMDAVecGetArray(fda, Coor, &coor);
    
    DMDAVecGetArray(fda, ICsi, &icsi);
    DMDAVecGetArray(fda, JEta, &jeta);
    DMDAVecGetArray(fda, KZet, &kzet);
    
    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    
    DMDAVecGetArray(da,  Aj,  &aj);
    DMDAVecGetArray(da, d_lNvert, &nvert);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    DMDAVecGetArray(fda, d_lUcont, &lucont);
    DMDAVecGetArray(fda, d_Ucat,  &ucat);
    DMDAVecGetArray(fda, d_lUcat_old,  &lucat_o);
    
    PetscInt i_periodic = d_grid->isIPeriodic();
    PetscInt j_periodic = d_grid->isJPeriodic();
    PetscInt k_periodic = d_grid->isKPeriodic();
    PetscInt ii_periodic = d_grid->isIIPeriodic();
    PetscInt jj_periodic = d_grid->isJJPeriodic();
    PetscInt kk_periodic = d_grid->isKKPeriodic();

    // important
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
                        lucont[k][j][i] = lucont[c][b][a];
                    }
                }
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
                if (nvert[k][j][i] < 0.1 ) {
                    mat[0][0] = (csi[k][j][i].x);
                    mat[0][1] = (csi[k][j][i].y);
                    mat[0][2] = (csi[k][j][i].z);
                          
                    mat[1][0] = (eta[k][j][i].x);
                    mat[1][1] = (eta[k][j][i].y);
                    mat[1][2] = (eta[k][j][i].z);
                          
                    mat[2][0] = (zet[k][j][i].x);
                    mat[2][1] = (zet[k][j][i].y);
                    mat[2][2] = (zet[k][j][i].z);
            
            
                    int iLL =i-2,  iL =i-1,  iR= i,  iRR =i+1;
                    int jLL =j-2,  jL =j-1,  jR= j,  jRR =j+1;
                    int kLL=k-2, kL=k-1, kR=k, kRR=k+1;
            
                    q[0] = 0.5 * ( lucont[k][j][iL].x + lucont[k][j][iR].x);
                    q[1] = 0.5 * ( lucont[k][jL][i].y + lucont[k][jR][i].y);
                    q[2] = 0.5 * ( lucont[kL][j][i].z + lucont[kR][j][i].z);
            
                    det = mat[0][0]*(mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1]) -
                          mat[0][1]*(mat[1][0]*mat[2][2]-mat[1][2]*mat[2][0]) +
                          mat[0][2]*(mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0]);

                    det0 = q[0]*(mat[1][1]*mat[2][2] - mat[1][2]*mat[2][1]) -
                           q[1]*(mat[0][1]*mat[2][2] - mat[0][2]*mat[2][1]) +
                           q[2]*(mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1]);

                    det1 =-q[0]*(mat[1][0]*mat[2][2] - mat[1][2]*mat[2][0]) +
                           q[1]*(mat[0][0]*mat[2][2] - mat[0][2]*mat[2][0]) -
                           q[2]*(mat[0][0]*mat[1][2] - mat[0][2]*mat[1][0]);
 
                    det2 = q[0]*(mat[1][0]*mat[2][1] - mat[1][1]*mat[2][0]) -
                           q[1]*(mat[0][0]*mat[2][1] - mat[0][1]*mat[2][0]) +
                           q[2]*(mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]);

                    ucat[k][j][i].x = det0 / det;
                    ucat[k][j][i].y = det1 / det;
                    ucat[k][j][i].z = det2 / det;
                }
            }
    
    DMDAVecRestoreArray(fda, d_Ucat,  &ucat);
    
    DMGlobalToLocalBegin(fda, d_Ucat, INSERT_VALUES, d_lUcat);
    DMGlobalToLocalEnd(fda, d_Ucat, INSERT_VALUES, d_lUcat);
    
    
    
    if (d_grid->isPeriodic()) {
        DMDAVecGetArray(fda, d_Ucat,  &ucat);
        DMDAVecGetArray(fda, d_lUcat,  &lucat);
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
                        lucat[k][j][i] = lucat[c][b][a];
                        ucat[k][j][i] = lucat[c][b][a];
                    }
               }
        DMDAVecRestoreArray(fda, d_Ucat,  &ucat);
        DMDAVecRestoreArray(fda, d_lUcat,  &lucat);    
    
        //second call is needed
        DMGlobalToLocalBegin(fda, d_Ucat, INSERT_VALUES, d_lUcat);
        DMGlobalToLocalEnd(fda, d_Ucat, INSERT_VALUES, d_lUcat);
    
    }
    
    
    //second call is needed
    DMGlobalToLocalBegin(fda, d_Ucat, INSERT_VALUES, d_lUcat);
    DMGlobalToLocalEnd(fda, d_Ucat, INSERT_VALUES, d_lUcat);


    
    DMDAVecGetArray(fda, d_lUcat,  &lucat);
    DMDAVecGetArray(fda, d_Ucat,  &ucat);
    
    PetscReal ***ustar;
    DMDAVecGetArray(da, d_lUstar, &ustar);

    
    PetscReal ***p;
    DMDAVecGetArray(da, d_lP, &p);

    for (k=zs; k<ze; k++)
        for (j=ys; j<ye; j++)
            for (i=xs; i<xe; i++) {
                int solid_flag=0;
 
                //Not moving and in solid
                if ( !d_movefsi && !d_rotatefsi && 
                     (int)(nvert[k][j][i]+0.1)==3 ) {
                     Set(&ucat[k][j][i], 0);
                     continue;
                }
    

               // wall function for boundary
               if ( j!=0 && j!=my-1 && k!=0 && k!=mz-1 && 
                   ( (d_grid->getBC(0)==-1 && i==1) || 
                     (d_grid->getBC(1)==-1 && i==mx-2) ) ) {

                   double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + 
                                       csi[k][j][i].y*csi[k][j][i].y + 
                                       csi[k][j][i].z*csi[k][j][i].z );
                   double sb, sc; 
                   double ni[3], nj[3], nk[3];
                   Cmpnts Uc, Ua, Ub;
            
                   Ua.x = Ua.y = Ua.z = 0;
                   sb = 0.5/aj[k][j][i]/area;
            
                   Ub = lucat[k][j][i]; 

                   if (i==1) {    
                       Uc = lucat[k][j][i+1];  
                       sc = 1.0/aj[k][j][i]/area + 0.5/aj[k][j][i+1]/area;
                   } else {
                       Uc = lucat[k][j][i-1]; 
                       sc = 1.0/aj[k][j][i]/area + 0.5/aj[k][j][i-1]/area;
                   } 

                   Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], 
                                    ni, nj, nk);
                   if (i==mx-2) {
                       ni[0]*=-1, ni[1]*=-1, ni[2]*=-1;
                   }
            
                   double nu = 1./d_ren;
            
                   double nx=ni[0], ny=ni[1], nz=ni[2];

                   int i1 = i+1, j1 = j, k1 = k;

                   if (i==mx-2) {
                       i1 = i-1; j1 = j; k1 = k;
                   }

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

                    Compute_dscalar_center(i1, j1, k1, mx, my, mz, 
                                           p, nvert, &dpdc, &dpde, &dpdz );

                    Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                         eta0, eta1, eta2, 
                                         zet0, zet1, zet2, ajc, 
                                         dpdc, dpde, dpdz, 
                                         &dp_dx, &dp_dy, &dp_dz);

                    if (!d_dp_wm) {
                        dp_dx=0.0; dp_dy=0.0; dp_dz=0.0;
                    }

                    double Ub_x = Ub.x - Ua.x;
                    double Ub_y = Ub.y - Ua.y;
                    double Ub_z = Ub.z - Ua.z;
                    double un = Ub_x * nx + Ub_y * ny + Ub_z * nz;
                    double Ub_t_x = Ub_x - un * nx;
                    double Ub_t_y = Ub_y - un * ny;
                    double Ub_t_z = Ub_z - un * nz;
                    double Ub_t_mag = sqrt(Ub_t_x*Ub_t_x + 
                                           Ub_t_y*Ub_t_y + 
                                           Ub_t_z*Ub_t_z );
                     
                    if (d_roughness_size>1.e-19) 
                        ustar[k][j][i] = WallFunctions::utau_wf(
                                             nu, d_roughness_size, 
                                             sb, Ub_t_mag);
                    else 
                        ustar[k][j][i] = d_wallf->find_utau_Cabot(
                                              nu, Ub_t_mag, 
                                              sb, 0.01, 0);
                    Cmpnts Ughost;

                    double nu_t=0.0;

                    if (Ub_t_mag<1.e-20) {
                        ustar[k][j][i]=0;
                        Set ( &Ughost, 0);
                    } else {

                        double tau_w = ustar[k][j][i]*ustar[k][j][i];
                        double Ughost_t_mag = Ub_t_mag - 
                                              2*tau_w*sb / (nu+nu_t);
                        Ughost.x = Ub_t_x / (Ub_t_mag) * Ughost_t_mag;
                        Ughost.y = Ub_t_y / (Ub_t_mag) * Ughost_t_mag;
                        Ughost.z = Ub_t_z / (Ub_t_mag) * Ughost_t_mag;
                    
                        if (nvert[k][j][i]>0.1) Set (&Ughost, 0);
                    }
            
                   if (i==1) ucat[k][j][i-1] = Ughost;
                   else ucat[k][j][i+1] = Ughost;

                   solid_flag=1;
                }
        
                // wall function for boundary
                if ( i!=0 && i!=mx-1 && k!=0 && k!=mz-1 && 
                   ( (d_grid->getBC(2)==-1 && j==1) || 
                     (d_grid->getBC(3)==-1 &&  j==my-2) ) ) {

                    double area = sqrt( eta[k][j][i].x*eta[k][j][i].x + 
                                        eta[k][j][i].y*eta[k][j][i].y + 
                                        eta[k][j][i].z*eta[k][j][i].z );
                    double sb, sc; 
                    double ni[3], nj[3], nk[3];
                    Cmpnts Uc, Ua, Ub;
            
                    Ua.x = Ua.y = Ua.z = 0;
                    sb = 0.5/aj[k][j][i]/area;
            
                    Ub = lucat[k][j][i];
    
                    if (j==1) {    
                        Uc = lucat[k][j+1][i];   
                        sc = 1.0/aj[k][j][i]/area + 0.5/aj[k][j+1][i]/area;
                    } else {
                        Uc = lucat[k][j-1][i];  
                        sc = 1.0/aj[k][j][i]/area + 0.5/aj[k][j-1][i]/area;
                    }


                    Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], 
                                     ni, nj, nk);
                    if (j==my-2) {
                        nj[0]*=-1, nj[1]*=-1, nj[2]*=-1;
                    }
             
                    double nu = 1./d_ren;

                    double nx=nj[0], ny=nj[1], nz=nj[2];
 
                    int i1 = i, j1 = j+1, k1 = k;

                    if (j==my-2) {
                        i1 = i; j1 = j-1; k1 = k;
                    }

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

                    Compute_dscalar_center(i1, j1, k1, mx, my, mz, 
                                           p, nvert, &dpdc, &dpde, &dpdz );

                    Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                         eta0, eta1, eta2, 
                                         zet0, zet1, zet2, ajc, 
                                         dpdc, dpde, dpdz, 
                                         &dp_dx, &dp_dy, &dp_dz);

                    if (!d_dp_wm) {
                        dp_dx=0.0; dp_dy=0.0; dp_dz=0.0;
                    }


                    double Ub_x = Ub.x - Ua.x; 
                    double Ub_y = Ub.y - Ua.y;
                    double Ub_z = Ub.z - Ua.z;
                    double un = Ub_x * nx + Ub_y * ny + Ub_z * nz;
                    double Ub_t_x = Ub_x - un * nx; 
                    double Ub_t_y = Ub_y - un * ny;
                    double  Ub_t_z = Ub_z - un * nz;
                    double Ub_t_mag = sqrt( Ub_t_x*Ub_t_x + 
                                            Ub_t_y*Ub_t_y + Ub_t_z*Ub_t_z );
            

                    Cmpnts Utmp;
                    if (d_roughness_size>1.e-19) 
                        ustar[k][j][i] = WallFunctions::utau_wf(
                                             nu, d_roughness_size, 
                                             sb, Ub_t_mag);
                    else 
                        ustar[k][j][i] = d_wallf->find_utau_Cabot(
                                             nu, Ub_t_mag, 
                                             sb, 0.01, 0);
                    Cmpnts Ughost;
        
                    double nu_t=0.0;

    
                    if (Ub_t_mag<1.e-20) {
                        ustar[k][j][i]=0;
                        Set ( &Ughost, 0);
                    } else {

                        double tau_w = ustar[k][j][i]*ustar[k][j][i];
 
                        double Ughost_t_mag = Ub_t_mag - 
                                              2*tau_w*sb / (nu+nu_t);
                        Ughost.x = Ub_t_x / (Ub_t_mag) * Ughost_t_mag;
                        Ughost.y = Ub_t_y / (Ub_t_mag) * Ughost_t_mag;
                        Ughost.z = Ub_t_z / (Ub_t_mag) * Ughost_t_mag;
                    }
                
                    if (j==1) ucat[k][j-1][i] = Ughost;
                    else ucat[k][j+1][i] = Ughost;
                
                    solid_flag=1;
                } 

    
                if (d_grid->getBC(3)==13 && j==my-1) { //slip top wall
                    ucat[k][j][i].x = ucat[k][j-1][i].x;
                    ucat[k][j][i].z = ucat[k][j-1][i].z;
                    ucat[k][j][i].y = -ucat[k][j-1][i].y;
                }
                if (d_grid->getBC(3)==14 && j==my-1) { //slip top wall
                    ucat[k][j][i].x = ucat[k][j-1][i].x;
                    ucat[k][j][i].y = ucat[k][j-1][i].y;
                    ucat[k][j][i].z = -ucat[k][j-1][i].z;
                }
        
                /*slip BC*/
                if (d_grid->getBC(0)==10 && i==0 && (j!=0 && k!=0) ) {
                    double g[3][3], G[3][3];
                    g[0][0]=csi[k][j][i+1].x;
                    g[0][1]=csi[k][j][i+1].y;
                    g[0][2]=csi[k][j][i+1].z;
                    g[1][0]=eta[k][j][i+1].x; 
                    g[1][1]=eta[k][j][i+1].y; 
                    g[1][2]=eta[k][j][i+1].z;
                  
                    g[2][0]=zet[k][j][i+1].x; 
                    g[2][1]=zet[k][j][i+1].y; 
                    g[2][2]=zet[k][j][i+1].z;
             
                    Calculate_Covariant_metrics(g, G);
                    double xcsi=G[0][0], ycsi=G[1][0], zcsi=G[2][0];
                    double nx = - xcsi, ny = - ycsi, nz = - zcsi;
                    double sum=sqrt(nx*nx+ny*ny+nz*nz);
                    nx /= sum, ny /= sum, nz /= sum;
            
                    Cmpnts U = ucat[k][j][i+1];
                    double un = U.x*nx + U.y*ny + U.z*nz;
                    ucat[k][j][i].x = U.x - 2 * un * nx;
                    ucat[k][j][i].y = U.y - 2 * un * ny;
                    ucat[k][j][i].z = U.z - 2 * un * nz;
            
                    if ( nvert[k][j][i+1]>0.1 ) Set(&ucat[k][j][i],0);
                    if (solid_flag) Set(&ucat[k][j][i], 0);
                }
        
                if (d_grid->getBC(1)==10 && i==mx-1 && (j!=0 && k!=0) ) {
                    double g[3][3], G[3][3];
                    g[0][0]=csi[k][j][i-1].x; 
                    g[0][1]=csi[k][j][i-1].y; 
                    g[0][2]=csi[k][j][i-1].z;
 
                    g[1][0]=eta[k][j][i-1].x; 
                    g[1][1]=eta[k][j][i-1].y; 
                    g[1][2]=eta[k][j][i-1].z;
 
                    g[2][0]=zet[k][j][i-1].x; 
                    g[2][1]=zet[k][j][i-1].y;
                    g[2][2]=zet[k][j][i-1].z;
            
                    Calculate_Covariant_metrics(g, G);
                    double xcsi=G[0][0], ycsi=G[1][0], zcsi=G[2][0];
                    double nx = xcsi, ny = ycsi, nz = zcsi;
                    double sum=sqrt(nx*nx+ny*ny+nz*nz);
                    nx /= sum, ny /= sum, nz /= sum;
            
                    Cmpnts U = ucat[k][j][i-1];
                    double un = U.x*nx + U.y*ny + U.z*nz;
                    ucat[k][j][i].x = U.x - 2 * un * nx;
                    ucat[k][j][i].y = U.y - 2 * un * ny;
                    ucat[k][j][i].z = U.z - 2 * un * nz;
            
                    if ( nvert[k][j][i-1]>0.1 ) Set(&ucat[k][j][i],0);
                    if (solid_flag) Set(&ucat[k][j][i], 0);
                }
        
                if (d_grid->getBC(2)==10 && j==0 && (i!=0 && k!=0) ) {
                    double g[3][3], G[3][3];
                    g[0][0]=csi[k][j+1][i].x;
                    g[0][1]=csi[k][j+1][i].y;
                    g[0][2]=csi[k][j+1][i].z;
                    g[1][0]=eta[k][j+1][i].x;
                    g[1][1]=eta[k][j+1][i].y; 
                    g[1][2]=eta[k][j+1][i].z;
                    g[2][0]=zet[k][j+1][i].x;
                    g[2][1]=zet[k][j+1][i].y;
                    g[2][2]=zet[k][j+1][i].z;
            
                    Calculate_Covariant_metrics(g, G);
                    double xeta=G[0][1], yeta=G[1][1], zeta=G[2][1];
                    double nx = - xeta, ny = - yeta, nz = - zeta;
                    double sum=sqrt(nx*nx+ny*ny+nz*nz);
                    nx /= sum, ny /= sum, nz /= sum;
            
                    Cmpnts U = ucat[k][j+1][i];
                    double un = U.x*nx + U.y*ny + U.z*nz;
                    ucat[k][j][i].x = U.x - 2 * un * nx;
                    ucat[k][j][i].y = U.y - 2 * un * ny;
                    ucat[k][j][i].z = U.z - 2 * un * nz;
            
                    if (nvert[k][j+1][i]>0.1 ) Set(&ucat[k][j][i],0);
                    if (solid_flag) Set(&ucat[k][j][i], 0);
                }
        
                if (std::abs(d_grid->getBC(3))==10 && 
                    j==my-1 && (i!=0 && k!=0) ) {

                   double g[3][3], G[3][3];
                   g[0][0]=csi[k][j-1][i].x; 
                   g[0][1]=csi[k][j-1][i].y;
                   g[0][2]=csi[k][j-1][i].z;
                   g[1][0]=eta[k][j-1][i].x;
                   g[1][1]=eta[k][j-1][i].y; 
                   g[1][2]=eta[k][j-1][i].z;
                   g[2][0]=zet[k][j-1][i].x;
                   g[2][1]=zet[k][j-1][i].y;
                   g[2][2]=zet[k][j-1][i].z;
            
                   Calculate_Covariant_metrics(g, G);
                   double xeta=G[0][1], yeta=G[1][1], zeta=G[2][1];
                   double nx = xeta, ny = yeta, nz = zeta;
                   double sum=sqrt(nx*nx+ny*ny+nz*nz);
                   nx /= sum, ny /= sum, nz /= sum;
            
                   Cmpnts U = ucat[k][j-1][i];
                   double un = U.x*nx + U.y*ny + U.z*nz;
                   ucat[k][j][i].x = U.x - 2 * un * nx;
                   ucat[k][j][i].y = U.y - 2 * un * ny;
                   ucat[k][j][i].z = U.z - 2 * un * nz;
            
                   if ( nvert[k][j-1][i]>0.1 ) Set(&ucat[k][j][i],0);
                   if (solid_flag) Set(&ucat[k][j][i], 0);
                }
        
                /* noslip BC */
                if (i==0 && d_grid->getBC(0)==1 && (j!=0 && k!=0) ) {
                    AxC(-1, lucat[k][j][i+1], &ucat[k][j][i]);
                    solid_flag=1;
                }
                if (i==mx-1 && d_grid->getBC(1)==1 && (j!=0 && k!=0) ) {
                    AxC(-1, lucat[k][j][i-1], &ucat[k][j][i]);
                    solid_flag=1;
                }
        
                if (j==0 && d_grid->getBC(2)==1 && (i!=0 && k!=0) ) {
                    AxC(-1, lucat[k][j+1][i], &ucat[k][j][i]);
                    solid_flag=1;
                }
                if (j==my-1 && d_grid->getBC(3)==1 && (i!=0 && k!=0) ) {
                    AxC(-1, lucat[k][j-1][i], &ucat[k][j][i]);
                    solid_flag=1;
                }
        
                if (k==0 && d_grid->getBC(4)==1 && (i!=0 && j!=0) ) {
                    AxC(-1, lucat[k+1][j][i], &ucat[k][j][i]);
                    solid_flag=1;
                }
                if (k==mz-1 && d_grid->getBC(5)==1  && (i!=0 && j!=0) ) {
                    AxC(-1, lucat[k-1][j][i], &ucat[k][j][i]);
                    solid_flag=1;
                }
        
                //cavity problem 
                if (j==my-1 && d_grid->getBC(3)==2) {
                    if (solid_flag) {
                        ucat[k][j][i].x=2.0-lucat_o[k][j-1][i].x;
                        ucat[k][j][i].y=-lucat_o[k][j-1][i].y;
                        ucat[k][j][i].z=-lucat_o[k][j-1][i].z;
                        //Set(&ucat[k][j][i], 0);
                    } else {
                        ucat[k][j][i].x=2.0-lucat[k][j-1][i].x;
                        ucat[k][j][i].y=-lucat[k][j-1][i].y;
                        ucat[k][j][i].z=-lucat[k][j-1][i].z;
                    }
                }
        
                // couette flow j=0
                if (j==0 && d_grid->getBC(2)==12) {
                    ucat[k][j][i].x=-lucat[k][j+1][i].x;
                    ucat[k][j][i].y=-lucat[k][j+1][i].y;
                    ucat[k][j][i].z=2.0-lucat[k][j+1][i].z;
            
                    if (solid_flag) Set(&ucat[k][j][i], 0);
                } 
        
                // couette flow j=my-1
                if (j==my-1 && d_grid->getBC(3)==12) {
                    ucat[k][j][i].x=-lucat[k][j-1][i].x;
                    ucat[k][j][i].y=-lucat[k][j-1][i].y;
                    ucat[k][j][i].z=2.0-lucat[k][j-1][i].z;
            
                    if (solid_flag) Set(&ucat[k][j][i], 0);
                }
        
                // body-fitted cylinder : inflow & outflow
                if (d_grid->getBC(0)==11 && i==0 && 
                    (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) {
                    double zc = (coor[k][j][i+1].z + coor[k-1][j][i+1].z + 
                              coor[k][j-1][i+1].z + coor[k-1][j-1][i+1].z)*0.25;
                    if( zc <= 0 ) {
                        ucat[k][j][i].x = - lucat[k][j][i+1].x;
                        ucat[k][j][i].y = - lucat[k][j][i+1].y;
                        ucat[k][j][i].z = 2.0 - lucat[k][j][i+1].z;
                
                        if (solid_flag) Set(&ucat[k][j][i], 0);
                    }
                }
        
        

                /* outflow */
                if (d_grid->getBC(3)==4 && j==my-1 && 
                    (i!=0 && i!=mx-1 && k!=0 && k!=mz-1) ) {
                    ucat[k][j][i] = lucat[k][j-1][i];
                }

                if (d_grid->getBC(5)==4 && k==mz-1 && 
                    (i!=0 && i!=mx-1 && j!=0 && j!=my-1) ) {
                    ucat[k][j][i] = lucat[k-1][j][i];    // xiaolei add
                    if ( nvert[k-1][j][i]>0.1 ) Set(&ucat[k][j][i],0);
                    if (solid_flag) Set(&ucat[k][j][i], 0);
                }
            }
        
   

    DMDAVecRestoreArray(da, d_lP, &p);
    DMDAVecRestoreArray(da, d_lUstar, &ustar);
    DMDAVecRestoreArray(fda, d_lUcont, &lucont);
    DMDAVecRestoreArray(fda, d_lUcat,  &lucat);
    DMDAVecRestoreArray(fda, d_Ucat,  &ucat);
    DMDAVecRestoreArray(fda, d_lUcat_old,  &lucat_o);
    
    DMGlobalToLocalBegin(fda, d_Ucat, INSERT_VALUES, d_lUcat);
    DMGlobalToLocalEnd(fda, d_Ucat, INSERT_VALUES, d_lUcat);
    

    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    
    DMDAVecRestoreArray(fda, ICsi, &icsi);
    DMDAVecRestoreArray(fda, JEta, &jeta);
    DMDAVecRestoreArray(fda, KZet, &kzet);
  
    DMDAVecRestoreArray(da,  Aj,  &aj);
    DMDAVecRestoreArray(da, d_lNvert, &nvert);
    
    DMDAVecRestoreArray(fda, Coor, &coor);
    
    if (d_grid->isPeriodic()) {
        DMDAVecGetArray(fda, d_Ucat,  &ucat);
        DMDAVecGetArray(fda, d_lUcat,  &lucat);
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
                        lucat[k][j][i] = lucat[c][b][a];
                        ucat[k][j][i] = lucat[c][b][a];
                    }
                }
        DMDAVecRestoreArray(fda, d_Ucat,  &ucat);
        DMDAVecRestoreArray(fda, d_lUcat,  &lucat);

        DMGlobalToLocalBegin(fda, d_Ucat, INSERT_VALUES, d_lUcat); 
        DMGlobalToLocalEnd(fda, d_Ucat, INSERT_VALUES, d_lUcat);
    }
}

void UData::Average(PetscInt ti)
{
    //Leave if no averaging
    if (!d_averaging) return;

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
  
    Cmpnts ***ucat, ***csi, ***eta, ***zet;
    Cmpnts ***u2sum, ***u_cross_sum;
    Cmpnts ***vort_sum, ***vort2_sum, ***uuusum;
    Cmpnts ***u2sum_phase, ***u_cross_sum_phase;
    Cmpnts ***vort_sum_phase, ***vort2_sum_phase, ***uuusum_phase;
    PetscReal ***p;
    PetscReal ***aj, ***nvert;
    PetscReal ***p2sum, ***du2sum, ***udpsum;
    PetscReal ***p2sum_phase, ***du2sum_phase, ***udpsum_phase;
    
    Vec Csi = d_grid->getlCsi();
    Vec Eta = d_grid->getlEta();
    Vec Zet = d_grid->getlZet();
    Vec Aj = d_grid->getlAj(); 

    VecAXPY(d_Ucat_sum, 1., d_Ucat);
    VecAXPY(d_P_sum, 1., d_P);
   
    PetscPrintf(PETSC_COMM_WORLD, "...Averaging (%d)...\n", ti); 
    double max_norm;
    VecMax(d_Ucat_sum, &i, &max_norm);
    PetscPrintf(PETSC_COMM_WORLD, "...Max Average Ucat = %e \n", 
                max_norm / (double)ti);
        
    if (d_phase_averaging && (ti+d_ti_lastsave)%d_phase_averaging==0) {
        VecAXPY(d_Ucat_sum_phase, 1., d_Ucat);
        VecAXPY(d_P_sum_phase, 1., d_P);
    }
    
    DMDAVecGetArray(fda, d_lUcat, &ucat);
    DMDAVecGetArray(da, d_lNvert, &nvert);
    DMDAVecGetArray(da, d_lP, &p);

    DMDAVecGetArray(fda, Csi, &csi);
    DMDAVecGetArray(fda, Eta, &eta);
    DMDAVecGetArray(fda, Zet, &zet);
    DMDAVecGetArray(da, Aj, &aj);
    
    DMDAVecGetArray(fda, d_Ucat_square_sum, &u2sum);
    DMDAVecGetArray(fda, d_Ucat_cross_sum, &u_cross_sum);
    
    if (d_phase_averaging) {
        DMDAVecGetArray(fda, d_Ucat_square_sum_phase, &u2sum_phase);
        DMDAVecGetArray(fda, d_Ucat_cross_sum_phase, &u_cross_sum_phase);
    }
   
    if (d_averaging >= 2)  {
        DMDAVecGetArray(da, d_P_square_sum, &p2sum);
        if (d_phase_averaging) {
            DMDAVecGetArray(da, d_P_square_sum_phase, &p2sum_phase);
        }
    } 
 
    if (d_averaging >= 3) {
        
        DMDAVecGetArray(da, d_Udp_sum, &udpsum);
        DMDAVecGetArray(da, d_dU2_sum, &du2sum);
        DMDAVecGetArray(fda, d_UUU_sum, &uuusum);
        DMDAVecGetArray(fda, d_Vort_sum, &vort_sum);
        DMDAVecGetArray(fda, d_Vort_square_sum, &vort2_sum);
        if (d_phase_averaging) {
            DMDAVecGetArray(da, d_Udp_sum_phase, &udpsum_phase);
            DMDAVecGetArray(da, d_dU2_sum_phase, &du2sum_phase);
            DMDAVecGetArray(fda, d_UUU_sum_phase, &uuusum_phase);
            DMDAVecGetArray(fda, d_Vort_sum_phase, &vort_sum_phase);
            DMDAVecGetArray(fda, d_Vort_square_sum_phase, &vort2_sum_phase);
        } 
    }
    
    for (k=lzs; k<lze; k++)
        for (j=lys; j<lye; j++)
            for (i=lxs; i<lxe; i++) {
 
                PetscReal U = ucat[k][j][i].x;
                PetscReal V = ucat[k][j][i].y;
                PetscReal W = ucat[k][j][i].z;

                u2sum[k][j][i].x += U*U; 
                u2sum[k][j][i].y += V*V; 
                u2sum[k][j][i].z += W*W; 

                if (d_averaging>=2) 
                    p2sum[k][j][i] += p[k][j][i] * p[k][j][i];

                u_cross_sum[k][j][i].x += U*V;
                u_cross_sum[k][j][i].y += V*W; 
                u_cross_sum[k][j][i].z += W*U;
            }

        
    if (d_phase_averaging && (ti+d_ti_lastsave)%d_phase_averaging==0) {
        for (k=lzs; k<lze; k++)
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {
 
                    PetscReal U = ucat[k][j][i].x;
                    PetscReal V = ucat[k][j][i].y;
                    PetscReal W = ucat[k][j][i].z;

                    u2sum_phase[k][j][i].x += U*U;
                    u2sum_phase[k][j][i].y += V*V;
                    u2sum_phase[k][j][i].z += W*W; 
                    u_cross_sum_phase[k][j][i].x += U*V; 
                    u_cross_sum_phase[k][j][i].y += V*W; 
                    u_cross_sum_phase[k][j][i].z += W*U; 

                    if (d_averaging>=2) 
                        p2sum_phase[k][j][i] += p[k][j][i] * p[k][j][i];
                } 
    }

    if (d_averaging>=3) {
        PetscInt i_p = d_grid->isIPeriodic();
        PetscInt j_p = d_grid->isJPeriodic();
        PetscInt k_p = d_grid->isKPeriodic();
        PetscInt ii_p = d_grid->isIIPeriodic();
        PetscInt jj_p = d_grid->isJJPeriodic();
        PetscInt kk_p = d_grid->isKKPeriodic();

        for (k=lzs; k<lze; k++)
            for (j=lys; j<lye; j++)
                for (i=lxs; i<lxe; i++) {

                    PetscReal U = ucat[k][j][i].x;
                    PetscReal V = ucat[k][j][i].y;
                    PetscReal W = ucat[k][j][i].z;

                    PetscReal dudc, dvdc, dwdc;
                    PetscReal dude, dvde, dwde;
                    PetscReal dudz, dvdz, dwdz;

                    PetscReal du_dx, du_dy, du_dz;
                    PetscReal dv_dx, dv_dy, dv_dz;
                    PetscReal dw_dx, dw_dy, dw_dz;
                    PetscReal dpdc, dpde, dpdz;
                    PetscReal dp_dx, dp_dy, dp_dz;

                    PetscReal csi0 = csi[k][j][i].x; 
                    PetscReal csi1 = csi[k][j][i].y;
                    PetscReal csi2 = csi[k][j][i].z;
                    PetscReal eta0 = eta[k][j][i].x;
                    PetscReal eta1 = eta[k][j][i].y;
                    PetscReal eta2 = eta[k][j][i].z;
                    PetscReal zet0 = zet[k][j][i].x;
                    PetscReal zet1 = zet[k][j][i].y;
                    PetscReal zet2 = zet[k][j][i].z;
                    PetscReal ajc = aj[k][j][i];
                
                    Compute_du_center(i, j, k, 
                                      mx, my, mz, 
                                      ucat, nvert,
                                      i_p, ii_p, j_p, jj_p, k_p, kk_p, 
                                      &dudc, &dvdc, &dwdc, 
                                      &dude, &dvde, &dwde, 
                                      &dudz, &dvdz, &dwdz);
                    Compute_dscalar_center(i, j, k, 
                                           mx, my, mz, 
                                           p, nvert, 
                                           &dpdc, &dpde, &dpdz);
                    Compute_du_dxyz(csi0, csi1, csi2, 
                                    eta0, eta1, eta2, 
                                    zet0, zet1, zet2, 
                                    ajc, 
                                    dudc, dvdc, dwdc, 
                                    dude, dvde, dwde, 
                                    dudz, dvdz, dwdz, 
                                    &du_dx, &dv_dx, &dw_dx, 
                                    &du_dy, &dv_dy, &dw_dy, 
                                    &du_dz, &dv_dz, &dw_dz);
                    Compute_dscalar_dxyz(csi0, csi1, csi2, 
                                         eta0, eta1, eta2, 
                                         zet0, zet1, zet2, 
                                         ajc, 
                                         dpdc, dpde, dpdz, 
                                         &dp_dx, &dp_dy, &dp_dz);

                    PetscReal vort_x = dw_dy - dv_dz;
                    PetscReal vort_y = du_dz - dw_dx;
                    PetscReal vort_z = dv_dx - du_dy;
            
                    vort_sum[k][j][i].x += vort_x;
                    vort_sum[k][j][i].y += vort_y;
                    vort_sum[k][j][i].z += vort_z;
            
                    vort2_sum[k][j][i].x += vort_x*vort_x;
                    vort2_sum[k][j][i].y += vort_y*vort_y;
                    vort2_sum[k][j][i].z += vort_z*vort_z;

                    udpsum[k][j][i] += U * dp_dx + V * dp_dy + W * dp_dz;

                    du2sum[k][j][i] += du_dx*du_dx+du_dy*du_dy+du_dz*du_dz;
                    du2sum[k][j][i] += dv_dx*dv_dx+dv_dy*dv_dy+dv_dz*dv_dz;
                    du2sum[k][j][i] += dw_dx*dw_dx+dw_dy*dw_dy+dw_dz*dw_dz;
            
                    uuusum[k][j][i].x += (U*U + V*V + W*W) * U;
                    uuusum[k][j][i].y += (U*U + V*V + W*W) * V;
                    uuusum[k][j][i].z += (U*U + V*V + W*W) * W;

                    if (d_phase_averaging && 
                        (ti+d_ti_lastsave)%d_phase_averaging==0) {

                        vort_sum_phase[k][j][i].x += vort_x;
                        vort_sum_phase[k][j][i].y += vort_y;
                        vort_sum_phase[k][j][i].z += vort_z;
              
                        vort2_sum_phase[k][j][i].x += vort_x*vort_x;
                        vort2_sum_phase[k][j][i].y += vort_y*vort_y;
                        vort2_sum_phase[k][j][i].z += vort_z*vort_z;

                        udpsum_phase[k][j][i] += 
                            U * dp_dx + V * dp_dy + W * dp_dz;

                        du2sum_phase[k][j][i] += 
                            du_dx*du_dx+du_dy*du_dy+du_dz*du_dz;
                        du2sum_phase[k][j][i] += 
                            dv_dx*dv_dx+dv_dy*dv_dy+dv_dz*dv_dz;
                        du2sum_phase[k][j][i] +=
                            dw_dx*dw_dx+dw_dy*dw_dy+dw_dz*dw_dz;
            
                        uuusum_phase[k][j][i].x += (U*U + V*V + W*W) * U;
                        uuusum_phase[k][j][i].y += (U*U + V*V + W*W) * V;
                        uuusum_phase[k][j][i].z += (U*U + V*V + W*W) * W;

                    }
            
                }
        
        
    }
    
    DMDAVecRestoreArray(fda, d_lUcat, &ucat);
    DMDAVecRestoreArray(da, d_lNvert, &nvert);
    DMDAVecRestoreArray(da, d_lP, &p);

    DMDAVecRestoreArray(da, Aj, &aj);
    DMDAVecRestoreArray(fda, Csi, &csi);
    DMDAVecRestoreArray(fda, Eta, &eta);
    DMDAVecRestoreArray(fda, Zet, &zet);
    
    DMDAVecRestoreArray(fda, d_Ucat_square_sum, &u2sum);
    DMDAVecRestoreArray(fda, d_Ucat_cross_sum, &u_cross_sum);
    
    if (d_phase_averaging) {
        DMDAVecRestoreArray(fda, d_Ucat_square_sum_phase, &u2sum_phase);
        DMDAVecRestoreArray(fda, d_Ucat_cross_sum_phase, &u_cross_sum_phase);
    }

    if (d_averaging >= 2) {
        DMDAVecRestoreArray(da, d_P_square_sum, &p2sum);
        if (d_phase_averaging) {
            DMDAVecRestoreArray(da, d_P_square_sum_phase, &p2sum_phase);
        }
    }
    
    if (d_averaging >= 3) {
        DMDAVecRestoreArray(da, d_Udp_sum, &udpsum);
        DMDAVecRestoreArray(da, d_dU2_sum, &du2sum);
        DMDAVecRestoreArray(fda, d_UUU_sum, &uuusum);
        DMDAVecRestoreArray(fda, d_Vort_sum, &vort_sum);
        DMDAVecRestoreArray(fda, d_Vort_square_sum, &vort2_sum);
        if (d_phase_averaging) {
            DMDAVecRestoreArray(da, d_Udp_sum_phase, &udpsum_phase);
            DMDAVecRestoreArray(da, d_dU2_sum_phase, &du2sum_phase);
            DMDAVecRestoreArray(fda, d_UUU_sum_phase, &uuusum_phase);
            DMDAVecRestoreArray(fda, d_Vort_sum_phase, &vort_sum_phase);
            DMDAVecRestoreArray(fda, d_Vort_square_sum_phase,&vort2_sum_phase);
        }
    }
    
    PetscPrintf(PETSC_COMM_WORLD, "...Averaging Done \n");
    MPI_Barrier(PETSC_COMM_WORLD);
}

void UData::ReadFile(char *filen, Vec U)
{
    
    PetscViewer viewer;
    PetscPrintf(PETSC_COMM_WORLD, "Reading %s ... \n", filen);
    if (d_read_hdf5) {
        PetscViewerHDF5Open(PETSC_COMM_WORLD,filen,FILE_MODE_READ,&viewer);
    } else
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,filen,FILE_MODE_READ,&viewer); 
    VecLoad(U, viewer);
    PetscViewerDestroy(&viewer);

}

void UData::WriteFile(char *filen, Vec U)
{
    
    PetscViewer viewer;
    PetscPrintf(PETSC_COMM_WORLD, "Writing %s ... \n", filen);
    if (d_write_hdf5) { 
        PetscViewerHDF5Open(PETSC_COMM_WORLD,filen,FILE_MODE_WRITE,&viewer);
    } else {
        PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
        PetscViewerBinarySetMPIIO(viewer);
        PetscViewerSetType(viewer, PETSCVIEWERBINARY);
        PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
        PetscViewerFileSetName(viewer, filen);
    }

    VecView(U, viewer);
    PetscViewerDestroy(&viewer);
}

void UData::PhaseNumber(PetscInt ti, PetscInt *phase, PetscInt *previous_ti)
{
    PetscInt n=0, t;

    for (PetscInt  i=d_ti_lastsave+1; i<=d_ti_lastsave+ti; i++) {
        if (i%d_phase_averaging==0) {
            n++;
            t = i - d_ti_lastsave;
        }
    }
    
    *phase = n;
    *previous_ti = t;
}


PetscErrorCode UData::ReadFromInput()
{
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
    PetscOptionsGetInt(PETSC_NULL, "-tiou", &d_tiout_ufield, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-tieu", &d_tiend_ufield, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-ti_lastsave", &d_ti_lastsave, PETSC_NULL);


    PetscOptionsGetInt(PETSC_NULL, "-rstart", &d_tistart, &d_restart);

    PetscOptionsGetInt(PETSC_NULL, "-totalsteps", &d_tisteps, PETSC_NULL);

    //if 1 do averaging; always begin with -rstart 0
    PetscOptionsGetInt(PETSC_NULL, "-averaging", &d_averaging, PETSC_NULL);  
    //Period of phase averaging 
    PetscOptionsGetInt(PETSC_NULL, "-phase_averaging", &d_phase_averaging, 
                       PETSC_NULL);      
    d_phase_averaging = std::max(d_phase_averaging, (PetscInt)0);
    if (d_averaging==0) d_phase_averaging=0;

    PetscOptionsGetReal(PETSC_NULL, "-ren", &d_ren, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-dt", &d_dt, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-St", &d_St, PETSC_NULL);
    d_dt_inflow = d_dt;
    PetscOptionsGetReal(PETSC_NULL, "-dt_inflow", &d_dt_inflow, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, "-roughness", &d_roughness_size, 
                        &d_rough_set);
    PetscOptionsGetInt(PETSC_NULL, "-dp_wm", &d_dp_wm, PETSC_NULL);

    PetscOptionsGetInt(PETSC_NULL, "-imm", &d_immersed, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-fsi", &d_movefsi, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, "-rfsi", &d_rotatefsi, PETSC_NULL);
  
    PetscOptionsGetString(PETSC_NULL,"-path", d_path, 256, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-field_path", 
                          d_fieldpath, 256, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-average_path", 
                          d_avepath, 256, PETSC_NULL);
    PetscOptionsGetString(PETSC_NULL,"-phase_path", 
                          d_phpath, 256, PETSC_NULL);
}
