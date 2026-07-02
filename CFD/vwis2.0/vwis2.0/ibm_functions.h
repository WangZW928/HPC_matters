#ifndef included_ibm_functions
#define included_ibm_functions

#include "petsc.h"
#include "functions.h"

//A 2d struct for some ibm stuff
typedef struct {
        PetscReal       x, y;
} Cpt2D;


//IBM Info
typedef struct {
    int i1, j1, k1;
    int i2, j2, k2;
    int i3, j3, k3;
    PetscReal cr1, cr2, cr3; // coefficients
    PetscReal d_i; // distance to interception point on grid cells
    int imode; // interception mode

    int ni, nj, nk;     // fluid cell
    PetscReal d_s; // shortest distance to solid surfaces
    Cmpnts pmin;
    int cell; // shortest distance surface cell
    PetscReal cs1, cs2, cs3;


    //Not sure if any of these are needed....
    //
    int i11, j11, k11;
    int i22, j22, k22;
    int i33, j33, k33;
    PetscReal cr11, cr22, cr33; // coefficients
    PetscReal d_ii; // distance to interception point on grid cells
    int iimode; // interception mode
    PetscReal cs11, cs22, cs33;

    int ii1, jj1, kk1;
    int ii2, jj2, kk2;
    int ii3, jj3, kk3;
    PetscReal  ct1, ct2, ct3; // coefficients
    int smode; // interception mode
  
    int ii11, jj11, kk11;
    int ii22, jj22, kk22;
    int ii33, jj33, kk33;
    PetscReal ct11, ct22, ct33; // coefficients
    PetscReal d_ss; // distance to interception point on grid cells
    int ssmode; // interception mode

} IBMInfo;


typedef struct {
    int nbnumber;
    int n_v, n_elmt;    // number of vertices and number of elements
    int my_n_v, my_n_elmt;      // seokkoo, my proc
    int *nv1, *nv2, *nv3;       // Node index of each triangle
    PetscReal *nf_x, *nf_y, *nf_z;    // Normal direction
    PetscReal *x_bp, *y_bp, *z_bp;    // Coordinates of IBM surface nodes
    PetscReal *x_bp0, *y_bp0, *z_bp0;
    PetscReal *x_bp_i, *y_bp_i, *z_bp_i;
    PetscReal *x_bp_o, *y_bp_o, *z_bp_o;
    Cmpnts *u, *uold, *urm1;

    PetscReal *dA;         // area of an element
    PetscReal *nt_x, *nt_y, *nt_z; //tangent dir
    PetscReal *ns_x, *ns_y, *ns_z; //azimuthal dir
    PetscReal *cent_x,*cent_y,*cent_z;

    // for radius check
    Cmpnts *qvec;
    PetscReal *radvec;

    PetscReal *count;
    PetscReal *shear;
    PetscReal *mean_shear;
    PetscReal *reynolds_stress1;
    PetscReal *reynolds_stress2;
    PetscReal *reynolds_stress3;
    PetscReal *pressure;
    Cmpnts *rel_velocity;   // flow velocity - body velocity

    PetscReal *Tmprt_lagr, *Ftmprt_lagr, *tmprt;
    // force at the IB surface points (lagrange points)
    PetscReal *F_lagr_x, *F_lagr_y, *F_lagr_z;
    // force at the IB surface points (lagrange points) 
    PetscReal *Ft_lagr_avg, *Fa_lagr_avg, *Fr_lagr_avg; 
    PetscReal *U_lagr_x, *U_lagr_y, *U_lagr_z;
    PetscReal *Urelmag; // relative incoming velocity for actuator model 

    Cmpnts *Urel; // vector of the relative incoming velocity
    Cmpnts *Uinduced; // vector of the induced velocity
    Cmpnts *circulation; // circulation vector on the blade
    Cmpnts *liftdirection; // direction of the lift 
    Cmpnts *rotationdirection; // direction of the lift 
    Cmpnts *Urel_mean; // vector of the relative incoming velocity
    Cmpnts *Uinduced_mean; // vector of the induced velocity
    Cmpnts *circulation_mean; // circulation vector on the blade
    Cmpnts *liftdirection_mean; // direction of the lift 

    int *i_min, *i_max, *j_min, *j_max, *k_min, *k_max;

    // twist angle and chord length at each grid point
    PetscReal *angle_attack, *angle_twist, *chord_blade; 
    PetscReal *CD, *CL;
    PetscReal pitch[3];  // Maximum number of blades: 3
    PetscReal U_ref;
    PetscReal *dhx, *dhy, *dhz;
    PetscReal CD_bluff;
    PetscReal friction_factor, pressure_factor;
    PetscReal *frictionfactor;
    PetscReal axialforcecoefficient, tangentialforcecoefficient;
    PetscReal axialprojectedarea, tangentialprojectedarea;
    PetscReal dh;
    PetscReal indf_axis, Tipspeedratio, CT, indf_tangent;
    // force at the IB surface points (lagrange points)
    PetscReal *Fr_mean, *Fa_mean, *Ft_mean; 
    PetscReal *Ur_mean, *Ua_mean, *Ut_mean;
    PetscReal *AOA_mean, *Urelmag_mean;
    PetscReal *AOAAOA_mean, *FFa_mean, *FFt_mean;
    PetscReal *centIP_x, *centIP_y, *centIP_z;
    PetscInt *iIP_min, *iIP_max, *jIP_min, *jIP_max, *kIP_min, *kIP_max;
    PetscReal *U_IPlagr_x, *U_IPlagr_y, *U_IPlagr_z, *P_IPlagr;

    PetscReal *dh_IP;

    PetscReal *Nut_lagr, *Shear_lagr_x, *Shear_lagr_y, *Shear_lagr_z;
    PetscReal *ShearDesired_lagr_x, *ShearDesired_lagr_y, *ShearDesired_lagr_z;
    PetscReal *UU_lagr_x, *UU_lagr_y, *UU_lagr_z;

    PetscReal *random_color;

    int num_cf;
    // Readed friction coefficient from a file  HARD CODING
    PetscReal r_in[200], cf_in[200]; 

    int *color;
    // actuator line element index for each actuator surface element 
    int *s2l; 

} IBMNodes;
  
//Update with std::list instead
typedef struct node {
        int Node;
        struct node *next;
} node;

typedef struct list{
        node *head;
} LIST;

typedef struct list_node {
        int     index;
        struct list_node *next;
} Node_List;

typedef struct IBMListNode {
        IBMInfo ibm_intp;
        struct IBMListNode* next;
} IBMListNode;

typedef struct IBMList {
        IBMListNode *head;
} IBMList;


typedef struct {
    PetscReal S_new[6],S_old[6],S_real[6],S_realm1[6];
    PetscReal S_ang_n[6],S_ang_o[6],S_ang_r[6],S_ang_rm1[6];
    PetscReal red_vel, damp, mu_s; // reduced vel, damping coeff, mass coeff
    PetscReal F_x,F_y,F_z, A_tot; //Forces & Area
    PetscReal F_x_old,F_y_old,F_z_old; //Forces & Area
    PetscReal F_x_real,F_y_real,F_z_real; //Forces & Area
    PetscReal M_x,M_y,M_z; // Moments
    PetscReal M_x_old,M_y_old,M_z_old; //Forces & Area
    PetscReal M_x_real,M_y_real,M_z_real; //Forces & Area
    PetscReal M_x_rm2,M_y_rm2,M_z_rm2; //Forces & Area
    PetscReal M_x_rm3,M_y_rm3,M_z_rm3; //Forces & Area
    PetscReal x_c,y_c,z_c; // center of rotation(mass)
    PetscReal Mdpdn_x, Mdpdn_y,Mdpdn_z;
    PetscReal Mdpdn_x_old, Mdpdn_y_old,Mdpdn_z_old;

    PetscReal Mx_applied,My_applied,Mz_applied; // applied Moments 
    // Aitkin's iteration
    PetscReal    dS[6],dS_o[6],atk,atk_o;
    // for force calculation
    //SurfElmtInfo  *elmtinfo;
    //IBMInfo       *fsi_intp;

    //Max & Min of ibm domain where forces are calculated
    PetscReal Max_xbc,Min_xbc;
    PetscReal Max_ybc,Min_ybc;
    PetscReal Max_zbc,Min_zbc;

    // CV bndry
    int CV_ys,CV_ye,CV_zs,CV_ze;

    PetscReal omega_x, omega_y, omega_z;
    PetscReal nx_tb, ny_tb, nz_tb; // direction vector of rotor axis rotor_model
    PetscReal angvel_z, angvel_x, angvel_y, angvel_axis;
    PetscReal x_c0, y_c0, z_c0;

    //Turbine Stuff
    PetscReal Torque_generator, J_rotation, CP_max, TSR_max;
    PetscReal r_rotor, Torque_aero, ang_axis, angvel_fixed, Force_axis;
    int rotate_alongaxis;
    //nacelle stuff
    PetscReal xnacelle_upstreamend, ynacelle_upstreamend, znacelle_upstreamend;

} FSInfo;





#define EPSILON 1.e-15//0.00000001
#define CROSS(dest, v1, v2) \
    dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
    dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
    dest[2] = v1[0] * v2[1] - v1[1] * v2[0];

#define DOT(v1, v2) (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])

#define SUB(dest, v1, v2) \
    dest[0] = v1[0] - v2[0]; \
    dest[1] = v1[1] - v2[1]; \
    dest[2] = v1[2] - v2[2];


PetscErrorCode randomdirection(Cmpnts p, PetscInt ip, PetscInt jp,
                               PetscReal xbp_min, PetscReal ybp_min,
                               PetscReal zbp_max, PetscReal dcx, PetscReal dcy,
                               PetscReal dir[3],PetscInt seed);
int intsect_triangle(PetscReal orig[3], PetscReal dir[3],
                     PetscReal vert0[3], PetscReal vert1[3],
                     PetscReal vert2[3],
                     PetscReal *t, PetscReal *u, PetscReal *v);
int ISSameSide2D(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3);
int ISInsideTriangle2D(Cpt2D p, Cpt2D pa, Cpt2D pb, Cpt2D pc);
int ISPointInTriangle(Cmpnts p, Cmpnts p1, Cmpnts p2, Cmpnts p3,
                      PetscReal nfx, PetscReal nfy, PetscReal nfz);
PetscErrorCode Dis_P_Line(Cmpnts p, Cmpnts p1, Cmpnts p2,
                          Cmpnts *po, PetscReal *d);
PetscErrorCode triangle_intp2(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3,
                              IBMInfo *ibminfo);
PetscErrorCode triangle_intp(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3,
                             IBMInfo *ibminfo);
PetscErrorCode triangle_intp3D(double x, double y, double z,
                               double x1, double y1, double z1,
                               double x2, double y2, double z2,
                               double x3, double y3, double z3,
                               IBMInfo *ibminfo);
PetscErrorCode triangle_intpp(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3,
                              IBMInfo *ibminfo);
PetscBool ISLineTriangleIntp(Cmpnts p1, Cmpnts p2, IBMNodes *ibm, int ln_v);
void DestroyIBMList(IBMList *ilist);
void AddIBMNode(IBMList *ilist, IBMInfo ibm_intp);
void InitIBMList(IBMList *ilist);
void destroy(LIST *ilist);
void insertnode(LIST *ilist, int Node);
void initlist(LIST *ilist);


#endif
