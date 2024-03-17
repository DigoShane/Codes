//=====================================================================
//This code is taken from K Chad Sockwells website
//https://people.sc.fsu.edu/~kcs12j/
//code for a G-L vortex dynamics simulation using Trilinos
//=====================================================================
#include <sstream>
#include <string>
# include <cstdlib>
#include <stdlib.h>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <cmath>
# include <ctime>
# include "AztecOO_config.h"
# include "mpi.h"
# include "Epetra_MpiComm.h"
# include "Epetra_Map.h"
# include "AztecOO.h"
# include "Epetra_CrsMatrix.h"
# include "Epetra_CrsGraph.h"
# include "Epetra_FECrsMatrix.h"
# include "Epetra_FECrsGraph.h"
#include "Epetra_BlockMap.h"
# include "Epetra_FEVector.h"
# include "Epetra_Time.h"
#include "Epetra_SerialDenseVector.h" 
#include "Epetra_SerialDenseMatrix.h" 
#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"
#include "Teuchos_ParameterList.hpp"
//#include "PrecondOperator.h"
using namespace std;
//prototypes for subs/functions ect. 
int main ( int argc, char *argv[] );
void assemble_jac(double psir_new[], double psii_new[],double a1_new[],double a2_new[],double xc[],double yc[],                
            int indx[], int node[], double basis_der_vertex, double basis_der_midpt, double quad_weight[],  
            int nx, int ny, int n_band, int n_triangles, int n_local, int n_nodes, int n_quad,  
            int n_unknowns, int unknowns_node, int lde, int ldn, double xi, double lambda, double dt ,int tau,
			int nnz_row[], int ncc,const Epetra_MpiComm Comm , Epetra_FECrsMatrix &K, int iam ,int flag, int eps )  ;

void eval_space_qdpt( int triangle,    
           double psir_new[],double psii_new[], double a1_new[], double a2_new[],   
           double xc[], double yc[],double basis_der_vertex,double basis_der_midpt, 
           int node[],int n_local,int i_quad,int lde, int n_nodes,
           double& psir,double& psir_x,double& psir_y,double&  psii,double& psii_x,double& psii_y,   
           double& a1,double& a1_x,double& a1_y,double& a2,double& a2_x,double& a2_y ) ;

void eval_space_time_qdpt( int triangle,double psir_old[],double psii_old[],  
           double a1_old[],double a2_old[],double psir_new[],
		   double psii_new[],double a1_new[],double a2_new[],   
           double xc[],double yc[],double basis_der_vertex,double basis_der_midpt,  
           int node[],int n_local,int i_quad,int lde,  int n_nodes,
           double& psir,double& psir_x,double& psir_y,
		   double& psir_t,double& psii,double& psii_x,double& psii_y,double& psii_t,  
           double& a1,double& a1_x,double& a1_y,double& a1_t,
		   double& a2,double& a2_x,double& a2_y,double& a2_t) ;

void four_to_one(double a[],double b[],double c[],double d[],
			int indx[],int n_nodes,int n_unknowns,int ldn, double e[], int iam);
void geometry  (double x_length,double y_length, int nx, int ny,int  flag_write,  
                      int lde,int  ldn,int   n_local,double n_quad,int& unknowns_node,      
                      double xc[],double yc[],double& basis_der_vertex,double& basis_der_midpt,
					  double quad_weight[],int indx[],int node[],
					  int& n_band,int& n_triangles,int& n_nodes,int& n_unknowns, int& ind_num, int *&nnz_row,
					  double &jac);
void ind_find(int &count,int nnz_row[], int num, int i_nodes, int indx[]);
void st_set_up(int indx[],int node[],int n_triangles,int unknowns_node,int n_unknowns,int n_local,
                 int n_nodes,int n_quad,int ldn,int lde, Epetra_CrsGraph & Graph, int iam);					  
void st_set_up_2(int indx[],int node[],int n_triangles,int unknowns_node,int n_unknowns,int n_local,
                 int n_nodes,int n_quad,int ldn,int lde,Epetra_FECrsMatrix& K);
					  
void init( int ldn, double psir[],double psii[],double a1[],double a2[] );
 
void nonlinear_solver(int  nx,int  ny,int n_band,int n_triangles,int n_local,int n_nodes, 
       int n_quad,int  n_unknowns,int  unknowns_node,int lde,int ldn,  
       double xi,double lambda,double h_external,double tol_newton,double n_newton,double time_cur,  
       double &dt,double dt_min,double dt_max,int flag_jacobian,int flag_write,double xc[],double yc[],   
       double basis_der_vertex,double basis_der_midpt,double quad_weight[],int indx[],int node[],          
       double psir_old[],double psii_old[],double a1_old[],double a2_old[], 
       double psir_new[],double psii_new[],double a1_new[],double a2_new[],int& n_nonlinear,double tau,int nnz_row[], 
	   int ncc, const Epetra_MpiComm Comm ,int iam , int &flag,Epetra_FECrsMatrix &K,Epetra_Vector &F, Epetra_Vector &x,AztecOO &solver,
		double &s_st_t,double &j_st_t, int eps, Epetra_Map &Map);
void one_to_four (double f[],int indx[],int n_nodes,int n_unknowns,int ldn,
			double a[],double b[],double c[],double d[]);
void output (double psir_new[],double psii_new[],double a1_new[],double a2_new[],int *node,int lde,int nx,int ny, 
                   double xi,double lambda,double h_external,double time_cur,int n_count,double quad_weight[],
				   double basis_der_midpt,double basis_der_vertex,int n_triangles,int n_local,int n_nodes,int n_quad,
				   double xc[], double yc[],int &flag, string s,double jac );  			
void quad_basis_fn(int it,int i_local,int i_quad,  
       double basis_der_vertex,double basis_der_midpt,double&  basis,double& basis_x,double& basis_y ) ;
void residual (double psir_old[],double psii_old[],double a1_old[],double a2_old[],    
            double psir_new[],double psii_new[],double a1_new[],double a2_new[],
			double xc[],double yc[],int indx[],int node[],   
            double basis_der_vertex,double basis_der_midpt,double quad_weight[],int nx,int ny,  
            int n_triangles,int n_local,int  n_nodes,int n_quad,int  n_unknowns,    
            int unknowns_node,int lde,int ldn,double xi,double lambda,double h_external,
			double dt,double resid_psir[],double resid_psii[],double resid_a1[],
			double resid_a2[],double tau,int eps, double time_cur ) ;
void r8ge_fs ( int n, double a[], double x[] );
double r8_abs ( double x );
int st_to_cc_size ( int nst, int ist[], int jst[] );
void st_to_cc_index ( int nst, int ist[], int jst[], int ncc, int n, 
  int icc[], int ccc[] );
int *i4vec_copy_new ( int n, int a1[] );
void i4vec2_sort_a ( int n, int a1[], int a2[] );
void i4vec2_sorted_uniquely ( int n1, int a1[], int b1[], int n2, int a2[], int b2[] );
void sort_heap_external ( int n, int &indx, int &i, int &j, int isgn );
int i4vec2_compare ( int n, int a1[], int a2[], int i, int j );
int i4vec2_sorted_unique_count ( int n, int a1[], int a2[] );
void qbf (int q_point, int element, int inode, double xc[],double yc[],
  int element_node[], int element_num, int nnodes,
  int node_num, double &b, double &dbdx, double &dbdy );

int main ( int argc, char *argv[])

//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*

// 
//          TIME-DEPENDENT SUPERCONDUCTIVTY PROGRAM 
//                  2-D, RECTANGULAR DOMAIN
//               ZERO ELECTRIC POTENTIAL GAUGE
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//        MODELS INCLUDED: 
//            1. GINZBURG-LANDAU
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//      	DISCRETIZATION: FINITE ELEMENT METHOD IN SPACE
//                       USING QUADRATICS ON TRIANGLES
//                       AND BACKWARD EULER IN TIME 
//       SOLUTION: NEWTON'S METHOD or Fixed Jacobian 
//
//
//
// 
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*

//

//
// INPUT  
//
{

MPI_Init(&argc,&argv);
Epetra_MpiComm Comm( MPI_COMM_WORLD );


double x_length = 10., y_length = 10.;    // size of the box   
double lambda = 0.5  ;      // coherence length
double xi = 0.05;//0.1            // penetration depth
double kappa = lambda/xi ;  // Ginzburg Landau parameter
double h_external= 0.15*kappa;//1.0;//2.0;//0.5;//0.5*kappa   //  External field
double tau=1.0;//0.4;
double eps=1.0;
double time_init=0.0, time_final= 5000000.0;  // initial and final times
double dt_max = 50.;//5000.       ; // maximum time step allowed
double dt_init = 0.1      ; // initial time step  
double dt_min = 0.001     ; // minimum time step allowed
double dt_factor = 1.5    ; // factor to increase time step
double tol_newton_loose = .1e-6;//6  ; // Newton tolerance  (initial)
double tol_newton_tight = .1e-11 ; // Newton tolerance  (final)
int    n_newton = 8               ; // maximum allowable newton steps
int    n_timesteps=10000000           ;    // maximum number of allowable timesteps
int    nx = 721, ny =721; // number of nodes in x and y directions
int    lde = (nx-1) * (ny-1) * 2  ; // leading dimension of arrays dimensioned by element number
int    ldn = (2* nx-1)*(2* ny-1)  ; // leading dimension of arrays dimensioned by node number 
int    n_local = 6                ; // number of local  nodes per element
int    n_quad = 3;//3                 ; // number of quadrature points per element
int    unknowns_node = 4          ; // number of unknowns per node (psi real, imag, A1, A2)
int    flag_write = 4             ; // flag for write statements (=0 least, =5 most)
int    flag_init = 0              ; // flag for initialization (=0 generate initial conditions,
                              //     = 1 read from file)
int    flag_output = 25;//25         ;   // flag for writing data to file every # of time steps
int    flag_jacobian = 1       ;    // update jacobian every # of iterations (=1 => Newton)
int    flag_steady = 15        ;     // flag for determining steady state
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
// GEOMETRY
//
double xc[ldn],yc[ldn];         // x and y coordinates of nodes 
double  quad_weight[n_quad];   // area of triangles * quadrature weight (1/3) for Midpoint rule
double  basis_der_vertex;//value of derivative of basis fn centered at vertex at qd pt
double  basis_der_midpt; //value of derivative of basis fn centered at midpt at qd pt
int indx[ldn*4];                 // unknown numbering array
int  node[lde*n_local];            // array associating local and global nodes
int   n_band;            //  half bandwidth +1  
int   n_triangles;       //  number of triangles
int   n_nodes;           //  number of global nodes
int   n_unknowns;        //  number of unknowns
int   ind_num,ncc;
int *ia, *ja;
int *icc;
int *ccc;
int *nnz_row;
double s_st_t,j_st_t;
double jac;

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//  SOLUTION
//
double  psir_old[ldn], psii_old[ldn];      // order PARAMETER at previous timestep 
double  a1_old[ldn], a2_old[ldn];          // magnetic potential at previous timestep
double  psir_new[ldn], psii_new[ldn];      // order PARAMETER at current Newton step
double  a1_new[ldn], a2_new[ldn];          // magnetic potential at current Newton step
double  dt;
double  tol_newton;                                  // current Newton tolerance
int    n_nonlinear;     // number of nonlinear iterations required to solve at given timestep 
int    n_steady;        // counter for reaching steady state
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//  LOCAL VARIABLES
//
int i, k;
int n_count;       // count for number of time steps in a row that nonlinear solver converged in < criteria
int n_call_output,flag; // number of calls to output
double  time_cur;
 ifstream output1;
ofstream output2;
ofstream output3;
ofstream output4;
ofstream output5;
ofstream output6;
int sum;
int iam;
double nlt_st,nlt_fin,nlt, start,finish;
stringstream ss;
string s;
 
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
MPI_Comm_rank(MPI_COMM_WORLD, &iam);
//h_external=atof(argv[1]);

if (iam==0){
//ss <<argv[1] ;
//ss >> s;
s="hey";
output1.open ( "init_out.data" , std::ifstream::in);
    output2.open ( "order_test.data", std::ofstream::out | std::ofstream::trunc);
  output3.open ( "current_test.data", std::ofstream::out | std::ofstream::trunc);
 output4.open ( "magnetic_test.data", std::ofstream::out | std::ofstream::trunc);
 output5.open ( "init_out_test.data", ofstream::out | ofstream::trunc);
output6.open ( "time_step_test.data", ofstream::out | ofstream::trunc);

}
n_nonlinear=0;
n_count = 0;
nlt=0.0;
flag=1;//0
s_st_t=0.0;
j_st_t=0.0;
//
//  SET UP GEOMETRY   
//  

 start=MPI_Wtime();
 
 geometry(  x_length, y_length, nx, ny, flag_write, lde, ldn,  
              n_local, n_quad,  unknowns_node,   
              xc, yc, basis_der_vertex, basis_der_midpt, quad_weight,   
              indx, node, n_band, n_triangles, n_nodes, n_unknowns, ind_num,nnz_row,jac );
			
			  
	sum=0;
for ( i= 0; i < n_unknowns; i++ ){

	sum+=nnz_row[i];
}

Epetra_Map Map(n_unknowns,0,Comm);
 
 Epetra_CrsGraph Graph(Copy,Map,nnz_row);
 	 st_set_up( indx,node,n_triangles,unknowns_node,n_unknowns,n_local,
                n_nodes,n_quad,ldn,lde, Graph, iam);

 Epetra_FECrsMatrix K(Copy,Graph);


Epetra_Vector F(Map);
Epetra_Vector x(Map);

 	Epetra_LinearProblem problem(&K, &x, &F);
	   
 
 AztecOO solver(problem);

solver.SetAztecOption(AZ_output, AZ_warnings);


   
//
if (iam==0){
cout<< "number of nodes    = "<<  n_nodes      <<endl;
cout<< "number of unknowns = " << n_unknowns   <<endl;
cout<< "number of triangles = " << n_triangles <<endl;
cout<< "bandwidth = "           <<n_band       <<endl;
cout<< "num of non 0= "			<< ncc         <<endl;
cout<< "sum of nnz_row= "<< sum<<endl;
}
// 
//   * * INITIALIZE for TIMESTEPPING  * * 
//
if( flag_init==0 )  // generate initial conditions 
{
init( ldn, psir_old, psii_old, a1_old, a2_old  ) ;
    time_cur = time_init ;
}	
  else                    // read initial conditions from file
  {
  output1>>time_cur;
  cout<<"TIME IS "<< time_cur<<endl;
      for ( k= 0; k< n_nodes ;k++ ){
  
  output1>>psir_old[k];
  output1>>psii_old[k];
  output1>>a1_old[k]  ;
  output1>>a2_old[k]  ;
  }
	}

//
for ( i= 0; i < n_nodes; i++ ){
 // set initial guess for nonlinear system to be solution at previous timestep
  psir_new[i] = psir_old[i];
  psii_new[i] = psii_old[i];
  a1_new[i] = a1_old[i];
  a2_new[i] = a2_old[i] ;
}
  
// 
//   * *  TIMESTEPPING LOOP  * *
//
dt = dt_init ;
n_call_output = 0;           // counter for calls to output
tol_newton = tol_newton_loose;
//
for ( k= 0; k< n_timesteps; k++ ){
          // begin time step loop
  time_cur = time_cur + dt;
  if ( time_cur > time_final ) time_cur = time_final;
  if (iam ==0){
  cout<< "^^^^^^^^^^^^^^^^^*"<<endl;
  cout<< " current time is "<< time_cur<< " Time step: "<< k+1<<endl;
  }
//
// Newton solver to assemble nonlinear system and solve
//
nlt_st=MPI_Wtime();
 nonlinear_solver ( nx, ny, n_band, n_triangles, n_local, n_nodes,  
       n_quad,  n_unknowns, unknowns_node, lde, ldn,  
       xi, lambda, h_external, tol_newton, n_newton, time_cur,  
       dt, dt_min, dt_max, flag_jacobian, flag_write, xc, yc,    
       basis_der_vertex, basis_der_midpt, quad_weight, indx, node,    
       psir_old, psii_old, a1_old, a2_old, 
       psir_new, psii_new, a1_new, a2_new, n_nonlinear,tau,nnz_row, ncc,Comm, iam,flag ,
	   K,F, x,solver,s_st_t, j_st_t,eps, Map) ;
nlt_fin=MPI_Wtime();
nlt+=nlt_fin-nlt_st;

output6<<fixed << setprecision(8) << k<<" "<<dt<<" "<< n_nonlinear<<endl;

if (iam==0) cout<<"Total NL time: "<< nlt<<endl;
//
// Compute output parameters if necessary  (call nonlinear solver again to obtain solution to tighter tolerance)
//
 if ( ( (k+1)% flag_output ) == 0){
     n_call_output = n_call_output + 1   ;
	 if (iam==0){
      output (psir_new, psii_new, a1_new, a2_new, node,  
          lde, nx, ny, xi, lambda, h_external, time_cur, n_call_output, quad_weight,  
          basis_der_midpt, basis_der_vertex, n_triangles, n_local, n_nodes,  n_quad,xc,yc,flag,s,jac) ;
		  }
  }
 //
 if (time_cur == time_final ){
     n_call_output = n_call_output + 1;
	 if (iam==0){
      output(psir_new, psii_new, a1_new, a2_new, node, 
          lde, nx, ny, xi, lambda, h_external, time_cur, n_call_output, quad_weight,  
          basis_der_midpt, basis_der_vertex, n_triangles, n_local, n_nodes,  n_quad,xc,yc,flag,s,jac) ;
     cout<< " ^^ SOLUTION HAS REACHED FINAL TIME ^* "<<endl;
	 }
	  MPI_Finalize() ;
     return 0;
 }
 
//
// Determine if steady state solution has been reached
//  First tighten tolerance
  if ( n_nonlinear == 0 ){
      n_steady = n_steady + 1;
//
      if ( n_steady == flag_steady ){
//
           if ( tol_newton == tol_newton_loose){
               tol_newton = tol_newton_tight;
               cout<< " tightened Newton tolerance"<<endl;
               n_steady = 0;
			   }
             else{
			 if (iam==0){
               cout<< "SOLUTION HAS REACHED STEADY STATE "<<endl;
			   finish=MPI_Wtime();
			   cout<< "time for sim:"<< finish-start<<endl;
               output(psir_new, psii_new, a1_new, a2_new, node, 
                lde, nx, ny, xi, lambda, h_external, time_cur, n_call_output, quad_weight,  
                basis_der_midpt, basis_der_vertex, n_triangles, n_local, n_nodes,  n_quad,xc,yc,flag,s,jac) ;
				}
				
				 MPI_Finalize() ;
				 
               return 0;
           }
//
      }
//      
}           
   else{
     n_steady = 0;
 }
       
//
// Determine whether time step should be increased
//    criteria for Newton:  3 steps in row with < = 2 iterations
//    criteria for fixed jacobian : 3 steps in row with < = 3 iterations
  if ( flag_jacobian == 1){    // Newton's method
//
      if ( n_nonlinear <= 2 ){
	  
          n_count = n_count + 1;
		  if (iam==0) cout<<n_count<<endl;
		  }
        else{
          n_count = 0;
            }
			}
//    
    else {     // fixed jacobian iteration
//
     if ( n_nonlinear <= 3 ){
          n_count = n_count + 1;
        }
		else{
          n_count = 0;
      }
//
  }
//
  if ( n_count == 3) {
      dt = min( dt * dt_factor, dt_max )   ;
      cout<< " increased time step to " <<dt<<endl;

      n_count = 0;
  }
//
// get ready for next time step
for ( i= 0; i< n_nodes; i++ ){
  
    psir_old[i] = psir_new[i];
    psii_old[i] = psii_new[i];
    a1_old[i] = a1_new[i]; 
    a2_old[i] = a2_new[i];
  }
//  
}                            // end time step loop
 MPI_Finalize() ;
cout<< " maximum number of timesteps is reached without reaching steady state"<<endl;
//
 
//
return 0;
}   
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
void assemble_jac(double psir_new[], double psii_new[],double a1_new[],double a2_new[],double xc[],double yc[],                
            int indx[], int node[], double basis_der_vertex, double basis_der_midpt, double quad_weight[],  
            int nx, int ny, int n_band, int n_triangles, int n_local, int n_nodes, int n_quad,  
            int n_unknowns, int unknowns_node, int lde, int ldn, double xi, double lambda, double dt,int tau,
			int nnz_row[], int ncc,const Epetra_MpiComm Comm, Epetra_FECrsMatrix &K, int iam,int flag,int eps)  
{			
// 
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//  ASSEMBLES JACOBIAN MATRIX                    
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*

//
 
//
// LOCAL VARIABLES
int   i_triangle, i_quad, i_local, i_global, j_local, j_global, k, l,  
             row_number, column_number, i_position,ii,jj,i1;
double one_over_lambdasq, one_over_kappa, xi_sq,  weight,
       psir, psir_x, psir_y,  
       psii, psii_x, psii_y,  
       a1, a1_x, a1_y, a2, a2_x, a2_y,   psi_sq, psisq_over_lambdasq, a_sq,   
       xi_psir, xi_psii, xi_psir_x, xi_psir_y, xi_psii_x, xi_psii_y,   
       two_psir, two_psii, a1_over_lambda, a2_over_lambda,  asq_over_lambdasq,  
       a11_term, a12_term, a23_term, a13_term, a14_term, a22_term, a24_term,  
       test, test_x, test_y, test_over_dt, a_dot_grad_test, test_over_lambda,  
       trial, trial_x, trial_y, trial_test_over_dt, laplacian, xisq_times_lap,   
       a_dot_grad_trial, trial_over_lambda, test_times_trial,kappa,y_node ;
//
double temp[4*4];
double temp2[1];
double J;
int indices[2];

// 
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//  
//  initialization
if (iam==0) cout<<"Inn Jacobian"<<endl;
Epetra_SerialDenseMatrix k_local(1,1);// maybe (0,0)?
Epetra_IntSerialDenseVector local_indices_r(1); //maybe (0)?
Epetra_IntSerialDenseVector local_indices_c(1); //maybe (0)?


J=0.0000000000000;
  one_over_lambdasq = 1.0 / ( lambda * lambda);
  xi_sq = xi * xi  ;
  one_over_kappa = xi / lambda;
  kappa=lambda/xi;
  if(flag==1)
  K.Scale(0.0);
 if(iam==0) cout<<"past set_up_2"<<endl;
//
//  assemble matrix by calculating contributions elementwise
// 
int NumProc = Comm.NumProc();

  for ( i_triangle = iam*n_triangles/NumProc; i_triangle< iam*n_triangles/NumProc+n_triangles/NumProc; i_triangle++ ){
//
for ( i_quad= 0; i_quad< n_quad; i_quad++ ){
     // loop over quadrature points

//                            // evaluate solution & derivatives at quad pt.
      eval_space_qdpt( i_triangle,    
           psir_new, psii_new, a1_new, a2_new,                
           xc, yc, basis_der_vertex, basis_der_midpt, 
           node, n_local, i_quad, lde, n_nodes,              
           psir, psir_x, psir_y, psii, psii_x, psii_y,      
           a1, a1_x, a1_y, a2, a2_x, a2_y );

		   
      psi_sq = psir*psir + psii*psii;
      psisq_over_lambdasq = psi_sq * one_over_lambdasq;
      a_sq = a1*a1 + a2*a2;  
      xi_psir = xi * psir;         xi_psii = xi * psii;    
      xi_psir_x = xi * psir_x;     xi_psir_y = xi * psir_y;
      xi_psii_x = xi * psii_x;     xi_psii_y = xi * psii_y ;
      two_psir = 2.0 * psir;   two_psii = 2.0 * psii;	 
      a1_over_lambda = a1 / lambda;      a2_over_lambda = a2 / lambda;  
      asq_over_lambdasq = a_sq * one_over_lambdasq;   
      a11_term = psi_sq + 2.0 * psir * psir + asq_over_lambdasq - tau;
      a22_term = psi_sq + 2.0 * psii * psii + asq_over_lambdasq - tau;
      a23_term = two_psii * a1_over_lambda + xi_psir_x;
      a13_term = two_psir * a1_over_lambda - xi_psii_x; 
      a14_term = two_psir * a2_over_lambda - xi_psii_y;
      a24_term = two_psii * a2_over_lambda + xi_psir_y; 
//
for ( i_local= 0; i_local < n_local; i_local++ ){
       // loop over test functions 
        i_global = node[i_triangle*n_local+i_local];
//                                   // evaluate test function at quadrature pt
		  qbf (i_quad, i_triangle, i_local, xc,yc,
			      node, lde, n_local,
			   n_nodes, test, test_x, test_y );
	

        test_over_dt = test / dt ;//* 0.1
        test_over_lambda = test / lambda;
        
        a_dot_grad_test = a1 * test_x + a2 * test_y;
//
for ( j_local= 0; j_local < n_local; j_local++ ){
        
          j_global = node[i_triangle*n_local+j_local];
//
	qbf (i_quad, i_triangle, j_local, xc,yc,
			      node, lde, n_local,
			   n_nodes, trial, trial_x, trial_y );
			
          laplacian = test_x * trial_x + test_y * trial_y;
          xisq_times_lap = xi_sq * laplacian ;         
          test_times_trial = test * trial ;
          trial_test_over_dt = trial * test_over_dt  ;
          a_dot_grad_trial = a1 * trial_x + a2 * trial_y;
          trial_over_lambda = trial / lambda;   
          
// 
          if ( test != 0.0 ) {


				
              temp[0*4+0] =  xisq_times_lap + a11_term * test_times_trial  
                                      + trial_test_over_dt;
              temp[0*4+1] =  test * ( two_psir * psii * trial  
                          - one_over_kappa * a_dot_grad_trial )   
                          + one_over_kappa * trial * a_dot_grad_test ;
						  
              temp[0*4+2] = ( test * a13_term  + xi_psii * test_x ) * trial_over_lambda ;
              temp[0*4+3] = ( test * a14_term  + xi_psii * test_y ) * trial_over_lambda ;
//
              temp[1*4+0] = trial * ( two_psir * psii * test - one_over_kappa * a_dot_grad_test )   
                          + one_over_kappa * test * a_dot_grad_trial;
              temp[1*4+1] = xisq_times_lap + a22_term * test_times_trial  
                                     + trial_test_over_dt ;
              temp[1*4+2] = ( test *  a23_term - xi_psir * test_x ) * trial_over_lambda ;
              temp[1*4+3] = ( test *  a24_term - xi_psir * test_y ) * trial_over_lambda ;
//

				   // 
              temp[2*4+0] = ( trial * a13_term + xi_psii * trial_x ) * test_over_lambda ;
              temp[2*4+1] =  (trial * a23_term - xi_psir * trial_x ) * test_over_lambda ;
              temp[2*4+2] = eps*test_x * trial_x + test_y * trial_y + test_times_trial * psisq_over_lambdasq   
                                + trial_test_over_dt * one_over_lambdasq  ; 
              temp[2*4+3] = eps*test_x * trial_y - test_y * trial_x; 
//
              temp[3*4+0] = ( trial * a14_term + xi_psii * trial_y ) * test_over_lambda  ;
              temp[3*4+1] = ( trial * a24_term - xi_psir * trial_y ) * test_over_lambda  ;
              temp[3*4+2] = eps*test_y * trial_x - test_x * trial_y  ;
              temp[3*4+3] =test_x * trial_x + eps*test_y * trial_y+ test_times_trial * psisq_over_lambdasq  
                        + trial_test_over_dt * one_over_lambdasq;
            }
			else{
              temp[0*4+0] =  xisq_times_lap  ;
              temp[0*4+1] =  one_over_kappa * trial * a_dot_grad_test ;
              temp[0*4+2] = ( xi_psii * test_x ) * trial_over_lambda ;
              temp[0*4+3] = ( xi_psii * test_y ) * trial_over_lambda ;
//
              temp[1*4+0] = trial * ( - one_over_kappa * a_dot_grad_test )  ;
              temp[1*4+1] = xisq_times_lap  ;
              temp[1*4+2] = ( - xi_psir * test_x ) * trial_over_lambda ;
              temp[1*4+3] = ( - xi_psir * test_y ) * trial_over_lambda ;
//
              temp[2*4+0] = 0.0000000000000000;
              temp[2*4+1] = 0.0000000000000000;
              temp[2*4+2] = eps*test_x * trial_x + test_y * trial_y; ;// can  i just put laplacian here and do away with other part?
              temp[2*4+3] = eps*test_x * trial_y - test_y * trial_x ;
//
              
              temp[3*4+0] = 0.0000000000000000;
              temp[3*4+1] = 0.0000000000000000;
              temp[3*4+2] = eps*test_y * trial_x - test_x * trial_y  ;
              temp[3*4+3] = test_x * trial_x + test_y * trial_y*eps;
          }
//


for ( ii= 0; ii < 4; ii++ ){
  for ( jj= 0; jj < 4; jj++ ){
    
          temp[ii*4+jj] = sqrt(dt)*quad_weight[i_quad] * temp[ii*4+jj];    
		  
		  row_number = indx[i_global*4+ii];
		  column_number = indx[j_global*4+jj];
		  			
		  }
		  }
//   

for ( k= 0; k < unknowns_node; k++ ){
          
	    row_number = indx[i_global*4+k];
//
            if ( row_number >= 0 ){
                for ( l= 0; l < unknowns_node; l++ ){
                  column_number = indx[j_global*4+l];
				  //i1 = max ( 1, j-n_band-1 );
	          if ( column_number >= 0 ) {
			  
			  indices[0]=row_number;
			  indices[1]=column_number;
			  
						local_indices_r(0)=indices[0];
						local_indices_c(0)=indices[1];
						k_local(0,0)=temp[k*4+l];
						temp2[0]=temp[k*4+l];

						if( flag==0)K.InsertGlobalValues(local_indices_r,local_indices_c,k_local);
						if( flag==1) K.SumIntoGlobalValues(local_indices_r,local_indices_c,k_local);
                  }
                }
            }
//
          }

        }
//
      }
//
    }
//
  }
 
if (iam==0) cout<<"before global asseble"<<endl;
  K.GlobalAssemble();
  
  K.OptimizeStorage ();
if (iam==0)  cout<<"after global asseble"<<endl;
 
 return;
}
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*

void eval_space_qdpt( int triangle,    
           double psir_new[],double psii_new[], double a1_new[], double a2_new[],   
           double xc[], double yc[],double basis_der_vertex,double basis_der_midpt, 
           int node[],int n_local,int i_quad,int lde, int n_nodes,
           double& psir,double& psir_x,double& psir_y,double&  psii,double& psii_x,double& psii_y,   
           double& a1,double& a1_x,double& a1_y,double& a2,double& a2_x,double& a2_y )  
{		   
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//            DETERMINES THE VALUE OF ORDER PARAMETER,  
//                       MAGNETIC POTENTIAL,
//          AND THEIR SPACE DERIVATIVES AT A QUADRATURE POINT
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//
//  LOCAL VARIABLES
int   i_local, i_global;  
double psir_node, psii_node, a1_node, a2_node,  
                      basis, basis_x, basis_y;
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*  
//
psir_x = 0.0;  psir_y = 0.0;
psii_x = 0.0;  psii_y = 0.0;
a1_x =0.0;  a1_y = 0.0;
a2_x =0.0;  a2_y = 0.0;
psir = 0.0;  psii = 0.0;
a1 =0.0;  a2 = 0.0;


  for ( i_local= 0; i_local < n_local; i_local++ ){   // loop over local nodes
    i_global = node[triangle*n_local+i_local] ;
    psir_node = psir_new[i_global];  psii_node = psii_new[i_global];
    a1_node = a1_new[i_global];  a2_node = a2_new[i_global];
                          // evaluate basis function & derivatives at quad pt
	qbf (i_quad, triangle, i_local, xc,yc,
			      node, lde, n_local,
			   n_nodes, basis, basis_x, basis_y );
			
// 
// determine  space derivatives
//
	if ( basis != 0.0 ) {
        psir = psir + psir_node * basis;  psii = psii + psii_node * basis;
        a1 = a1 + a1_node * basis;  a2 = a2 + a2_node * basis;
    }
    if ( basis_x != 0.0 ) {
        psir_x = psir_x + psir_node * basis_x;  psii_x = psii_x + psii_node * basis_x;
        a1_x = a1_x + a1_node * basis_x;  a2_x = a2_x + a2_node * basis_x;
    }
//
    if ( basis_y != 0.0 ) {
        psir_y = psir_y + psir_node * basis_y;  psii_y = psii_y + psii_node * basis_y      ;
        a1_y = a1_y + a1_node * basis_y;  a2_y = a2_y + a2_node * basis_y;
    }
// 
  }
//
//
 return;
}

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*

void eval_space_time_qdpt( int triangle,double psir_old[],double psii_old[],  
           double a1_old[],double a2_old[],double psir_new[],
		   double psii_new[],double a1_new[],double a2_new[],   
           double xc[],double yc[],double basis_der_vertex,double basis_der_midpt,  
           int node[],int n_local,int i_quad,int lde, int n_nodes, 
           double& psir,double& psir_x,double& psir_y,
		   double& psir_t,double& psii,double& psii_x,double& psii_y,double& psii_t,  
           double& a1,double& a1_x,double& a1_y,double& a1_t,
		   double& a2,double& a2_x,double& a2_y,double& a2_t)  
{		   
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//           EVALUATES THE ORDER PARAMETER, MAGNETIC POTENTIAL,  
//          AND THEIR SPACE AND TIME DERIVATIVES AT A QUADRATURE POINT
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
 
//
//
//  LOCAL VARIABLES
int   i_local, i_global  ;
double    psir_node, psii_node, a1_node, a2_node,   
    psir_node_old, psii_node_old, a1_node_old, a2_node_old,   
                      basis, basis_x, basis_y, tpsir_t;
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
// 
//
psir_x = 0.0;  psir_y = 0.0;
psii_x = 0.0;  psii_y = 0.0;
a1_x =0.0;  a1_y = 0.0;
a2_x =0.0;  a2_y = 0.0 ;
psir = 0.0;  psii = 0.0;
a1 =0.0;  a2 = 0.0;
psir_t = 0.0;  psii_t = 0.0;
a1_t =0.0;  a2_t = 0.0;

psir_t = psir_new[ node[ triangle*n_local+ i_quad+3 ] ] - psir_old[ node[ triangle*n_local+ i_quad+3] ];
psii_t = psii_new[ node[ triangle*n_local+ i_quad+3 ] ] - psii_old[ node[ triangle*n_local+ i_quad+3 ] ];
a1_t   = a1_new[ node[ triangle*n_local+ i_quad+3 ] ] -   a1_old[ node[ triangle*n_local+ i_quad+3 ] ];
a2_t   = a2_new[ node[ triangle*n_local+ i_quad+3 ] ] -   a2_old[ node[ triangle*n_local+ i_quad+3 ] ];

//   
//      
for ( i_local= 0; i_local < n_local; i_local++ ){    // loop over local nodes
    i_global = node[triangle*n_local+i_local] ;
    psir_node = psir_new[i_global];  psii_node = psii_new[i_global];
    a1_node = a1_new[i_global];  a2_node = a2_new[i_global];
    
	qbf (i_quad, triangle, i_local, xc,yc,
			      node, lde, n_local,
		   n_nodes, basis, basis_x, basis_y );

//
// determine  space derivatives  
//

	if ( basis != 0.0 ) {
        psir = psir + psir_node * basis;  psii = psii + psii_node * basis;
        a1 = a1 + a1_node * basis;  a2 = a2 + a2_node * basis;
			  
    }
    if ( basis_x != 0.0 ) {
        psir_x = psir_x + psir_node * basis_x;  psii_x = psii_x + psii_node * basis_x;
        a1_x = a1_x + a1_node * basis_x;  a2_x = a2_x + a2_node * basis_x;
    }
//
    if ( basis_y != 0.0 ) {
        psir_y = psir_y + psir_node * basis_y;  psii_y = psii_y + psii_node * basis_y      ;
        a1_y = a1_y + a1_node * basis_y;  a2_y = a2_y + a2_node * basis_y;
    }

  }
// 
 
 
 return;
}

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^* 
//

void four_to_one(double a[],double b[],double c[],double d[],
			int indx[],int n_nodes,int n_unknowns,int ldn,double e[], int iam)
{			
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//         Takes 4 vectors of length number of nodes and
//         makes 1 vector dimensioned by number of unknowns 
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
 
//  LOCAL VARIABLES
int  k, kk;

  kk=-1 ;
  for ( k= 0; k< n_nodes; k++ ){
  
    kk = kk + 4;
    //e[kk-3] = a[k] ;
	e[kk-3] = a[k] ;
	
	
    //e[kk-2] = b[k] ;
    e[kk-2] = b[k] ;
//
    if( indx[k*4+2] >= 0) {
        //e[kk-1] = c[k] ;
        e[kk-1] = c[k] ;
		}
      else{
        kk=kk-1;
    }
//
    if( indx[k*4+3] >= 0){
        //e[kk] = d[k] ;
        e[kk] = d[k] ;// migth have issues with skipping indc
		
      }
	  else{
        kk=kk-1;
    }
//
  }
//
 
 return;
}

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*  
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
void geometry  (double x_length,double y_length, int nx, int ny,int  flag_write,  
                      int lde,int  ldn,int   n_local,double n_quad,int& unknowns_node,      
                      double xc[],double yc[],double& basis_der_vertex,double& basis_der_midpt,
					  double quad_weight[],int indx[],int node[],
					  int& n_band,int& n_triangles,int& n_nodes,int& n_unknowns, int& ind_num,int *&nnz_row,
						double &jac)
{                      
                      
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//                  ROUTINE TO SET UP GEOMETRIC INFORMATION
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
 
//
// Local variables
int  i_triangle, i_nodes, i_unknowns, n_col, j, midpt_col,   
           n_row, i, midpt_row, i_triangle_plus_one,node_plus_nrow,   
           node1, node2, node3, node4, node5, node6,            
           node_plus_two_nrow, it, i_local, i_global, i_unknown,  
           j_local, j_global, j_unknown,  j_minus_i, ip, n, k,    
           i1, i2, i3, j1, j2, j3,counter; 
double    dx, dy, yy, xx, x1, x2, x3, y1, y2, y3, 
                      xm1, xm2, xm3, ym1, ym2, ym3,    
                      det, b, c, s, t, x, y;
                      
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//      
n_row = nx + nx -1;  n_col = ny + ny -1;     ;  //  number of nodes in row and column
dx = x_length/double(nx-1);  dy = y_length/double(ny-1) ;    //  delta x, delta y for quadratics  
//      
yy = -dy/2. ;
i_triangle = -2;//-1 ;      // counter for triangles
i_nodes = -1;//0      ;   // counter for global nodes
i_unknowns = 0   ;      // counter for unknown number
ind_num=0;
//
// 
counter=0;
for ( j= 0;  j< n_col; j++ )
{

    midpt_col = (j+1)%2 ;   // flag to determine if at vertical midpoint node (when midpt_col=0)
	//cout<<j<<midpt_col<<endl;
    yy = yy + dy / 2. ;
    xx = - dx / 2.;
// 

    for ( i= 0;  i< n_row; i++ )
	{
	
      midpt_row = (i+1)%2;
      xx = xx+dx/2.      ;
      i_nodes = i_nodes + 1 ;
//
// set x and y coordinates of node
      xc[i_nodes]=xx;    yc[i_nodes]=yy ;
// 
// set unknown number at node  (assumes 4 unknowns per node)
      i_unknowns = i_unknowns + 4 ;  
      indx[i_nodes*4+0] = i_unknowns-4 ;  // unknown for psi real (order PARAMETER)
      indx[i_nodes*4+1] = i_unknowns-3  ; // unknown for psi imaginary (order PARAMETER)
      indx[i_nodes*4+2] = i_unknowns-2 ;  // unknown for x component of magnetic potential  
      indx[i_nodes*4+3] = i_unknowns-1   ;  // unknown for y component of magnetic potential

	  //  
      if ( i==1 || i==(n_row-1) ) { // adjust for Neumann boundary condition at two sides
          i_unknowns = i_unknowns-1;
	  indx[i_nodes*4+2] = -1;
          indx[i_nodes*4+3] = indx[i_nodes*4+3]-1;
      }
//
      if( j==1 || j==(n_col-1) ) {   // adjust for Neumann boundary condition at top and bottom
          i_unknowns = i_unknowns-1;
          indx[i_nodes*4+3] = -1;
      }
//

// set node array

      if ( midpt_row==1 && midpt_col==1) {   // check to make sure you are at vertex, not midpoint
//

          if ( (i != (n_row-1)) && (j != (n_col-1)) ) {   // don't set up node info if at last node in row or column  
              i_triangle = i_triangle + 2;			  
              i_triangle_plus_one = i_triangle + 1;
              node_plus_nrow = i_nodes + n_row;
              node_plus_two_nrow = i_nodes + n_row + n_row;
              node[i_triangle*n_local+1-1] = i_nodes;
              node[i_triangle*n_local+2-1] = node_plus_two_nrow;
              node[i_triangle*n_local+3-1] = node_plus_two_nrow + 2;
              node[i_triangle*n_local+4-1] =  node_plus_nrow;
              node[i_triangle*n_local+5-1] = node_plus_two_nrow + 1;
              node[i_triangle*n_local+6-1] =  node_plus_nrow + 1;
              node[i_triangle_plus_one*n_local+1-1] = i_nodes;
              node[i_triangle_plus_one*n_local+2-1] = node_plus_two_nrow + 2;
              node[i_triangle_plus_one*n_local+3-1] = i_nodes + 2;
              node[i_triangle_plus_one*n_local+4-1] =  node_plus_nrow + 1;
              node[i_triangle_plus_one*n_local+5-1] =  node_plus_nrow + 2;
              node[i_triangle_plus_one*n_local+6-1] = i_nodes + 1;
			
			
         }
     }
//   
	
														
				
	 }

//
   }

  n_nodes = i_nodes+1 ;                     // set number of global nodes
  n_triangles = i_triangle_plus_one+1 ;      // set number of triangles
  n_unknowns = i_unknowns ;               // set number of unknowns 
  
  i_nodes=0;
  nnz_row=new int [n_unknowns];
for ( j= 0;  j< n_col; j++ )
{

    midpt_col = (j+1)%2 ;   // flag to determine if at vertical midpoint node (when midpt_col=0)
	//cout<<j<<midpt_col<<endl;
 
 
// 

    for ( i= 0;  i< n_row; i++ )
	{
	
      midpt_row = (i+1)%2;
   
      i_nodes = i_nodes + 1 ;
//
  
  
  if ( (i==0 &&j==0) || (i==n_row-1 &&j==n_col-1)){
		
				ind_find(counter,nnz_row ,9, i_nodes, indx);
				
									}
//###########################									
               													
	 else if ( (i==n_row-1&&j==0) || (i==0 &&j==n_col-1)){
										ind_find(counter,nnz_row ,6, i_nodes, indx);
										
													}
													
    else if ( indx[i_nodes*4+2]<0)			{
	   if ( midpt_row==1 && midpt_col==1) {
		ind_find(counter,nnz_row ,12, i_nodes, indx);
											}
        else{
         ind_find(counter,nnz_row ,6, i_nodes, indx);
			}
		
										}
    else if ( indx[i_nodes*4+3]<0)			{
	   if ( midpt_row==1 && midpt_col==1) {
		ind_find(counter,nnz_row ,12, i_nodes, indx);
											}
        else{
         ind_find(counter,nnz_row ,6, i_nodes, indx);
			}
	
										}
   	else	{
				if ( midpt_row==1 && midpt_col==1) {
		ind_find(counter,nnz_row ,19, i_nodes, indx);
											}
        else{
         ind_find(counter,nnz_row ,9, i_nodes, indx);
			}
	

			}
	   	
}
}  
  

  jac=.5 * ( xc[3-1] - xc[1-1] ) * ( yc[n_row + n_row+1-1] - yc[1-1] );//*0.16666666666666666667; // / 3. ;                                
  // area of triangle times quadarture weight   
  
  
 /*quad_weight[0] =0.16666666666666666667*jac;
 quad_weight[1] =0.16666666666666666667*jac;
 quad_weight[2] =0.16666666666666666667*jac;
 quad_weight[3] =0.16666666666666666667*jac;
 quad_weight[4] = 0.16666666666666666667*jac;
 quad_weight[5] = 0.16666666666666666667*jac;*/
  
  
  
  /*quad_weight[0] =0.22500000000000000*jac;
  quad_weight[1] =0.12593918054482717*jac;
  quad_weight[2] =0.12593918054482717*jac;
  quad_weight[3] =0.12593918054482717*jac;
  quad_weight[4] =0.13239415278850616*jac;
  quad_weight[5] =0.13239415278850616*jac;
  quad_weight[6] =0.13239415278850616*jac;*/
  
  quad_weight[0] =1.0/3.*jac;
  quad_weight[1] =1.0/3.*jac;
  quad_weight[2] =1.0/3.*jac;
  
  
  
  
  
//
//  set half bandwidth for coefficient matrix  
//

  n_band = 0;
for ( i_triangle= 0;  i_triangle< n_triangles; i_triangle++ ){
  

//

	for ( i_local= 0;  i_local< n_local; i_local++ ){
      i_global = node[i_triangle*n_local+ i_local];
//

	  for ( i_unknown = 0;  i_unknown < unknowns_node; i_unknown ++ ){
        i = indx[i_global*4+i_unknown];
			
        if( i >= 0 ) {
//
            
			for ( j_local= 0;  j_local< n_local; j_local++ ){
              j_global = node[i_triangle*n_local+ j_local];
//
              
			  for ( j_unknown= 0;  j_unknown< unknowns_node; j_unknown++ ){
                j = indx[j_global*4+j_unknown] ;
                if (j>=0)   ind_num+=1;
				
				
                if(i <= j) {    // only use upper half 
                    j_minus_i = j - i ;
                    if( j_minus_i  > n_band ) n_band = j_minus_i  ;
                }
//
              }
//
            }
//
        }
//
      }
//
    }
//
  }
//
n_band=n_band + 1  ;       // n_band computed is number of upper diagonals 
//
 
 return;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void ind_find(int &counter,int nnz_row[], int num, int i_nodes, int indx[])
{



			nnz_row[counter]=0;
				nnz_row[counter]+=num*2;
				if ( indx[i_nodes*4+2]>=0){
				nnz_row[counter]+=num;
									}
				if ( indx[i_nodes*4+3]>=0){
				nnz_row[counter]+=num;
									}
			    counter+=1;
				////////////////////////////////////
				nnz_row[counter]=0;
				nnz_row[counter]+=num*2;
				if ( indx[i_nodes*4+2]>=0){
				nnz_row[counter]+=num	;							
									}
				if ( indx[i_nodes*4+3]>=0){
				nnz_row[counter]+=num;
									}
			    counter+=1;
				//////////////////////
				if ( indx[i_nodes*4+2]>=0){
				nnz_row[counter]=0;
				nnz_row[counter]+=num*3;
				if ( indx[i_nodes*4+3]>=0){
				nnz_row[counter]+=num	;							
									}
			    counter+=1;
				
										}
				//////////////////////////////
				if ( indx[i_nodes*4+3]>=0){
				nnz_row[counter]=0;
				nnz_row[counter]+=num*3;
				if ( indx[i_nodes*4+2]>=0){
				nnz_row[counter]+=num;
									}
			    counter+=1;
										}

				
return;
}
//
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
void st_set_up(int indx[],int node[],int n_triangles,int unknowns_node,int n_unknowns,int n_local,
                 int n_nodes,int n_quad,int ldn,int lde, Epetra_CrsGraph & Graph, int iam)
{
//////////////////////////////////////////////////////////////
 //Local variables
int i_triangle, i_nodes, i_unknowns, n_col, j, midpt_col,   
           n_row, i,  i_local, i_global, i_unknown,  
           j_local, j_global, j_unknown,  j_minus_i, k;
          
int counter;
//////////////////////////////////////////////////////////////




 counter=0;
  
  for ( i_triangle= 0;  i_triangle< n_triangles; i_triangle++ ){

//

	for ( i_local= 0;  i_local< n_local; i_local++ ){
      i_global = node[i_triangle*n_local+ i_local];
//

	  for ( i_unknown = 0;  i_unknown < unknowns_node; i_unknown ++ ){
        i = indx[i_global*4+i_unknown];
			
        if( i >=0 ) {
//
            
			for ( j_local= 0;  j_local< n_local; j_local++ ){
              j_global = node[i_triangle*n_local+ j_local];
//
              
			  for ( j_unknown= 0;  j_unknown< unknowns_node; j_unknown++ ){
                j = indx[j_global*4+j_unknown] ;
                 if( j >= 0 ) {
					Graph.InsertGlobalIndices(i,1,&j);

				 }
       
                }
//
              }
//
            }
//
        }
//
      }
//
    }
//
Graph.FillComplete();
Graph.OptimizeStorage();

return;
}				
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
void st_set_up_2(int indx[],int node[],int n_triangles,int unknowns_node,int n_unknowns,int n_local,
                 int n_nodes,int n_quad,int ldn,int lde,Epetra_FECrsMatrix &K)
{
//////////////////////////////////////////////////////////////
 //Local variables
int i_triangle, i_nodes, i_unknowns, n_col, j, midpt_col,   
           n_row, i,  i_local, i_global, i_unknown,  
           j_local, j_global, j_unknown,  j_minus_i, k;
          
int counter;
//////////////////////////////////////////////////////////////


Epetra_SerialDenseMatrix k_local(1,1);// maybe (0,0)?
Epetra_IntSerialDenseVector local_indices_r(1); //maybe (0)?
Epetra_IntSerialDenseVector local_indices_c(1); //maybe (0)?

 counter=0;
  
  for ( i_triangle= 0;  i_triangle< n_triangles; i_triangle++ ){
//

	for ( i_local= 0;  i_local< n_local; i_local++ ){
      i_global = node[i_triangle*n_local+ i_local];
//

	  for ( i_unknown = 0;  i_unknown < unknowns_node; i_unknown ++ ){
        i = indx[i_global*4+i_unknown];
			
        if( i >=0 ) {
//
            
			for ( j_local= 0;  j_local< n_local; j_local++ ){
              j_global = node[i_triangle*n_local+ j_local];
//
              
			  for ( j_unknown= 0;  j_unknown< unknowns_node; j_unknown++ ){
                j = indx[j_global*4+j_unknown] ;
                 if( j >= 0 ) {
				 
				 						local_indices_r(0)=i;
						local_indices_c(0)=j;
						k_local(0,0)=0.0;
	 K.ReplaceGlobalValues(local_indices_r,local_indices_c,k_local);//(row_number,1,temp[k*4+l], column_number);
				 

				    
                     }
       
                }
//
              }
//
            }
//
        }
//
      }
//
    }
//
K.GlobalAssemble();
K.OptimizeStorage ();
return;
}				

void init( int ldn, double psir[],double psii[],double a1[],double a2[] )
{
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//         SET-UP INITIAL CONDITIONS
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
 
//
//  INPUT
//int    ldn;    // leading dimension of array dimensioned by global nodes
//
//  OUTPUT
//double  psir[ldn], psii[ldn];  // order parameter at previous timestep 
//double  a1[ldn], a2[ldn];      // magnetic potential at previous timestep 
int i;
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
// Set magnetic potential to zero and order PARAMETER so that (psi)^2 = 1
// 
for ( i = 0; i < ldn; i ++ ){
  psir[i] = 0.8000000000000000;
  psii[i] = 0.600000000000000;
  a1[i] = 0.000000000000000000000000000;
  a2[i] = 0.00000000000000000000000000;
}
 
 return;
}

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//

void nonlinear_solver(int  nx,int  ny,int n_band,int n_triangles,int n_local,int n_nodes, 
       int n_quad,int  n_unknowns,int  unknowns_node,int lde,int ldn,  
       double xi,double lambda,double h_external,double tol_newton,double n_newton,double time_cur,  
       double &dt,double dt_min,double dt_max,int flag_jacobian,int flag_write,double xc[],double yc[],   
       double basis_der_vertex,double basis_der_midpt,double quad_weight[],int indx[],int node[],          
       double psir_old[],double psii_old[],double a1_old[],double a2_old[], 
       double psir_new[],double psii_new[],double a1_new[],double a2_new[],int& n_nonlinear,double tau,
       int nnz_row[],int ncc, const Epetra_MpiComm Comm	, int iam , int &flag ,
       Epetra_FECrsMatrix &K,Epetra_Vector &F, Epetra_Vector &x,AztecOO &solver,
	    double &s_st_t,double &j_st_t, int eps, Epetra_Map &Map )
{       

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//                     NEWTON ITERATION
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
 
//
//
// Allocatable variables
//double *a_coeff;     // coefficient matrix 
//double *a_lu;     // coefficient matrix 
double rhs[n_unknowns];           // right hand side vector
//double *rhs_new;           // right hand side vector
double resid_a1[n_nodes], resid_a2[n_nodes];      // residual for magnetic potential
double resid_psir[n_nodes], resid_psii[n_nodes];  // residual for order potential 
double delta_a1[n_nodes], delta_a2[n_nodes];      // increment for magnetic potential
double delta_psir[n_nodes], delta_psii[n_nodes];  // increment for order potential 
//
//  Local variables
int    k, kk, n_rhs,  info, n_lband, i,ii,array1[4],array2[4],indi[n_unknowns];
	   
double    norm_resid_psi,  norm_resid_a, norm_resid,
			s_st,s_fin,j_st,j_fin,nl_st,nl_fin,nl_per_it ;
double  temp1, temp2,*temp;

// 
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
 start:;  //CONTINUE
 nl_st=MPI_Wtime();
//
// Newton iteration loop 
//
   Teuchos::ParameterList MLList; // keep here not above
/*ML_Epetra::SetDefaults("DD", MLList);
MLList.set("max levels",2);
 MLList.set("increasing or decreasing","increasing"); 
MLList.set("aggregation: type", "ParMETIS");//metis doesnt work with DD (i guess)
MLList.set("aggregation: nodes per aggregate", 16);
MLList.set("smoother: pre or post", "both");
MLList.set("coarse: type","Gauss-Seidel"); // gS works better 
//MLList.set("smoother: type", "Gauss-Seidel");
MLList.set("smoother: type", "Aztec");
*/

//PrecondOperator PrecondOp(Comm,&Map);

/*
// set default values for smoothed aggregation in MLList
ML_Epetra::SetDefaults("SA",MLList); //  DD-ML gets iter from 96 to 65 at start.  WAS SA.. DD-ML-LU take more iters but it QUICK IN TImE!!!!!
// overwrite with user’s defined parameters     // although fast i think it used alot of mem.
MLList.set("max levels",6);//2 was 6 for SA
//MLList.set("PDE equations",4); // new
MLList.set("increasing or decreasing","decreasing");
MLList.set("aggregation: type", "ParMETIS");//MIS// but metis works here??
						// ^ I NEED TO SET THIS TO MIS OR SOMETHING BESIDES PARMETIS TO AVOID KEY PROBLEEM... altealst for SA
//MLList.set("aggregation: smoothing sweeps", 3);//new
//MLList.set("repartition: partitioner","ParMETIS"); // new
MLList.set("coarse: type","Gauss-Seidel");//"Amesos-KLU");
MLList.set("smoother: type","Aztec");//  seems to work good! WAS NOT HERE
*/

// gets 30 fucking iters on start....
ML_Epetra::SetDefaults("DD-ML",MLList); //  DD-ML gets iter from 96 to 65 at start.  WAS SA.. DD-ML-LU take more iters but it QUICK IN TImE!!!!!
// overwrite with user’s defined parameters     // although fast i think it used alot of mem.
MLList.set("max levels",3);//2 was 6 for SA
MLList.set("PDE equations",4); // lowered iters tremendously..
MLList.set("increasing or decreasing","decreasing");
MLList.set("aggregation: type", "METIS");//MIS// but metis works here??
						// ^ I NEED TO SET THIS TO MIS OR SOMETHING BESIDES PARMETIS TO AVOID KEY PROBLEEM... altealst for SA
//MLList.set("aggregation: smoothing sweeps", 3);//new
//MLList.set("repartition: partitioner","ParMETIS"); // new
MLList.set("coarse: type","Gauss-Seidel");//"Amesos-KLU");
MLList.set("smoother: type","Aztec");//  seems to work good! WAS NOT HERE

//for DD and DD-ML and DD-ML-LU
Teuchos::RCP<vector<int>> options = Teuchos::rcp(new vector<int>(AZ_OPTIONS_SIZE));
Teuchos::RCP<vector<double>> params = Teuchos::rcp(new vector<double>(AZ_PARAMS_SIZE));
//AZ_defaults(options,params); // complains about rcp vectors...
AZ_defaults(&(*options)[0],&(*params)[0]);
(*options)[AZ_precond] = AZ_dom_decomp;
(*options)[AZ_subdomain_solve] = AZ_icc;
MLList.set("smoother: Aztec options", options);
MLList.set("smoother: Aztec params", params);
//ML_Epetra::SetDefaults("DD",MLList,options,params); you might need this as well




// create the preconditioner
 ML_Epetra::MultiLevelPreconditioner* MLPrec =
new ML_Epetra::MultiLevelPreconditioner(K, MLList, false);


 for ( k = 0; k < n_newton; k ++ ){

norm_resid_psi=0.000000000000000;
norm_resid_a = 0.000000000000000;
norm_resid = 0.000000000000000; 

 // should be elements lde not ncc ???
//Epetra_Map Map(-1, n_unknowns, 0, Comm);// worked with 4*lde so tri nunk
//Epetra_Vector F(Map);

 j_st=MPI_Wtime(); 
//
// compute residual -- right hand side of Newton linear system 
    residual ( psir_old, psii_old, a1_old, a2_old,    
            psir_new, psii_new, a1_new, a2_new, xc, yc, indx, node,    
            basis_der_vertex, basis_der_midpt, quad_weight, nx, ny,  
            n_triangles, n_local,  n_nodes, n_quad,  n_unknowns,   
            unknowns_node, lde, ldn, xi, lambda, h_external, dt,    
            resid_psir, resid_psii, resid_a1, resid_a2,tau,eps,time_cur) ;
			
// 
// calculate norm of residual 
 for ( ii = 0; ii < n_nodes; ii ++ ){
    norm_resid_psi +=  (   resid_psir[ii]*resid_psir[ii]     
                       +  resid_psii[ii]*resid_psii[ii]    )   ;
    norm_resid_a +=(  resid_a1[ii]*resid_a1[ii]        
                       +  resid_a2[ii]*resid_a2[ii]   ) ;
     
    //print *, "norm residuals", norm_resid_psi, norm_resid_a
     }
	 norm_resid_psi=sqrt(norm_resid_psi / double(n_nodes)  );
	 norm_resid_a = sqrt(norm_resid_a / double(n_nodes)  )   ;          
	 norm_resid = max ( norm_resid_psi, norm_resid_a );
//    


// output Newton residual information 
    if(flag_write >= 2) {
	   
	   if (iam==0){
        cout<< " After "<< k<< "Nonlinear iterates "<<endl;
        cout<< " Norms of residuals for psi and A : "<< norm_resid_psi<<", "<< norm_resid_a  <<endl;
		cout<<"Jacobian and resid time: "<< j_st_t<<"Solver time: "<<s_st_t<<endl;
		
		}
    }
//
    //if ( norm_resid > 10^4 ) GO TO 100  ;      // residual too large so reduce time step
//
//check to see if solution is good enough; if so, output; if not, do a Newton step  //
// 
    if ( norm_resid < tol_newton ){       // solution good enough so write out GFE, etc.
	nl_fin=MPI_Wtime();
	   nl_per_it=nl_fin-nl_st;
	   if (iam==0){
        cout<< "      NONLINEAR ITERATION CONVERGED in "<< k<< " steps "<<endl;
		cout<<" NL time for this solve: "<<nl_per_it<<endl;
		}
        n_nonlinear = k;//-1;
		delete MLPrec;		

       return;
     }
	 else{            // perform Newton iteration
//
// put 4 residual vectors into one rhs vector
     four_to_one( resid_psir, resid_psii, resid_a1, resid_a2,  
                       indx, n_nodes, n_unknowns, ldn, rhs,iam ) ;
   //  four_to_one( psir_new, psii_new, a1_new, a2_new,  
     //                  indx, n_nodes, n_unknowns, ldn, rhs,x,iam ) ;
	 for( ii=0; ii<n_unknowns; ii++){
	 indi[ii]=ii;
	 }
	if (iam==0) cout<<"PAST II loop"<<endl;
F.ReplaceGlobalValues(n_unknowns, rhs,indi);
     four_to_one( psir_new, psii_new, a1_new, a2_new,  
                       indx, n_nodes, n_unknowns, ldn, rhs,iam ) ;
x.ReplaceGlobalValues(n_unknowns, rhs,indi);
	//				   F.GlobalAssemble(); 
//                         x=F;
if (iam==0) cout<<"PAST REPLACE VECTORS"<<endl;
			
//
// assemble jacobian
     if( ((k+1)%flag_jacobian)==0 )  
	 {
	 
         assemble_jac( psir_new, psii_new, a1_new, a2_new, xc, yc,    
            indx, node, basis_der_vertex, basis_der_midpt,quad_weight,  
            nx, ny, n_band, n_triangles, n_local, n_nodes, n_quad, n_unknowns,    
            unknowns_node, lde, ldn, xi, lambda, dt,tau, nnz_row,ncc, Comm, K, iam ,flag, eps )  ;
			
			
			}
if (iam==0) cout<<"PAST JAcobian"<<endl;

             j_fin=MPI_Wtime() ;

       n_rhs=1 ;
       n_lband=n_band-1 ;
 s_st=MPI_Wtime() ;
 

 
 

MLPrec->ComputePreconditioner(); 

solver.SetPrecOperator(MLPrec);

if( iam==0) cout<<"Past compute Precond"<<endl;


 solver.Iterate(10000, tol_newton);
 if (iam==0)
 cout << "Solver performed " << solver.NumIters() << " iterations." << endl;
         	

MLPrec->DestroyPreconditioner();

s_fin=MPI_Wtime() ;		
/*for (ii=0; ii<x.MyLength(); i++){			 
MPI_Allgather(&x[ii], 1, 
     MPI_DOUBLE, &rhs[iam*x.MyLength()+ii], 1,
     MPI_DOUBLE, MPI_COMM_WORLD )	;								 
	 }
	 */
	 
flag=1;

  
	 temp=new double[x.MyLength()];
	 for( ii=0; ii<x.MyLength(); ii++){
	 temp[ii]=x[ii];//+iam*x.MyLength()];
	 //cout<<iam<<"' "<<x[ii]<<", "<<ii<<endl;
	 }
	 
Comm.GatherAll	(	temp, rhs, x.MyLength() )	;
	delete [] temp;
	
//
//  break vector of length number of unknowns into 4 vectors of
//       length number of nodes
       one_to_four( rhs,indx, n_nodes, n_unknowns, ldn, 
            delta_psir, delta_psii, delta_a1, delta_a2 )  ;     
//
// increment to old Newton iterate to get new Newton iterate   
for ( ii = 0; ii < n_nodes; ii ++ ){
        psir_new[ii] = psir_new[ii] + delta_psir[ii];
        psii_new[ii] = psii_new[ii] + delta_psii[ii];
        a1_new[ii] = a1_new[ii] + delta_a1[ii];
        a2_new[ii] = a2_new[ii] +  delta_a2[ii] ;         
}
		//

    }         // end for Newton   

s_st_t+=s_fin-s_st;
j_st_t+=j_fin-j_st;

    }       // end iteration loop for Newton's method

	//  
//  Newton iteration failed to converge 
    cout<< "* // Nonlinear iteration failed to converge* // " <<endl;
//
// halve the time step and re-start Newton iteration 
// 
 //CONTINUE

    time_cur = time_cur - dt;
    dt=.5000000000000000*dt;
    time_cur = time_cur + dt;
    cout<<  "halved the timestep, dt= "<< dt<<endl;
    cout<< " ^* current time is  "<< time_cur<<endl;
//
    if ( dt < dt_min ) {
       cout<< " delta t too small so stop "<<endl;
        exit (EXIT_FAILURE);
	}
     else{
for ( ii = 0; ii < n_nodes; ii ++ ){	 
	psir_new[ii] = psir_old[ii];
	psii_new[ii] = psii_old[ii] ;
	a1_new[ii] = a1_old[ii] ;
	a2_new[ii] = a2_old[ii]   ;
	}
	
        goto start;
		
    }

//
 
 return;
}

//
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//

void one_to_four (double f[],int indx[],int n_nodes,int n_unknowns,int ldn,
             double a[],double b[],double c[],double d[])
			 {
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//         Takes 1 vectors (f)   of length number of unknowns  and
//         makes 4 vector (a,b,c,d) dimensioned by number of nodes 
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
// 
//
//
//  LOCAL VARIABLES
int  k, kk,ii;
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//
for ( ii = 0; ii < n_nodes; ii ++ ){
  a[ii] = 0.0 ;
  b[ii] = 0.0 ;
  c[ii] = 0.0 ;
  d[ii] = 0.0 ;
  }
// 
  k=0;
  for ( kk = 0; kk < n_nodes; kk ++ ){
  
    

    a[kk] = f[k];
    k = k + 1;

    b[kk] = f[k] ;
// 
   if( indx[kk*4+2] >= 0) {
       k = k+1;

       c[kk] = f[k];
   }
//
   if( indx[kk*4+3] >= 0) {
       k = k+1;

       d[kk] = f[k];
   }
//
k = k + 1;
  }
//
 
 return;
}

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
void output (double psir_new[],double psii_new[],double a1_new[],double a2_new[],int *node,int lde,int nx,int ny, 
                   double xi,double lambda,double h_external,double time_cur,int n_count,double quad_weight[],
				   double basis_der_midpt,double basis_der_vertex,int n_triangles,int n_local,int n_nodes,int n_quad,
				    double xc[], double yc[],int &flag, string s,double jac)  
{
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*c
// COMPUTE MAGNITUDE OF THE ORDER PARAMETER, THE GIBBS FREE ENERGY, AVERAGE
//    FIELD,  BULK MAGNETIZATION, MAGNETIC FIELD,AND COMPONENTS OF THE CURRENT                
//
//
//  LOCAL VARIABLES
int    i_triangle, i_quad, i, j, k, i_start, i_end; 
int    n_row, n_col,ii; 
double order_parameter[n_nodes];
double current_1[n_triangles], current_2[n_triangles], magnetic_field[n_triangles];
double   psir, psir_x, psir_y, psii, psii_x, psii_y, a1, a1_x, a1_y, a2, a2_x, a2_y,  
                     psi_sq, a_sq, curl_a, grad_psir_sq,grad_psii_sq, a_dot_grad_psir, a_dot_grad_psii, 
                     gibbs_free_energy, average_magnetic_field, bulk_magnetization, one_over_lambda,  
                     area_triangle, total_area,one_over_lambdasq;
ifstream output1;
ofstream output2;
ofstream output3;
ofstream output4;
ofstream output5;


//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*  
//


output1.open ( "init_out.data" , ofstream::out | ofstream::app);
   output2.open ( "order_test.data", ofstream::out | ofstream::app);
 output3.open ( "current_test.data", ofstream::out | ofstream::app);
output4.open ( "magnetic_test.data", ofstream::out | ofstream::app);
output5.open ( "init_out_test.data", ofstream::out | ofstream::trunc);


//flag=1;

one_over_lambda = 1. / lambda;
one_over_lambdasq = one_over_lambda * one_over_lambda;
area_triangle =jac;// quad_weight * 3.0;
//
//  compute magnitude of order parameter 
for ( ii = 0; ii < n_nodes; ii ++ ){
order_parameter[ii] =  sqrt(  psir_new[ii]*psir_new[ii] + psii_new[ii]*psii_new[ii]  );
}
//
//  compute Gibbs free energy, bulk magnetic field, bulk magnetization
// 
  gibbs_free_energy = 0.0;
  average_magnetic_field = 0.0;
  bulk_magnetization = 0.0;
//
 for ( i_triangle = 0; i_triangle < n_triangles ;i_triangle++ ){
  
    magnetic_field[i_triangle] = 0.0;
    current_1[i_triangle] = 0.0;
    current_2[i_triangle] = 0.0;
//
 for ( i_quad = 0; i_quad < n_quad ;i_quad++ ){
    
      eval_space_qdpt( i_triangle,    
           psir_new, psii_new, a1_new, a2_new,                
           xc, yc, basis_der_vertex, basis_der_midpt, 
           node, n_local, i_quad, lde, n_nodes,               
           psir, psir_x, psir_y, psii, psii_x, psii_y,      
           a1, a1_x, a1_y, a2, a2_x, a2_y )   ;
      psi_sq = psir*psir + psii*psii;
      a_sq = a1*a1 + a2*a2 ;
      curl_a = a2_x - a1_y;
      grad_psir_sq = psir_x * psir_x + psir_y * psir_y;   grad_psii_sq = psii_x * psii_x + psii_y * psii_y;
      a_dot_grad_psir = a1 * psir_x + a2 * psir_y;
      a_dot_grad_psii = a1 * psii_x + a2 * psii_y;
//
      gibbs_free_energy = gibbs_free_energy   
          +(  psi_sq * ( .5 * psi_sq + a_sq * one_over_lambdasq - 1. ) 
          +  curl_a  * ( curl_a - 2. * h_external )  
          + xi * ( xi * ( grad_psir_sq + grad_psii_sq )  
          + 2. *  one_over_lambda * ( psii * a_dot_grad_psir - psir * a_dot_grad_psii )  )
		  )* quad_weight[i_quad];
//
      magnetic_field[i_triangle] = magnetic_field[i_triangle] + curl_a* quad_weight[i_quad];
      current_1[i_triangle] = current_1[i_triangle]   
             + (xi * ( psir * psii_x - psii * psir_x ) - psi_sq * a1 * one_over_lambda)
			 * quad_weight[i_quad];   
      current_2[i_triangle] = current_2 [i_triangle]  
             + (xi * ( psir * psii_y - psii * psir_y ) - psi_sq * a2 * one_over_lambda)
			 * quad_weight[i_quad]; 
//
    }
//
    
// set  magnetic field and current over an element
// and also contributions to average field  
// 
    magnetic_field[i_triangle] = magnetic_field[i_triangle] / area_triangle;
    current_1[i_triangle] = current_1[i_triangle] / area_triangle;
    current_2[i_triangle] = current_2[i_triangle] / area_triangle;
    average_magnetic_field = average_magnetic_field + magnetic_field[i_triangle];
   
//
  }
//
// set average field over region and bulk magnetization 
//
  total_area = area_triangle * double(n_triangles);
  average_magnetic_field = average_magnetic_field / double(n_triangles) ;
  bulk_magnetization = h_external - average_magnetic_field;
  
//
// print out results & write psi_sq, current and magnetic field vectors to file
  cout<< "Gibbs free energy = "<< gibbs_free_energy<<endl;
  cout<< "Average magnetic field = "<< average_magnetic_field<<endl;
  cout<< "Bulk magnetization = "<< bulk_magnetization<<endl;
  
//
// write out results in Mathematica format to file
  //if (n_count <= 9) {          // write out name of array
      //write(3,301) n_count
	//  }
    //else if (n_count >= 10 && n_count <= 99)
      //write(3,311) n_count
    //else
     // write(3,321) n_count
  //END if
//
  n_row = 2 * nx - 1;    n_col = 2 * ny - 1;
  for ( k= 0; k< n_col ;k++ ){
  
      i_end = (k+1) * n_row  ;
      i_start =  i_end - n_row;//
//      write(3,*) '{'
      //write(3,331) ( order_parameter[i], i = i_start, i_end - 1 );
	  for ( i = i_start; i< i_end-1  ;i++ ){
	   output2<<fixed << setprecision(16) << order_parameter[i]<<" ";
	  }
//
      if (k+1 == n_col) {
          //write(3,341)  order_parameter[i_end];
		  output2<<fixed << setprecision(16) << order_parameter[i_end-1]<<endl;
		  }
        else{
          //write(3,351)  order_parameter[i_end];
		  		  output2<<fixed << setprecision(16) << order_parameter[i_end-1]<<endl;
      } 
//
  }
  output5<<fixed << setprecision(16) << time_cur<<endl;
  for ( k= 0; k< n_nodes ;k++ ){
  
  output5<<fixed << setprecision(16) << psir_new[k]<<endl;
  output5<<fixed << setprecision(16) << psii_new[k]<<endl;
  output5<<fixed << setprecision(16) << a1_new[k]<<endl;
  output5<<fixed << setprecision(16) << a2_new[k]<<endl;
  
  }
//
/*
for ( i_triangle= 0; i_triangle < n_triangles ;i_triangles++ ){
  
    write(4,*) current_1[i_triangle], current_2[i_triangle];
    write(5,*) magnetic_field[i_triangle] ;
  }
//
301  format('  order',i1,':={' )
311  format('  order',i2,':={' )
321  format('  order',i3,':={' )
331         format(8(f8.5))
341           format(f8.5)
351           format(f8.5)
//

//
 */
 return;
}

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//

void quad_basis_fn(int it,int i_local,int i_quad,  
       double basis_der_vertex,double basis_der_midpt,double  &basis,
	   double &basis_x,double &basis_y ) 
{	   
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
//  quadratic basis functions
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
 
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
// reset mods and -1 to quads and i_locals
  if ( i_local <= 3-1 ){          // evaluate basis functions centered at verticies      
//
      basis = 0.0; basis_x = 0.0;  basis_y= 0.0;
      if (((it+1)% 2) == 0 ){ 
          if ( i_local == 1-1){ 
              basis_x = -basis_der_vertex ;
              if ( i_quad == 2-1 ) basis_x = basis_der_vertex ;
			  }
            else if (i_local == 2-1) {
                  basis_y = basis_der_vertex ;
                  if ( i_quad == 3 -1) basis_y = -basis_der_vertex ;
				  }
            else{
              basis_x = basis_der_vertex ;  basis_y=-basis_der_vertex  ;
              if ( i_quad == 1-1) {
                  basis_x = -basis_der_vertex ;
                  basis_y =  basis_der_vertex ;
								}
				  }
//            END if
//        END if
//           write(0,*)"even", it, i_local,i_quad ;
//           write(0,*)  basis, basis,  basis_x, basis_x,  basis_y,basis_y   ;
		}
        else{
          if ( i_local == 1-1) {
              basis_y = -basis_der_vertex ;
			  
              if ( i_quad == 2 -1) basis_y = basis_der_vertex ;
			  }
            else if (i_local == 2-1) {
              basis_x = -basis_der_vertex ;  basis_y = basis_der_vertex ;
              if ( i_quad == 3-1) {
                  basis_x = basis_der_vertex ;
                  basis_y = -basis_der_vertex ;
				  }
				  }
//            END if
            else{
              basis_x = basis_der_vertex ;
              if ( i_quad == 1 -1) basis_x = -basis_der_vertex ;
			  }
  //     END if
//          write(0,*) basis, basis, basis_x, basis_x, basis_y, basis_y
    //END if
//                      
//
           }
          }
    else{                 // evaluate basis functions centered at midpoints of sides     
      basis = 0.0; basis_x = 0.0;  basis_y= 0.0;
      if ( (it+1)%2 == 0 ) {
          if ( i_local == 4-1) {
              basis_x = -basis_der_midpt;   basis_y = basis_der_midpt;
			  
              if ( i_quad == 1 -1) basis  = 1.;
              if ( i_quad == 2 -1) basis_y = 0.; 
              if ( i_quad == 3 -1) basis_x = 0. ;
			  }
            else if ( i_local == 5-1) {
               basis_x =  basis_der_midpt;   basis_y =  basis_der_midpt;
                  if ( i_quad == 1 -1) basis_y = -basis_der_midpt;
                  if ( i_quad == 2 -1) {
                      basis = 1.;
                      basis_y = 0.;
					  }
      //          END if 
                  if ( i_quad == 3 -1) basis_x = 0 ;
				 } 
				 
            else{
              basis_x = basis_der_midpt;  basis_y=basis_der_midpt ;
              if ( i_quad == 1 -1) basis_y = -basis_der_midpt;
              if ( i_quad == 2 -1) {
                  basis_x = -basis_der_midpt;
                  basis_y = 0;}
        //    END if 
              if ( i_quad == 3 -1) {
                  basis = 1.;
                  basis_x = 0.;
                  basis_y = -basis_der_midpt;}
				  }
				  }
          //  END if
      //  END if
//           write(0,*)"even", it, i_local,i_quad 
//          write(0,*)  basis, basis,  basis_x, basis_x,  basis_y,basis_y   
			
        else{
          if ( i_local == 4-1) {
              basis_x = -basis_der_midpt;
              if ( i_quad == 1-1 ) basis = 1.;
              if ( i_quad == 2 -1) {
                  basis_x = 0.;  basis_y = -basis_der_midpt ;
				  }
        //    END if
					
              if ( i_quad == 3 -1) basis_y = basis_der_midpt ;
			     }
            else if (i_local == 5-1){ 
              basis_x = basis_der_midpt;  basis_y = basis_der_midpt;
              if ( i_quad == 1 -1) basis_y = 0.;
              if ( i_quad == 2 -1) {
                  basis  = 1.;  basis_x = 0.;
				  }
				  
          //  END if
              if ( i_quad == 3 -1) basis_x = -basis_der_midpt ;
			  }
            else{
              basis_x = basis_der_midpt; basis_y = -basis_der_midpt;
              if ( i_quad == 1 -1) basis_y = 0;
              if ( i_quad == 2 -1) basis_x  = 0. ;
              if ( i_quad == 3 -1) basis  = 1.0;
			  }
			 
//        END if
//          write(0,*)"midpt", basis, basis, basis_x, basis_x, basis_y, basis_y
//    END if
		}
     }
//END if
 
 return;
}

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*

void residual (double psir_old[],double psii_old[],double a1_old[],double a2_old[],    
            double psir_new[],double psii_new[],double a1_new[],double a2_new[],
			double xc[],double yc[],int indx[],int node[],   
            double basis_der_vertex,double basis_der_midpt,double quad_weight[],int nx,int ny,  
            int n_triangles,int n_local,int  n_nodes,int n_quad,int  n_unknowns,    
            int unknowns_node,int lde,int ldn,double xi,double lambda,double h_external,
			double dt,double resid_psir[],double resid_psii[],double resid_a1[],
			double resid_a2[],double tau , int eps,double time_cur) 
{
//            
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^* 
//
//              CALCULATES RESIDUAL VECTOR OF NEWTON ITERATE
//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
 
//
//  Local variables
int    i_triangle , i_quad, i_local, i_global,ii;
double     psir, psir_x, psir_y, psir_t,  
          psii, psii_x, psii_y, psii_t,  psi_sq,  
          a1, a1_x, a1_y, a1_t, a2, a2_x, a2_y, a2_t, a_sq, xi_sq,  
          one_over_lambda, one_over_lambda_sq, one_over_kappa, basis_over_dt, 
          basis_over_lambda, psisq_plus_asq_over_lambdasq,  
          basis, basis_x, basis_y , test_term_psir, test_term_psii, 
          test_term_a1, test_term_a2,y_node;
double J,kappa;
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*
//
xi_sq = xi* xi;
one_over_lambda = 1. / lambda;
one_over_lambda_sq = one_over_lambda * one_over_lambda;
one_over_kappa = xi / lambda;
kappa=lambda/xi;
J=0.0000000000000;
// zero arrays 
for ( ii = 0; ii < n_nodes; ii ++ ){
  resid_psir[ii] = 0.0 ;
  resid_psii[ii] = 0.0 ;
  resid_a1[ii] = 0.0 ;
  resid_a2[ii] = 0.0 ;
  }
//
// calculate residual  
//
  
	
for ( i_triangle= 0; i_triangle < n_triangles ;i_triangle++ ){
             // loop over elements 
//
for ( i_quad= 0; i_quad < n_quad ;i_quad++ ){
         // loop over quadrature points                
                                  // evaluate solution and derivative at quad pt
          eval_space_time_qdpt (  i_triangle ,  
               psir_old, psii_old, a1_old, a2_old, 
               psir_new, psii_new, a1_new, a2_new,  
               xc, yc, basis_der_vertex, basis_der_midpt, 
               node, n_local, i_quad, lde, n_nodes,        
               psir, psir_x, psir_y, psir_t, psii, psii_x, psii_y, psii_t,  
               a1, a1_x, a1_y, a1_t, a2, a2_x, a2_y, a2_t   );

// 
          psi_sq  = psir*psir + psii*psii ;
          a_sq = a1*a1 + a2*a2 ;
          psisq_plus_asq_over_lambdasq = psi_sq + a_sq * one_over_lambda_sq          ;
// 
for ( i_local= 0; i_local < n_local; i_local++ ){
              // loop over local nodes
            i_global = node[i_triangle*n_local+i_local];
                                 // determine basis function at quadrature point                            
  	qbf (i_quad, i_triangle, i_local, xc,yc,
			      node, lde, n_local,
		   n_nodes, basis, basis_x, basis_y );


                basis_over_dt = basis / dt ;//* 0.1
                basis_over_lambda = basis * one_over_lambda ;
//
                test_term_psir = -psir * basis * ( psisq_plus_asq_over_lambdasq - tau ) 
                   +  basis *( a1 * psii_x + a2  * psii_y) * one_over_kappa   
                   -  psir_t * basis_over_dt;//-J*psii*basis*yc[node[i_triangle*n_local+i_quad+3]]*kappa;
                test_term_psii = -psii * basis * ( psisq_plus_asq_over_lambdasq - tau )      
                   -  basis * (a1 *psir_x + a2 *  psir_y  ) * one_over_kappa     
                   -  psii_t * basis_over_dt;// +J*psir*basis*yc[node[i_triangle*n_local+i_quad+3]]*kappa  ;
                test_term_a1 = 	- (  psi_sq * a1 * one_over_lambda    
                   +  xi * (psii * psir_x - psir * psii_x)  ) * basis_over_lambda   
                   -  a1_t * basis_over_dt * one_over_lambda_sq   ;
                test_term_a2 = - (  psi_sq * a2 * one_over_lambda    
                   +  xi * (psii * psir_y - psir * psii_y )  ) * basis_over_lambda   
                   -  a2_t * basis_over_dt * one_over_lambda_sq ;//+J*basis 

			//
// residual for psi-real test function 

            resid_psir[i_global] = resid_psir[i_global]     
                +sqrt(dt)*quad_weight[i_quad]*
				(- xi_sq * (psir_x * basis_x + psir_y * basis_y)  
                + test_term_psir     
                - psii * (a1  * basis_x + a2 * basis_y ) * one_over_kappa) ;
//
// residual for psi-imaginary test function 
            resid_psii[i_global] = resid_psii[i_global]   
			    +sqrt(dt)*quad_weight[i_quad]*(
                - xi_sq * (psii_x * basis_x + psii_y * basis_y)  
                +  test_term_psii     
                +  psir * (a1 * basis_x  + a2 * basis_y ) * one_over_kappa);   //
// residual for A-1 test function  
            if ( indx[i_global*4+2] >= 0 )    
               resid_a1[i_global] = resid_a1[i_global]   
			   +sqrt(dt)*quad_weight[i_quad]*(
			   -(h_external)
				  * basis_y - eps*(a1_x + a2_y ) * basis_x     
                 +  ( a2_x - a1_y ) * basis_y + test_term_a1); 
//
// residual for A-2 test function 
             if ( indx[i_global*4+3] >= 0 )  
                resid_a2[i_global] = resid_a2[i_global]  
				+sqrt(dt)*quad_weight[i_quad]*(		
                 h_external
				 * basis_x - eps*(a1_x + a2_y ) * basis_y     
                 -  ( a2_x - a1_y ) * basis_x + test_term_a2) ; 

			  
				 //
          }
//

        }
//  
      }
// 

/*for ( ii = 0; ii < n_nodes; ii ++ ){
 resid_psir[ii] = resid_psir[ii] * quad_weight;
 resid_psii[ii] = resid_psii[ii] * quad_weight;
 resid_a1[ii] = resid_a1[ii] * quad_weight;
 resid_a2[ii] = resid_a2[ii] * quad_weight;
 }*/
//
 
  return;
}

//
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*

//****************************************************************************80
int st_to_cc_size ( int nst, int ist[], int jst[] )

//****************************************************************************80
//
//  Purpose:
//
//    ST_TO_CC_SIZE sizes CC indexes based on ST data.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    15 July 2014
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NST, the number of ST elements.
//
//    Input, int IST[NST], JST[NST], the ST rows and columns.
//
//    Output, int ST_TO_CC_SIZE, the number of CC elements.
//
{
  int *ist2;
  int *jst2;
  int ncc;
//
//  Make copies so the sorting doesn't confuse the user.
//
  ist2 = i4vec_copy_new ( nst, ist );
  jst2 = i4vec_copy_new ( nst, jst );
//
//  Sort by column first, then row.
//
  i4vec2_sort_a ( nst, jst2, ist2 );
//
//  Count the unique pairs.
//
  ncc = i4vec2_sorted_unique_count ( nst, jst2, ist2 );

  delete [] ist2;
  delete [] jst2;

  return ncc;
}
void st_to_cc_index ( int nst, int ist[], int jst[], int ncc, int n, 
  int icc[], int ccc[] )

//****************************************************************************80
//
//  Purpose:
//
//    ST_TO_CC_INDEX creates CC indices from ST data.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    15 July 2014
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NST, the number of ST elements.
//
//    Input, int IST[NST], JST[NST], the ST rows and columns.
//
//    Input, int NCC, the number of CC elements.
//
//    Input, int N, the number of columns in the matrix.
//
//    Output, int ICC[NCC], the CC rows.
//
//    Output, int CCC[N+1], the compressed CC columns.
//
// IM GOING TO FLIP thiS SO IT COMPRESSES row not CC, need to know convention you can get non zeros

{
  int *ist2;
  int j;
  int *jcc;
  int jhi;
  int jlo;
  int *jst2;
  int k;
//
//  Make copies so the sorting doesn't confuse the user.
//
  ist2 = i4vec_copy_new ( nst, ist );
  jst2 = i4vec_copy_new ( nst, jst );
//
//  Sort the elements.
//
  i4vec2_sort_a ( nst, jst2, ist2 );
//
//  Get the unique elements.
//
  jcc = new int[ncc];
  i4vec2_sorted_uniquely ( nst, jst2, ist2, ncc, jcc, ccc );
//
//  Compress the \x column x\ -> row index.
//
  icc[0] = 0;
  jlo = 0;
  for ( k = 0; k < ncc; k++ )
  {
    jhi = jcc[k];
    if ( jhi != jlo )
    {
      for ( j = jlo + 1; j <= jhi; j++ )
      {
        icc[j] = k;
      }
      jlo = jhi;
    }
  }
  jhi = n;
  for ( j = jlo + 1; j <= jhi; j++ )
  {
    icc[j] = ncc;
  }

  delete [] ist2;
  delete [] jcc;
  delete [] jst2;

  return;
}
//****************************************************************************80
int *i4vec_copy_new ( int n, int a1[] )

//****************************************************************************80
//
//  Purpose:
//
//    I4VEC_COPY_NEW copies an I4VEC.
//
//  Discussion:
//
//    An I4VEC is a vector of I4's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    04 July 2008
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of entries in the vectors.
//
//    Input, int A1[N], the vector to be copied.
//
//    Output, int I4VEC_COPY_NEW[N], the copy of A1.
//
{
  int *a2;
  int i;

  a2 = new int[n];

  for ( i = 0; i < n; i++ )
  {
    a2[i] = a1[i];
  }
  return a2;
}
//****************************************************************************80
void i4vec2_sort_a ( int n, int a1[], int a2[] )

//****************************************************************************80
//
//  Purpose:
//
//    I4VEC2_SORT_A ascending sorts an I4VEC2.
//
//  Discussion:
//
//    Each item to be sorted is a pair of integers (I,J), with the I
//    and J values stored in separate vectors A1 and A2.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of items of data.
//
//    Input/output, int A1[N], A2[N], the data to be sorted..
//
{
  int i;
  int indx;
  int isgn;
  int j;
  int temp;
//
//  Initialize.
//
  i = 0;
  indx = 0;
  isgn = 0;
  j = 0;
//
//  Call the external heap sorter.
//
  for ( ; ; )
  {
    sort_heap_external ( n, indx, i, j, isgn );
//
//  Interchange the I and J objects.
//
    if ( 0 < indx )
    {
      temp    = a1[i-1];
      a1[i-1] = a1[j-1];
      a1[j-1] = temp;

      temp    = a2[i-1];
      a2[i-1] = a2[j-1];
      a2[j-1] = temp;
    }
//
//  Compare the I and J objects.
//
    else if ( indx < 0 )
    {
      isgn = i4vec2_compare ( n, a1, a2, i, j );
    }
    else if ( indx == 0 )
    {
      break;
    }
  }
  return;
}
//****************************************************************************80
void i4vec2_sorted_uniquely ( int n1, int a1[], int b1[], int n2, int a2[], 
  int b2[] )

//****************************************************************************80
//
//  Purpose:
//
//    I4VEC2_SORTED_UNIQUELY keeps the unique elements in an I4VEC2.
//
//  Discussion:
//
//    Item I is stored as the pair A1(I), A2(I).
//
//    The items must have been sorted, or at least it must be the
//    case that equal items are stored in adjacent vector locations.
//
//    If the items were not sorted, then this routine will only
//    replace a string of equal values by a single representative.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    15 July 2014
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N1, the number of items.
//
//    Input, int A1[N1], B1[N1], the input array.
//
//    Input, int N2, the number of unique items.
//
//    Input, int A2[N2], B2[N2], the output array of unique items.
//
{
  int i1;
  int i2;

  i1 = 0;
  i2 = 0;

  if ( n1 <= 0 )
  {
    return;
  }

  a2[i2] = a1[i1];
  b2[i2] = b1[i1];

  for ( i1 = 1; i1 < n1; i1++ )
  {
    if ( a1[i1] != a2[i2] || b1[i1] != b2[i2] )
    {
      i2 = i2 + 1;
      a2[i2] = a1[i1];
      b2[i2] = b1[i1];
    }
  }

  return;
}
//****************************************************************************80
void sort_heap_external ( int n, int &indx, int &i, int &j, int isgn )

//****************************************************************************80
//
//  Purpose:
//
//    SORT_HEAP_EXTERNAL externally sorts a list of items into ascending order.
//
//  Discussion:
//
//    The actual list is not passed to the routine.  Hence it may
//    consist of integers, reals, numbers, names, etc.  The user,
//    after each return from the routine, will be asked to compare or
//    interchange two items.
//
//    The current version of this code mimics the FORTRAN version,
//    so the values of I and J, in particular, are FORTRAN indices.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    06 January 2013
//
//  Author:
//
//    Original FORTRAN77 version by Albert Nijenhuis, Herbert Wilf.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Albert Nijenhuis, Herbert Wilf,
//    Combinatorial Algorithms,
//    Academic Press, 1978, second edition,
//    ISBN 0-12-519260-6.
//
//  Parameters:
//
//    Input, int N, the length of the input list.
//
//    Input/output, int &INDX.
//    The user must set INDX to 0 before the first call.
//    On return,
//      if INDX is greater than 0, the user must interchange
//      items I and J and recall the routine.
//      If INDX is less than 0, the user is to compare items I
//      and J and return in ISGN a negative value if I is to
//      precede J, and a positive value otherwise.
//      If INDX is 0, the sorting is done.
//
//    Output, int &I, &J.  On return with INDX positive,
//    elements I and J of the user's list should be
//    interchanged.  On return with INDX negative, elements I
//    and J are to be compared by the user.
//
//    Input, int ISGN. On return with INDX negative, the
//    user should compare elements I and J of the list.  If
//    item I is to precede item J, set ISGN negative,
//    otherwise set ISGN positive.
//
{
  static int i_save = 0;
  static int j_save = 0;
  static int k = 0;
  static int k1 = 0;
  static int n1 = 0;
//
//  INDX = 0: This is the first call.
//
  if ( indx == 0 )
  {

    i_save = 0;
    j_save = 0;
    k = n / 2;
    k1 = k;
    n1 = n;
  }
//
//  INDX < 0: The user is returning the results of a comparison.
//
  else if ( indx < 0 )
  {
    if ( indx == -2 )
    {
      if ( isgn < 0 )
      {
        i_save = i_save + 1;
      }
      j_save = k1;
      k1 = i_save;
      indx = -1;
      i = i_save;
      j = j_save;
      return;
    }

    if ( 0 < isgn )
    {
      indx = 2;
      i = i_save;
      j = j_save;
      return;
    }

    if ( k <= 1 )
    {
      if ( n1 == 1 )
      {
        i_save = 0;
        j_save = 0;
        indx = 0;
      }
      else
      {
        i_save = n1;
        j_save = 1;
        n1 = n1 - 1;
        indx = 1;
      }
      i = i_save;
      j = j_save;
      return;
    }
    k = k - 1;
    k1 = k;
  }
//
//  0 < INDX: the user was asked to make an interchange.
//
  else if ( indx == 1 )
  {
    k1 = k;
  }

  for ( ; ; )
  {

    i_save = 2 * k1;

    if ( i_save == n1 )
    {
      j_save = k1;
      k1 = i_save;
      indx = -1;
      i = i_save;
      j = j_save;
      return;
    }
    else if ( i_save <= n1 )
    {
      j_save = i_save + 1;
      indx = -2;
      i = i_save;
      j = j_save;
      return;
    }

    if ( k <= 1 )
    {
      break;
    }

    k = k - 1;
    k1 = k;
  }

  if ( n1 == 1 )
  {
    i_save = 0;
    j_save = 0;
    indx = 0;
    i = i_save;
    j = j_save;
  }
  else
  {
    i_save = n1;
    j_save = 1;
    n1 = n1 - 1;
    indx = 1;
    i = i_save;
    j = j_save;
  }

  return;
}
//****************************************************************************80
int i4vec2_compare ( int n, int a1[], int a2[], int i, int j )

//****************************************************************************80
//
//  Purpose:
//
//    I4VEC2_COMPARE compares pairs of integers stored in two vectors.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of data items.
//
//    Input, int A1[N], A2[N], contain the two components of each item.
//
//    Input, int I, J, the items to be compared.  These values will be
//    1-based indices for the arrays A1 and A2.
//
//    Output, int I4VEC2_COMPARE, the results of the comparison:
//    -1, item I < item J,
//     0, item I = item J,
//    +1, item J < item I.
//
{
  int isgn;

  isgn = 0;

  if ( a1[i-1] < a1[j-1] )
  {
    isgn = -1;
  }
  else if ( a1[i-1] == a1[j-1] )
  {
    if ( a2[i-1] < a2[j-1] )
    {
      isgn = -1;
    }
    else if ( a2[i-1] < a2[j-1] )
    {
      isgn = 0;
    }
    else if ( a2[j-1] < a2[i-1] )
    {
      isgn = +1;
    }
  }
  else if ( a1[j-1] < a1[i-1] )
  {
    isgn = +1;
  }

  return isgn;
}
//****************************************************************************80
int i4vec2_sorted_unique_count ( int n, int a1[], int a2[] )

//****************************************************************************80
//
//  Purpose:
//
//    I4VEC2_SORTED_UNIQUE_COUNT counts unique elements in an I4VEC2.
//
//  Discussion:
//
//    Item I is stored as the pair A1(I), A2(I).
//
//    The items must have been sorted, or at least it must be the
//    case that equal items are stored in adjacent vector locations.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 July 2014
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of items.
//
//    Input, int A1[N], A2[N], the array of N items.
//
//    Output, int I4VEC2_SORTED_UNIQUE_COUNT, the number of unique items.
//
{
  int i;
  int iu;
  int unique_num;

  unique_num = 0;

  if ( n <= 0 )
  {
    return unique_num;
  }

  iu = 0;
  unique_num = 1;

  for ( i = 1; i < n; i++ )
  {
    if ( a1[i] != a1[iu] ||
         a2[i] != a2[iu] )
    {
      iu = i;
      unique_num = unique_num + 1;
    }
  }

  return unique_num;
}
//****************************************************************************80
void qbf (int q_point, int element, int inode, double xc[],double yc[],
  int element_node[], int element_num, int nnodes,
  int node_num, double &b, double &dbdx, double &dbdy )
//What i need to do is just read in quad number and figure out which thing i should be doing.
// I.e. skip all the crap with x and y and just do r and s on the unit triangle.
//****************************************************************************80
//
//  Purpose:
//
//    QBF evaluates the quadratic basis functions.
//
//  Discussion:
//
//    This routine assumes that the "midpoint" nodes are, in fact,
//    exactly the average of the two extreme nodes.  This is NOT true
//    for a general quadratic triangular element.
//
//    Assuming this property of the midpoint nodes makes it easy to
//    determine the values of (R,S) in the reference element that
//    correspond to (X,Y) in the physical element.
//
//    Once we know the (R,S) coordinates, it's easy to evaluate the
//    basis functions and derivatives.
//
//  The physical element T6:
//
//    In this picture, we don't mean to suggest that the bottom of
//    the physical triangle is horizontal.  However, we do assume that
//    each of the sides is a straight line, and that the intermediate
//    points are exactly halfway on each side.
//
//    |
//    |
//    |        3
//    |       / \
//    |      /   \
//    Y     6     5
//    |    /       \
//    |   /         \
//    |  1-----4-----2
//    |
//    +--------X-------->
//
//  Reference element T6:
//
//    In this picture of the reference element, we really do assume
//    that one side is vertical, one horizontal, of length 1.
//
//    |
//    |
//    1  3
//    |  |\
//    |  | \
//    S  6  5
//    |  |   \
//    |  |    \
//    0  1--4--2
//    |
//    +--0--R--1-------->
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    23 September 2008
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, Y, the (global) coordinates of the point
//    at which the basis function is to be evaluated.
//
//    Input, int ELEMENT, the index of the element which contains the point.
//
//    Input, int INODE, the local index (between 1 and 6) that
//    specifies which basis function is to be evaluated.
//
//    Input, double NODE_XY[2*NODE_NUM], the nodes.
//
//    Input, int ELEMENT_NODE[NNODES*ELEMENT_NUM];
//    ELEMENT_NODE(I,J) is the global index of local node I in element J.
//
//    Input, int ELEMENT_NUM, the number of elements.
//
//    Input, int NNODES, the number of nodes used to form one element.
//
//    Input, int NODE_NUM, the number of nodes.
//
//    Output, double *B, *DBDX, *DBDY, the value of the basis function
//    and its X and Y derivatives at (X,Y).
//
{
  double dbdr;
  double dbds;
  double det;
  double drdx;
  double drdy;
  double dsdx;
  double dsdy;
  int i;
  double r;
  double s;
  double xn[6];
  double yn[6];

  for ( i = 0; i < 6; i++ )
  {
    xn[i] = xc[(element_node[i+(element)*nnodes])];
    yn[i] = yc[(element_node[i+(element)*nnodes])];
  }
//
//  Determine the (R,S) coordinates corresponding to (X,Y).
//
//  What is happening here is that we are solving the linear system:
//
//    ( X2-X1  X3-X1 ) * ( R ) = ( X - X1 )
//    ( Y2-Y1  Y3-Y1 )   ( S )   ( Y - Y1 )
//
//  by computing the inverse of the coefficient matrix and multiplying
//  it by the right hand side to get R and S.
//
//  The values of dRdX, dRdY, dSdX and dSdY are easily from the formulas
//  for R and S.
//
//3 pooint
  if (q_point==0){
r= 0.50000000000000000000;
s=0.000000000000000000000;

				}
if (q_point==1){
r=0.500000000000000000000;
s= 0.50000000000000000000;
				}
if (q_point==2){
r= 0.00000000000000000000;				
s= 0.50000000000000000000;				
}

  
  
  /*
  // 6 point 3rd order 
if (q_point==0){
r=0.659027622374092;
s=0.231933368553031;

				}
if (q_point==1){
r=0.659027622374092;
s=  0.109039009072877;
				}
if (q_point==2){
r=0.231933368553031;
s=0.659027622374092;
				}
if (q_point==3){
r=0.231933368553031;
s=  0.109039009072877;
				}
if (q_point==4){
r=0.109039009072877;
s=  0.659027622374092;
				}
if (q_point==5){
r=0.109039009072877;
s=  0.231933368553031;
				}
*/

// 7 point 5th  order
/*if (q_point==0){
r=0.33333333333333333;
s=0.33333333333333333;

				}
if (q_point==1){
r= 0.79742698535308720;
s=  0.10128650732345633;
				}
if (q_point==2){
r=0.10128650732345633;
s=0.79742698535308720;
				}
if (q_point==3){

r=0.10128650732345633;
s=  0.10128650732345633;
				}
if (q_point==4){
 
r=0.05971587178976981;
s=  0.47014206410511505;
				}
if (q_point==5){
 
r=0.47014206410511505;
s=0.05971587178976981;
	}
if (q_point==6){

r=0.47014206410511505;
s=0.47014206410511505;

				}
*/


  det =   ( xn[1] - xn[0] ) * ( yn[2] - yn[0] )
        - ( xn[2] - xn[0] ) * ( yn[1] - yn[0] );

//  r = ( ( yn[2] - yn[0] ) * ( x     - xn[0] )
//      + ( xn[0] - xn[2] ) * ( y     - yn[0] ) ) / det;

  drdx = ( yn[2] - yn[0] ) / det;
  drdy = ( xn[0] - xn[2] ) / det;

  //s = ( ( yn[0] - yn[1] ) * ( x     - xn[0] )
      //+ ( xn[1] - xn[0] ) * ( y     - yn[0] ) ) / det;

  dsdx = ( yn[0] - yn[1] ) / det;
  dsdy = ( xn[1] - xn[0] ) / det;
//
//  The basis functions can now be evaluated in terms of the
//  reference coordinates R and S.  It's also easy to determine
//  the values of the derivatives with respect to R and S.
//
  if ( inode == 0 )
  {
    b   =   2.0E+00 *     ( 1.0E+00 - r - s ) * ( 0.5E+00 - r - s );
    dbdr = - 3.0E+00 + 4.0E+00 * r + 4.0E+00 * s;
    dbds = - 3.0E+00 + 4.0E+00 * r + 4.0E+00 * s;
  }
  else if ( inode == 1 )
  {
    b   =   2.0E+00 * r * ( r - 0.5E+00 );
    dbdr = - 1.0E+00 + 4.0E+00 * r;
    dbds =   0.0E+00;
  }
  else if ( inode == 2 )
  {
    b   =   2.0E+00 * s * ( s - 0.5E+00 );
    dbdr =   0.0E+00;
    dbds = - 1.0E+00               + 4.0E+00 * s;
  }
  else if ( inode == 3 )
  {
    b   =   4.0E+00 * r * ( 1.0E+00 - r - s );
    dbdr =   4.0E+00 - 8.0E+00 * r - 4.0E+00 * s;
    dbds =           - 4.0E+00 * r;
  }
  else if ( inode == 4 )
  {
    b   =   4.0E+00 * r * s;
    dbdr =                           4.0E+00 * s;
    dbds =             4.0E+00 * r;
  }
  else if ( inode == 5 )
  {
    b   =   4.0E+00 * s * ( 1.0E+00 - r - s );
    dbdr =                         - 4.0E+00 * s;
    dbds =   4.0E+00 - 4.0E+00 * r - 8.0E+00 * s;
  }
  else
  {
    cout << "\n";
    cout << "QBF - Fatal error!\n";
    cout << "  Request for local basis function INODE = " << inode << "\n";
    exit ( 1 );
  }
//
//  We need to convert the derivative information from (R(X,Y),S(X,Y))
//  to (X,Y) using the chain rule.
//
  dbdx = dbdr * drdx + dbds * dsdx;
  dbdy = dbdr * drdy + dbds * dsdy;

  return;
}
//****************************************************************************80