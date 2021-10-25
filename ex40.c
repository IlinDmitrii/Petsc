
static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -n <mesh_y>       : number of mesh points in y-direction\n\n\
  -o : Specify output filename for solution (will be petsc binary format or paraview format if the extension is .vts) \n\n" ;

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;  /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;     /* linear solver context */
  PetscRandom    rctx;     /* random number generator context */
  PetscReal      norm;     /* norm of solution error */
  PetscLogDouble t1,t2,dt;
  PetscInt       i,n = 100,its,rstart,rend,col[5]; 
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscScalar    value[5];
  PetscMPIInt    rank;
  //char           filename[PETSC_MAX_PATH_LEN];
  //PetscBool      set;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  //ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  PetscTime(&t1);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

   ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
   


if (!rstart) {
  rstart = 1;
  i      = 0; col[0] = 0; col[1] = 1;col[2]=2; value[0] = 100.0; value[1] = 10.0;value[2]=1.0;
  ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr); 
}
if (rstart==1) {
  rstart = 2;
  i      = 1; col[0] = 0; col[1] = 1;col[2]=2;col[3]=3; value[0] = 10.0; value[1] = 100.0;value[2]=10.0;value[3]=1.0;
  ierr   = MatSetValues(A,1,&i,4,col,value,INSERT_VALUES);CHKERRQ(ierr); 
}
if (rend == n) {
  rend = n-1;
  i    = n-1; col[0] = n-3; col[1] = n-2;col[2]=n-1; value[0] = 1.0; value[1] = 10.0;value[2]=100.0;
  ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
}
if (rend == n-1) {
  rend = n-2;
  i    = n-2; col[0] = n-4; col[1] = n-3;col[2]=n-2;col[3]=n-1; value[0] = 1.0; value[1] = 10.0;value[2]=100.0;value[3]=10.0;
  ierr = MatSetValues(A,1,&i,4,col,value,INSERT_VALUES);CHKERRQ(ierr);
}

value[0] = 1.0; value[1] = 10.0; value[2] = 100.0;value[3] = 10.0;value[4] = 1.0;
for (i=rstart; i<rend; i++) {
  col[0] = i-2; col[1] = i-1; col[2] = i;col[3] = i+1;col[4] = i+2;
  ierr   = MatSetValues(A,1,&i,5,col,value,INSERT_VALUES);CHKERRQ(ierr);
}




  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr=MatView(A,	PETSC_VIEWER_STDOUT_WORLD);
  //ierr=MatView(A,	PETSC_VIEWER_STDOUT_WORLD);
  /* ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr); */

 /* PetscOptionsGetString(NULL,NULL,"-o",filename,sizeof(filename),&set);
  if (set) {
    char        *ext;
    PetscViewer viewer;
     PetscBool   flg;
     PetscViewerCreate(PETSC_COMM_WORLD,&viewer);
     PetscStrrchr(filename,'.',&ext);
    PetscStrcmp("vts",ext,&flg);
    if (flg) {
       PetscViewerSetType(viewer,PETSCVIEWERVTK);
     } else {
       PetscViewerSetType(viewer,PETSCVIEWERBINARY);
    }
     PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);
    PetscViewerFileSetName(viewer,filename);
     MatView(A,viewer);
    PetscViewerDestroy(&viewer);
   }*/



  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,n);CHKERRQ(ierr);//m
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);


  ierr = PetscOptionsGetBool(NULL,NULL,"-random_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(u,rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(u,1.0);CHKERRQ(ierr);
  }
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

  
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);


  ierr = KSPSetTolerances(ksp,(0.0),(0.0),PETSC_DEFAULT,
                          PETSC_DEFAULT);CHKERRQ(ierr);//1.e-2/((n+1)

/*ierr = KSPSetTolerances(ksp,(0.0),PETSC_DEFAULT,PETSC_DEFAULT,
  PETSC_DEFAULT);CHKERRQ(ierr);//1.e-2/((n+1)*/
//*(n+1)
 
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  //ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);


  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its);CHKERRQ(ierr);

 
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);  ierr = MatDestroy(&A);CHKERRQ(ierr);

   PetscTime(&t2);CHKERRQ(ierr);
  dt=t2-t1;
  printf("all time %f\n",dt);
  ierr = PetscFinalize();

  return ierr;
}

