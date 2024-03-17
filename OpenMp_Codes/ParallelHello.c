//OpenMP program to print Hello World
//using C language


#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv)
{
    //Beginning of parallel region
    #pragma omp parallel
    {
        printf("YoYoHello, World!.... from thread = %d\n",omp_get_thread_num());
    }
    return 0;
    //Ending of parallel region
}
