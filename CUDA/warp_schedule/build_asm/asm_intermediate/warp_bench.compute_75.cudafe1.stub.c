#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _1eef773f_13_warp_bench_cu_51737f3e
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "warp_bench.fatbin.c"
extern void __device_stub__Z18warp_stress_kernelPfii(float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z18warp_stress_kernelPfii(float *__par0, int __par1, int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 12UL);__cudaLaunch(((char *)((void ( *)(float *, int, int))warp_stress_kernel)));}
# 23 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/warp_schedule/src/warp_bench.cu"
void warp_stress_kernel( float *__cuda_0,int __cuda_1,int __cuda_2)
# 23 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/warp_schedule/src/warp_bench.cu"
{__device_stub__Z18warp_stress_kernelPfii( __cuda_0,__cuda_1,__cuda_2);
# 33 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/warp_schedule/src/warp_bench.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/warp_schedule/build_asm/asm_intermediate/warp_bench.compute_75.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T12) {  __nv_dummy_param_ref(__T12); __nv_save_fatbinhandle_for_managed_rt(__T12); __cudaRegisterEntry(__T12, ((void ( *)(float *, int, int))warp_stress_kernel), _Z18warp_stress_kernelPfii, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
