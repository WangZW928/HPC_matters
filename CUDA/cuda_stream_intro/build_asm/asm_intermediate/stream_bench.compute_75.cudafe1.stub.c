#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _e720258b_15_stream_bench_cu_36090dad
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "stream_bench.fatbin.c"
extern void __device_stub__Z10vector_addPKfS0_Pfii(const float *, const float *, float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z10vector_addPKfS0_Pfii(const float *__par0, const float *__par1, float *__par2, int __par3, int __par4){__cudaLaunchPrologue(5);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 28UL);__cudaLaunch(((char *)((void ( *)(const float *, const float *, float *, int, int))vector_add)));}
# 22 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_stream_intro/src/stream_bench.cu"
void vector_add( const float *__cuda_0,const float *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4)
# 22 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_stream_intro/src/stream_bench.cu"
{__device_stub__Z10vector_addPKfS0_Pfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 34 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_stream_intro/src/stream_bench.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_stream_intro/build_asm/asm_intermediate/stream_bench.compute_75.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T18) {  __nv_dummy_param_ref(__T18); __nv_save_fatbinhandle_for_managed_rt(__T18); __cudaRegisterEntry(__T18, ((void ( *)(const float *, const float *, float *, int, int))vector_add), _Z10vector_addPKfS0_Pfii, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
