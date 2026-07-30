#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _64891681_23_mem_coalescing_bench_cu_778ef60a
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "mem_coalescing_bench.fatbin.c"
extern void __device_stub__Z18stride_read_kernelPKfPfii(const float *, float *, int, int);
extern void __device_stub__Z18offset_read_kernelPKfPfii(const float *, float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z18stride_read_kernelPKfPfii(const float *__par0, float *__par1, int __par2, int __par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 20UL);__cudaLaunch(((char *)((void ( *)(const float *, float *, int, int))stride_read_kernel)));}
# 23 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/memory_coalescing_intro/src/mem_coalescing_bench.cu"
void stride_read_kernel( const float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 23 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/memory_coalescing_intro/src/mem_coalescing_bench.cu"
{__device_stub__Z18stride_read_kernelPKfPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/memory_coalescing_intro/build_asm/asm_intermediate/mem_coalescing_bench.compute_75.cudafe1.stub.c"
void __device_stub__Z18offset_read_kernelPKfPfii( const float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(const float *, float *, int, int))offset_read_kernel))); }
# 30 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/memory_coalescing_intro/src/mem_coalescing_bench.cu"
void offset_read_kernel( const float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 30 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/memory_coalescing_intro/src/mem_coalescing_bench.cu"
{__device_stub__Z18offset_read_kernelPKfPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/memory_coalescing_intro/build_asm/asm_intermediate/mem_coalescing_bench.compute_75.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T10) {  __nv_dummy_param_ref(__T10); __nv_save_fatbinhandle_for_managed_rt(__T10); __cudaRegisterEntry(__T10, ((void ( *)(const float *, float *, int, int))offset_read_kernel), _Z18offset_read_kernelPKfPfii, (-1)); __cudaRegisterEntry(__T10, ((void ( *)(const float *, float *, int, int))stride_read_kernel), _Z18stride_read_kernelPKfPfii, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
