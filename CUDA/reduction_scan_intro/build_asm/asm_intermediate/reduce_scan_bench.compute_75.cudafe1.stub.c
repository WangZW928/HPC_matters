#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _4e18b9ce_20_reduce_scan_bench_cu_0b89627b
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "reduce_scan_bench.fatbin.c"
extern void __device_stub__Z26block_reduce_shared_kernelPKfPfi(const float *, float *, int);
extern void __device_stub__Z27block_reduce_shuffle_kernelPKfPfi(const float *, float *, int);
extern void __device_stub__Z30exclusive_scan_blelloch_kernelPKfPfi(const float *, float *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z26block_reduce_shared_kernelPKfPfi(const float *__par0, float *__par1, int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(const float *, float *, int))block_reduce_shared_kernel)));}
# 23 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
void block_reduce_shared_kernel( const float *__cuda_0,float *__cuda_1,int __cuda_2)
# 23 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
{__device_stub__Z26block_reduce_shared_kernelPKfPfi( __cuda_0,__cuda_1,__cuda_2);
# 39 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/build_asm/asm_intermediate/reduce_scan_bench.compute_75.cudafe1.stub.c"
void __device_stub__Z27block_reduce_shuffle_kernelPKfPfi( const float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *, float *, int))block_reduce_shuffle_kernel))); }
# 49 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
void block_reduce_shuffle_kernel( const float *__cuda_0,float *__cuda_1,int __cuda_2)
# 49 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
{__device_stub__Z27block_reduce_shuffle_kernelPKfPfi( __cuda_0,__cuda_1,__cuda_2);
# 69 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/build_asm/asm_intermediate/reduce_scan_bench.compute_75.cudafe1.stub.c"
void __device_stub__Z30exclusive_scan_blelloch_kernelPKfPfi( const float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *, float *, int))exclusive_scan_blelloch_kernel))); }
# 71 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
void exclusive_scan_blelloch_kernel( const float *__cuda_0,float *__cuda_1,int __cuda_2)
# 71 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
{__device_stub__Z30exclusive_scan_blelloch_kernelPKfPfi( __cuda_0,__cuda_1,__cuda_2);
# 102 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/src/reduce_scan_bench.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/reduction_scan_intro/build_asm/asm_intermediate/reduce_scan_bench.compute_75.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T17) {  __nv_dummy_param_ref(__T17); __nv_save_fatbinhandle_for_managed_rt(__T17); __cudaRegisterEntry(__T17, ((void ( *)(const float *, float *, int))exclusive_scan_blelloch_kernel), _Z30exclusive_scan_blelloch_kernelPKfPfi, (-1)); __cudaRegisterEntry(__T17, ((void ( *)(const float *, float *, int))block_reduce_shuffle_kernel), _Z27block_reduce_shuffle_kernelPKfPfi, (-1)); __cudaRegisterEntry(__T17, ((void ( *)(const float *, float *, int))block_reduce_shared_kernel), _Z26block_reduce_shared_kernelPKfPfi, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
