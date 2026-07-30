#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _3dd96517_22_bank_conflict_bench_cu_6d4a1acb
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "bank_conflict_bench.fatbin.c"
extern void __device_stub__Z20shared_stride_kernelPfii(float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z20shared_stride_kernelPfii(float *__par0, int __par1, int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 12UL);__cudaLaunch(((char *)((void ( *)(float *, int, int))shared_stride_kernel)));}
# 27 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/shared_memory_bank_conflict/src/bank_conflict_bench.cu"
void shared_stride_kernel( float *__cuda_0,int __cuda_1,int __cuda_2)
# 27 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/shared_memory_bank_conflict/src/bank_conflict_bench.cu"
{__device_stub__Z20shared_stride_kernelPfii( __cuda_0,__cuda_1,__cuda_2);
# 49 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/shared_memory_bank_conflict/src/bank_conflict_bench.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/shared_memory_bank_conflict/build_asm/asm_intermediate/bank_conflict_bench.compute_89.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T12) {  __nv_dummy_param_ref(__T12); __nv_save_fatbinhandle_for_managed_rt(__T12); __cudaRegisterEntry(__T12, ((void ( *)(float *, int, int))shared_stride_kernel), _Z20shared_stride_kernelPfii, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
