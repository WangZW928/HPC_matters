#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _2ba4f9e3_15_kernel_types_cu_542d12f9
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kernel_types.fatbin.c"
extern void __device_stub__Z20compute_bound_kernelPKfPfii(const float *, float *, int, int);
extern void __device_stub__Z19memory_bound_kernelPKfS0_Pfi(const float *, const float *, float *, int);
extern void __device_stub__Z20latency_bound_kernelPKiPfii(const int *, float *, int, int);
extern void __device_stub__Z22launch_overhead_kernelPf(float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z20compute_bound_kernelPKfPfii(const float *__par0, float *__par1, int __par2, int __par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 20UL);__cudaLaunch(((char *)((void ( *)(const float *, float *, int, int))compute_bound_kernel)));}
# 23 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
void compute_bound_kernel( const float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 23 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
{__device_stub__Z20compute_bound_kernelPKfPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 34 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/build_asm/asm_intermediate/kernel_types.compute_75.cudafe1.stub.c"
void __device_stub__Z19memory_bound_kernelPKfS0_Pfi( const float *__par0,  const float *__par1,  float *__par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(const float *, const float *, float *, int))memory_bound_kernel))); }
# 36 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
void memory_bound_kernel( const float *__cuda_0,const float *__cuda_1,float *__cuda_2,int __cuda_3)
# 36 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
{__device_stub__Z19memory_bound_kernelPKfS0_Pfi( __cuda_0,__cuda_1,__cuda_2,__cuda_3);


}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/build_asm/asm_intermediate/kernel_types.compute_75.cudafe1.stub.c"
void __device_stub__Z20latency_bound_kernelPKiPfii( const int *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(const int *, float *, int, int))latency_bound_kernel))); }
# 41 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
void latency_bound_kernel( const int *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 41 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
{__device_stub__Z20latency_bound_kernelPKiPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 53 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/build_asm/asm_intermediate/kernel_types.compute_75.cudafe1.stub.c"
void __device_stub__Z22launch_overhead_kernelPf( float *__par0) {  __cudaLaunchPrologue(1); __cudaSetupArgSimple(__par0, 0UL); __cudaLaunch(((char *)((void ( *)(float *))launch_overhead_kernel))); }
# 55 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
void launch_overhead_kernel( float *__cuda_0)
# 55 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/src/kernel_types.cu"
{__device_stub__Z22launch_overhead_kernelPf( __cuda_0);

}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/kernel_type_playground/build_asm/asm_intermediate/kernel_types.compute_75.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T39) {  __nv_dummy_param_ref(__T39); __nv_save_fatbinhandle_for_managed_rt(__T39); __cudaRegisterEntry(__T39, ((void ( *)(float *))launch_overhead_kernel), _Z22launch_overhead_kernelPf, (-1)); __cudaRegisterEntry(__T39, ((void ( *)(const int *, float *, int, int))latency_bound_kernel), _Z20latency_bound_kernelPKiPfii, (-1)); __cudaRegisterEntry(__T39, ((void ( *)(const float *, const float *, float *, int))memory_bound_kernel), _Z19memory_bound_kernelPKfS0_Pfi, (-1)); __cudaRegisterEntry(__T39, ((void ( *)(const float *, float *, int, int))compute_bound_kernel), _Z20compute_bound_kernelPKfPfii, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
