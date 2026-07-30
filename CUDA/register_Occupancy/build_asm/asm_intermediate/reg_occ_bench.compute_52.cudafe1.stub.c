#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _187bab00_16_reg_occ_bench_cu_0a8bb7b3
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "reg_occ_bench.fatbin.c"
extern void __device_stub__Z14kernel_low_regPKfPfii(const float *, float *, int, int);
extern void __device_stub__Z15kernel_high_regPKfPfii(const float *, float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z14kernel_low_regPKfPfii(const float *__par0, float *__par1, int __par2, int __par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 20UL);__cudaLaunch(((char *)((void ( *)(const float *, float *, int, int))kernel_low_reg)));}
# 27 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/register_Occupancy/src/reg_occ_bench.cu"
void kernel_low_reg( const float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 27 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/register_Occupancy/src/reg_occ_bench.cu"
{__device_stub__Z14kernel_low_regPKfPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 37 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/register_Occupancy/src/reg_occ_bench.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/register_Occupancy/build_asm/asm_intermediate/reg_occ_bench.compute_52.cudafe1.stub.c"
void __device_stub__Z15kernel_high_regPKfPfii( const float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(const float *, float *, int, int))kernel_high_reg))); }
# 39 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/register_Occupancy/src/reg_occ_bench.cu"
void kernel_high_reg( const float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 39 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/register_Occupancy/src/reg_occ_bench.cu"
{__device_stub__Z15kernel_high_regPKfPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 64 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/register_Occupancy/src/reg_occ_bench.cu"
}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/register_Occupancy/build_asm/asm_intermediate/reg_occ_bench.compute_52.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T23) {  __nv_dummy_param_ref(__T23); __nv_save_fatbinhandle_for_managed_rt(__T23); __cudaRegisterEntry(__T23, ((void ( *)(const float *, float *, int, int))kernel_high_reg), _Z15kernel_high_regPKfPfii, (-1)); __cudaRegisterEntry(__T23, ((void ( *)(const float *, float *, int, int))kernel_low_reg), _Z14kernel_low_regPKfPfii, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
