#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_MODULE_ID _6e240b22_14_graph_bench_cu_f23da181
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "graph_bench.fatbin.c"
extern void __device_stub__Z8add_biasPffi(float *, float, int);
extern void __device_stub__Z5scalePffi(float *, float, int);
extern void __device_stub__Z4reluPfi(float *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z8add_biasPffi(float *__par0, float __par1, int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 12UL);__cudaLaunch(((char *)((void ( *)(float *, float, int))add_bias)));}
# 21 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/src/graph_bench.cu"
void add_bias( float *__cuda_0,float __cuda_1,int __cuda_2)
# 21 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/src/graph_bench.cu"
{__device_stub__Z8add_biasPffi( __cuda_0,__cuda_1,__cuda_2);


}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/build_asm/asm_intermediate/graph_bench.compute_89.cudafe1.stub.c"
void __device_stub__Z5scalePffi( float *__par0,  float __par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 12UL); __cudaLaunch(((char *)((void ( *)(float *, float, int))scale))); }
# 26 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/src/graph_bench.cu"
void scale( float *__cuda_0,float __cuda_1,int __cuda_2)
# 26 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/src/graph_bench.cu"
{__device_stub__Z5scalePffi( __cuda_0,__cuda_1,__cuda_2);


}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/build_asm/asm_intermediate/graph_bench.compute_89.cudafe1.stub.c"
void __device_stub__Z4reluPfi( float *__par0,  int __par1) {  __cudaLaunchPrologue(2); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaLaunch(((char *)((void ( *)(float *, int))relu))); }
# 31 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/src/graph_bench.cu"
void relu( float *__cuda_0,int __cuda_1)
# 31 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/src/graph_bench.cu"
{__device_stub__Z4reluPfi( __cuda_0,__cuda_1);


}
# 1 "/home/wangzw/agent-workspace/Projects/HPC_matters/CUDA/cuda_graph_intro/build_asm/asm_intermediate/graph_bench.compute_89.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T10) {  __nv_dummy_param_ref(__T10); __nv_save_fatbinhandle_for_managed_rt(__T10); __cudaRegisterEntry(__T10, ((void ( *)(float *, int))relu), _Z4reluPfi, (-1)); __cudaRegisterEntry(__T10, ((void ( *)(float *, float, int))scale), _Z5scalePffi, (-1)); __cudaRegisterEntry(__T10, ((void ( *)(float *, float, int))add_bias), _Z8add_biasPffi, (-1)); }
static void __sti____cudaRegisterAll(void) {  ____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
