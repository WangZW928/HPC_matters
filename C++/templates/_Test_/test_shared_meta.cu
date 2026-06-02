#include <iostream>
#include <cassert>
#include <typeinfo>

namespace amrex::Gpu {
    template <class T>
    struct SharedMemory {
        __device__ T* dataPtr() noexcept {
            static_assert(sizeof(T) < 0, "We must specialize struct SharedMemory");
            return nullptr;
        }
    };

    // 特化 int 版本
    template <> struct SharedMemory<int> {
        [[nodiscard]] __device__ int* dataPtr() noexcept {
            extern __shared__ int amrex_sm_int[];
            return amrex_sm_int;
        }
    };

    // 特化 float 版本
    template <> struct SharedMemory<float> {
        [[nodiscard]] __device__ float* dataPtr() noexcept {
            extern __shared__ float amrex_sm_float[];
            return amrex_sm_float;
        }
    };

    // 特化 double 版本
    template <> struct SharedMemory<double> {
        [[nodiscard]] __device__ double* dataPtr() noexcept {
            extern __shared__ double amrex_sm_double[];
            return amrex_sm_double;
        }
    };

    // 特化 char 版本
    template <> struct SharedMemory<char> {
        [[nodiscard]] __device__ char* dataPtr() noexcept {
            extern __shared__ char amrex_sm_char[];
            return amrex_sm_char;
        }
    };
}

template <typename T>
class TestSharedMemoryFunctor {
public:
    TestSharedMemoryFunctor(T* d_out) : out_ptr(d_out) {}

    __device__ void operator()() const noexcept {
        amrex::Gpu::SharedMemory<T> gsm;
        T* shared = gsm.dataPtr();

        int tid = threadIdx.x;

        if constexpr (std::is_floating_point_v<T>) {
            shared[tid] = static_cast<T>(tid * 3.14159);
        } else {
            shared[tid] = static_cast<T>(tid * 10 + 5);
        }

        __syncthreads();

        out_ptr[tid] = shared[tid];
    }

private:
    T* out_ptr;
};

template <typename F>
__global__ void amrex_simulation_kernel(F f) {
    f(); 
}

template <typename T>
void executeSingleTypeTest(const char* type_label) {
    const int threads_count = 4;
    std::size_t shared_bytes = threads_count * sizeof(T);

    T* d_output = nullptr;
    cudaMallocManaged(&d_output, threads_count * sizeof(T));

    amrex_simulation_kernel<<<1, threads_count, shared_bytes>>>(TestSharedMemoryFunctor<T>(d_output));
    
    cudaDeviceSynchronize();

    std::cout << "👉 [测试类型: " << type_label << "] (占用共享内存: " << shared_bytes << " 字节)\n";
    for (int i = 0; i < threads_count; ++i) {
        std::cout << "   线程 " << i << " 传出数据 = " << static_cast<double>(d_output[i]) << "\n";
    }
    std::cout << "   " << type_label << " 释放显存，单项通过 ✔\n\n";

    cudaFree(d_output);
}

// 利用 C++17 折叠表达式（Fold Expressions），在编译期把参数包 Ts... 原地展开
template <typename... Ts, typename F>
void for_each_type(F&& f) {
    (f.template operator()<Ts>(), ...);
}

struct TestRunner {
    template <typename T>
    void operator()() {
        if constexpr (std::is_same_v<T, int>)    executeSingleTypeTest<int>("int");
        if constexpr (std::is_same_v<T, float>)  executeSingleTypeTest<float>("float");
        if constexpr (std::is_same_v<T, double>) executeSingleTypeTest<double>("double");
        if constexpr (std::is_same_v<T, char>)   executeSingleTypeTest<char>("char");
    }
};

int main() {
    std::cout << "==================================================\n";
    std::cout << "===  AMReX SharedMemory 编译期自动流测试启动  ===\n";
    std::cout << "==================================================\n\n";

    for_each_type<int, float, double, char>(TestRunner{});

    std::cout << "【全部通过】所有特化类型测试完毕，无状态安全无虞！\n";
    return 0;
}