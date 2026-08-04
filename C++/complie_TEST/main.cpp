#include "vector_add.h"

#include <cstdio>
#include <memory>

int main()
{
    constexpr std::size_t N = 64;

    auto a_ptr = std::make_unique<int[]>(N);
    auto b_ptr = std::make_unique<int[]>(N);

    vectorAdd(a_ptr.get(), b_ptr.get(), N);

    long long checksum = 0;

    for (std::size_t i = 0; i < N; ++i) {
        checksum += b_ptr[i];
    }

    std::printf("b[0] = %d\n", b_ptr[0]);
    std::printf("b[63] = %d\n", b_ptr[N - 1]);
    std::printf("checksum = %lld\n", checksum);

    return checksum == 2080 ? 0 : 1;
}