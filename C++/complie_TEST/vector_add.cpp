#include "vector_add.h"

void vectorAdd(int* __restrict__ a,
               int* __restrict__ b,
               std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1;
        b[i] = static_cast<int>(i);
    }

    for (std::size_t i = 0; i < n; ++i) {
        b[i] += a[i];
    }
}