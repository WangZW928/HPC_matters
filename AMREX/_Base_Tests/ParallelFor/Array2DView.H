#ifndef BASE_TESTS_ARRAY2DVIEW_H_
#define BASE_TESTS_ARRAY2DVIEW_H_

#include "Backend.H"

#include <cstddef>

template <typename T>
struct Array2DView
{
    T* data = nullptr;
    int nx = 0;
    int ny = 0;

    BASE_GPU_HOST_DEVICE
    T& operator() (int i, int j) const noexcept
    {
        return data[i + j * nx];
    }

    BASE_GPU_HOST_DEVICE
    std::size_t size () const noexcept
    {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    }
};

#endif
