#ifndef BASE_TESTS_2DHEAT_ARRAY2DVIEW_H_
#define BASE_TESTS_2DHEAT_ARRAY2DVIEW_H_

#include "Backend.H"

#include <cstddef>

template <typename T>
struct Array2DView
{
    T* data = nullptr;
    int xlo = 0;
    int xhi = -1;
    int ylo = 0;
    int yhi = -1;
    int pitch = 0;

    BASE_GPU_HOST_DEVICE
    T& operator() (int i, int j) const noexcept
    {
        return data[(i - xlo) + (j - ylo) * pitch];
    }

    BASE_GPU_HOST_DEVICE
    std::size_t size () const noexcept
    {
        return static_cast<std::size_t>(xhi - xlo + 1) * static_cast<std::size_t>(yhi - ylo + 1);
    }
};

#endif
