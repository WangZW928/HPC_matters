#ifndef BASE_TESTS_PRINT2D_H_
#define BASE_TESTS_PRINT2D_H_

#include <cstddef>
#include <iostream>
#include <vector>

inline void print2D (std::vector<double> const& host, int nx, int ny, char const* label)
{
    std::cout << label << '\n';
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            std::cout << host[static_cast<std::size_t>(i + j * nx)] << ' ';
        }
        std::cout << '\n';
    }
}
    
#endif
