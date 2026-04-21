#ifndef BASE_TESTS_2DHEAT_PRINT2D_H_
#define BASE_TESTS_2DHEAT_PRINT2D_H_

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

inline void interiorMinMax (std::vector<double> const& host, int nx, int ny, int ng,
                            double& vmin, double& vmax)
{
    int pitch = nx + 2 * ng;
    vmin = host[static_cast<std::size_t>(ng + ng * pitch)];
    vmax = vmin;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int offset = (i + ng) + (j + ng) * pitch;
            double v = host[static_cast<std::size_t>(offset)];
            if (v < vmin) { vmin = v; }
            if (v > vmax) { vmax = v; }
        }
    }
}

inline void writeInteriorCSV (std::vector<double> const& host, int nx, int ny, int ng,
                              std::string const& filename)
{
    int pitch = nx + 2 * ng;
    std::ofstream os(filename);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int offset = (i + ng) + (j + ng) * pitch;
            os << host[static_cast<std::size_t>(offset)];
            if (i + 1 < nx) {
                os << ',';
            }
        }
        os << '\n';
    }
}

inline void writeInteriorPPM (std::vector<double> const& host, int nx, int ny, int ng,
                              std::string const& filename)
{
    double vmin = 0.0;
    double vmax = 0.0;
    interiorMinMax(host, nx, ny, ng, vmin, vmax);

    int pitch = nx + 2 * ng;
    std::ofstream os(filename);
    os << "P3\n";
    os << nx << ' ' << ny << "\n255\n";

    double scale = (vmax > vmin) ? 1.0 / (vmax - vmin) : 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int offset = (i + ng) + (j + ng) * pitch;
            double v = host[static_cast<std::size_t>(offset)];
            double t = (vmax > vmin) ? (v - vmin) * scale : 0.0;

            int r = 0;
            int g = 0;
            int b = 0;
            if (t < 0.5) {
                double s = t / 0.5;
                g = static_cast<int>(255.0 * s);
                b = static_cast<int>(255.0 * (1.0 - s));
             } else {
                double s = (t - 0.5) / 0.5;
                r = static_cast<int>(255.0 * s);
                g = static_cast<int>(255.0 * (1.0 - s));
            }
            os << r << ' ' << g << ' ' << b << ' ';
        }
        os << '\n';
    }
}

#endif
