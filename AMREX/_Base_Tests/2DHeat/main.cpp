#include "Backend.H"
#include "Buffer.H"
#include "ParallelFor2D.H"
#include "Print2D.H"

#include <cmath>
#include <iostream>
#include <utility>

#ifndef BASE_TESTS_PI
#define BASE_TESTS_PI 3.14159265358979323846
#endif

#ifndef BASE_TESTS_TWO_PI
#define BASE_TESTS_TWO_PI (2.0 * BASE_TESTS_PI)
#endif

int main ()
{
    int n_cell = 128;
    int n_iter = 2000; 
    int nghost = 1;
    int print_every = 100;

    double dx = 1.0 / static_cast<double>(n_cell);
    double dy = 1.0 / static_cast<double>(n_cell);
    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);
    double denom = 2.0 * (idx2 + idy2);

    int total_nx = n_cell + 2 * nghost;
    int total_ny = n_cell + 2 * nghost;
    int total_ncell = total_nx * total_ny;

    Buffer<double> u_old(total_ncell, defaultMemorySpace());
    Buffer<double> u_new(total_ncell, defaultMemorySpace());
    Buffer<double> rhs(total_ncell, defaultMemorySpace());
    u_old.fillZero();
    u_new.fillZero();
    rhs.fillZero();

    auto old_view = u_old.view2D(n_cell, n_cell, nghost);
    auto new_view = u_new.view2D(n_cell, n_cell, nghost);
    auto rhs_view = rhs.view2D(n_cell, n_cell, nghost);

    // Build a zero-mean source term for Neumann Poisson.
    parallelFor2D(n_cell, n_cell, [=] BASE_GPU_DEVICE (int i, int j) noexcept {
        double x = (static_cast<double>(i) + 0.5) * dx;
        double y = (static_cast<double>(j) + 0.5) * dy;
        rhs_view(i, j) = cos(BASE_TESTS_TWO_PI * x) * cos(BASE_TESTS_TWO_PI * y);
    });
    backendSynchronize();

    for (int iter = 1; iter <= n_iter; ++iter) {
        // Homogeneous Neumann BC via ghost fill.
        parallelFor2D(1, n_cell, [=] BASE_GPU_DEVICE (int, int j) noexcept {
            old_view(-1, j) = old_view(0, j);
            old_view(n_cell, j) = old_view(n_cell - 1, j);
        });
        parallelFor2D(n_cell, 1, [=] BASE_GPU_DEVICE (int i, int) noexcept {
            old_view(i, -1) = old_view(i, 0);
            old_view(i, n_cell) = old_view(i, n_cell - 1);
        });
        backendSynchronize();

        parallelFor2D(n_cell, n_cell, [=] BASE_GPU_DEVICE (int i, int j) noexcept {
            if (i == 0 && j == 0) {
                // Pin one DoF for Neumann problem null space.
                new_view(i, j) = 0.0;
                return;
            }

            double ue = old_view(i + 1, j);
            double uw = old_view(i - 1, j);
            double un = old_view(i, j + 1);
            double us = old_view(i, j - 1);

            // Jacobi update for -Laplace(u) = rhs.
            new_view(i, j) = ((ue + uw) * idx2 + (un + us) * idy2 + rhs_view(i, j)) / denom;
        });  
        backendSynchronize();

        // Jacobi double-buffer rotation:
        // u_new (just computed) becomes next iteration's u_old.
        // This depends on Buffer move ctor/assignment (Buffer is non-copyable).
        //std::swap(u_old, u_new);
        std::swap(old_view, new_view);

        if (iter % print_every == 0 || iter == 1 || iter == n_iter) {
            std::cout << backendLabel() << " Jacobi iter " << iter << " / " << n_iter << "\n";
        }
    }

    auto host = u_old.copyToHostVector();
    writeInteriorCSV(host, n_cell, n_cell, nghost, "poisson_final.csv");
    writeInteriorPPM(host, n_cell, n_cell, nghost, "poisson_final.ppm");

    std::cout << "Wrote poisson_final.csv\n";
    std::cout << "Wrote poisson_final.ppm\n";

    return 0;
}
