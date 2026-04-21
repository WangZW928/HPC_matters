#include "Backend.H"
#include "Buffer.H"
#include "ParallelFor2D.H"
#include "Print2D.H"

int main ()
{
    int nx = 8;
    int ny = 4;
    int ncell = nx * ny;

    Buffer<double> buffer(ncell, defaultMemorySpace());
    buffer.fillZero();

    auto view = buffer.view2D(nx, ny);

    parallelFor2D(nx, ny, [=] BASE_GPU_DEVICE (int i, int j) noexcept {
        view(i, j) = 100.0 + i + 10.0 * j;
    });

    backendSynchronize();

    auto host = buffer.copyToHostVector();
    print2D(host, nx, ny, backendLabel());

    return 0;
}
