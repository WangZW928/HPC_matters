#include "Backend.H"
#include "Buffer.H"
#include "Reduce.H"

#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

int main ()
{
    int n = 8;

    std::vector<Vec2> host_vec(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        host_vec[static_cast<std::size_t>(i)] = Vec2{1.0 + 0.25 * i, 0.5 * i};
    }

    Buffer<Vec2> buffer(n, defaultMemorySpace());
    buffer.copyFromHostVector(host_vec);

    Vec2 const* data = buffer.data();
    Reducer reducer;
    auto result = reducer.eval(n, [=] BASE_GPU_DEVICE (int i) noexcept -> ReduceTuple {
        Vec2 v = data[i];
        double mag = sqrt(v.x * v.x + v.y * v.y);
        return {mag, mag};
    });

    double sum_mag = std::get<0>(result);
    double max_mag = std::get<1>(result);

    std::cout << backendLabel() << " local reduction\n";
    std::cout << "sum(|v|) = " << sum_mag << "\n";
    std::cout << "max(|v|) = " << max_mag << "\n";

    return 0;
}
