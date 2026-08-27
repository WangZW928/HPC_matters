#include "sph_shared.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_ok(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
}

void allocate_particles(sph::ParticleArrays& p, int count) {
    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&p.pos), count * sizeof(float3)),
            "cudaMalloc pos");
    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&p.vel), count * sizeof(float3)),
            "cudaMalloc vel");
    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&p.rho), count * sizeof(float)),
            "cudaMalloc rho");
    cuda_ok(cudaMalloc(reinterpret_cast<void**>(&p.press), count * sizeof(float)),
            "cudaMalloc press");
}

void free_particles(sph::ParticleArrays& p) {
    cudaFree(p.pos);
    cudaFree(p.vel);
    cudaFree(p.rho);
    cudaFree(p.press);
    p = {};
}

void initialize_particles(std::vector<float3>& pos, std::vector<float3>& vel) {
    // Fibonacci 球面 + 均匀体积分布，生成近似均匀的薄球壳水层。
    constexpr float golden_angle = 2.39996322972865332f;
    const int n = static_cast<int>(pos.size());
    for (int i = 0; i < n; ++i) {
        const float u = (i + 0.5f) / n;
        const float z = 1.0f - 2.0f * u;
        const float phi = golden_angle * i;
        const float xy = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float radial_u = std::fmod(i * 0.61803398875f, 1.0f);
        const float inner3 = std::pow(sph::WATER_INNER_RADIUS, 3.0f);
        const float outer3 = std::pow(sph::WATER_OUTER_RADIUS, 3.0f);
        const float r = std::cbrt(inner3 + radial_u * (outer3 - inner3));
        pos[i] = make_float3(r * xy * std::cos(phi), r * xy * std::sin(phi),
                             r * z);
        vel[i] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

void write_frame(const std::filesystem::path& path,
                 const sph::ParticleArrays& particles, int count, float time) {
    std::vector<float3> pos(count), vel(count);
    std::vector<float> rho(count), press(count);
    cuda_ok(cudaMemcpy(pos.data(), particles.pos, count * sizeof(float3),
                       cudaMemcpyDeviceToHost),
            "copy positions");
    cuda_ok(cudaMemcpy(vel.data(), particles.vel, count * sizeof(float3),
                       cudaMemcpyDeviceToHost),
            "copy velocities");
    cuda_ok(cudaMemcpy(rho.data(), particles.rho, count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy densities");
    cuda_ok(cudaMemcpy(press.data(), particles.press, count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy pressures");

    std::ofstream out(path, std::ios::binary);
    const std::uint32_t magic = 0x53504831;  // "SPH1"
    const std::uint32_t n = static_cast<std::uint32_t>(count);
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&time), sizeof(time));
    out.write(reinterpret_cast<const char*>(pos.data()), count * sizeof(float3));
    out.write(reinterpret_cast<const char*>(vel.data()), count * sizeof(float3));
    out.write(reinterpret_cast<const char*>(rho.data()), count * sizeof(float));
    out.write(reinterpret_cast<const char*>(press.data()), count * sizeof(float));

    float min_radius = std::numeric_limits<float>::max();
    float max_radius = 0.0f;
    float max_speed = 0.0f;
    float min_density = std::numeric_limits<float>::max();
    float max_density = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float radius = std::sqrt(pos[i].x * pos[i].x + pos[i].y * pos[i].y +
                                       pos[i].z * pos[i].z);
        const float speed = std::sqrt(vel[i].x * vel[i].x + vel[i].y * vel[i].y +
                                      vel[i].z * vel[i].z);
        min_radius = std::min(min_radius, radius);
        max_radius = std::max(max_radius, radius);
        max_speed = std::max(max_speed, speed);
        min_density = std::min(min_density, rho[i]);
        max_density = std::max(max_density, rho[i]);
    }
    std::cout << "  diagnostics: radius=[" << min_radius << ", " << max_radius
              << "], max_speed=" << max_speed << ", density=[" << min_density
              << ", " << max_density << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int count = argc > 1 ? std::stoi(argv[1]) : 300000;
    const int steps = argc > 2 ? std::stoi(argv[2]) : 5000;
    const int export_every = argc > 3 ? std::stoi(argv[3]) : 50;
    const std::filesystem::path output = argc > 4 ? argv[4] : "output";
    if (count <= 0 || steps < 0 || export_every <= 0)
        throw std::invalid_argument("参数必须满足 N>0, steps>=0, export_every>0");
    std::filesystem::create_directories(output);

    sph::ParticleArrays particles, scratch;
    sph::GridArrays grid;
    float3* acceleration = nullptr;
    try {
        allocate_particles(particles, count);
        allocate_particles(scratch, count);
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&grid.hash),
                           count * sizeof(std::uint32_t)),
                "cudaMalloc hash");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&grid.index),
                           count * sizeof(std::uint32_t)),
                "cudaMalloc index");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&grid.cell_start),
                           sph::NUM_CELLS * sizeof(std::uint32_t)),
                "cudaMalloc cell_start");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&grid.cell_end),
                           sph::NUM_CELLS * sizeof(std::uint32_t)),
                "cudaMalloc cell_end");
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&acceleration),
                           count * sizeof(float3)),
                "cudaMalloc acceleration");

        std::vector<float3> host_pos(count), host_vel(count);
        initialize_particles(host_pos, host_vel);
        cuda_ok(cudaMemcpy(particles.pos, host_pos.data(), count * sizeof(float3),
                           cudaMemcpyHostToDevice),
                "initialize positions");
        cuda_ok(cudaMemcpy(particles.vel, host_vel.data(), count * sizeof(float3),
                           cudaMemcpyHostToDevice),
                "initialize velocities");
        cuda_ok(cudaMemset(particles.rho, 0, count * sizeof(float)),
                "initialize density");
        cuda_ok(cudaMemset(particles.press, 0, count * sizeof(float)),
                "initialize pressure");

        const float shell_volume =
            4.0f * sph::PI / 3.0f *
            (std::pow(sph::WATER_OUTER_RADIUS, 3.0f) -
             std::pow(sph::WATER_INNER_RADIUS, 3.0f));
        const float particle_mass = sph::REST_DENSITY * shell_volume / count;
        std::cout << "N=" << count << ", particle_mass=" << particle_mass
                  << ", dt=" << sph::TIME_STEP
                  << ", tidal_scale=" << sph::TIDAL_FORCE_SCALE << '\n';

        float time = 0.0f;
        for (int step = 0; step <= steps; ++step) {
            sph::build_and_sort_grid(particles, scratch, grid, count);
            sph::compute_density_pressure(particles, grid, count, particle_mass);
            if (step % export_every == 0) {
                cuda_ok(cudaDeviceSynchronize(), "synchronize before export");
                const auto file = output / ("frame_" + std::to_string(step) + ".bin");
                write_frame(file, particles, count, time);
                std::cout << "exported " << file << '\n';
            }
            if (step == steps) break;
            sph::compute_forces(particles, grid, acceleration, count,
                                particle_mass, time);
            sph::integrate(particles, acceleration, count, sph::TIME_STEP);
            time += sph::TIME_STEP;
        }
        cuda_ok(cudaDeviceSynchronize(), "final synchronize");
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << '\n';
        free_particles(particles);
        free_particles(scratch);
        cudaFree(grid.hash);
        cudaFree(grid.index);
        cudaFree(grid.cell_start);
        cudaFree(grid.cell_end);
        cudaFree(acceleration);
        return 1;
    }

    free_particles(particles);
    free_particles(scratch);
    cudaFree(grid.hash);
    cudaFree(grid.index);
    cudaFree(grid.cell_start);
    cudaFree(grid.cell_end);
    cudaFree(acceleration);
    return 0;
}
