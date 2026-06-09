#include "sph_shared.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace sph {
namespace {

__host__ __device__ inline float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 mul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline int3 grid_coord(float3 p) {
    return make_int3(static_cast<int>(floorf((p.x - GRID_MIN) / CELL_SIZE)),
                     static_cast<int>(floorf((p.y - GRID_MIN) / CELL_SIZE)),
                     static_cast<int>(floorf((p.z - GRID_MIN) / CELL_SIZE)));
}

__device__ inline bool valid_cell(int3 c) {
    return c.x >= 0 && c.x < GRID_DIM && c.y >= 0 && c.y < GRID_DIM &&
           c.z >= 0 && c.z < GRID_DIM;
}

__device__ inline std::uint32_t cell_hash(int3 c) {
    return static_cast<std::uint32_t>((c.z * GRID_DIM + c.y) * GRID_DIM + c.x);
}

// 3D Cubic Spline 核，支撑半径为 h。
__device__ inline float cubic_kernel(float r) {
    const float q = r / SMOOTHING_LENGTH;
    if (q >= 1.0f) return 0.0f;
    const float sigma = 8.0f / (PI * SMOOTHING_LENGTH * SMOOTHING_LENGTH *
                                SMOOTHING_LENGTH);
    if (q < 0.5f) return sigma * (1.0f - 6.0f * q * q + 6.0f * q * q * q);
    const float t = 1.0f - q;
    return sigma * 2.0f * t * t * t;
}

__device__ inline float3 cubic_gradient(float3 rij, float r) {
    if (r <= 1.0e-7f || r >= SMOOTHING_LENGTH) return make_float3(0, 0, 0);
    const float q = r / SMOOTHING_LENGTH;
    const float sigma = 8.0f / (PI * SMOOTHING_LENGTH * SMOOTHING_LENGTH *
                                SMOOTHING_LENGTH);
    float dwdq;
    if (q < 0.5f)
        dwdq = sigma * (-12.0f * q + 18.0f * q * q);
    else {
        const float t = 1.0f - q;
        dwdq = -sigma * 6.0f * t * t;
    }
    return mul(rij, dwdq / (SMOOTHING_LENGTH * r));
}

__global__ void kernel_build_grid(const float3* pos, std::uint32_t* hash,
                                  std::uint32_t* index, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int3 c = grid_coord(pos[i]);
    hash[i] = valid_cell(c) ? cell_hash(c) : EMPTY_CELL;
    index[i] = static_cast<std::uint32_t>(i);
}

__global__ void kernel_reorder(const ParticleArrays src, ParticleArrays dst,
                               const std::uint32_t* sorted_index, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::uint32_t old = sorted_index[i];
    dst.pos[i] = src.pos[old];
    dst.vel[i] = src.vel[old];
    dst.rho[i] = src.rho[old];
    dst.press[i] = src.press[old];
}

__global__ void kernel_find_cell_ranges(const std::uint32_t* hash,
                                        std::uint32_t* cell_start,
                                        std::uint32_t* cell_end, int count) {
    extern __shared__ std::uint32_t shared_hash[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t h = i < count ? hash[i] : EMPTY_CELL;
    shared_hash[threadIdx.x + 1] = h;
    if (threadIdx.x == 0)
        shared_hash[0] = i > 0 ? hash[i - 1] : EMPTY_CELL;
    __syncthreads();

    if (i >= count || h == EMPTY_CELL) return;
    if (i == 0 || h != shared_hash[threadIdx.x]) cell_start[h] = i;
    if (i == count - 1 || h != hash[i + 1]) cell_end[h] = i + 1;
}

__global__ void kernel_compute_density_pressure(
    ParticleArrays particles, const std::uint32_t* cell_start,
    const std::uint32_t* cell_end, int count, float particle_mass) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float3 pi = particles.pos[i];
    const int3 center = grid_coord(pi);
    float rho = 0.0f;

    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int3 c = make_int3(center.x + dx, center.y + dy,
                                         center.z + dz);
                if (!valid_cell(c)) continue;
                const std::uint32_t h = cell_hash(c);
                const std::uint32_t begin = cell_start[h];
                if (begin == EMPTY_CELL) continue;
                for (std::uint32_t j = begin; j < cell_end[h]; ++j) {
                    const float r = sqrtf(dot(sub(pi, particles.pos[j]),
                                              sub(pi, particles.pos[j])));
                    rho += particle_mass * cubic_kernel(r);
                }
            }
    rho = fmaxf(rho, 0.1f * REST_DENSITY);
    particles.rho[i] = rho;
    // 自由表面不承受张力。将负压截断为零，避免 tensile instability
    // 将粒子拉成团并数值喷射出去。
    particles.press[i] =
        fmaxf(EOS_B * (powf(rho / REST_DENSITY, EOS_GAMMA) - 1.0f), 0.0f);
}

__global__ void kernel_compute_forces(
    const ParticleArrays particles, const std::uint32_t* cell_start,
    const std::uint32_t* cell_end, float3* acceleration, int count,
    float particle_mass, float time) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float3 pi = particles.pos[i];
    const int3 center = grid_coord(pi);
    const float rhoi = fmaxf(particles.rho[i], 0.1f * REST_DENSITY);
    const float pi_term = particles.press[i] / (rhoi * rhoi);
    float3 acc = make_float3(0, 0, 0);

    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int3 c = make_int3(center.x + dx, center.y + dy,
                                         center.z + dz);
                if (!valid_cell(c)) continue;
                const std::uint32_t h = cell_hash(c);
                const std::uint32_t begin = cell_start[h];
                if (begin == EMPTY_CELL) continue;
                for (std::uint32_t j = begin; j < cell_end[h]; ++j) {
                    if (j == static_cast<std::uint32_t>(i)) continue;
                    const float3 rij = sub(pi, particles.pos[j]);
                    const float r = sqrtf(dot(rij, rij));
                    if (r >= SMOOTHING_LENGTH || r < 1.0e-7f) continue;
                    const float rhoj =
                        fmaxf(particles.rho[j], 0.1f * REST_DENSITY);
                    const float pressure_term =
                        pi_term + particles.press[j] / (rhoj * rhoj);
                    acc = add(acc, mul(cubic_gradient(rij, r),
                                       -particle_mass * pressure_term));
                }
            }

    const float radius = fmaxf(sqrtf(dot(pi, pi)), 1.0e-6f);
    acc = add(acc, mul(pi, -EARTH_GM / (radius * radius * radius)));

    const float angle = MOON_OMEGA * time;
    const float3 moon =
        make_float3(MOON_ORBIT_RADIUS * cosf(angle),
                    MOON_ORBIT_RADIUS * sinf(angle), 0.0f);
    const float3 to_moon = sub(moon, pi);
    const float moon_r = fmaxf(sqrtf(dot(to_moon, to_moon)), 1.0e-6f);
    // 地心坐标系是随地球质心加速的非惯性系，需要减去月球对地心的共同加速度。
    // 剩余项才是使近侧与远侧水体产生差异形变的潮汐加速度。
    const float3 moon_at_particle =
        mul(to_moon, MOON_GM / (moon_r * moon_r * moon_r));
    const float3 moon_at_earth =
        mul(moon, MOON_GM /
                      (MOON_ORBIT_RADIUS * MOON_ORBIT_RADIUS *
                       MOON_ORBIT_RADIUS));
    acc = add(acc, mul(sub(moon_at_particle, moon_at_earth),
                       TIDAL_FORCE_SCALE));

    // 带阻尼的平滑地表支撑。相比高次 Lennard-Jones 势，它不会在粒子稍微
    // 穿透时产生数百倍重力的瞬时脉冲。
    const float gap = radius - EARTH_RADIUS;
    if (gap < BOUNDARY_RANGE) {
        const float penetration = BOUNDARY_RANGE - gap;
        const float radial_velocity = dot(particles.vel[i], pi) / radius;
        const float repel = BOUNDARY_STIFFNESS * penetration -
                            BOUNDARY_DAMPING * fminf(radial_velocity, 0.0f);
        acc = add(acc, mul(pi, repel / radius));
    }
    // 仅在粒子明显离开正常水层后启用柔和恢复力，抑制离群粒子继续飞散。
    if (radius > WATER_OUTER_RADIUS + 0.025f) {
        const float excess = radius - (WATER_OUTER_RADIUS + 0.025f);
        acc = add(acc, mul(pi, -OUTER_RESTORE_STRENGTH * excess / radius));
    }
    // 防止极少数局部密度异常产生单步压力尖峰。
    const float acc_norm = sqrtf(dot(acc, acc));
    if (acc_norm > MAX_ACCELERATION) acc = mul(acc, MAX_ACCELERATION / acc_norm);
    acceleration[i] = acc;
}

__global__ void kernel_integrate(ParticleArrays particles,
                                 const float3* acceleration, int count,
                                 float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // Symplectic Euler：先更新速度，再用新速度更新位置。
    float3 v = add(particles.vel[i], mul(acceleration[i], dt));
    // 数值阻尼耗散初始化阶段的径向振荡；不会产生向外推力。
    v = mul(v, fmaxf(0.0f, 1.0f - VELOCITY_DAMPING * dt));
    const float speed = sqrtf(dot(v, v));
    if (speed > MAX_SPEED) v = mul(v, MAX_SPEED / speed);
    float3 p = add(particles.pos[i], mul(v, dt));
    const float r = fmaxf(sqrtf(dot(p, p)), 1.0e-6f);
    if (r < WATER_INNER_RADIUS) {
        const float3 n = mul(p, 1.0f / r);
        p = mul(n, WATER_INNER_RADIUS);
        const float vn = dot(v, n);
        if (vn < 0.0f) v = sub(v, mul(n, (1.0f + BOUNDARY_RESTITUTION) * vn));
    } else if (r > WATER_ESCAPE_RADIUS) {
        // 最后一道数值保护，防止极端压力脉冲把粒子送出空间哈希网格。
        const float3 n = mul(p, 1.0f / r);
        p = mul(n, WATER_ESCAPE_RADIUS);
        const float vn = dot(v, n);
        if (vn > 0.0f) v = sub(v, mul(n, (1.0f + OUTER_RESTITUTION) * vn));
    }
    particles.pos[i] = p;
    particles.vel[i] = v;
}

}  // namespace

void check_cuda(const char* stage) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", stage,
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void build_and_sort_grid(ParticleArrays& particles, ParticleArrays& scratch,
                         GridArrays& grid, int count) {
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_build_grid<<<blocks, BLOCK_SIZE>>>(particles.pos, grid.hash,
                                              grid.index, count);
    check_cuda("kernel_build_grid");
    thrust::device_ptr<std::uint32_t> keys(grid.hash);
    thrust::device_ptr<std::uint32_t> values(grid.index);
    thrust::sort_by_key(keys, keys + count, values);
    kernel_reorder<<<blocks, BLOCK_SIZE>>>(particles, scratch, grid.index,
                                           count);
    check_cuda("kernel_reorder");
    std::swap(particles, scratch);
    cudaMemset(grid.cell_start, 0xff, NUM_CELLS * sizeof(std::uint32_t));
    cudaMemset(grid.cell_end, 0, NUM_CELLS * sizeof(std::uint32_t));
    kernel_find_cell_ranges<<<blocks, BLOCK_SIZE,
                              (BLOCK_SIZE + 1) * sizeof(std::uint32_t)>>>(
        grid.hash, grid.cell_start, grid.cell_end, count);
    check_cuda("kernel_find_cell_ranges");
}

void compute_density_pressure(const ParticleArrays& particles,
                              const GridArrays& grid, int count,
                              float particle_mass) {
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_compute_density_pressure<<<blocks, BLOCK_SIZE>>>(
        particles, grid.cell_start, grid.cell_end, count, particle_mass);
    check_cuda("kernel_compute_density_pressure");
}

void compute_forces(const ParticleArrays& particles, const GridArrays& grid,
                    float3* acceleration, int count, float particle_mass,
                    float time) {
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_compute_forces<<<blocks, BLOCK_SIZE>>>(
        particles, grid.cell_start, grid.cell_end, acceleration, count,
        particle_mass, time);
    check_cuda("kernel_compute_forces");
}

void integrate(ParticleArrays& particles, const float3* acceleration, int count,
               float dt) {
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_integrate<<<blocks, BLOCK_SIZE>>>(particles, acceleration, count,
                                             dt);
    check_cuda("kernel_integrate");
}

}  // namespace sph
