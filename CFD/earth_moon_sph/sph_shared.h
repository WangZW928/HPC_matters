#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// 无量纲单位：长度以地球半径 Re 为 1，质量以地球质量 Me 为 1，
// 时间以 sqrt(Re^3 / (G Me)) 为 1。这样 GM_e=1，可避免 SI 单位下 float 精度恶化。
namespace sph {

constexpr float PI = 3.14159265358979323846f;
constexpr float EARTH_RADIUS = 1.0f;
constexpr float EARTH_GM = 1.0f;
constexpr float MOON_MASS_RATIO = 0.012300037f;
constexpr float MOON_GM = EARTH_GM * MOON_MASS_RATIO;
constexpr float MOON_ORBIT_RADIUS = 60.3f;
constexpr float MOON_OMEGA =
    0.0021480f;  // sqrt((GM_e + GM_m) / R_m^3)
// 真实潮汐加速度仅约为地表重力的 1e-7，短时教学动画中不可见。
// 默认放大差分潮汐力；设为 1.0f 可恢复真实比例。
constexpr float TIDAL_FORCE_SCALE = 1000000.0f;

constexpr float REST_DENSITY = 1.0f;
constexpr float SOUND_SPEED = 4.0f;
constexpr float EOS_GAMMA = 7.0f;
constexpr float EOS_B = REST_DENSITY * SOUND_SPEED * SOUND_SPEED / EOS_GAMMA;

constexpr float SMOOTHING_LENGTH = 0.055f;
constexpr float PARTICLE_MASS = 1.0e-5f;  // 启动时按实际粒子数和水层体积覆盖
constexpr float TIME_STEP = 4.0e-4f;
constexpr int BLOCK_SIZE = 256;

constexpr float WATER_INNER_RADIUS = 1.012f;
constexpr float WATER_OUTER_RADIUS = 1.075f;
// 数值安全层：正常潮汐自由表面不应触及此半径，仅拦截非物理逃逸粒子。
constexpr float WATER_ESCAPE_RADIUS = 1.16f;
constexpr float OUTER_RESTORE_STRENGTH = 120.0f;
constexpr float BOUNDARY_RANGE = 0.030f;
constexpr float BOUNDARY_STIFFNESS = 180.0f;
constexpr float BOUNDARY_DAMPING = 10.0f;
constexpr float VELOCITY_DAMPING = 0.35f;
constexpr float MAX_ACCELERATION = 50.0f;
constexpr float MAX_SPEED = 1.0f;
constexpr float BOUNDARY_RESTITUTION = 0.02f;
constexpr float OUTER_RESTITUTION = 0.05f;

// 网格只需包围水粒子，不必包围远处月球。
constexpr float GRID_MIN = -1.25f;
constexpr float GRID_MAX = 1.25f;
constexpr float CELL_SIZE = SMOOTHING_LENGTH;
constexpr int GRID_DIM = 46;
constexpr std::uint32_t NUM_CELLS = GRID_DIM * GRID_DIM * GRID_DIM;
constexpr std::uint32_t EMPTY_CELL = 0xffffffffu;

struct ParticleArrays {
    float3* pos = nullptr;
    float3* vel = nullptr;
    float* rho = nullptr;
    float* press = nullptr;
};

struct GridArrays {
    std::uint32_t* hash = nullptr;
    std::uint32_t* index = nullptr;
    std::uint32_t* cell_start = nullptr;
    std::uint32_t* cell_end = nullptr;
};

// CUDA 主机端包装函数，由 main.cpp 调用。
void build_and_sort_grid(ParticleArrays& particles, ParticleArrays& scratch,
                         GridArrays& grid, int count);
void compute_density_pressure(const ParticleArrays& particles,
                              const GridArrays& grid, int count,
                              float particle_mass);
void compute_forces(const ParticleArrays& particles, const GridArrays& grid,
                    float3* acceleration, int count, float particle_mass,
                    float time);
void integrate(ParticleArrays& particles, const float3* acceleration, int count,
               float dt);
void check_cuda(const char* stage);

}  // namespace sph
