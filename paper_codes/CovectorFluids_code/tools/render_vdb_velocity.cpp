#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

openvdb::FloatGrid::Ptr readFloatGrid(const std::string& path, const std::string& name)
{
    openvdb::io::File file(path);
    file.open();
    openvdb::GridBase::Ptr base = file.readGrid(name);
    file.close();
    auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
    if (!grid) throw std::runtime_error("Expected FloatGrid '" + name + "' in " + path);
    return grid;
}

std::string framePath(const std::string& folder, const std::string& component, int frame)
{
    char suffix[64];
    std::snprintf(suffix, sizeof(suffix), "/%s_render_%04d.vdb", component.c_str(), frame);
    return folder + suffix;
}

void heatColor(float value, float scale, unsigned char& r, unsigned char& g, unsigned char& b)
{
    float x = scale > 0.0f ? std::min(value / scale, 1.0f) : 0.0f;
    x = std::sqrt(x);
    const float stops[5][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.05f, 0.15f, 0.85f},
        {0.0f, 0.90f, 1.0f},
        {1.0f, 0.90f, 0.0f},
        {1.0f, 1.0f, 1.0f}
    };
    float position = x * 4.0f;
    int segment = std::min(static_cast<int>(position), 3);
    float t = position - static_cast<float>(segment);
    r = static_cast<unsigned char>(255.0f * ((1.0f - t) * stops[segment][0] + t * stops[segment + 1][0]));
    g = static_cast<unsigned char>(255.0f * ((1.0f - t) * stops[segment][1] + t * stops[segment + 1][1]));
    b = static_cast<unsigned char>(255.0f * ((1.0f - t) * stops[segment][2] + t * stops[segment + 1][2]));
}

void writePpm(const std::string& path, int width, int height,
              const std::vector<float>& top, const std::vector<float>& side, float scale)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Unable to write " + path);
    out << "P6\n" << width << " " << height << "\n255\n";

    const int topHeight = static_cast<int>(top.size()) / width;
    const int sideHeight = static_cast<int>(side.size()) / width;
    for (int y = topHeight - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            unsigned char rgb[3];
            heatColor(top[x + width * y], scale, rgb[0], rgb[1], rgb[2]);
            out.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }
    for (int z = sideHeight - 1; z >= 0; --z) {
        for (int x = 0; x < width; ++x) {
            unsigned char rgb[3];
            heatColor(side[x + width * z], scale, rgb[0], rgb[1], rgb[2]);
            out.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <vdb-folder> <ppm-folder> <first-frame> <last-frame>\n";
        return 1;
    }

    const std::string input = argv[1];
    const std::string output = argv[2];
    const int first = std::stoi(argv[3]);
    const int last = std::stoi(argv[4]);
    openvdb::initialize();

    float fixedScale = 0.0f;
    std::cout << "frame,max_vorticity\n";
    for (int frame = first; frame <= last; ++frame) {
        auto u = readFloatGrid(framePath(input, "vel_x", frame), "vel_x");
        auto v = readFloatGrid(framePath(input, "vel_y", frame), "vel_y");
        auto w = readFloatGrid(framePath(input, "vel_z", frame), "vel_z");
        const auto ubox = u->evalActiveVoxelBoundingBox();
        const int nx = ubox.dim().x() - 1;
        const int ny = ubox.dim().y();
        const int nz = ubox.dim().z();
        if (nx <= 1 || ny <= 1 || nz <= 1) throw std::runtime_error("Invalid velocity grid dimensions");

        std::vector<float> top(static_cast<size_t>(nx) * ny, 0.0f);
        std::vector<float> side(static_cast<size_t>(nx) * nz, 0.0f);
        auto ua = u->getConstAccessor();
        auto va = v->getConstAccessor();
        auto wa = w->getConstAccessor();
        float frameMax = 0.0f;

        for (int k = 0; k < nz - 1; ++k) {
            for (int j = 0; j < ny - 1; ++j) {
                for (int i = 0; i < nx - 1; ++i) {
                    const float curlX = wa.getValue({i, j + 1, k}) - wa.getValue({i, j, k})
                                      - va.getValue({i, j, k + 1}) + va.getValue({i, j, k});
                    const float curlY = ua.getValue({i, j, k + 1}) - ua.getValue({i, j, k})
                                      - wa.getValue({i + 1, j, k}) + wa.getValue({i, j, k});
                    const float curlZ = va.getValue({i + 1, j, k}) - va.getValue({i, j, k})
                                      - ua.getValue({i, j + 1, k}) + ua.getValue({i, j, k});
                    const float magnitude = std::sqrt(curlX * curlX + curlY * curlY + curlZ * curlZ);
                    top[i + nx * j] = std::max(top[i + nx * j], magnitude);
                    side[i + nx * k] = std::max(side[i + nx * k], magnitude);
                    frameMax = std::max(frameMax, magnitude);
                }
            }
        }

        if (frame == first) fixedScale = frameMax;
        char outputPath[1024];
        std::snprintf(outputPath, sizeof(outputPath), "%s/vorticity_%04d.ppm", output.c_str(), frame);
        writePpm(outputPath, nx, ny + nz, top, side, fixedScale);
        std::cout << frame << "," << std::setprecision(9) << frameMax << "\n";
    }
    return 0;
}

