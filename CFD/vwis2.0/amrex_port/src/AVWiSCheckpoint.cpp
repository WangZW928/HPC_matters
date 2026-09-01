#include "AVWiSSolver.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef AVWIS_LOCKED_VERSION
#define AVWIS_LOCKED_VERSION "unknown"
#endif
#ifndef AVWIS_LOCKED_GIT_SHA
#define AVWIS_LOCKED_GIT_SHA "unknown"
#endif

namespace {

constexpr int checkpoint_schema_version = 2;
constexpr char checkpoint_magic[] = "AVWIS_CARTESIAN_CHECKPOINT";
constexpr char legacy_checkpoint_magic[] = "VWIS_AMREX_CARTESIAN_CHECKPOINT";

struct Payload {
    std::string name;
    std::string location;
    int components;
    amrex::MultiFab* data;
};

[[noreturn]] void reject(std::string const& reason)
{
    throw std::runtime_error("P8 checkpoint rejected: " + reason);
}

void require_cpu_single_rank()
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        reject("checkpoint/restart supports exactly one MPI rank");
    }
#ifdef AMREX_USE_GPU
    reject("checkpoint/restart is validated only for an AMReX CPU build");
#endif
}

std::string real_text(amrex::Real value)
{
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10) << value;
    return stream.str();
}

std::vector<std::string> words(std::string const& value)
{
    std::istringstream stream(value);
    std::vector<std::string> result;
    for (std::string token; stream >> token;) result.push_back(token);
    return result;
}

long long signed_integer(std::string const& text, char const* key)
{
    std::size_t used = 0;
    long long value = 0;
    try { value = std::stoll(text, &used); }
    catch (...) { reject(std::string("invalid integer for ") + key); }
    if (used != text.size()) reject(std::string("invalid integer for ") + key);
    return value;
}

std::uint64_t unsigned_integer(std::string const& text, char const* key)
{
    if (text.empty() || text[0] == '-') reject(std::string("invalid unsigned integer for ") + key);
    std::size_t used = 0;
    unsigned long long value = 0;
    try { value = std::stoull(text, &used); }
    catch (...) { reject(std::string("invalid unsigned integer for ") + key); }
    if (used != text.size()) reject(std::string("invalid unsigned integer for ") + key);
    return static_cast<std::uint64_t>(value);
}

amrex::Real real_number(std::string const& text, char const* key)
{
    std::size_t used = 0;
    long double value = 0.0;
    try { value = std::stold(text, &used); }
    catch (...) { reject(std::string("invalid real for ") + key); }
    if (used != text.size() || !std::isfinite(value)) {
        reject(std::string("invalid real for ") + key);
    }
    return static_cast<amrex::Real>(value);
}

class StrictHeader {
public:
    explicit StrictHeader(std::filesystem::path const& path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input) reject("cannot open Header");
        for (std::string line; std::getline(input, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            m_lines.push_back(line);
        }
        if (!input.eof()) reject("cannot read Header");
    }

    void exact(std::string const& expected)
    {
        if (m_at >= m_lines.size() || m_lines[m_at++] != expected) {
            reject("malformed or incompatible Header line " + std::to_string(m_at));
        }
    }

    void exact_or_legacy(std::string const& expected, std::string const& legacy)
    {
        if (m_at >= m_lines.size()) reject("missing Header identity line");
        std::string const actual = m_lines[m_at++];
        if (actual != expected && actual != legacy) {
            reject("malformed or incompatible Header line " + std::to_string(m_at));
        }
    }

    std::string value(std::string const& key)
    {
        if (m_at >= m_lines.size()) reject("missing Header key " + key);
        std::string const prefix = key + " ";
        std::string const& line = m_lines[m_at++];
        if (line.rfind(prefix, 0) != 0 || line.size() == prefix.size()) {
            reject("expected Header key " + key);
        }
        return line.substr(prefix.size());
    }

    void end()
    {
        exact("END");
        if (m_at != m_lines.size()) reject("trailing Header content");
    }

private:
    std::vector<std::string> m_lines;
    std::size_t m_at = 0;
};

void same(std::string const& actual, std::string const& expected, char const* key)
{
    if (actual != expected) reject(std::string(key) + " mismatch");
}

void same_integer(std::string const& actual, long long expected, char const* key)
{
    if (signed_integer(actual, key) != expected) reject(std::string(key) + " mismatch");
}

void same_real(std::string const& actual, amrex::Real expected, char const* key)
{
    if (real_number(actual, key) != expected) reject(std::string(key) + " mismatch");
}

amrex::Real max_difference(amrex::MultiFab const& lhs, amrex::MultiFab const& rhs,
                           int components, int nghost = 0)
{
    amrex::MultiFab difference(lhs.boxArray(), lhs.DistributionMap(), components, nghost);
    amrex::MultiFab::Copy(difference, lhs, 0, 0, components, nghost);
    amrex::MultiFab::Subtract(difference, rhs, 0, 0, components, nghost);
    amrex::Real result = 0.0;
    for (int comp = 0; comp < components; ++comp) {
        result = amrex::max(result, difference.norm0(comp, nghost, true));
    }
    amrex::ParallelDescriptor::ReduceRealMax(result);
    return result;
}

} // namespace

void AVWiSSolver::write_checkpoint(std::string const& path) const
{
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Cartesian) {
        reject("C2.2 mapped mapping provenance is not part of the Cartesian schema");
    }
    require_cpu_single_rank();
    if (path.empty()) reject("empty checkpoint path");

    std::filesystem::path root(path);
    if (std::filesystem::exists(root)) reject("destination already exists: " + path);
    if (!std::filesystem::create_directory(root) ||
        !std::filesystem::create_directory(root / "Level_0")) {
        reject("cannot create checkpoint directory: " + path);
    }

    std::vector<Payload> payloads = {
        {"P", "cell", 1, const_cast<amrex::MultiFab*>(&m_p)},
        {"Phi", "cell", 1, const_cast<amrex::MultiFab*>(&m_phi)},
        {"Nvert", "cell", 1, const_cast<amrex::MultiFab*>(&m_nvert)},
        {"Ucat", "cell", AMREX_SPACEDIM, const_cast<amrex::MultiFab*>(&m_ucat)},
        {"Ucat_old", "cell", AMREX_SPACEDIM, const_cast<amrex::MultiFab*>(&m_ucat_old)},
        {"Ucat_older", "cell", AMREX_SPACEDIM, const_cast<amrex::MultiFab*>(&m_ucat_older)}
    };
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        std::string axis(1, static_cast<char>('x' + dir));
        payloads.push_back({"Ucont_" + axis, axis + "-face", 1,
                            const_cast<amrex::MultiFab*>(&m_ucont[dir])});
        payloads.push_back({"Ucont_" + axis + "_old", axis + "-face", 1,
                            const_cast<amrex::MultiFab*>(&m_ucont_old[dir])});
        payloads.push_back({"Ucont_" + axis + "_older", axis + "-face", 1,
                            const_cast<amrex::MultiFab*>(&m_ucont_older[dir])});
    }

    for (auto const& payload : payloads) {
        amrex::VisMF::Write(*payload.data, (root / "Level_0" / payload.name).string());
    }

    std::ofstream output(root / "Header", std::ios::binary | std::ios::trunc);
    if (!output) reject("cannot create Header");
    output << checkpoint_magic << '\n'
           << "schema_version " << checkpoint_schema_version << '\n'
           << "amrex_release " << AVWIS_LOCKED_VERSION << '\n'
           << "amrex_git_sha " << AVWIS_LOCKED_GIT_SHA << '\n'
           << "amrex_runtime_version " << amrex::Version() << '\n'
           << "dimension " << AMREX_SPACEDIM << '\n'
           << "real_bytes " << sizeof(amrex::Real) << '\n'
           << "real_digits " << std::numeric_limits<amrex::Real>::digits << '\n'
           << "backend CPU\nlevels 1\nranks 1\ncoordinate_system Cartesian\n"
           << "time " << real_text(m_time) << '\n'
           << "step " << m_step << '\n'
           << "history_depth " << m_history_depth << '\n';

    auto const& domain = m_geom.Domain();
    output << "domain";
    for (int d = 0; d < AMREX_SPACEDIM; ++d) output << ' ' << domain.smallEnd(d);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) output << ' ' << domain.bigEnd(d);
    output << '\n' << "prob_lo";
    for (int d = 0; d < AMREX_SPACEDIM; ++d) output << ' ' << real_text(m_geom.ProbLo(d));
    output << '\n' << "prob_hi";
    for (int d = 0; d < AMREX_SPACEDIM; ++d) output << ' ' << real_text(m_geom.ProbHi(d));
    output << '\n' << "dx";
    for (int d = 0; d < AMREX_SPACEDIM; ++d) output << ' ' << real_text(m_dx[d]);
    output << '\n' << "periodic";
    for (int d = 0; d < AMREX_SPACEDIM; ++d) output << ' ' << m_geom.isPeriodic(d);
    output << '\n' << "box_count " << m_ba.size() << '\n';
    for (int index = 0; index < m_ba.size(); ++index) {
        auto const& box = m_ba[index];
        output << "box_" << index;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) output << ' ' << box.smallEnd(d);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) output << ' ' << box.bigEnd(d);
        output << '\n';
    }

    output << "boundary_enabled " << static_cast<int>(m_boundary.enabled) << '\n';
    for (int slot = 0; slot < 2 * AMREX_SPACEDIM; ++slot) {
        auto const& side = m_boundary.sides[slot];
        output << "bc_" << slot << ' ' << cartesian_bc_name(side.velocity) << ' '
               << side.legacy_code << ' ' << real_text(side.pressure) << '\n';
    }
    output << "inlet_profile " << m_boundary.inlet_profile << '\n'
           << "inlet_target_flux " << real_text(m_boundary.inlet_target_flux) << '\n'
           << "profile_offset " << real_text(m_boundary.profile_offset) << '\n'
           << "profile_slope_0 " << real_text(m_boundary.profile_slope_0) << '\n'
           << "profile_slope_1 " << real_text(m_boundary.profile_slope_1) << '\n'
           << "constrain_outlet_flux " << static_cast<int>(m_boundary.constrain_outlet_flux) << '\n'
           << "moving_wall_velocity "
           << real_text(m_boundary.moving_wall_velocity[0]) << ' '
           << real_text(m_boundary.moving_wall_velocity[1]) << ' '
           << real_text(m_boundary.moving_wall_velocity[2]) << '\n'
           << "payload_count " << payloads.size() << '\n';
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        auto const& payload = payloads[index];
        output << "payload_" << index << ' ' << payload.name << ' ' << payload.location << ' '
               << payload.components << ' ' << m_nghost << " Level_0/" << payload.name << '\n';
    }
    output << "END\n";
    output.close();
    if (!output) reject("failed while writing Header");
    amrex::Print() << "AVWiS P8 checkpoint written: " << path
                   << " time=" << m_time << " step=" << m_step
                   << " history_depth=" << m_history_depth << "\n";
}

void AVWiSSolver::read_checkpoint(std::string const& path)
{
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Cartesian) {
        reject("C2.2 mapped mapping provenance is not part of the Cartesian schema");
    }
    require_cpu_single_rank();
    if (path.empty()) reject("empty checkpoint path");
    std::filesystem::path root(path);
    StrictHeader header(root / "Header");
    header.exact_or_legacy(checkpoint_magic, legacy_checkpoint_magic);
    const int disk_schema_version = static_cast<int>(
        signed_integer(header.value("schema_version"), "schema_version"));
    if (disk_schema_version != 1 && disk_schema_version != checkpoint_schema_version) {
        reject("schema_version mismatch");
    }
    same(header.value("amrex_release"), AVWIS_LOCKED_VERSION, "AMReX release");
    same(header.value("amrex_git_sha"), AVWIS_LOCKED_GIT_SHA, "AMReX git SHA");
    same(header.value("amrex_runtime_version"), amrex::Version(), "AMReX runtime version");
    same_integer(header.value("dimension"), AMREX_SPACEDIM, "dimension");
    same_integer(header.value("real_bytes"), sizeof(amrex::Real), "Real byte width");
    same_integer(header.value("real_digits"), std::numeric_limits<amrex::Real>::digits, "Real precision");
    same(header.value("backend"), "CPU", "backend");
    same_integer(header.value("levels"), 1, "level count");
    same_integer(header.value("ranks"), 1, "rank count");
    same(header.value("coordinate_system"), "Cartesian", "coordinate system");

    amrex::Real restored_time = real_number(header.value("time"), "time");
    std::uint64_t restored_step = unsigned_integer(header.value("step"), "step");
    int restored_history = static_cast<int>(signed_integer(header.value("history_depth"), "history_depth"));
    if (restored_time < 0.0 || restored_history < 1 || restored_history > 3 ||
        restored_history > static_cast<int>(std::min<std::uint64_t>(3, restored_step + 1))) {
        reject("invalid time/step/history_depth state");
    }

    auto check_int_vector = [&](std::string const& key, std::vector<int> const& expected) {
        auto token = words(header.value(key));
        if (token.size() != expected.size()) reject(key + " length mismatch");
        for (std::size_t i = 0; i < token.size(); ++i) {
            if (signed_integer(token[i], key.c_str()) != expected[i]) reject(key + " mismatch");
        }
    };
    auto check_real_vector = [&](std::string const& key, std::vector<amrex::Real> const& expected) {
        auto token = words(header.value(key));
        if (token.size() != expected.size()) reject(key + " length mismatch");
        for (std::size_t i = 0; i < token.size(); ++i) {
            if (real_number(token[i], key.c_str()) != expected[i]) reject(key + " mismatch");
        }
    };

    auto const& domain = m_geom.Domain();
    std::vector<int> expected_domain;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) expected_domain.push_back(domain.smallEnd(d));
    for (int d = 0; d < AMREX_SPACEDIM; ++d) expected_domain.push_back(domain.bigEnd(d));
    check_int_vector("domain", expected_domain);
    std::vector<amrex::Real> prob_lo, prob_hi, dx;
    std::vector<int> periodic;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        prob_lo.push_back(m_geom.ProbLo(d)); prob_hi.push_back(m_geom.ProbHi(d));
        dx.push_back(m_dx[d]); periodic.push_back(m_geom.isPeriodic(d));
    }
    check_real_vector("prob_lo", prob_lo);
    check_real_vector("prob_hi", prob_hi);
    check_real_vector("dx", dx);
    check_int_vector("periodic", periodic);
    same_integer(header.value("box_count"), m_ba.size(), "box count");
    for (int index = 0; index < m_ba.size(); ++index) {
        std::vector<int> expected_box;
        auto const& box = m_ba[index];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) expected_box.push_back(box.smallEnd(d));
        for (int d = 0; d < AMREX_SPACEDIM; ++d) expected_box.push_back(box.bigEnd(d));
        check_int_vector("box_" + std::to_string(index), expected_box);
    }

    same_integer(header.value("boundary_enabled"), static_cast<int>(m_boundary.enabled), "boundary enabled");
    for (int slot = 0; slot < 2 * AMREX_SPACEDIM; ++slot) {
        auto token = words(header.value("bc_" + std::to_string(slot)));
        auto const& side = m_boundary.sides[slot];
        if (token.size() != 3 || token[0] != cartesian_bc_name(side.velocity) ||
            signed_integer(token[1], "legacy BC") != side.legacy_code ||
            real_number(token[2], "BC pressure") != side.pressure) reject("BC mismatch");
    }
    same(header.value("inlet_profile"), m_boundary.inlet_profile, "inlet profile");
    same_real(header.value("inlet_target_flux"), m_boundary.inlet_target_flux, "inlet target flux");
    same_real(header.value("profile_offset"), m_boundary.profile_offset, "profile offset");
    same_real(header.value("profile_slope_0"), m_boundary.profile_slope_0, "profile slope 0");
    same_real(header.value("profile_slope_1"), m_boundary.profile_slope_1, "profile slope 1");
    same_integer(header.value("constrain_outlet_flux"), static_cast<int>(m_boundary.constrain_outlet_flux),
                 "outlet flux constraint");
    if (disk_schema_version >= 2) {
        check_real_vector("moving_wall_velocity", {
            m_boundary.moving_wall_velocity[0], m_boundary.moving_wall_velocity[1],
            m_boundary.moving_wall_velocity[2]});
    }

    std::vector<Payload> payloads = {
        {"P", "cell", 1, &m_p}, {"Phi", "cell", 1, &m_phi},
        {"Nvert", "cell", 1, &m_nvert},
        {"Ucat", "cell", AMREX_SPACEDIM, &m_ucat},
        {"Ucat_old", "cell", AMREX_SPACEDIM, &m_ucat_old},
        {"Ucat_older", "cell", AMREX_SPACEDIM, &m_ucat_older}
    };
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        std::string axis(1, static_cast<char>('x' + dir));
        payloads.push_back({"Ucont_" + axis, axis + "-face", 1, &m_ucont[dir]});
        payloads.push_back({"Ucont_" + axis + "_old", axis + "-face", 1, &m_ucont_old[dir]});
        payloads.push_back({"Ucont_" + axis + "_older", axis + "-face", 1, &m_ucont_older[dir]});
    }
    same_integer(header.value("payload_count"), payloads.size(), "payload count");
    std::vector<std::filesystem::path> files;
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        auto token = words(header.value("payload_" + std::to_string(index)));
        auto const& payload = payloads[index];
        std::string expected_file = "Level_0/" + payload.name;
        if (token.size() != 5 || token[0] != payload.name || token[1] != payload.location ||
            signed_integer(token[2], "payload components") != payload.components ||
            signed_integer(token[3], "payload ghost width") != m_nghost || token[4] != expected_file) {
            reject("payload manifest mismatch for " + payload.name);
        }
        files.push_back(root / expected_file);
    }
    header.end();

    // Validate every VisMF header before reading any payload into live state.
    amrex::IntVect expected_ngrow(AMREX_D_DECL(m_nghost, m_nghost, m_nghost));
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        if (!amrex::VisMF::Exist(files[index].string())) reject("missing payload " + payloads[index].name);
        amrex::VisMF disk(files[index].string());
        if (disk.nComp() != payloads[index].components || disk.nGrowVect() != expected_ngrow ||
            disk.boxArray() != payloads[index].data->boxArray()) {
            reject("VisMF layout/components/ghost mismatch for " + payloads[index].name);
        }
    }
    for (std::size_t index = 0; index < payloads.size(); ++index) {
        amrex::VisMF::Read(*payloads[index].data, files[index].string());
    }

    m_time = restored_time;
    m_step = restored_step;
    m_history_depth = restored_history;
    mark_valid_modified();
    if (m_boundary.enabled) apply_boundary_pipeline("checkpoint-restart");
    else fill_ghost_cells();
    amrex::Print() << "AVWiS P8 checkpoint restored: " << path
                   << " time=" << m_time << " step=" << m_step
                   << " history_depth=" << m_history_depth << "\n";
}
