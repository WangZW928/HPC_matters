#ifndef BASE_TESTS_2DHEAT_BUFFER_H_
#define BASE_TESTS_2DHEAT_BUFFER_H_

#include "Array2DView.H"
#include "Backend.H"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

enum class MemorySpace {
    Host,
    Device
};

inline constexpr MemorySpace defaultMemorySpace () noexcept
{
#if defined(__CUDACC__)
    return MemorySpace::Device;
#else
    return MemorySpace::Host;
#endif
}

template <class T>
class Buffer
{
public:
    Buffer () = default;

    Buffer (int an, MemorySpace aspace)
        : m_n(an), m_space(aspace)
    {
        allocate();
    }

    ~Buffer () { release(); }

    Buffer (Buffer const&) = delete;
    Buffer& operator= (Buffer const&) = delete;


    // NOTE:
    // This move ctor is required by the current solver flow.
    // main.cpp uses std::swap(u_old, u_new) for Jacobi double-buffering.
    // Because copy operations are deleted, swap relies on move semantics.
    // Do not remove this unless you also redesign swap/buffering logic.
    Buffer (Buffer&& rhs) noexcept
        : m_ptr(rhs.m_ptr), m_n(rhs.m_n), m_space(rhs.m_space)
    {
        rhs.m_ptr = nullptr;
        rhs.m_n = 0;
        rhs.m_space = MemorySpace::Host;
    }

    // NOTE:
    // Same reason as move ctor: std::swap(Buffer, Buffer) needs move assignment
    // for a non-copyable type. Removing this will break compilation where swap
    // is used for old/new buffer rotation.
    Buffer& operator= (Buffer&& rhs) noexcept
    {
        if (this != &rhs) {
            release();
            m_ptr = rhs.m_ptr;
            m_n = rhs.m_n;
            m_space = rhs.m_space;
            rhs.m_ptr = nullptr;
            rhs.m_n = 0;
            rhs.m_space = MemorySpace::Host;
        }
        return *this;
    }

    T* data () noexcept { return m_ptr; }
    T const* data () const noexcept { return m_ptr; }
    int size () const noexcept { return m_n; }
    MemorySpace space () const noexcept { return m_space; }

    Array2DView<T> view2D (int nx, int ny, int ng) const noexcept
    {
        int total_nx = nx + 2 * ng;
        return Array2DView<T>{m_ptr, -ng, nx + ng - 1, -ng, ny + ng - 1, total_nx};
    }

    void fillZero ()
    {
#if defined(__CUDACC__)
        if (m_space == MemorySpace::Device) {
            cudaMemset(m_ptr, 0, static_cast<std::size_t>(m_n) * sizeof(T));
            return;
        }
#endif
        for (int i = 0; i < m_n; ++i) {
            m_ptr[i] = T{};
        }
    }

    std::vector<T> copyToHostVector () const
    {
        std::vector<T> host(static_cast<std::size_t>(m_n));
#if defined(__CUDACC__)
        if (m_space == MemorySpace::Device) {
            cudaMemcpy(host.data(), m_ptr, static_cast<std::size_t>(m_n) * sizeof(T), cudaMemcpyDeviceToHost);
            return host;
        }
#endif
        for (int i = 0; i < m_n; ++i) {
            host[static_cast<std::size_t>(i)] = m_ptr[i];
        }
        return host;
    }

private:
    void allocate ()
    {
        if (m_n <= 0) { return; }
#if defined(__CUDACC__)
        if (m_space == MemorySpace::Device) {
            cudaMalloc(&m_ptr, static_cast<std::size_t>(m_n) * sizeof(T));
            return;
        }
#else
        if (m_space == MemorySpace::Device) {
            throw std::runtime_error("Device memory requires CUDA compilation");
        }
#endif
        m_ptr = new T[static_cast<std::size_t>(m_n)];
    }

    void release ()
    {
        if (!m_ptr) { return; }
#if defined(__CUDACC__)
        if (m_space == MemorySpace::Device) {
            cudaFree(m_ptr);
        } else {
            delete[] m_ptr;
        }
#else
        delete[] m_ptr;
#endif
        m_ptr = nullptr;
        m_n = 0;
    }

    T* m_ptr = nullptr;
    int m_n = 0;
    MemorySpace m_space = MemorySpace::Host;
};

#endif
