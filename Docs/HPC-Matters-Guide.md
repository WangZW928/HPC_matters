# HPC Matters — Personal Learning Notes

## Overview

`HPC_matters` is a personal study repository for high-performance computing, scientific machine learning, C++, CUDA, and numerical methods. It is not a formal course package. The notes are written as a learning trail: first build intuition with small numerical and PINN examples, then move into C++ systems topics, and finally explore CUDA performance through focused microbenchmarks and a larger SPH simulation.

The repository has three main learning tracks. The CUDA track has since been moved to the standalone repository [`A-simply-CUDA-tutorial`](https://github.com/WangZW928/A-simply-CUDA-tutorial); the historical paths below identify the corresponding projects there.

- `lessons/`: Physics-Informed Neural Networks, numerical PDE examples, conservation constraints, and Hamiltonian systems.
- `C++/`: C++ memory layout, memory pools, allocators, pointer basics, and template metaprogramming.
- `A-simply-CUDA-tutorial/` (standalone repository): CUDA execution, streams, graphs, memory coalescing, bank conflicts, register pressure, occupancy, warp scheduling, profiling plans, and CUDA examples.

## Repository Structure

```text
HPC_matters/
├── lessons/
│   ├── section2.md
│   ├── section3.md
│   ├── section3_pinn_heat1d.ipynb
│   ├── section4.md
│   ├── section4_respinn_heat1d.ipynb
│   ├── section5_conservation_pinn_advection.ipynb
│   ├── section6_conservation_pinn_burgers.ipynb
│   ├── section6_conservation_pinn_burgers.py
│   ├── section7_double_pendulum_hamiltonian.ipynb
│   ├── section7_double_pendulum_hamiltonian.py
│   ├── build_section_notebooks.py
│   ├── record.md
│   └── requirements.txt
├── C++/
│   ├── memory_pool/
│   ├── alignment_demo/
│   ├── templates/
│   └── TheCherno/pointer/
└── A-simply-CUDA-tutorial/  (standalone repository)
│   ├── cuda_stream_intro/
│   ├── cuda_graph_intro/
│   ├── memory_coalescing_intro/
│   ├── shared_memory_bank_conflict/
│   ├── register_Occupancy/
│   ├── warp_schedule/
│   ├── earth_moon_sph/
│   ├── nsight_compute_intro/
│   ├── nsight_systems_intro/
│   ├── kernel_type_playground/
│   └── reduction_scan_intro/
└── CPU/nand2Tetris/
```

## Part 1: PINN & Scientific Computing (`lessons/`)

### Section 2: Heat Equation, FDM, and PINN Motivation

`section2.md` introduces the one-dimensional heat equation:

```math
u_t = \alpha u_{xx}, \quad x \in [0,L], \ t > 0
```

with a typical initial condition `u(x,0)=sin(pi x)` and zero Dirichlet boundaries. It explains why computers need discretization, then derives the explicit finite-difference update:

```math
u_i^{n+1}=u_i^n+r(u_{i+1}^n-2u_i^n+u_{i-1}^n),
\quad r=\alpha\Delta t/\Delta x^2
```

The main concept is the bridge from continuous PDEs to numerical arrays, and then from numerical residuals to PINN loss functions.

### Section 3: First Working PINN for 1D Heat

Files: `section3.md`, `section3_pinn_heat1d.ipynb`, `section3_pinn_heat1d.executed.ipynb`.

Physics problem:

```math
u_t=\alpha u_{xx}, \quad x,t \in [0,1], \quad \alpha=0.1
```

with:

```math
u(x,0)=\sin(\pi x), \quad u(0,t)=u(1,t)=0
```

and analytic reference:

```math
u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x)
```

Key PINN concepts:

- Represent the unknown function by `u_theta(x,t)`.
- Use automatic differentiation for `u_t`, `u_x`, and `u_xx`.
- Define the PDE residual `f = u_t - alpha * u_xx`.
- Combine IC, BC, and PDE losses.
- Evaluate by heatmaps, time slices, and relative L2 error.

Representative model snippet:

```python
class PINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_hidden=3,
                 dropout_rate=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))
```

Representative residual and training pattern:

```python
def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

def pde_residual(model, x, t, alpha):
    u = model(x, t)
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_xx = gradients(u_x, x)
    return u_t - alpha * u_xx

loss = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_f * loss_f
loss.backward()
optimizer.step()
```

Libraries used: PyTorch, NumPy, Matplotlib. The repository uses PyTorch for all neural-network experiments; TensorFlow is not used in the current lessons.

### Section 4: Res-PINN for 1D Heat

Files: `section4.md`, `section4_respinn_heat1d.ipynb`.

This section keeps the heat-equation problem, sampling, and loss definition from Section 3, but compares a plain MLP PINN with a residual-network PINN.

Important distinction:

- PDE residual: `f_theta = u_t - alpha u_xx`, a physics constraint.
- ResNet residual connection: `h_{l+1}=h_l+F_l(h_l)`, an optimization aid.

Representative residual block:

```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim=64, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(residual + out)
```

The experiment compares total loss, IC loss, BC loss, PDE loss, training time, relative L2 error, heatmaps, and time slices. The pedagogical conclusion is careful: residual connections may improve optimization stability, but they do not mathematically guarantee better PDE accuracy.

### Section 5: Conservation PINN for Linear Advection

File: `section5_conservation_pinn_advection.ipynb`.

Physics problem:

```math
u_t + c u_x = 0, \quad x,t \in [0,1]
```

with periodic boundary conditions:

```math
u(0,t)=u(1,t), \quad u_x(0,t)=u_x(1,t)
```

and initial condition:

```math
u(x,0)=1+0.5\sin(2\pi x)
```

The exact solution is a translation:

```math
u(x,t)=1+0.5\sin(2\pi(x-ct))
```

The new concept is conservation of mass:

```math
M(t)=\int_0^1 u(x,t)\,dx = M(0)
```

Conservation loss is approximated by a uniform quadrature grid:

```python
def conservation_mass(model, cons_x_t, t_value):
    t = torch.ones_like(cons_x_t) * t_value
    return torch.mean(model(cons_x_t, t))

def conservation_loss(model, cons_x_t, cons_t_t, M0):
    mass_errors = []
    for t_value in cons_t_t:
        mass = conservation_mass(model, cons_x_t, t_value)
        mass_errors.append((mass - M0) ** 2)
    return torch.mean(torch.stack(mass_errors))
```

This notebook sets up a baseline PINN and a Conservation-PINN using the same MLP, same training points, and same optimizer. Metrics include relative L2 error and conservation drift `|M_theta(t)-M0|`, including long-time extrapolation beyond the training interval.

### Section 6: Conservation PINN for Viscous Burgers

Files: `section6_conservation_pinn_burgers.ipynb`, `section6_conservation_pinn_burgers.py`, `section6_conservation_pinn_burgers.md`.

Physics problem:

```math
u_t + u u_x = \nu u_{xx}, \quad x \in [0,1], \quad t \in [0,2]
```

also written in conservative form:

```math
u_t+\left(\frac{u^2}{2}\right)_x=\nu u_{xx}
```

with periodic boundary conditions and `u(x,0)=1+0.5 sin(2 pi x)`. The code trains on `t in [0,1]` and evaluates out to `t in [0,2]`.

The reference solution is generated by a periodic pseudo-spectral solver with RK4:

```python
wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
flux_hat = np.fft.fft(0.5 * values**2)
flux_x = np.fft.ifft(1j * wave_numbers * flux_hat).real
u_xx = np.fft.ifft(-(wave_numbers**2) * u_hat).real
return -flux_x + nu * u_xx
```

PINN residual:

```python
def burgers_residual(model, x, t, nu):
    u = model(x, t)
    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_xx = gradient(u_x, x)
    return u_t + u * u_x - nu * u_xx
```

Training compares:

- Baseline PINN with `lambda_cons=0`.
- Conservation-PINN with `lambda_cons>0`.

The code includes device selection for MPS, CUDA, and CPU, but uses float32 for PINN training because MPS does not support the needed float64 path well.

### Section 7: Double Pendulum Hamiltonian and Canonical Transformations

Files: `section7_double_pendulum_hamiltonian.ipynb`, `section7_double_pendulum_hamiltonian.py`.

This section moves from PINNs to Hamiltonian mechanics. It studies a planar double pendulum in canonical variables:

```math
z=(q_1,q_2,p_1,p_2)^T
```

The code uses PyTorch automatic differentiation to compute Hamilton's equations:

```python
def hamiltonian_rhs(state, energy_function, params):
    z = torch.tensor(state, dtype=torch.float64, device="cpu", requires_grad=True)
    energy = energy_function(z, params)
    grad_h = torch.autograd.grad(energy, z)[0]
    dzdt = torch.cat((grad_h[2:], -grad_h[:2]))
    return dzdt.detach().numpy()
```

The canonical transformation is generated by:

```math
F_2(q,P)=q_1P_1+(q_2-q_1)P_2
```

which gives:

```math
Q_1=q_1,\quad Q_2=q_2-q_1,\quad P_1=p_1+p_2,\quad P_2=p_2
```

The code verifies the symplectic condition `A^T J A = J`, integrates both coordinate systems with RK4, maps the transformed result back to original coordinates, and compares state agreement and Hamiltonian drift. This section intentionally uses CPU float64 because tiny energy drift is more important than GPU speed.

## Part 2: C++ HPC Learning

### Memory Pool

Directory: `C++/memory_pool/`.

The memory-pool project is a header-oriented C++ learning library with examples. It demonstrates the core HPC idea of replacing many small heap allocations with pooled allocation patterns: allocate a larger region up front, then satisfy repeated smaller allocations from that region with predictable, low-overhead bookkeeping.

Components:

- `FixedBlockMemoryPool`: one large aligned buffer split into equal blocks and managed through a free list.
- `ObjectPool<T>`: type-safe object construction/destruction over `FixedBlockMemoryPool`, with RAII `unique_ptr` handles.
- `LinearMemoryPool`: bump-pointer allocator with `reset()` for phase-based allocation.
- `PoolAllocator<T>` and `PoolVector<T>`: STL-compatible allocator using `LinearMemoryPool`.
- `CudaDeviceBuffer<T>` and `CudaManagedAllocator<T>`: optional CUDA helpers enabled by `MEMORY_POOL_ENABLE_CUDA`.

Core fixed-block allocation pattern:

```cpp
void* allocate() {
    if (free_list_ == nullptr) throw std::bad_alloc{};
    FreeBlock* block = free_list_;
    free_list_ = free_list_->next;
    ++used_count_;
    return block;
}

void deallocate(void* ptr) {
    if (ptr == nullptr) return;
    auto* block = static_cast<FreeBlock*>(ptr);
    block->next = free_list_;
    free_list_ = block;
    --used_count_;
}
```

This is the central memory-pool trick: an unused block is not holding a live object, so the pool temporarily treats the first bytes of that block as a `FreeBlock` node. Allocation pops the list head. Deallocation pushes the returned block back onto the list. This gives constant-time block reuse for fixed-size allocations.

Typed object construction:

```cpp
template <typename... Args>
T* create(Args&&... args) {
    void* raw = memory_pool_.allocate();
    try {
        return new (raw) T(std::forward<Args>(args)...);
    } catch (...) {
        memory_pool_.deallocate(raw);
        throw;
    }
}
```

`ObjectPool<T>` separates raw storage from object lifetime:

- `FixedBlockMemoryPool::allocate()` gives raw bytes.
- Placement `new` constructs a real `T` object in those bytes.
- `destroy()` calls `ptr->~T()` and returns the block to the pool.
- `make_handle()` wraps the object in `std::unique_ptr<T, Deleter>` so destruction is automatic.

The CPU demo shows that a destroyed object's block is reused:

```text
after p1.reset(): used = 1 blocks
after creating p3:
p1 old address = ...
p3 address     = ... same old address ...
```

That reuse is exactly why pools matter in HPC-style code. Instead of asking the general heap for thousands or millions of tiny allocations, the program pays for a larger buffer once and then reuses addresses with simple local bookkeeping. This reduces allocator overhead, reduces fragmentation pressure, and can improve cache locality when recently freed blocks are reused soon.

The linear allocator demonstrates a different lifetime model:

```cpp
void* allocate(std::size_t bytes, std::size_t alignment) {
    auto* base = static_cast<std::byte*>(buffer_);
    std::uintptr_t current = reinterpret_cast<std::uintptr_t>(base + offset_);
    const std::size_t adjustment = alignment_adjustment(current, alignment);

    if (offset_ + adjustment + bytes > capacity_) {
        throw std::bad_alloc{};
    }

    offset_ += adjustment;
    void* result = base + offset_;
    offset_ += bytes;
    return result;
}
```

`LinearMemoryPool` is a bump-pointer allocator. It does not recycle individual allocations. Instead, `deallocate()` is intentionally a no-op and `reset()` releases everything at once by moving the offset back to zero. This fits phase-based workloads: allocate temporary arrays during one simulation step, finish the step, then reset the whole arena.

`PoolAllocator<T>` adapts `LinearMemoryPool` to STL containers:

```cpp
template <typename T>
class PoolAllocator {
public:
    using value_type = T;

    explicit PoolAllocator(std::shared_ptr<LinearMemoryPool> pool) noexcept
        : pool_(std::move(pool)) {}

    T* allocate(std::size_t n) {
        if (pool_ == nullptr) {
            throw std::bad_alloc{};
        }

        void* raw = pool_->allocate(n * sizeof(T), alignof(T));
        return static_cast<T*>(raw);
    }

    void deallocate(T* ptr, std::size_t n) noexcept {
        pool_->deallocate(ptr, n * sizeof(T));
    }
};

template <typename T>
using PoolVector = std::vector<T, PoolAllocator<T>>;
```

The important idea is that `std::vector` still owns object construction, destruction, size, capacity, and iteration semantics. The custom allocator only changes where the vector gets raw storage.

#### CUDA support in this project

The CUDA code is optional. `CMakeLists.txt` builds `memory_pool_cuda_demo` only when a CUDA compiler is available and defines `MEMORY_POOL_ENABLE_CUDA` for that target:

```cmake
add_executable(memory_pool_cuda_demo src/cuda_demo.cu)
target_include_directories(memory_pool_cuda_demo PRIVATE include)
target_compile_features(memory_pool_cuda_demo PRIVATE cuda_std_17)
target_compile_definitions(memory_pool_cuda_demo PRIVATE MEMORY_POOL_ENABLE_CUDA)
```

The CUDA demo squares integer arrays with a simple kernel:

```cpp
__global__ void square_kernel(int* values, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        values[index] = values[index] * values[index];
    }
}
```

The important design warning from the README is that GPU memory is not automatically normal CPU memory. `cudaMalloc` device pointers are excellent for kernels, but they are not valid backing storage for ordinary CPU-side `std::vector` access. Use `CudaDeviceBuffer<T>` for kernel-facing buffers and `CudaManagedVector<T>` only when unified memory semantics are intended.

#### GPU memory types comparison

CUDA exposes several memory allocation models. They are easy to confuse because each returns a pointer, but the pointer means different things depending on where the physical memory lives and which processor may touch it directly.

| Memory type | Allocation API | Where it physically lives | CPU direct read/write? | GPU access model | Pool relationship |
|---|---|---|---|---|---|
| Device memory | `cudaMalloc` | GPU VRAM | No. CPU must use `cudaMemcpy` or related transfer APIs. | Fastest normal kernel access, with no PCIe transfer during compute. | Best backing store for GPU-side pools and kernel-facing buffers. Pool metadata is usually managed on the host. |
| Pinned/page-locked host memory | `cudaMallocHost` or `cudaHostAlloc` | Host RAM, locked so the OS cannot swap it out | Yes. CPU can read/write like normal host memory. | GPU copy engines can DMA directly without a pageable-memory staging copy. | Best for transfer/staging pools, especially repeated async host-to-device and device-to-host transfers. |
| Unified/managed memory | `cudaMallocManaged` | Migrates between CPU and GPU on demand | Yes, after required synchronization when GPU work has written it. | GPU can access the same pointer, but page faults and migration may occur. | Good for learning, prototypes, and complex access patterns; less predictable than explicit device memory plus pinned staging. |

##### Device memory: `cudaMalloc`

Device memory is allocated in GPU VRAM. Kernels can read/write it efficiently, and during kernel execution there is no PCIe transfer overhead because the data is already on the GPU. The CPU cannot directly dereference this pointer. Host code must use `cudaMemcpy`, `cudaMemcpyAsync`, kernels, or other CUDA APIs to move or operate on the data.

This project's `CudaDeviceBuffer<T>` is a small RAII wrapper around `cudaMalloc`/`cudaFree`:

```cpp
template <typename T>
class CudaDeviceBuffer {
public:
    explicit CudaDeviceBuffer(std::size_t size)
        : size_(size) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
    }

    ~CudaDeviceBuffer() {
        cudaFree(data_);
    }

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    std::size_t size() const {
        return size_;
    }

    std::size_t bytes() const {
        return size_ * sizeof(T);
    }

private:
    std::size_t size_{0};
    T* data_{nullptr};
};
```

The demo uses it in the explicit-copy style:

```cpp
memory_pool::CudaDeviceBuffer<int> device_values(size);

memory_pool::check_cuda(cudaMemcpy(device_values.data(),
                                   host_in.data(),
                                   device_values.bytes(),
                                   cudaMemcpyHostToDevice));

square_kernel<<<1, 32>>>(device_values.data(), size);
memory_pool::check_cuda(cudaGetLastError());
memory_pool::check_cuda(cudaDeviceSynchronize());

memory_pool::check_cuda(cudaMemcpy(host_out.data(),
                                   device_values.data(),
                                   device_values.bytes(),
                                   cudaMemcpyDeviceToHost));
```

Use this model when the data is kernel-facing and performance matters. Do not use `cudaMalloc` memory as a CPU-side `std::vector` backing store: the CPU cannot safely load/store through that pointer as ordinary RAM.

##### Pinned/page-locked host memory: `cudaMallocHost`

Pinned memory is allocated in host RAM, but the pages are locked. The OS cannot page them out to disk. The CPU can read/write the pointer directly, like normal host memory.

The GPU still does not treat plain pinned memory as normal VRAM. Its main value is transfer efficiency: the CUDA driver and GPU copy engine can DMA directly between pinned host memory and device memory. This is what enables useful `cudaMemcpyAsync` overlap with streams.

Pinned memory has costs:

- Allocation/deallocation is more expensive than ordinary `malloc` or `new`.
- Overuse reduces memory available to the OS page cache and can hurt the whole system.
- It should usually be reserved for transfer buffers, not for every host-side allocation.

The project does not currently include a pinned host wrapper, but it would fit naturally beside `CudaDeviceBuffer<T>`:

```cpp
template <typename T>
class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size)
        : size_(size) {
        memory_pool::check_cuda(cudaMallocHost(reinterpret_cast<void**>(&data_),
                                               size_ * sizeof(T)));
    }

    ~PinnedHostBuffer() {
        cudaFreeHost(data_);
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    std::size_t size() const {
        return size_;
    }

    std::size_t bytes() const {
        return size_ * sizeof(T);
    }

private:
    std::size_t size_{0};
    T* data_{nullptr};
};
```

Pinned memory is closely related to streams:

```cpp
cudaMemcpyAsync(device.data(),
                pinned_host.data(),
                pinned_host.bytes(),
                cudaMemcpyHostToDevice,
                stream);
```

With pinned memory, the async copy can be queued in a stream and performed by the GPU copy engine while other stream work runs, if the hardware and workload permit overlap.

Without pinned memory, ordinary pageable host memory forces the CUDA runtime to stage the transfer through an internal pinned buffer. In practice:

- `cudaMemcpy` blocks the host until the synchronous copy completes.
- `cudaMemcpyAsync` from pageable memory may still block or perform hidden staging before the queued device-side copy can proceed.
- The result is less reliable overlap and extra synchronization/copy overhead.

##### Unified/managed memory: `cudaMallocManaged`

Unified memory returns one pointer that both CPU and GPU code can use. This simplifies programming because there is no explicit `cudaMemcpy` in the basic version. The tradeoff is that physical pages migrate on demand.

The project's `CudaManagedAllocator<T>` adapts `cudaMallocManaged` to STL containers:

```cpp
template <typename T>
class CudaManagedAllocator {
public:
    using value_type = T;

    CudaManagedAllocator() noexcept = default;

    template <typename U>
    CudaManagedAllocator(const CudaManagedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        cudaFree(ptr);
    }

    template <typename U>
    bool operator==(const CudaManagedAllocator<U>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const CudaManagedAllocator<U>&) const noexcept {
        return false;
    }
};

template <typename T>
using CudaManagedVector = std::vector<T, CudaManagedAllocator<T>>;
```

The demo uses managed memory like a vector, then launches the kernel on the vector's data:

```cpp
memory_pool::CudaManagedVector<int> values;
values.reserve(8);
for (int i = 0; i < 8; ++i) {
    values.push_back(i + 1);
}

square_kernel<<<1, 32>>>(values.data(), static_cast<int>(values.size()));
memory_pool::check_cuda(cudaGetLastError());
memory_pool::check_cuda(cudaDeviceSynchronize());

std::cout << "managed vector values:";
for (int value : values) {
    std::cout << ' ' << value;
}
```

The `cudaDeviceSynchronize()` is not optional here. After a kernel writes managed memory, the CPU must wait for the kernel to finish before reading those values on the host. Without synchronization, the CPU may race GPU work or observe incomplete results.

Managed memory migration model:

- First GPU access to a CPU-resident page can fault and migrate that page to the GPU.
- First CPU access after GPU work can migrate the page back to the CPU.
- These page faults add latency and make performance less predictable.
- `cudaMemPrefetchAsync(ptr, bytes, device, stream)` can move pages to the GPU before the kernel needs them.
- `cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream)` can move pages back toward CPU access.
- `cudaMemAdvise` can provide hints such as preferred location or expected access patterns.
- Managed memory can oversubscribe GPU VRAM: the program may allocate more managed memory than physical VRAM, with the system migrating/evicting pages. This is convenient, but it can be much slower than keeping the working set resident.

Managed memory is therefore best for prototypes, learning, and algorithms where access patterns are complex or inconvenient to express with explicit copies. For production performance, explicit device memory plus pinned staging buffers is usually easier to reason about.

#### Memory transfer patterns

| Pattern | Host Memory | GPU Memory | API |
|---|---|---|---|
| Synchronous copy | Pageable | Device | `cudaMemcpy` on the default stream |
| Async copy | Pinned | Device | `cudaMemcpyAsync` + explicit stream |
| Zero-copy | Pinned + mapped | Direct access from GPU | `cudaHostAlloc` + `cudaHostGetDevicePointer` |
| Managed | Unified | Unified | `cudaMallocManaged` + optional prefetch |

Zero-copy is different from ordinary pinned staging. With mapped pinned memory, the GPU can obtain a device pointer to host memory. This can be useful for small or latency-sensitive access, but bandwidth and latency are typically worse than real device VRAM. For heavy kernel data, copy into device memory first.

#### Memory pool integration with GPU memory

Memory pools matter even more on the GPU side because `cudaMalloc` and `cudaFree` are slow compared with simple pointer arithmetic or free-list operations. A device allocation often involves driver/runtime work, synchronization constraints, and a round trip through CUDA's allocation machinery. Calling `cudaMalloc` repeatedly in the hot path can dominate the workload.

The usual GPU pool pattern is:

```text
startup / resize point:
    cudaMalloc one large device buffer

runtime:
    sub-allocate slices from that buffer
    launch kernels using slice pointers

phase boundary:
    reset the pool, or return slices to a host-managed free list

shutdown:
    cudaFree the large device buffer
```

A linear device-memory pool is the GPU analogue of `LinearMemoryPool`: allocate a large `CudaDeviceBuffer<std::byte>` once, then hand out aligned offsets for per-frame or per-iteration temporary buffers. At the end of the frame/iteration, reset the offset. This avoids per-temporary `cudaMalloc` calls.

Conceptually:

```cpp
memory_pool::CudaDeviceBuffer<std::byte> arena(total_bytes);
std::size_t offset = 0;

void* allocate_from_device_arena(std::size_t bytes, std::size_t alignment) {
    std::uintptr_t base = reinterpret_cast<std::uintptr_t>(arena.data());
    std::uintptr_t current = base + offset;
    std::size_t adjustment = alignment_adjustment(current, alignment);
    offset += adjustment;
    void* result = arena.data() + offset;
    offset += bytes;
    return result;
}
```

That is a host-side allocator returning device pointers. Kernels can use the returned pointers, but the host owns the pool metadata.

GPU memory pool challenges:

- Device memory pointers cannot be dereferenced by CPU code.
- GPU kernels generally should not call `malloc` for ordinary high-performance allocation patterns.
- Fully device-side dynamic allocation is possible in CUDA, but it has restrictions and is usually avoided in hot kernels.
- Pool metadata must be synchronized if multiple streams or kernels may use the same pool concurrently.
- Host-side pool management is simpler: decide offsets/free blocks on the CPU, pass raw device pointers into kernels.
- Stream-aware pools must ensure a block is not reused until all queued work using that block has completed. Events are commonly used for this.

Pinned memory can also back a CPU-side pool when entries need frequent asynchronous GPU transfer. For example, a simulation might maintain a pool of page-locked staging buffers. The CPU fills one staging block while the GPU is processing another block and a stream is copying a third block.

Managed memory can back STL-style containers with fewer code changes, but it is not a magic high-performance pool. It trades explicit copy control for page migration. If a pooled object is touched alternately by CPU and GPU, managed memory may ping-pong pages between processors.

#### Relationship to the CUDA stream benchmark

The CUDA stream intro project demonstrates why pinned host memory matters. In `A-simply-CUDA-tutorial/cuda_stream_intro/src/stream_bench.cu`, host arrays are allocated with `cudaMallocHost`:

```cpp
HostBuffers host;
CUDA_CHECK(cudaMallocHost(&host.a, total_bytes));
CUDA_CHECK(cudaMallocHost(&host.b, total_bytes));
CUDA_CHECK(cudaMallocHost(&host.out, total_bytes));
```

The default-stream path uses synchronous copies and kernels in sequence:

```cpp
CUDA_CHECK(cudaMemcpy(dev.a0, host.a, chunk_bytes, cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(dev.b0, host.b, chunk_bytes, cudaMemcpyHostToDevice));
vector_add<<<blocks, threads>>>(dev.a0, dev.b0, dev.c0, chunk_elems, iters);
CUDA_CHECK(cudaMemcpy(host.out, dev.c0, chunk_bytes, cudaMemcpyDeviceToHost));
```

The two-stream path queues async copies and kernels into two independent streams:

```cpp
CUDA_CHECK(cudaMemcpyAsync(dev.a0, host.a, chunk_bytes, cudaMemcpyHostToDevice, s0));
CUDA_CHECK(cudaMemcpyAsync(dev.b0, host.b, chunk_bytes, cudaMemcpyHostToDevice, s0));
vector_add<<<blocks, threads, 0, s0>>>(dev.a0, dev.b0, dev.c0, chunk_elems, iters);
CUDA_CHECK(cudaMemcpyAsync(host.out, dev.c0, chunk_bytes, cudaMemcpyDeviceToHost, s0));

CUDA_CHECK(cudaMemcpyAsync(dev.a1, host.a + chunk_elems, chunk_bytes, cudaMemcpyHostToDevice, s1));
CUDA_CHECK(cudaMemcpyAsync(dev.b1, host.b + chunk_elems, chunk_bytes, cudaMemcpyHostToDevice, s1));
vector_add<<<blocks, threads, 0, s1>>>(dev.a1, dev.b1, dev.c1, chunk_elems, iters);
CUDA_CHECK(cudaMemcpyAsync(host.out + chunk_elems, dev.c1, chunk_bytes, cudaMemcpyDeviceToHost, s1));
```

The recorded result in `A-simply-CUDA-tutorial/cuda_stream_intro/results/summary.txt` is:

```text
speedup(default/two_streams)=1.228732
```

The CSV shows:

```text
default mean:     3.36068 ms
two_streams mean: 2.73508 ms
device: NVIDIA GeForce RTX 4060 Laptop GPU
async engines: 1
```

The mechanism is not that pinned memory makes the kernel faster. The mechanism is that pinned memory lets CUDA queue real asynchronous DMA transfers. Once transfers are stream operations, the runtime can overlap copy and compute where the hardware has room to do so. A memory-pool version of this pattern would preallocate reusable pinned staging buffers instead of repeatedly calling `cudaMallocHost`.

#### When to use what

- Device kernel data: use `cudaMalloc` device memory, represented in this project by `CudaDeviceBuffer<T>`.
- Frequent host-to-device or device-to-host transfers: use `cudaMallocHost` pinned memory for staging buffers, especially with `cudaMemcpyAsync` and streams.
- Prototype or learning code: use `cudaMallocManaged`, represented here by `CudaManagedAllocator<T>` and `CudaManagedVector<T>`.
- Production code with complex but performance-sensitive access: prefer explicit device memory plus pinned staging buffers, then measure.
- Custom CPU memory pool whose entries need async GPU transfer: consider making the pool's backing buffer pinned host memory.
- Custom GPU temporary pool: allocate one large device buffer and sub-allocate offsets from it, often with a linear reset at phase boundaries.

The short version:

```text
cudaMalloc        -> best for data kernels will actively compute on
cudaMallocHost    -> best for CPU-visible transfer/staging buffers
cudaMallocManaged -> easiest shared pointer model, but migration can cost
memory pool       -> best when allocation frequency itself becomes overhead
```
### Alignment Demo

Directory: `C++/alignment_demo/`.

This example connects C++ object layout to hardware memory access. It compares:

```cpp
struct S1 {
    char a;
    int b;
    char c;
};

struct S2 {
    int b;
    char a;
    char c;
};

struct alignas(16) Vec4Like {
    float x, y, z, w;
};
```

The lesson:

- `alignof(T)` describes preferred object address multiples.
- Padding is inserted between members and at the end of structs.
- Member order can change `sizeof`.
- `alignas(16)` is relevant to SIMD-style and HPC-friendly layouts.
- Alignment affects speed, portability, and sometimes correctness on strict architectures.

### Templates: `_Base_`

Directory: `C++/templates/_Base_/`.

This project introduces templates as compile-time code generation. It covers:

- Function templates such as `add<T>` and `max_value<T>`.
- Class templates such as `Array<T, Size>`.
- Non-type template parameters.
- Template instantiation and type deduction.
- Why templates are usually defined in headers.

Representative code:

```cpp
template <typename T, int Size>
struct Array {
    T data[Size];

    constexpr int size() const { return Size; }
    T& operator[](int index) { return data[index]; }
};
```

The project explicitly teaches that `Array<int,16>` and `Array<int,32>` are different types, and that embedded arrays live inside the object rather than involving `new/delete`.

### Templates: `_If-Then-Else_`

Directory: `C++/templates/_If-Then-Else_/`.

This project introduces compile-time selection through specialization and type traits.

Compile-time if-then-else:

```cpp
template <bool Condition, typename Then, typename Else>
struct IfThenElse {
    using type = Then;
};

template <typename Then, typename Else>
struct IfThenElse<false, Then, Else> {
    using type = Else;
};
```

Other concepts:

- Full specialization: `TypeName<int>`.
- Partial specialization: `TypeName<T*>`, `TypeName<const T>`.
- Function template full specialization.
- Prefer overloads or `if constexpr` for many function-branching cases.
- `StoragePolicy<T>` choosing value vs reference based on `sizeof(T)`.

### Templates: `_Variadic-Templates_`

Directory: `C++/templates/_Variadic-Templates_/`.

This project introduces parameter packs and fold expressions:

```cpp
template <typename... Ts>
void print_with_fold(const Ts&... values) {
    ((std::cout << values << ' '), ...);
}

template <typename... Ts>
auto sum_all(Ts... values) {
    return (values + ... + 0);
}
```

It also covers:

- Recursive pack expansion.
- `sizeof...`.
- Type-list-like compile-time structures.
- Pack-based traits:

```cpp
template <typename... Ts>
struct AllIntegral : std::bool_constant<(std::is_integral_v<Ts> && ...)> {};
```

- Perfect-forwarding-style helpers such as `make_vector<T>(Ts&&...)`.

### Pointer Basics

Directory: `C++/TheCherno/pointer/`.

This is a minimal pointer example:

```cpp
int var = 8;
void* pointer = &var;
```

It is a small seed for understanding raw addresses before the memory-pool and CUDA buffer topics.

## Part 3: CUDA Deep Dives

All measured benchmark results currently come from an NVIDIA GeForce RTX 4060 Laptop GPU unless noted otherwise.

### CUDA Stream Intro

Directory: `A-simply-CUDA-tutorial/cuda_stream_intro/`.

GPU concept: streams as GPU work queues, pinned host memory, async copies, and copy/compute overlap.

Key APIs:

- `cudaMallocHost`
- `cudaMemcpyAsync`
- `cudaStreamCreate`
- Kernel launch with stream parameter: `<<<blocks, threads, 0, stream>>>`
- `cudaStreamSynchronize`
- `cudaEventRecord`, `cudaEventElapsedTime`

Kernel:

```cpp
__global__ void vector_add(const float* a, const float* b, float* c,
                           int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = a[idx];
    float y = b[idx];
    for (int i = 0; i < iters; ++i) {
        x = x * 1.000001f + y * 0.999999f;
        y = y * 1.0000001f + 0.000001f;
    }
    c[idx] = x + y;
}
```

Measurement approach:

- Allocate pinned host buffers with `cudaMallocHost`.
- Split input into two chunks.
- Compare default-stream serial H2D, kernel, D2H sequence against two explicit streams.
- Time with CUDA events, including stream wait events to measure both stream completions.

Results:

```text
default:     3.36068 ms
two_streams: 2.73508 ms
speedup:     1.228732x
asyncEngineCount: 1
```

Performance insight: streams do not make one kernel faster. They help arrange independent work so copies and kernels can overlap when the hardware and workload allow it.

### CUDA Graph Intro

Directory: `A-simply-CUDA-tutorial/cuda_graph_intro/`.

GPU concept: reduce launch overhead for fixed repeated workflows.

Kernels:

```cpp
__global__ void add_bias(float* x, float b, int n) { ... }
__global__ void scale(float* x, float s, int n) { ... }
__global__ void relu(float* x, int n) { ... }
```

Graph path:

```cpp
cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
add_bias<<<blocks, threads, 0, s>>>(d_x, 0.1f, n);
scale<<<blocks, threads, 0, s>>>(d_x, 1.01f, n);
relu<<<blocks, threads, 0, s>>>(d_x, n);
cudaStreamEndCapture(s, &graph);
cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(graph_exec, s);
```

Measurement approach:

- Run 3 kernels per iteration.
- Compare normal launches against graph replay.
- Use CUDA events and thousands of repeats.

Results:

```text
normal: 0.0461264 ms
graph:  0.0292238 ms
speedup: 1.578385x
```

Performance insight: CUDA Graph helps when workflows are fixed and repeated, especially when small kernels make CPU launch overhead visible.

### Memory Coalescing Intro

Directory: `A-simply-CUDA-tutorial/memory_coalescing_intro/`.

GPU concept: global-memory coalescing and warp access patterns.

Kernels:

```cpp
__global__ void stride_read_kernel(const float* in, float* out,
                                   int elements, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    out[idx] = in[idx * stride] * 1.000001f + 1.0f;
}

__global__ void offset_read_kernel(const float* in, float* out,
                                   int elements, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    out[idx] = in[idx + offset] * 1.000001f + 1.0f;
}
```

Measurement approach:

- Sweep strides `1,2,4,8,16,32`.
- Sweep offsets `0..32`.
- Estimate requested bandwidth as one read plus one write per output element.

Selected stride results:

```text
stride 1:  0.033293 ms, 251.96 GB/s requested
stride 2:  0.014959 ms, 560.78 GB/s requested
stride 4:  0.017677 ms, 474.54 GB/s requested
stride 8:  0.161139 ms,  52.06 GB/s requested
stride 16: 0.162884 ms,  51.50 GB/s requested
stride 32: 0.165860 ms,  50.58 GB/s requested
```

Performance insight: large strides destroy effective memory bandwidth because adjacent warp lanes access scattered addresses. Small-stride and offset cases are not perfectly monotone because cache, warmup, measurement noise, and architecture details also matter.

### Shared Memory Bank Conflict

Directory: `A-simply-CUDA-tutorial/shared_memory_bank_conflict/`.

GPU concept: shared memory is fast but banked; warp access patterns can serialize when lanes hit the same bank.

Kernel:

```cpp
__global__ void shared_stride_kernel(float* out, int stride, int iterations) {
    __shared__ float smem[2048];
    int lane = threadIdx.x & 31;
    int index = lane * stride;
    ...
    volatile float* vsmem = smem;
    for (int i = 0; i < iterations; ++i) {
        acc += vsmem[index];
    }
    out[global] = acc;
}
```

Measurement approach:

- One warp per block (`threads_per_block=32`).
- Sweep stride `{1,2,3,4,5,8,16,32}`.
- Estimate conflict degree by `gcd(stride, 32)`.
- Use `volatile` to keep repeated shared-memory loads visible to the benchmark.

Current measured summary:

```text
stride 1:  conflict 1,  mean 0.026636 ms, relative throughput 1.000
stride 2:  conflict 2,  mean 0.028317 ms, relative throughput 0.941
stride 4:  conflict 4,  mean 0.031017 ms, relative throughput 0.859
stride 8:  conflict 8,  mean 0.031620 ms, relative throughput 0.842
stride 16: conflict 16, mean 0.029615 ms, relative throughput 0.899
stride 32: conflict 32, mean 0.023859 ms, relative throughput 1.116
```

Performance insight: this benchmark is also a lesson in measurement design. The result is not a clean conflict-degree curve; very short kernel time, standard deviation, compiler behavior, and modern shared-memory behavior can obscure the textbook pattern. The README correctly treats this as a benchmark purity issue rather than a refutation of bank conflicts.

### Register Occupancy

Directory: `A-simply-CUDA-tutorial/register_Occupancy/`.

GPU concept: per-thread register count limits active blocks/warps per SM and therefore theoretical occupancy.

Kernels:

```cpp
__global__ void kernel_low_reg(const float* a, float* out, int n, int iters) {
    float x = a[i];
    for (int t = 0; t < iters; ++t) {
        x = x * 1.00001f + 0.0001f;
    }
    out[i] = x;
}

__global__ void kernel_high_reg(const float* a, float* out, int n, int iters) {
    float tmp[HIGH_REG_TMP_SIZE];
    ...
}
```

Measurement approach:

- Query kernel attributes with `cudaFuncGetAttributes`.
- Estimate occupancy with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
- Rebuild with different `HIGH_REG_TMP_SIZE` values through `scripts/sweep_registers.sh`.

Sweep summary:

```text
tmp  regs/thread  occupancy  avg_ms
8    16           1.000000   0.025771
16   37           1.000000   0.048488
24   40           1.000000   0.077240
32   40           1.000000   0.104110
48   72           0.500000   0.116234
64   95           0.333333   0.182034
80   96           0.333333   0.190219
96   128          0.333333   0.222773
128  168          0.166667   0.303928
```

Performance insight: register pressure first increases per-thread work and eventually cuts occupancy. Runtime rises strongly as register count grows, especially once occupancy drops below full residency.

### Warp Schedule

Directory: `A-simply-CUDA-tutorial/warp_schedule/`.

GPU concept: active warp count, blocks per SM, latency hiding, and throughput saturation.

Kernel:

```cpp
__global__ void warp_stress_kernel(float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = static_cast<float>(idx) * 1e-4f;
    for (int i = 0; i < iters; ++i) {
        x = x * 1.000001f + 0.00001f;
    }
    out[idx] = x;
}
```

Measurement approach:

- Sweep `blocks_per_sm` across `{1,2,4,8,16,24,32,48,64}`.
- Sweep `warps_per_block` from `1` to `32`.
- Report runtime and throughput as `total_warps / ms`.

Top configurations:

```text
blocks/SM  warps/block  threads/block  warps/ms
64         24           768            66912.19
64         20           640            66592.67
64         16           512            66575.65
48         22           704            66389.00
48         23           736            66320.08
```

Performance insight: throughput saturates once enough warps are available to hide latency. More warps are not automatically better; the useful target is enough occupancy for the kernel type without causing unnecessary resource pressure.

### Earth-Moon SPH

Directory: `CFD/earth_moon_sph/`.

GPU concept: a larger CUDA application combining particle simulation, spatial hashing, neighbor search, sorting, and visualization.

Numerical model:

- 3D weakly compressible SPH.
- Non-dimensional units with `R_e=1`, `GM_e=1`.
- Cubic spline kernel with `h=0.055`.
- Tait equation of state with `gamma=7`.
- Symmetric pressure force.
- Uniform-grid spatial hashing.
- Thrust sorting by cell hash.
- Symplectic Euler integration.
- Differential tidal force from the Moon, with default visualization scale `TIDAL_FORCE_SCALE=1000000`.

Important constants:

```cpp
constexpr float EARTH_RADIUS = 1.0f;
constexpr float MOON_MASS_RATIO = 0.012300037f;
constexpr float MOON_ORBIT_RADIUS = 60.3f;
constexpr float SMOOTHING_LENGTH = 0.055f;
constexpr float TIME_STEP = 4.0e-4f;
constexpr int BLOCK_SIZE = 256;
```

Main CUDA kernels:

- `kernel_build_grid`: compute cell hashes and particle indices.
- `kernel_reorder`: reorder SoA particle arrays after sorting.
- `kernel_find_cell_ranges`: find sorted start/end range per grid cell.
- `kernel_compute_density_pressure`: loop over 27 neighboring cells and evaluate SPH density and pressure.
- `kernel_compute_forces`: compute SPH pressure force, Earth gravity, differential Moon tide, boundary support, damping, and safety limits.
- `kernel_integrate`: symplectic Euler update with radial clamps.

Representative neighbor-search pattern:

```cpp
for (int dz = -1; dz <= 1; ++dz)
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int3 c = make_int3(center.x + dx, center.y + dy, center.z + dz);
        if (!valid_cell(c)) continue;
        uint32_t h = cell_hash(c);
        for (uint32_t j = cell_start[h]; j < cell_end[h]; ++j) {
            ...
        }
    }
```

The host code writes binary frames with magic `SPH1`, positions, velocities, density, and pressure. Visualization is handled by:

- `visualize.py`: Matplotlib 3D display or GIF through Pillow.
- `render_gif.py`: standard-library PPM renderer plus `ffmpeg`, faster than Matplotlib animation.

Performance insight: this is the repository's most realistic CUDA example. It introduces data-oriented SoA layout, spatial partitioning, `thrust::sort_by_key`, kernel staging, and the difference between teaching visualization and quantitatively valid physical modeling.

### Nsight Systems Intro

Directory: `A-simply-CUDA-tutorial/nsight_systems_intro/`.

This is currently a planning README. It will use existing stream benchmarks to inspect timeline behavior: memcpy, kernels, stream overlap, and whether assumed concurrency actually appears in the profiler.

Performance insight: use Nsight Systems for whole-program timelines and CPU/GPU scheduling visibility.

### Nsight Compute Intro

Directory: `A-simply-CUDA-tutorial/nsight_compute_intro/`.

This is currently a planning README. It targets single-kernel analysis: occupancy, stall reasons, memory throughput, register use, and mapping profiler counters back to benchmark behavior.

Performance insight: use Nsight Compute when runtime alone is not enough and the question is why a kernel is fast or slow.

### Kernel Type Playground

Directory: `A-simply-CUDA-tutorial/kernel_type_playground/`.

This is a planned project to classify kernels as:

- compute-bound
- memory-bound
- latency-bound
- launch-overhead-bound

The intended lesson is to diagnose the bottleneck before choosing optimizations such as occupancy tuning, memory coalescing, streams, or graphs.

### Reduction and Scan Intro

Directory: `A-simply-CUDA-tutorial/reduction_scan_intro/`.

This is a planned project for CUDA parallel primitives:

- reduction
- prefix sum / scan
- shared-memory optimization
- warp shuffle
- synchronization patterns

The goal is to move from microbenchmark kernels toward reusable HPC and deep-learning building blocks.

## Learning Progression Overview

The repository has a coherent learning path:

1. Start from PDE intuition and finite differences.
2. Learn basic PINN formulation using the heat equation.
3. Improve neural architectures with residual connections.
4. Add integral physics constraints for conservation laws.
5. Move from linear advection to nonlinear Burgers.
6. Explore Hamiltonian systems and coordinate transformations.
7. Build C++ memory and type-system foundations.
8. Study CUDA runtime mechanisms: streams and graphs.
9. Study CUDA hardware behavior: coalescing, bank conflicts, registers, occupancy, warp scheduling.
10. Apply several CUDA ideas in a larger SPH simulation.
11. Prepare for profiler-driven performance analysis with Nsight Systems and Nsight Compute.

## Key HPC Concepts Mastered Checklist

- [x] PDE discretization and finite-difference thinking.
- [x] PINN loss decomposition: IC, BC, PDE residual.
- [x] Automatic differentiation for first and second derivatives.
- [x] Residual networks as optimization tools.
- [x] Conservation constraints and integral diagnostics.
- [x] Pseudo-spectral reference solvers.
- [x] Hamiltonian equations and symplectic coordinate checks.
- [x] C++ alignment, padding, and object layout.
- [x] Fixed-block memory pools and free lists.
- [x] RAII object pools with placement new and explicit destruction.
- [x] STL-compatible custom allocators.
- [x] Template instantiation, specialization, and variadic templates.
- [x] CUDA kernel indexing and launch configuration.
- [x] CUDA events for timing.
- [x] Streams, async copies, and pinned memory.
- [x] CUDA Graph capture, instantiation, and replay.
- [x] Global memory coalescing.
- [x] Shared-memory bank conflict measurement issues.
- [x] Register pressure and theoretical occupancy.
- [x] Warp count, latency hiding, and throughput saturation.
- [x] SoA particle storage, spatial hashing, and neighbor search in CUDA SPH.
- [ ] Full Nsight Systems timeline study.
- [ ] Full Nsight Compute counter-based kernel study.
- [ ] Reduction and scan implementation.

## Technologies Covered

- Python: NumPy, Matplotlib, argparse, dataclasses, CSV output.
- PyTorch: `torch.nn`, `torch.autograd.grad`, Adam optimization, device selection for CPU/CUDA/MPS.
- Jupyter notebooks: lesson-style interactive experiments.
- C++17: templates, RAII, allocators, object layout, alignment.
- CUDA C++: kernels, memory management, streams, events, graphs, shared memory, occupancy APIs.
- Thrust: `thrust::sort_by_key` for SPH grid sorting.
- CMake: C++ and CUDA project builds.
- Shell scripting: benchmark sweeps.
- Visualization: Matplotlib, Pillow, ffmpeg-based GIF rendering.
- TensorFlow: not used in the current repository; PyTorch is the neural-network framework throughout the lessons.

## CUDA API and Technique Quick Reference

Memory:

- `cudaMalloc`
- `cudaFree`
- `cudaMallocHost`
- `cudaFreeHost`
- `cudaMallocManaged`
- `cudaMemcpy`
- `cudaMemcpyAsync`
- `cudaMemset`

Execution:

- `kernel<<<blocks, threads>>>`
- `kernel<<<blocks, threads, shared_bytes, stream>>>`
- `blockIdx.x`, `blockDim.x`, `threadIdx.x`
- `__global__`
- `__device__`
- `__host__ __device__`
- `__shared__`
- `__syncthreads()`

Streams and events:

- `cudaStreamCreate`
- `cudaStreamDestroy`
- `cudaStreamSynchronize`
- `cudaStreamWaitEvent`
- `cudaEventCreate`
- `cudaEventRecord`
- `cudaEventSynchronize`
- `cudaEventElapsedTime`
- `cudaEventDestroy`

Graphs:

- `cudaStreamBeginCapture`
- `cudaStreamEndCapture`
- `cudaGraphInstantiate`
- `cudaGraphLaunch`
- `cudaGraphExecDestroy`
- `cudaGraphDestroy`

Device and occupancy:

- `cudaSetDevice`
- `cudaGetDeviceProperties`
- `cudaGetLastError`
- `cudaDeviceSynchronize`
- `cudaFuncGetAttributes`
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`

CUDA performance techniques:

- Use pinned host memory for more effective async transfer.
- Use streams to expose overlap across independent chunks.
- Use CUDA Graphs for fixed, repeated launch sequences.
- Prefer coalesced global-memory access where adjacent lanes touch adjacent addresses.
- Use shared memory carefully; bank mapping and compiler optimizations affect measured behavior.
- Track register count because it can reduce active warps per SM.
- Treat occupancy as a means to latency hiding, not a goal by itself.
- Use profiler timelines and kernel counters to verify performance explanations.

## Final Notes

This repository is strongest as a personal learning notebook because it keeps code, notes, measurements, and interpretation together. The CUDA projects are especially useful because they turn abstract performance concepts into benchmarkable artifacts. The PINN lessons similarly keep the scientific-computing thread grounded by always tying equations to losses, code, plots, and metrics.
