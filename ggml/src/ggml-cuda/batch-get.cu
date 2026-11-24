#include "batch-get.cuh"
#include "common.cuh"
#include "ggml-cpp.h"

#include <climits>
#include <cstring>
#include <mutex>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct ggml_cuda_batch_get_descriptor {
    const unsigned char * src;
    unsigned char * dst;
    size_t size;
};

static __global__ void ggml_cuda_batch_get_kernel(const ggml_cuda_batch_get_descriptor * copies, size_t count) {
    // Blocks along y cooperate on each descriptor.
    const size_t index = blockIdx.x;
    if (index >= count) {
        return;
    }
    const size_t first = (size_t) blockIdx.y * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.y * blockDim.x;
    const auto copy = copies[index];
    // Use 16-, 8-, 4-, or 2-byte copies when both pointers meet that alignment; threads copy separate chunks.
    const uintptr_t alignment = (uintptr_t) copy.src | (uintptr_t) copy.dst;
    size_t tail = 0;
    if (alignment % 16 == 0) {
        for (size_t i = first; i < copy.size / 16; i += stride) {
            ggml_cuda_memcpy_1<16>(copy.dst + 16*i, copy.src + 16*i);
        }
        tail = copy.size / 16 * 16;
    } else if (alignment % 8 == 0) {
        for (size_t i = first; i < copy.size / 8; i += stride) {
            ggml_cuda_memcpy_1<8>(copy.dst + 8*i, copy.src + 8*i);
        }
        tail = copy.size / 8 * 8;
    } else if (alignment % 4 == 0) {
        for (size_t i = first; i < copy.size / 4; i += stride) {
            ggml_cuda_memcpy_1<4>(copy.dst + 4*i, copy.src + 4*i);
        }
        tail = copy.size / 4 * 4;
    } else if (alignment % 2 == 0) {
        for (size_t i = first; i < copy.size / 2; i += stride) {
            ggml_cuda_memcpy_1<2>(copy.dst + 2*i, copy.src + 2*i);
        }
        tail = copy.size / 2 * 2;
    }
    // Copy trailing bytes, or the entire range when no wider alignment is available.
    for (size_t i = tail + first; i < copy.size; i += stride) {
        ggml_cuda_memcpy_1<1>(copy.dst + i, copy.src + i);
    }
}

#if defined(_WIN32)
// On Windows a batched read pins no host memory while it runs: it goes through a stage, memory of the host that is
// pinned and mapped for the device once and kept, and the host copies from there to where the bytes belong.
//
// Why: under WDDM the graphics memory manager builds page tables for pinned host memory inside the device's own
// memory at the moment the memory is pinned, one table of 4 KiB for every 2 MiB, and it places them only in gaps that
// it finds by rules of its own. A read that registered its destination asked for some hundred to some thousand of
// them at every call (690 for a state of 1.4 GiB). A device whose memory is mapped in pieces while the program runs
// (a lazy buffer) fills those gaps; then no place is found for a table, the manager marks the device as failed
// ("Unrecoverable page in failure", device state D3DKMT_DEVICEEXECUTION_ERROR_OUTOFMEMORY), and every call to the
// device answers "out of memory" from then on, with gigabytes of it free. A server whose sequence memory was lazy
// stopped that way in the read of a state. The stage's tables are made once, when the backend of the device is
// made, and nothing is added for a read.
struct ggml_cuda_batch_get_stage {
    std::mutex mutex; // one read of a device goes through its stage at a time

    unsigned char * host = nullptr;   // pinned and mapped: the list of the small ranges, then the bytes
    unsigned char * device = nullptr; // where the device has it
    ggml_cuda_batch_get_descriptor * device_copies = nullptr; // the list on the device
    bool tried = false;
};

// the bytes of a stage, and the small ranges of one launch of the kernel
static constexpr size_t ggml_cuda_batch_get_stage_bytes = 64ull * 1024 * 1024;
static constexpr size_t ggml_cuda_batch_get_stage_copies = 128 * 1024;
static constexpr size_t ggml_cuda_batch_get_stage_list_bytes = ggml_cuda_batch_get_stage_copies * sizeof(ggml_cuda_batch_get_descriptor);

static ggml_cuda_batch_get_stage & ggml_cuda_batch_get_stage_of(int device) {
    static ggml_cuda_batch_get_stage stages[GGML_CUDA_MAX_DEVICES];
    return stages[device];
}

// With the stage's mutex held and its device current. False when the driver does not give the memory: it is not
// asked again, and reads of the device take the plain copies.
static bool ggml_cuda_batch_get_stage_make(ggml_cuda_batch_get_stage & stage) {
    if (stage.host != nullptr) {
        return true;
    }
    if (stage.tried) {
        return false;
    }
    stage.tried = true;

    void * host = nullptr;
    cudaError_t err = cudaHostAlloc(&host, ggml_cuda_batch_get_stage_list_bytes + ggml_cuda_batch_get_stage_bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                       (ggml_cuda_batch_get_stage_list_bytes + ggml_cuda_batch_get_stage_bytes) / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }

    void * device = nullptr;
    void * device_copies = nullptr;
    if (cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess ||
            cudaMalloc(&device_copies, ggml_cuda_batch_get_stage_list_bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(host);
        (void) cudaGetLastError();
        return false;
    }

    stage.host = static_cast<unsigned char *>(host);
    stage.device = static_cast<unsigned char *>(device);
    stage.device_copies = static_cast<ggml_cuda_batch_get_descriptor *>(device_copies);
    return true;
}
#else
struct ggml_cuda_batch_get_mapping {
    void * host = nullptr;
    unsigned char * device = nullptr;

    ggml_cuda_batch_get_mapping(void * host, size_t size) {
        const cudaError_t err = cudaHostRegister(host, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
            return;
        }
        this->host = host;
        if (cudaHostGetDevicePointer((void **) &device, host, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            device = nullptr;
            return;
        }
    }

    ggml_cuda_batch_get_mapping(const ggml_cuda_batch_get_mapping &) = delete;
    ggml_cuda_batch_get_mapping & operator=(const ggml_cuda_batch_get_mapping &) = delete;

    ~ggml_cuda_batch_get_mapping() {
        if (host) {
            // Release only the registration acquired by this object.
            CUDA_CHECK(cudaHostUnregister(host));
        }
    }
};
#endif // defined(_WIN32)

// The batched read is a faster way to the same bytes, and it asks more of the driver: the device writes into pinned
// memory of the host. When the driver refuses a step of it the read declines, and the caller reads the ranges one by
// one, as it does when there is no pinned memory to be had. Before, a refusal ended the program: a server on Windows
// whose system was at its commit limit (ggml_cuda_host_commit_free) got "out of memory" for a copy into the pinned
// memory with 2 GB of the device free, and with plain copies alone it ran on at the same limit.
// What was queued before the refusal writes into the pinned memory, so it is waited for before that memory is let
// go or used again.
static bool ggml_cuda_batch_get_decline(int device, cudaError_t err, const char * what, size_t n_reads, size_t n_bytes) {
    (void) cudaGetLastError();
    (void) cudaStreamSynchronize(cudaStreamPerThread);
    (void) cudaGetLastError();

    char note[256];
    ggml_cuda_memory_note(note, sizeof(note));

    GGML_LOG_WARN("%s: the driver refused the batched read of %zu ranges, %.2f MiB, on device %d (%s: %s); %s. "
                  "The ranges are read one by one\n",
                  __func__, n_reads, n_bytes / 1024.0 / 1024.0, device, what, cudaGetErrorString(err), note);
    return false;
}
#endif

void ggml_cuda_batch_get_prepare(int device) {
#if defined(_WIN32) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_cuda_set_device(device);

    ggml_cuda_batch_get_stage & stage = ggml_cuda_batch_get_stage_of(device);
    std::lock_guard<std::mutex> lock(stage.mutex);
    (void) ggml_cuda_batch_get_stage_make(stage);
#else
    GGML_UNUSED(device);
#endif
}

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(tensors);
    GGML_UNUSED(data);
    GGML_UNUSED(offsets);
    GGML_UNUSED(sizes);
    GGML_UNUSED(n_tensors);
    return false;
#else
    // Provisional dispatch values based on experiments.
    constexpr size_t min_batch_reads = 128;
    constexpr size_t max_gather_read_bytes = 128 * 1024;
    // Collect non-empty CPU destination ranges.
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    ranges.reserve(n_tensors);
    size_t payload_bytes = 0;
    // Number of non-empty reads whose size is at most max_gather_read_bytes.
    size_t n_reads_small = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        const uintptr_t address = (uintptr_t) data[i];
        GGML_ASSERT(address <= UINTPTR_MAX - sizes[i]);
        GGML_ASSERT(payload_bytes <= SIZE_MAX - sizes[i]);
        payload_bytes += sizes[i];
        n_reads_small += sizes[i] <= max_gather_read_bytes;
        ranges.emplace_back(address, address + sizes[i]);
    }
    if (ranges.empty()) {
        return true;
    }
    // Total number of non-empty reads.
    const size_t n_reads = ranges.size();
    if (n_reads > SIZE_MAX / sizeof(ggml_cuda_batch_get_descriptor) || n_reads > INT_MAX) {
        return false;
    }
    if (n_reads < 2 || (n_reads < min_batch_reads && n_reads_small == n_reads)) {
        return false;
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        // Preserve the ordered copy path when CPU destinations overlap.
        if (ranges[i].first < ranges[i - 1].second) {
            return false;
        }
    }
#if defined(_WIN32)
    ggml_cuda_batch_get_stage & stage = ggml_cuda_batch_get_stage_of(device);
    std::lock_guard<std::mutex> lock(stage.mutex);
    if (!ggml_cuda_batch_get_stage_make(stage)) {
        return false;
    }

    auto * copies = reinterpret_cast<ggml_cuda_batch_get_descriptor *>(stage.host);
    unsigned char * const stage_host = stage.host + ggml_cuda_batch_get_stage_list_bytes;
    unsigned char * const stage_device = stage.device + ggml_cuda_batch_get_stage_list_bytes;

    // What the device was given for the stage and where its bytes belong; `used` bytes of the stage are spoken for.
    struct placed {
        void * dst;
        size_t at;
        size_t size;
    };
    std::vector<placed> group;
    size_t used = 0;

    cudaError_t err = cudaSuccess;

    // Waits for the device, then the host copies the group on and the stage is free again.
    const auto flush = [&]() {
        err = cudaStreamSynchronize(cudaStreamPerThread);
        if (err != cudaSuccess) {
            return false;
        }
        for (const placed & p : group) {
            memcpy(p.dst, stage_host + p.at, p.size);
        }
        group.clear();
        used = 0;
        return true;
    };

    // Reads selected for gather; false means all reads use DMA.
    const bool gather = n_reads_small >= min_batch_reads;
    if (gather) {
        const int threads = 256;
        const size_t nsm = ggml_cuda_info().devices[device].nsm;
        size_t i = 0;
        while (i < n_tensors) {
            // as many small ranges as the list and the stage hold
            size_t count = 0;
            for (; i < n_tensors && count < ggml_cuda_batch_get_stage_copies; ++i) {
                if (sizes[i] == 0 || sizes[i] > max_gather_read_bytes) {
                    continue;
                }
                const size_t at = GGML_PAD(used, 16);
                if (at + sizes[i] > ggml_cuda_batch_get_stage_bytes) {
                    break;
                }
                copies[count++] = {(const unsigned char *) tensors[i]->data + offsets[i], stage_device + at, sizes[i]};
                group.push_back({data[i], at, sizes[i]});
                used = at + sizes[i];
            }
            if (count == 0) {
                break;
            }
            const size_t blocks_per_copy = std::min<size_t>(UINT16_MAX, std::max<size_t>(1, (nsm + count - 1) / count));
            const dim3 blocks((unsigned int) count, (unsigned int) blocks_per_copy);
            // Stream ordering keeps the kernel behind the copy of its list; the list is written again after the wait.
            err = cudaMemcpyAsync(stage.device_copies, copies, count * sizeof(copies[0]), cudaMemcpyHostToDevice, cudaStreamPerThread);
            if (err != cudaSuccess) {
                return ggml_cuda_batch_get_decline(device, err, "the copy of the list of small ranges", n_reads, payload_bytes);
            }
            ggml_cuda_batch_get_kernel<<<blocks, threads, 0, cudaStreamPerThread>>>(stage.device_copies, count);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return ggml_cuda_batch_get_decline(device, err, "the kernel that gathers the small ranges", n_reads, payload_bytes);
            }
            if (!flush()) {
                return ggml_cuda_batch_get_decline(device, err, "the wait for the small ranges", n_reads, payload_bytes);
            }
        }
    }
    // The other ranges: copied by the device into the stage, a range that is larger than the stage in parts.
    size_t n_dma = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0 || (gather && sizes[i] <= max_gather_read_bytes)) {
            continue;
        }
        size_t done = 0;
        while (done < sizes[i]) {
            if (used == ggml_cuda_batch_get_stage_bytes && !flush()) {
                return ggml_cuda_batch_get_decline(device, err, "the wait for the copies", n_reads, payload_bytes);
            }
            const size_t n = std::min(sizes[i] - done, ggml_cuda_batch_get_stage_bytes - used);
            err = n_dma > 0 && ggml_cuda_debug_refuse("batch-get") ? cudaErrorMemoryAllocation :
                cudaMemcpyAsync(stage_host + used, (const char *) tensors[i]->data + offsets[i] + done, n, cudaMemcpyDeviceToHost, cudaStreamPerThread);
            if (err != cudaSuccess) {
                return ggml_cuda_batch_get_decline(device, err, "a copy into the pinned memory", n_reads, payload_bytes);
            }
            group.push_back({(unsigned char *) data[i] + done, used, n});
            used += n;
            done += n;
            n_dma++;
        }
    }
    if (!group.empty() && !flush()) {
        return ggml_cuda_batch_get_decline(device, err, "the wait for the copies", n_reads, payload_bytes);
    }
    return true;
#else
    // Find the CPU span to register, including gaps between destinations.
    const uintptr_t begin = ranges.front().first;
    const uintptr_t end = ranges.back().second;
    if (end - begin - payload_bytes > payload_bytes) {
        return false;
    }
    // Register and map the existing CPU destination so the kernel can write to it directly.
    ggml_cuda_batch_get_mapping mapping((void *) begin, end - begin);
    if (!mapping.host) {
        return false;
    }
    // Reads selected for gather; zero means all reads use DMA.
    size_t n_reads_gather = mapping.device && n_reads_small >= min_batch_reads ? n_reads_small : 0;
    // Keep descriptor storage alive until the final stream synchronization.
    constexpr size_t max_batch_descriptors = 128 * 1024;
    const size_t bytes = std::min(n_reads_gather, max_batch_descriptors) * sizeof(ggml_cuda_batch_get_descriptor);
    ggml_backend_buffer_ptr descriptors(n_reads_gather ? ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), bytes) : nullptr);
    if (!descriptors) {
        n_reads_gather = 0;
    }
    const auto host_type = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_ptr host_descriptors(n_reads_gather ? ggml_backend_buft_alloc_buffer(host_type, bytes) : nullptr);
    if (!host_descriptors || ggml_backend_buffer_get_type(host_descriptors.get()) != host_type) {
        n_reads_gather = 0;
    }
    cudaError_t err = cudaSuccess;
    if (n_reads_gather) {
        auto * copies = static_cast<ggml_cuda_batch_get_descriptor *>(ggml_backend_buffer_get_base(host_descriptors.get()));
        auto * device_copies = static_cast<ggml_cuda_batch_get_descriptor *>(ggml_backend_buffer_get_base(descriptors.get()));
        const int threads = 256;
        const size_t nsm = ggml_cuda_info().devices[device].nsm;
        size_t i = 0;
        for (size_t first = 0; first < n_reads_gather; first += max_batch_descriptors) {
            const size_t n_batch = std::min(n_reads_gather - first, max_batch_descriptors);
            size_t count = 0;
            for (; i < n_tensors && count < n_batch; ++i) {
                if (sizes[i] == 0 || sizes[i] > max_gather_read_bytes) {
                    continue;
                }
                copies[count++] = {(const unsigned char *) tensors[i]->data + offsets[i],
                        mapping.device + ((uintptr_t) data[i] - begin), sizes[i]};
            }
            const size_t blocks_per_copy = std::min<size_t>(UINT16_MAX, std::max<size_t>(1, (nsm + n_batch - 1) / n_batch));
            const dim3 blocks((unsigned int) n_batch, (unsigned int) blocks_per_copy);
            // Stream ordering keeps each kernel ahead of the next descriptor overwrite.
            err = cudaMemcpyAsync(device_copies, copies, n_batch * sizeof(copies[0]), cudaMemcpyHostToDevice, cudaStreamPerThread);
            if (err != cudaSuccess) {
                return ggml_cuda_batch_get_decline(device, err, "the copy of the list of small ranges", n_reads, payload_bytes);
            }
            ggml_cuda_batch_get_kernel<<<blocks, threads, 0, cudaStreamPerThread>>>(device_copies, n_batch);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return ggml_cuda_batch_get_decline(device, err, "the kernel that gathers the small ranges", n_reads, payload_bytes);
            }
            if (first + n_batch < n_reads_gather) {
                // Wait before refilling the pinned host descriptor buffer.
                err = cudaStreamSynchronize(cudaStreamPerThread);
                if (err != cudaSuccess) {
                    return ggml_cuda_batch_get_decline(device, err, "the wait for the small ranges", n_reads, payload_bytes);
                }
            }
        }
    }
    // Keep the destination pinned for DMA too; overlapping destinations already declined above.
    size_t n_dma = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] && (!n_reads_gather || sizes[i] > max_gather_read_bytes)) {
            err = n_dma > 0 && ggml_cuda_debug_refuse("batch-get") ? cudaErrorMemoryAllocation :
                cudaMemcpyAsync(data[i], (const char *) tensors[i]->data + offsets[i], sizes[i], cudaMemcpyDeviceToHost, cudaStreamPerThread);
            if (err != cudaSuccess) {
                return ggml_cuda_batch_get_decline(device, err, "a copy into the registered memory", n_reads, payload_bytes);
            }
            n_dma++;
        }
    }
    // Wait for GPU writes before releasing the CPU mapping and descriptor buffers.
    err = cudaStreamSynchronize(cudaStreamPerThread);
    if (err != cudaSuccess) {
        return ggml_cuda_batch_get_decline(device, err, "the wait for the copies", n_reads, payload_bytes);
    }
    return true;
#endif // defined(_WIN32)
#endif
}
