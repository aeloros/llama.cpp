#include "card-memory.cuh"

#if defined(_WIN32) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>

// NVML, the management library that comes with the driver: nvidia-smi prints its figures. It is loaded at run time,
// so nothing is linked against it, and where it is missing the caller keeps the figure it has.
namespace {

struct nvml_memory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

typedef int (*nvml_init_fn)(void);
typedef int (*nvml_handle_fn)(const char * pci_bus_id, void ** device);
typedef int (*nvml_memory_fn)(void * device, nvml_memory * memory);

struct nvml_lib {
    nvml_handle_fn handle = nullptr;
    nvml_memory_fn memory = nullptr;

    nvml_lib() {
        const HMODULE lib = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (lib == nullptr) {
            return;
        }
        const auto init = (nvml_init_fn)   GetProcAddress(lib, "nvmlInit_v2");
        const auto h    = (nvml_handle_fn) GetProcAddress(lib, "nvmlDeviceGetHandleByPciBusId_v2");
        const auto m    = (nvml_memory_fn) GetProcAddress(lib, "nvmlDeviceGetMemoryInfo");
        if (init == nullptr || h == nullptr || m == nullptr || init() != 0) {
            return;
        }
        handle = h;
        memory = m;
    }
};

} // namespace

bool ggml_cuda_card_memory_free(const char * pci_bus_id, size_t * free) {
    static const nvml_lib nvml; // loaded by the first reading
    if (nvml.memory == nullptr) {
        return false;
    }

    void *      device = nullptr;
    nvml_memory memory = {};
    if (nvml.handle(pci_bus_id, &device) != 0 || nvml.memory(device, &memory) != 0) {
        return false;
    }

    *free = (size_t) memory.free;
    return true;
}

bool ggml_cuda_host_commit_free(size_t * free) {
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        return false;
    }

    *free = (size_t) status.ullAvailPageFile;
    return true;
}

#else

bool ggml_cuda_card_memory_free(const char *, size_t *) {
    return false;
}

bool ggml_cuda_host_commit_free(size_t *) {
    return false;
}

#endif
