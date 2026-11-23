// Lazy backend buffers: the address range is reserved when the buffer is allocated, and its memory is taken and given
// back in ranges while tensors live in it (ggml_backend_buft_alloc_buffer_lazy). Before memory goes back, the backend
// that computed on it is synchronized: that is the caller's part.
//
// usage: test-lazy-buffer [<device name> ...]
// Every device is tried, and the CPU buffer type. A device that cannot make lazy buffers is skipped, unless it is
// named on the command line: then it has to.
//
// usage: test-lazy-buffer --reads <device of A> <device of B> [key=value ...]
// Two threads read the tensors of a state each, at the same time (run_reads). Not part of the run without arguments.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { printf("  FAILED: %s (line %d)\n", #cond, __LINE__); n_fail++; } } while (0)

static const size_t MiB = 1024*1024;

// whether a page of host memory is taken: 1, 0, or -1 where the system cannot be asked
static int host_page_taken(const void * p) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0) {
        return -1;
    }
    return mbi.State == MEM_COMMIT ? 1 : 0;
#else
    (void) p;
    return -1;
#endif
}

static void test_buft(ggml_backend_buffer_type_t buft, ggml_backend_t backend, const char * name, bool must) {
    const size_t unit = ggml_backend_buft_lazy_granularity(buft);
    if (unit == 0) {
        if (must) {
            printf("%s: FAILED: this buffer type cannot make lazy buffers\n", name);
            n_fail++;
        } else {
            printf("%s: skipped, this buffer type cannot make lazy buffers\n", name);
        }
        return;
    }

    printf("%s: lazy buffers in units of %zu bytes\n", name, unit);

    const bool is_host = ggml_backend_buft_is_host(buft);

    // a buffer that is not lazy counts as taken whole
    {
        ggml_backend_buffer_ptr nb(ggml_backend_buft_alloc_buffer(buft, MiB));
        CHECK(nb != nullptr);
        CHECK(!ggml_backend_buffer_is_lazy(nb.get()));
        CHECK(ggml_backend_buffer_committed(nb.get()) == ggml_backend_buffer_get_size(nb.get()));
        CHECK(ggml_backend_buffer_commit_size(nb.get(), 0, MiB) == 0);
        CHECK(ggml_backend_buffer_commit(nb.get(), 0, MiB));
        ggml_backend_buffer_decommit(nb.get(), 0, MiB);
        CHECK(ggml_backend_buffer_committed(nb.get()) == ggml_backend_buffer_get_size(nb.get()));
    }

    const size_t size = 64*MiB;

    ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer_lazy(buft, size));
    CHECK(buf != nullptr);
    if (!buf) {
        return;
    }

    CHECK(ggml_backend_buffer_is_lazy(buf.get()));
    CHECK(ggml_backend_buffer_get_type(buf.get()) == buft);
    CHECK(ggml_backend_buffer_get_size(buf.get()) >= size);
    CHECK(ggml_backend_buffer_committed(buf.get()) == 0);

    char * base = (char *) ggml_backend_buffer_get_base(buf.get());
    CHECK(base != nullptr);

    // nothing is taken: clearing the buffer must not touch it
    ggml_backend_buffer_clear(buf.get(), 0);

    // a tensor of 16 MiB that starts 8 MiB and a bit into the buffer
    const int64_t n      = 4*MiB;
    const size_t  nbytes = n*sizeof(float);
    const size_t  off    = 8*MiB + 4096;

    ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));

    ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    CHECK(ggml_backend_tensor_alloc(buf.get(), a, base + off) == GGML_STATUS_SUCCESS);

    // a commit takes every unit that the range touches
    const size_t first    = off/unit*unit;
    const size_t last     = (off + nbytes + unit - 1)/unit*unit;
    const size_t expected = last - first;

    CHECK(ggml_backend_buffer_commit_size(buf.get(), off, nbytes) == expected);
    CHECK(ggml_backend_buffer_commit(buf.get(), off, nbytes));
    CHECK(ggml_backend_buffer_committed(buf.get()) == expected);

    // taking it again takes nothing
    CHECK(ggml_backend_buffer_commit_size(buf.get(), off, nbytes) == 0);
    CHECK(ggml_backend_buffer_commit(buf.get(), off, nbytes));
    CHECK(ggml_backend_buffer_committed(buf.get()) == expected);

    if (is_host && host_page_taken(base) >= 0) {
        CHECK(host_page_taken(base + off) == 1);
        CHECK(host_page_taken(base + off + nbytes - 1) == 1);
        CHECK(host_page_taken(base + first - 1) == 0);
        CHECK(host_page_taken(base + last) == 0);
        CHECK(host_page_taken(base + size - 1) == 0);
    }

    // data goes in and comes back
    std::vector<float> data(n);
    for (int64_t i = 0; i < n; i++) {
        data[i] = (float) (i % 1000)*0.5f;
    }
    std::vector<float> back(n, -1.0f);

    ggml_backend_tensor_set(a, data.data(), 0, nbytes);
    ggml_backend_tensor_get(a, back.data(), 0, nbytes);
    CHECK(memcmp(data.data(), back.data(), nbytes) == 0);

    // the backend computes on it
    {
        ggml_init_params gp = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
        ggml_context_ptr gctx(ggml_init(gp));

        ggml_tensor * b  = ggml_scale(gctx.get(), a, 2.0f);
        ggml_cgraph * gf = ggml_new_graph(gctx.get());
        ggml_build_forward_expand(gf, b);

        ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
        CHECK(ggml_gallocr_alloc_graph(galloc.get(), gf));
        CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        std::vector<float> res(n, -1.0f);
        ggml_backend_tensor_get(b, res.data(), 0, nbytes);

        bool ok = true;
        for (int64_t i = 0; i < n; i++) {
            ok = ok && res[i] == 2.0f*data[i];
        }
        CHECK(ok);
    }

    // giving back a range in the middle gives back the units that lie whole within it, and the rest keeps its data
    const size_t hole_off  = off + 4*MiB + 12345;
    const size_t hole_size = 4*MiB;
    const size_t hole_first = (hole_off + unit - 1)/unit*unit;
    const size_t hole_last  = (hole_off + hole_size)/unit*unit;
    const size_t hole       = hole_last > hole_first ? hole_last - hole_first : 0;

    ggml_backend_synchronize(backend);
    ggml_backend_buffer_decommit(buf.get(), hole_off, hole_size);
    CHECK(ggml_backend_buffer_committed(buf.get()) == expected - hole);
    CHECK(ggml_backend_buffer_commit_size(buf.get(), off, nbytes) == hole);

    if (is_host && host_page_taken(base) >= 0 && hole > 0) {
        CHECK(host_page_taken(base + hole_first) == 0);
        CHECK(host_page_taken(base + hole_last - 1) == 0);
        CHECK(host_page_taken(base + hole_first - 1) == 1);
        CHECK(host_page_taken(base + hole_last) == 1);
    }

    {
        const size_t n_head = (hole_first - off)/sizeof(float);          // whole floats before the hole
        const size_t i_tail = (hole_last - off + sizeof(float) - 1)/sizeof(float); // first whole float after it

        std::vector<float> head(n_head, -1.0f);
        ggml_backend_tensor_get(a, head.data(), 0, n_head*sizeof(float));
        CHECK(memcmp(data.data(), head.data(), n_head*sizeof(float)) == 0);

        std::vector<float> tail(n - i_tail, -1.0f);
        ggml_backend_tensor_get(a, tail.data(), i_tail*sizeof(float), (n - i_tail)*sizeof(float));
        CHECK(memcmp(data.data() + i_tail, tail.data(), (n - i_tail)*sizeof(float)) == 0);
    }

    // a clear covers what is taken, and only that
    ggml_backend_buffer_clear(buf.get(), 0);
    {
        std::vector<float> head(1024, -1.0f);
        ggml_backend_tensor_get(a, head.data(), 0, head.size()*sizeof(float));
        bool zero = true;
        for (float v : head) {
            zero = zero && v == 0.0f;
        }
        CHECK(zero);
    }

    // the hole is taken again: what is taken anew is zero, and the whole tensor is usable
    CHECK(ggml_backend_buffer_commit(buf.get(), off, nbytes));
    CHECK(ggml_backend_buffer_committed(buf.get()) == expected);
    if (hole > 0) {
        std::vector<uint8_t> z(hole, 0xff);
        ggml_backend_tensor_get(a, z.data(), hole_first - off, hole);
        bool zero = true;
        for (uint8_t v : z) {
            zero = zero && v == 0;
        }
        CHECK(zero);
    }
    ggml_backend_tensor_set(a, data.data(), 0, nbytes);
    ggml_backend_tensor_get(a, back.data(), 0, nbytes);
    CHECK(memcmp(data.data(), back.data(), nbytes) == 0);

    // a range that reaches past the end of the buffer is refused, and nothing is taken for it
    CHECK(!ggml_backend_buffer_commit(buf.get(), ggml_backend_buffer_get_size(buf.get()) - unit, 2*unit));
    CHECK(ggml_backend_buffer_committed(buf.get()) == expected);

    // everything goes back
    ggml_backend_synchronize(backend);
    ggml_backend_buffer_decommit(buf.get(), 0, ggml_backend_buffer_get_size(buf.get()));
    CHECK(ggml_backend_buffer_committed(buf.get()) == 0);

    if (is_host && host_page_taken(base) >= 0) {
        CHECK(host_page_taken(base + off) == 0);
    }
}

// several threads, each with a lazy buffer of its own, take and give back memory at the same time: what one does
// to its buffer must not reach the others (contexts on one device do this from their own threads)
static void test_threads(ggml_backend_buffer_type_t buft, const char * name) {
    const size_t unit = ggml_backend_buft_lazy_granularity(buft);
    if (unit == 0) {
        return;
    }

    const int    n_threads = 8;
    const int    n_rounds  = 60;
    const size_t n_units   = 32;

    std::atomic<int> n_bad{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer_lazy(buft, n_units*unit));
            if (!buf) {
                n_bad++;
                return;
            }

            ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
            ggml_context_ptr ctx(ggml_init(ip));

            ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) (n_units*unit));
            if (ggml_backend_tensor_alloc(buf.get(), a, ggml_backend_buffer_get_base(buf.get())) != GGML_STATUS_SUCCESS) {
                n_bad++;
                return;
            }

            const size_t n_data = std::min<size_t>(unit, 64*1024);

            std::vector<uint8_t> data(n_data, (uint8_t) (t + 1));
            std::vector<uint8_t> back(n_data);

            for (int round = 0; round < n_rounds; round++) {
                const size_t u = (size_t) ((round*7 + t*3) % (n_units - 2));

                if (!ggml_backend_buffer_commit(buf.get(), u*unit, 2*unit)) {
                    n_bad++;
                    return;
                }

                ggml_backend_tensor_set(a, data.data(), u*unit + unit - n_data/2, n_data); // across two units
                ggml_backend_tensor_get(a, back.data(), u*unit + unit - n_data/2, n_data);
                if (memcmp(data.data(), back.data(), n_data) != 0) {
                    n_bad++;
                }

                if (round % 3 == 0) {
                    ggml_backend_buffer_clear(buf.get(), 0);
                }
                if (round % 2 == 0) {
                    ggml_backend_buffer_decommit(buf.get(), u*unit, 2*unit);
                }
            }

            ggml_backend_buffer_decommit(buf.get(), 0, n_units*unit);
            if (ggml_backend_buffer_committed(buf.get()) != 0) {
                n_bad++;
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }

    printf("%s: %d threads with a lazy buffer each\n", name, n_threads);
    CHECK(n_bad == 0);
}

// A copy from one place of a buffer to another is done by the buffer's device, whichever device the thread used
// last. llama_memory_recurrent moves the state of a sequence between rows this way, from the thread of a server that
// may just have asked another device what it has free: with two GPUs the copy ran with that other device current.
static void test_copy_after(ggml_backend_dev_t dev, ggml_backend_dev_t other) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    const size_t unit = ggml_backend_buft_lazy_granularity(buft);
    if (unit == 0) {
        return;
    }

    const size_t n_units = 4;

    ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer_lazy(buft, n_units*unit));
    CHECK(buf != nullptr);
    if (!buf) {
        return;
    }

    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));

    ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) (n_units*unit));
    CHECK(ggml_backend_tensor_alloc(buf.get(), a, ggml_backend_buffer_get_base(buf.get())) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_buffer_commit(buf.get(), 0, n_units*unit));

    const size_t n       = std::min<size_t>(unit, 256*1024) + 5;
    const size_t off_src = 3;
    const size_t off_dst = 2*unit - 7; // across two units

    std::vector<uint8_t> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = (uint8_t) (i*31 + 7);
    }
    ggml_backend_tensor_set(a, data.data(), off_src, n);

    ggml_tensor * src = ggml_view_1d(ctx.get(), a, (int64_t) n, off_src);
    ggml_tensor * dst = ggml_view_1d(ctx.get(), a, (int64_t) n, off_dst);
    CHECK(ggml_backend_view_init(src) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_view_init(dst) == GGML_STATUS_SUCCESS);

    // the thread uses the other device last
    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(other, &free, &total);

    ggml_backend_tensor_copy(src, dst);

    std::vector<uint8_t> back(n);
    ggml_backend_tensor_get(a, back.data(), off_dst, n);
    CHECK(memcmp(data.data(), back.data(), n) == 0);

    ggml_backend_buffer_decommit(buf.get(), 0, n_units*unit);

    printf("%s: a copy inside a buffer after the thread used %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_name(other));
}

// Two threads read many tensors each to host memory with one call each, at the same time: two contexts of one process
// do that when each saves the state of a sequence (a server with a copy of its model on a second GPU). The tensors of
// a side are those of such a state: a few large ranges (the cells of a sequence in every layer), many of a few MiB and
// many small ones (the states of a recurrent memory).
//   lazy=a|b|ab|none     whose buffer is lazy, every unit of it a mapping of its own (a)
//   rounds=N             (40)
//   span=MS step=MS      B starts up to span ms after A, step ms more each round; every fourth round A after B (450, 37)
//   free-a=MIB free-b=   fill the device until this much of it is free beside the buffer of the side (no fill)
//   big-a=NxMIB mid-a=NxMIB small-a=NxBYTES, and the same with -b: the tensors of a side
//   solo=1               only A reads
//   margin=MIB           for the time of a read the process may commit this much more than it has (Windows; no limit)
// Each round prints what the process had committed before A's read, and how much more at most during it.

// what the process has committed, 0 where the system cannot be asked
static size_t reads_committed() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        return 0;
    }
    return pmc.PrivateUsage;
#else
    return 0;
#endif
}

// The process may commit this many bytes at most, 0 for no limit: a job object of its own, so that the limit is this
// process's and the rest of the system keeps what it has. False where that cannot be done.
static bool reads_commit_limit(size_t bytes) {
#ifdef _WIN32
    static std::mutex mutex;
    static HANDLE     job = nullptr;

    std::lock_guard<std::mutex> lock(mutex);

    if (!job) {
        job = CreateJobObjectW(nullptr, nullptr);
        if (!job || !AssignProcessToJobObject(job, GetCurrentProcess())) {
            return false;
        }
    }

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = {};
    if (bytes > 0) {
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit               = bytes;
    }
    return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info)) != 0;
#else
    (void) bytes;
    return false;
#endif
}

static std::atomic<size_t> reads_commit_max{0};
static std::atomic<bool>   reads_sampling{false};
static size_t              reads_margin_mib = 0;

// the most that the process has committed since reads_commit_max was set
static void reads_sampler() {
    while (reads_sampling) {
        const size_t c = reads_committed();
        size_t m = reads_commit_max.load();
        while (c > m && !reads_commit_max.compare_exchange_weak(m, c)) {
        }
        std::this_thread::yield();
    }
}

struct reads_side {
    const char *       tag      = "";
    ggml_backend_dev_t dev      = nullptr;
    bool               lazy     = false;
    bool               active   = true;
    size_t             free_mib = 0;
    size_t             n[3]     = { 32, 144, 144 };
    size_t             size[3]  = { 40*MiB, 3*MiB, 120*1000 };
};

struct reads_shared {
    std::mutex              mutex;
    std::condition_variable cv;
    int                     round   = -1; // the round that may start
    int                     n_ready = 0;  // sides that are set up
    int                     n_done  = 0;  // reads that have ended
    std::atomic<int>        n_bad{0};
};

static uint8_t reads_value(size_t i) {
    return (uint8_t) (i*37 + 11);
}

static long long reads_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
}

// the delay of a side in a round
static int reads_delay(int r, int span_ms, int step_ms, bool is_b) {
    const int  d      = span_ms > 0 ? (r*step_ms) % (span_ms + 1) : 0;
    const bool b_late = r % 4 != 3;
    return is_b == b_late ? d : 0;
}

static void reads_worker(const reads_side & side, bool is_b, reads_shared & sh, int n_rounds, int span_ms, int step_ms,
                         std::vector<std::array<long long, 4>> & times) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(side.dev);

    std::vector<size_t> full; // the size of every tensor
    for (int k = 0; k < 3; k++) {
        for (size_t i = 0; i < side.n[k]; i++) {
            full.push_back(side.size[k]);
        }
    }

    const size_t align = ggml_backend_buft_get_alignment(buft);

    std::vector<size_t> at; // where every tensor starts in the buffer
    size_t size_buf = 0;
    for (const size_t s : full) {
        at.push_back(size_buf);
        size_buf += GGML_PAD(s, align);
    }

    ggml_backend_buffer_ptr fill;
    if (side.free_mib > 0) {
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(side.dev, &free, &total);
        const size_t keep = side.free_mib*MiB + size_buf;
        if (free > keep + 64*MiB) {
            fill.reset(ggml_backend_buft_alloc_buffer(buft, free - keep));
        }
        printf("%s: %s had %zu MiB free, %zu MiB of it are filled\n", side.tag, ggml_backend_dev_name(side.dev),
               free/MiB, fill ? (free - keep)/MiB : (size_t) 0);
    }

    ggml_backend_buffer_ptr buf(side.lazy ? ggml_backend_buft_alloc_buffer_lazy(buft, size_buf)
                                          : ggml_backend_buft_alloc_buffer(buft, size_buf));
    if (!buf || (side.lazy && !ggml_backend_buffer_commit(buf.get(), 0, size_buf))) {
        printf("%s: FAILED: no buffer of %zu MiB on %s\n", side.tag, size_buf/MiB, ggml_backend_dev_name(side.dev));
        fflush(stdout);
        exit(1);
    }

    ggml_init_params ip = { ggml_tensor_overhead()*(full.size() + 1), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));

    char * base = (char *) ggml_backend_buffer_get_base(buf.get());

    // every tensor is read without its first and last 64 bytes, so that no range starts where a unit does
    std::vector<const ggml_tensor *> tensors;
    std::vector<size_t> offsets;
    std::vector<size_t> sizes;
    std::vector<size_t> data_offsets;
    size_t size_host = 0;
    {
        std::vector<uint8_t> pattern(*std::max_element(full.begin(), full.end()));
        for (size_t i = 0; i < full.size(); i++) {
            ggml_tensor * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) full[i]);
            if (ggml_backend_tensor_alloc(buf.get(), t, base + at[i]) != GGML_STATUS_SUCCESS) {
                printf("%s: FAILED: tensor %zu has no place\n", side.tag, i);
                fflush(stdout);
                exit(1);
            }
            memset(pattern.data(), reads_value(i), full[i]);
            ggml_backend_tensor_set(t, pattern.data(), 0, full[i]);

            tensors.push_back(t);
            offsets.push_back(64);
            sizes.push_back(full[i] - 128);
            data_offsets.push_back(size_host);
            size_host += full[i] - 128;
        }
    }

    {
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(side.dev, &free, &total);
        printf("%s: %zu tensors, %zu MiB in a %s buffer on %s, which has %zu MiB free now\n", side.tag, tensors.size(),
               size_host/MiB, side.lazy ? "lazy" : "whole", ggml_backend_dev_name(side.dev), free/MiB);
        fflush(stdout);
    }

    {
        std::lock_guard<std::mutex> lock(sh.mutex);
        sh.n_ready++;
    }
    sh.cv.notify_all();

    for (int r = 0; r < n_rounds; r++) {
        {
            std::unique_lock<std::mutex> lock(sh.mutex);
            sh.cv.wait(lock, [&]() { return sh.round >= r; });
        }

        const int d = reads_delay(r, span_ms, step_ms, is_b);
        if (d > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(d));
        }

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> host(size_host); // new memory, zeroed, as a server takes it for a state
        const long long ms_host = reads_ms(t0);

        const size_t committed = reads_committed();
        if (!is_b) {
            reads_commit_max = committed;
        }
        if (reads_margin_mib > 0 && !reads_commit_limit(committed + reads_margin_mib*MiB)) {
            printf("%s: FAILED: the process cannot be limited\n", side.tag);
            fflush(stdout);
            exit(1);
        }

        const auto t1 = std::chrono::steady_clock::now();
        ggml_backend_tensor_get_n(tensors.data(), host.data(), offsets.data(), sizes.data(), data_offsets.data(), tensors.size());
        const long long ms_read = reads_ms(t1);

        if (reads_margin_mib > 0) {
            reads_commit_limit(0);
        }
        const size_t committed_max = std::max(reads_commit_max.load(), reads_committed());

        for (size_t i = 0; i < tensors.size(); i++) {
            const uint8_t * p = host.data() + data_offsets[i];
            const uint8_t   v = reads_value(i);
            if (p[0] != v || p[sizes[i]/2] != v || p[sizes[i] - 1] != v) {
                sh.n_bad++;
            }
        }

        times[r] = { ms_host, ms_read, (long long) (committed/MiB), (long long) ((committed_max - committed)/MiB) };

        {
            std::lock_guard<std::mutex> lock(sh.mutex);
            sh.n_done++;
        }
        sh.cv.notify_all();
    }
}

static bool reads_pair(const std::string & val, size_t mult, size_t & n, size_t & size) {
    const size_t x = val.find('x');
    if (x == std::string::npos) {
        return false;
    }
    n    = (size_t) atoll(val.substr(0, x).c_str());
    size = (size_t) atoll(val.substr(x + 1).c_str())*mult;
    return size > 128 || n == 0;
}

static int run_reads(int argc, char ** argv) {
    if (argc < 4) {
        printf("usage: test-lazy-buffer --reads <device of A> <device of B> [key=value ...]\n");
        return 1;
    }

    reads_side a;
    reads_side b;
    a.tag  = "A";
    b.tag  = "B";
    a.lazy = true;
    a.dev  = ggml_backend_dev_by_name(argv[2]);
    b.dev  = ggml_backend_dev_by_name(argv[3]);
    b.size[0] = 45*MiB;
    b.n[1]    = 48;
    b.n[2]    = 48;

    if (!a.dev || !b.dev) {
        printf("FAILED: there is no device %s\n", a.dev ? argv[3] : argv[2]);
        return 1;
    }

    int n_rounds = 40;
    int span_ms  = 450;
    int step_ms  = 37;

    for (int i = 4; i < argc; i++) {
        const std::string arg = argv[i];
        const size_t eq = arg.find('=');
        const std::string key = arg.substr(0, eq);
        const std::string val = eq == std::string::npos ? "" : arg.substr(eq + 1);

        bool ok = eq != std::string::npos;
        if (key == "lazy") {
            a.lazy = val == "a" || val == "ab";
            b.lazy = val == "b" || val == "ab";
        } else if (key == "rounds") {
            n_rounds = atoi(val.c_str());
        } else if (key == "span") {
            span_ms = atoi(val.c_str());
        } else if (key == "step") {
            step_ms = atoi(val.c_str());
        } else if (key == "solo") {
            b.active = atoi(val.c_str()) == 0;
        } else if (key == "margin") {
            reads_margin_mib = (size_t) atoll(val.c_str());
        } else if (key == "free-a") {
            a.free_mib = (size_t) atoll(val.c_str());
        } else if (key == "free-b") {
            b.free_mib = (size_t) atoll(val.c_str());
        } else if (key == "big-a") {
            ok = ok && reads_pair(val, MiB, a.n[0], a.size[0]);
        } else if (key == "mid-a") {
            ok = ok && reads_pair(val, MiB, a.n[1], a.size[1]);
        } else if (key == "small-a") {
            ok = ok && reads_pair(val, 1, a.n[2], a.size[2]);
        } else if (key == "big-b") {
            ok = ok && reads_pair(val, MiB, b.n[0], b.size[0]);
        } else if (key == "mid-b") {
            ok = ok && reads_pair(val, MiB, b.n[1], b.size[1]);
        } else if (key == "small-b") {
            ok = ok && reads_pair(val, 1, b.n[2], b.size[2]);
        } else {
            ok = false;
        }
        if (!ok || n_rounds < 1) {
            printf("FAILED: argument %s\n", arg.c_str());
            return 1;
        }
    }

    reads_shared sh;
    std::vector<std::array<long long, 4>> times_a(n_rounds);
    std::vector<std::array<long long, 4>> times_b(n_rounds);

    reads_sampling = true;
    std::thread sampler(reads_sampler);

    // one side is set up after the other, so that each sees what the other took of a device they share
    std::thread ta(reads_worker, std::cref(a), false, std::ref(sh), n_rounds, span_ms, step_ms, std::ref(times_a));
    {
        std::unique_lock<std::mutex> lock(sh.mutex);
        sh.cv.wait(lock, [&]() { return sh.n_ready >= 1; });
    }
    std::thread tb;
    if (b.active) {
        tb = std::thread(reads_worker, std::cref(b), true, std::ref(sh), n_rounds, span_ms, step_ms, std::ref(times_b));
        std::unique_lock<std::mutex> lock(sh.mutex);
        sh.cv.wait(lock, [&]() { return sh.n_ready >= 2; });
    }

    const int n_active = b.active ? 2 : 1;

    for (int r = 0; r < n_rounds; r++) {
        const int d_a = reads_delay(r, span_ms, step_ms, false);
        const int d_b = reads_delay(r, span_ms, step_ms, true);

        // before the round: a refusal of the driver ends the program inside it
        printf("round %3d, %s %3d ms after the other: ", r, d_a > 0 ? "A" : "B", b.active ? std::max(d_a, d_b) : 0);
        fflush(stdout);

        {
            std::lock_guard<std::mutex> lock(sh.mutex);
            sh.round = r;
        }
        sh.cv.notify_all();
        {
            std::unique_lock<std::mutex> lock(sh.mutex);
            sh.cv.wait(lock, [&]() { return sh.n_done >= n_active*(r + 1); });
        }

        printf("A %4lld ms for its memory and %4lld for the read", times_a[r][0], times_a[r][1]);
        if (b.active) {
            printf(", B %4lld and %4lld", times_b[r][0], times_b[r][1]);
        }
        printf("; %lld MiB committed before A's read, at most %lld more during it\n", times_a[r][2], times_a[r][3]);
        fflush(stdout);
    }

    ta.join();
    if (tb.joinable()) {
        tb.join();
    }

    reads_sampling = false;
    sampler.join();

    printf("%d rounds, %d values that are wrong\n", n_rounds, sh.n_bad.load());
    return sh.n_bad == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--reads") == 0) {
        return run_reads(argc, argv);
    }

    std::vector<std::string> must;
    for (int i = 1; i < argc; i++) {
        must.push_back(argv[i]);
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev  = ggml_backend_dev_get(i);
        const char *       name = ggml_backend_dev_name(dev);

        bool is_must = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        for (const auto & m : must) {
            is_must = is_must || m == name;
        }

        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        if (!backend) {
            printf("%s: FAILED: no backend\n", name);
            n_fail++;
            continue;
        }

        test_buft(ggml_backend_dev_buffer_type(dev), backend.get(), name, is_must);
        test_threads(ggml_backend_dev_buffer_type(dev), name);
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        for (size_t j = 0; j < ggml_backend_dev_count(); j++) {
            test_copy_after(ggml_backend_dev_get(i), ggml_backend_dev_get(j));
        }
    }

    for (const auto & m : must) {
        if (ggml_backend_dev_by_name(m.c_str()) == nullptr) {
            printf("%s: FAILED: there is no such device\n", m.c_str());
            n_fail++;
        }
    }

    if (n_fail > 0) {
        printf("%d checks failed\n", n_fail);
        return 1;
    }

    printf("all checks passed\n");
    return 0;
}
