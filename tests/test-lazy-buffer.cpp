// Lazy backend buffers: the address range is reserved when the buffer is allocated, and its memory is taken and given
// back in ranges while tensors live in it (ggml_backend_buft_alloc_buffer_lazy).
//
// usage: test-lazy-buffer [<device name> ...]
// Every device is tried, and the CPU buffer type. A device that cannot make lazy buffers is skipped, unless it is
// named on the command line: then it has to.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
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

int main(int argc, char ** argv) {
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
