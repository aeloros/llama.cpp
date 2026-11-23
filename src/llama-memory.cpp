#include "llama-memory.h"

#include "llama-ext.h"
#include "llama-io.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

//
// llama_memory_bufs
//

ggml_backend_buffer_ptr llama_memory_bufs::alloc_buf(ggml_context * ctx, ggml_backend_buffer_type_t buft, bool no_alloc, const char * name) {
    ggml_backend_buffer_ptr buf;
    if (no_alloc) {
        buf.reset(ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0)); // dummy buffer
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            t->buffer = buf.get(); // set dummy buffer so that the backend scheduler won't try to allocate the tensors
        }
    } else {
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)); // real buffer
    }
    if (!buf) {
        throw std::runtime_error(std::string("failed to allocate buffer for ") + name);
    }
    return buf;
}

ggml_backend_buffer_t llama_memory_bufs::emplace(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name) {
    ggml_backend_buffer_ptr buf = alloc_buf(ctx.get(), buft, no_alloc, name);

    // initialize the buffer to avoid NaNs in the padding
    ggml_backend_buffer_clear(buf.get(), 0);

    bufs.emplace_back(std::move(ctx), std::move(buf));

    return bufs.back().second.get();
}

ggml_backend_buffer_t llama_memory_bufs::emplace_lazy(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name) {
    GGML_ASSERT(!no_alloc);

    const size_t unit = ggml_backend_buft_lazy_granularity(buft);
    GGML_ASSERT(unit > 0);

    // the tensors side by side, each from the start of a unit
    std::vector<std::pair<ggml_tensor *, size_t>> offs;
    size_t size = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
        if (t->view_src != nullptr) {
            continue;
        }
        size = (size + unit - 1)/unit*unit;
        offs.emplace_back(t, size);
        size += ggml_backend_buft_get_alloc_size(buft, t);
    }

    ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer_lazy(buft, size));
    if (!buf) {
        throw std::runtime_error(std::string("failed to reserve the address range of the buffer for ") + name);
    }

    char * base = (char *) ggml_backend_buffer_get_base(buf.get());

    for (const auto & [t, off] : offs) {
        if (ggml_backend_tensor_alloc(buf.get(), t, base + off) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("failed to place a tensor in the buffer for ") + name);
        }
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
        if (t->view_src != nullptr && ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("failed to place a view in the buffer for ") + name);
        }
    }

    bufs.emplace_back(std::move(ctx), std::move(buf));

    return bufs.back().second.get();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_bufs::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // what the buffer holds: its size, or what is taken of a lazy one
            ret[buft] += ggml_backend_buffer_committed(buf.get());
        }
    }
    return ret;
}

//
// llama_memory_lazy
//

// where a range lies in its buffer
static size_t llama_memory_lazy_offset(const llama_memory_lazy_range & r) {
    return (size_t) ((const char *) r.tensor->data - (const char *) ggml_backend_buffer_get_base(r.tensor->buffer)) + r.offset;
}

bool llama_memory_lazy_take(llama_memory_lazy & lazy, const std::vector<llama_memory_lazy_range> & ranges) {
    // what the ranges would add. A unit that two ranges touch is counted twice here, which errs to the safe side.
    size_t n_need = 0;
    for (const auto & r : ranges) {
        n_need += ggml_backend_buffer_commit_size(r.tensor->buffer, llama_memory_lazy_offset(r), r.size);
    }

    if (n_need == 0) {
        return true;
    }

    if (!lazy.fits(n_need)) {
        return false;
    }

    for (const auto & r : ranges) {
        ggml_backend_buffer_t buf = r.tensor->buffer;

        const size_t n_before = ggml_backend_buffer_committed(buf);
        const bool   ok       = ggml_backend_buffer_commit(buf, llama_memory_lazy_offset(r), r.size);

        lazy.taken += ggml_backend_buffer_committed(buf) - n_before;

        if (!ok) {
            return false;
        }
    }

    return true;
}

void llama_memory_lazy_keep(llama_memory_lazy & lazy, const std::vector<const ggml_tensor *> & tensors, const std::vector<llama_memory_lazy_range> & keep) {
    struct part {
        ggml_backend_buffer_t buf;
        size_t offset;
        size_t size;
    };

    // what is taken of the units that the ranges do not touch
    std::vector<part> parts;

    for (const ggml_tensor * t : tensors) {
        ggml_backend_buffer_t      buf  = t->buffer;
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);

        const size_t unit = ggml_backend_buft_lazy_granularity(buft);
        if (unit == 0 || !ggml_backend_buffer_is_lazy(buf)) {
            continue;
        }

        // the units of the tensor: [u0, u1)
        const size_t t_offset = (size_t) ((const char *) t->data - (const char *) ggml_backend_buffer_get_base(buf));
        const size_t u0       = t_offset/unit;
        const size_t u1       = (t_offset + ggml_backend_buft_get_alloc_size(buft, t) + unit - 1)/unit;

        // those that have to stay
        std::vector<std::pair<size_t, size_t>> kept;
        for (const auto & r : keep) {
            if (r.tensor == t && r.size > 0) {
                kept.emplace_back((t_offset + r.offset)/unit, (t_offset + r.offset + r.size + unit - 1)/unit);
            }
        }
        std::sort(kept.begin(), kept.end());

        auto add = [&](size_t from, size_t to) {
            const size_t size = (to - from)*unit;
            // anything taken there?
            if (to > from && ggml_backend_buffer_commit_size(buf, from*unit, size) < size) {
                parts.push_back({ buf, from*unit, size });
            }
        };

        size_t u = u0;
        for (const auto & [k0, k1] : kept) {
            if (k0 > u) {
                add(u, std::min(k0, u1));
            }
            u = std::max(u, k1);
        }
        if (u < u1) {
            add(u, u1);
        }
    }

    if (parts.empty()) {
        return;
    }

    if (lazy.synchronize) {
        lazy.synchronize();
    }

    for (const auto & p : parts) {
        const size_t n_before = ggml_backend_buffer_committed(p.buf);
        ggml_backend_buffer_decommit(p.buf, p.offset, p.size);

        lazy.taken -= n_before - ggml_backend_buffer_committed(p.buf);
    }
}

size_t llama_memory_lazy_size(const std::vector<llama_memory_lazy_range> & ranges) {
    // the units of each buffer that the ranges touch: [first, last)
    std::map<ggml_backend_buffer_t, std::vector<std::pair<size_t, size_t>>> units;
    for (const auto & r : ranges) {
        if (r.size == 0) {
            continue;
        }
        ggml_backend_buffer_t buf = r.tensor->buffer;

        const size_t unit = ggml_backend_buft_lazy_granularity(ggml_backend_buffer_get_type(buf));
        if (unit == 0) {
            continue;
        }
        const size_t offset = llama_memory_lazy_offset(r);

        units[buf].emplace_back(offset/unit, (offset + r.size + unit - 1)/unit);
    }

    size_t res = 0;
    for (auto & [buf, v] : units) {
        const size_t unit = ggml_backend_buft_lazy_granularity(ggml_backend_buffer_get_type(buf));

        std::sort(v.begin(), v.end());

        size_t last = 0;
        for (const auto & [u0, u1] : v) {
            const size_t from = std::max(u0, last);
            if (u1 > from) {
                res += (u1 - from)*unit;
                last = u1;
            }
        }
    }

    return res;
}

// collects the buffers of the tensors that a state write visits, without keeping the data
struct llama_memory_buffers_writer : public llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem) {
    if (mem == nullptr) {
        return {};
    }

    llama_memory_buffers_writer writer;

    mem->state_write(writer);

    return std::vector<ggml_backend_buffer_t>(writer.buffers.begin(), writer.buffers.end());
}
