#pragma once

#include "llama.h"
#include "llama-graph.h"

#include <map>
#include <memory>
#include <functional>
#include <vector>

struct llama_ubatch;
struct llama_memory_lazy_info; // llama-ext.h

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

// Lazy sequence memory of a context (llama_context_params::n_seq_budget). The memory modules take the memory of their
// cells and rows when sequences need it and give it back, and what they hold together stays within one budget.
struct llama_memory_lazy {
    size_t budget = 0; // bytes that the modules may take together; 0 until the context has set it: no limit
    size_t taken  = 0; // bytes taken now

    // Finishes the work of the context that is under way. Called before memory goes back: a graph that still runs
    // may read cells that are no longer in use, and memory that is given back under it costs the process the device.
    std::function<void()> synchronize;

    bool fits(size_t n_bytes) const {
        return budget == 0 || taken + n_bytes <= budget;
    }
};

// a part of a tensor that lives in a lazy buffer: size bytes from offset, in bytes from the start of the tensor
struct llama_memory_lazy_range {
    const ggml_tensor * tensor;
    size_t offset;
    size_t size;
};

// Take the memory of the ranges, within the budget. Returns false if the budget or the device does not have it: what
// the call took before it failed stays taken, and counted, until it is given back.
bool llama_memory_lazy_take(llama_memory_lazy & lazy, const std::vector<llama_memory_lazy_range> & ranges);

// Give back the memory of the tensors that the ranges do not need: every unit of a tensor that none of the ranges
// touches (a tensor in a lazy buffer of a memory module has its units to itself, see llama_memory_bufs::emplace_lazy).
// The work of the context that is under way is finished first (llama_memory_lazy::synchronize).
void llama_memory_lazy_keep(llama_memory_lazy & lazy, const std::vector<const ggml_tensor *> & tensors, const std::vector<llama_memory_lazy_range> & keep);

// The bytes that the ranges take in buffers that hold nothing else: the units that they touch, each counted once.
size_t llama_memory_lazy_size(const std::vector<llama_memory_lazy_range> & ranges);

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;

    // only simulate the memory allocations
    bool no_alloc;

    // not null: the cells and rows are to take their memory lazily, within this budget
    llama_memory_lazy * lazy;
};

// the tensor contexts and backend buffers of a memory instance
// with no_alloc, dummy buffers are attached instead of allocating the tensor data, so the memory use can still be measured
struct llama_memory_bufs {
    explicit llama_memory_bufs(bool no_alloc) : no_alloc(no_alloc) {}

    // allocate the tensors of ctx on buft, or attach a dummy buffer if no_alloc is set
    // returns the buffer, for logging
    ggml_backend_buffer_t emplace(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name);

    // as emplace, with a lazy buffer (ggml_backend_buft_alloc_buffer_lazy): the address range of the tensors is
    // reserved and no memory is taken. Every tensor starts on a unit of the buffer type, so that its memory is taken
    // and given back without regard to its neighbours.
    ggml_backend_buffer_t emplace_lazy(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name);

    // the memory used by the tensors: what the buffers hold, or the simulated sizes if no_alloc is set
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // allocate a single ctx buffer - for the memory instances that keep their own buffer layout
    static ggml_backend_buffer_ptr alloc_buf(ggml_context * ctx, ggml_backend_buffer_type_t buft, bool no_alloc, const char * name);

    bool empty() const { return bufs.empty(); }

    auto begin()       { return bufs.begin(); }
    auto end()         { return bufs.end();   }
    auto begin() const { return bufs.begin(); }
    auto end()   const { return bufs.end();   }

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> bufs;

    const bool no_alloc;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    // embd_all: keep all the tokens of a sequence in one ubatch (all tokens are output, or the attention is non-causal)
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    // add a stream of KV cells with its own size and buffer type, returns its id or -1 if not supported
    // alloc: false adds the stream without its memory, see stream_alloc
    virtual int32_t stream_add(ggml_backend_buffer_type_t buft, uint32_t size, bool alloc) {
        GGML_UNUSED(buft);
        GGML_UNUSED(size);
        GGML_UNUSED(alloc);
        return -1;
    }

    // An added stream can be without its memory: it then holds no cells, and no sequence takes its cells from it.
    // stream_alloc takes the memory, false if the device does not have it: the stream stays without.
    // stream_release gives it back, false while a sequence takes its cells from the stream.
    // Both return true when there is nothing left to do.
    virtual bool stream_alloc(uint32_t strm) {
        GGML_UNUSED(strm);
        return false;
    }

    virtual bool stream_release(uint32_t strm) {
        GGML_UNUSED(strm);
        return false;
    }

    // the size in bytes of an added stream's memory, taken or not; 0 for any other stream
    virtual size_t stream_size(uint32_t strm) const {
        GGML_UNUSED(strm);
        return 0;
    }

    // the stream that the new cells of seq_id are taken from, returns false if the sequence still has cells
    virtual bool seq_set_stream(llama_seq_id seq_id, uint32_t strm) {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(strm);
        return false;
    }

    //
    // lazy memory (llama_memory_lazy)
    //

    // what the memory holds lazily; false if it holds all of its memory from the start
    virtual bool lazy_info(llama_memory_lazy_info & info) const {
        GGML_UNUSED(info);
        return false;
    }

    // the bytes that the cells [0, n_cells) of stream 0 and the rows [0, n_rows) take together
    virtual size_t lazy_bytes(uint32_t n_cells, uint32_t n_rows) const {
        GGML_UNUSED(n_cells);
        GGML_UNUSED(n_rows);
        return 0;
    }

    // What the lazy part may take so that the memory holds no more than it would hold whole with n_rows sequences:
    // the bytes of all the cells and of n_rows rows, as tensors and not in the device's units, less what the rows
    // beyond n_rows hold from the start (the parts of a row that are not lazy). Can be negative for a part of a
    // memory; the context adds the parts up.
    virtual int64_t lazy_budget(uint32_t n_rows) const {
        GGML_UNUSED(n_rows);
        return 0;
    }

    // give back the memory above the cells and rows that are in use
    virtual void lazy_trim() {
    }

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
