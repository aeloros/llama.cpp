#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);

//
// KV streams
//

// Add a stream of n_cells KV cells (padded to a multiple of 256) on a device (NULL for the CPU) to a unified KV cache.
// alloc: false adds it without its memory, which llama_memory_stream_alloc takes later.
// Returns the stream id, or -1 if the memory does not support it.
LLAMA_API int32_t llama_memory_stream_add(llama_memory_t mem, ggml_backend_dev_t dev, uint32_t n_cells, bool alloc);

// An added stream can be without its memory: it then holds no cells, and no sequence takes its cells from it.
// llama_memory_stream_alloc takes the memory from the device. Returns false if the device does not have it: the
// stream stays without.
// llama_memory_stream_release gives it back. Returns false while a sequence takes its cells from the stream.
// Both return true when there is nothing left to do.
LLAMA_API bool llama_memory_stream_alloc  (llama_memory_t mem, int32_t strm);
LLAMA_API bool llama_memory_stream_release(llama_memory_t mem, int32_t strm);

// The size in bytes of an added stream's memory, whether the stream has it now or not.
LLAMA_API size_t llama_memory_stream_size(llama_memory_t mem, int32_t strm);

// Take the new cells of a sequence from a stream. Stream 0 is the one the cache was created with.
// Returns false if the sequence still has cells.
LLAMA_API bool llama_memory_seq_set_stream(llama_memory_t mem, llama_seq_id seq_id, int32_t strm);

// Take the further cells of a sequence from a stream, and leave the cells that it has where they are: the sequence
// then holds cells in two streams. Attention runs over each stream's cells where they are and the two results are
// combined (ggml_flash_attn_ext_rows_partial), which needs flash attention over the cells of each sequence, as a
// unified cache has it by default; a batch that cannot have it is refused. A copy and a removal cover both streams,
// and a state of the sequence holds the cells of both. Returns false if the memory cannot do that.
LLAMA_API bool llama_memory_seq_span_stream(llama_memory_t mem, llama_seq_id seq_id, int32_t strm);

// Whether a sequence that goes on in stream strm with llama_memory_seq_span_stream will be computed: the memory can
// hold a sequence in two streams, the stream has its memory, and every attention that has read the cache's cells so
// far reads both streams. The models whose attention is built by code of their own read one, and a batch of such a
// sequence fails there. False before the first batch, when no attention has read the cache yet.
LLAMA_API bool llama_memory_can_span(llama_memory_t mem, int32_t strm);

// The number of cells that a sequence holds in a stream of the KV cache, 0 for a memory without streams.
LLAMA_API int32_t llama_memory_seq_n_cells(llama_memory_t mem, llama_seq_id seq_id, int32_t strm);

// For tests of a cache with streams on two devices. The number of attention nodes of the context's last graph that
// ran on another device than the one that holds the cells they read, -1 without a graph. Attention has to run where
// the cells are: an attention that runs elsewhere has the K and V of its stream copied to it whole, for every layer
// of every batch.
LLAMA_API int32_t llama_context_n_attn_away(const struct llama_context * ctx);

//
// Lazy sequence memory (llama_context_params::n_seq_budget)
//
// The cells of stream 0 and the rows of the recurrent state take their memory when sequences need it, and what they
// hold together stays within a budget. The budget is what the same memory holds whole with n_seq_budget sequences,
// less what the sequences beyond that number hold from the start (the small states of their rows): a context with
// lazy memory holds no more of its device than one of n_seq_budget sequences without it. What is taken is counted in
// the device's units, so a memory that is nearly full holds a little less than the whole one: up to one unit less
// for each tensor. The memory below the marks is taken: a mark rises when a batch or a restored state needs cells
// or rows above it, and a batch that finds no memory for them is refused like one that finds no cells. A mark falls
// when the sequences above it are removed.

struct llama_memory_lazy_info {
    size_t   budget        = 0; // bytes that cells and rows may take together
    size_t   taken         = 0; // bytes taken now
    size_t   held          = 0; // bytes that the memory holds from the start beside them: the small states of every row
    uint32_t n_cells       = 0; // the cells of stream 0
    uint32_t n_cells_taken = 0; // the cell mark: the cells below it have their memory
    uint32_t n_cells_used  = 0; // those of them that sequences hold; the free ones below the mark are filled first
    size_t   cell_bytes    = 0; // of one cell
    uint32_t n_rows        = 0; // the rows of the recurrent state that are taken lazily, 0 if there are none
    uint32_t n_rows_taken  = 0; // the row mark
    uint32_t n_rows_used   = 0; // those of them that sequences hold; a new sequence takes the lowest free row
    size_t   row_bytes     = 0; // of one row, the part that is taken lazily
};

// false if the memory holds all of it from the start
LLAMA_API bool llama_memory_get_lazy_info(llama_memory_t mem, llama_memory_lazy_info * info);

// The bytes that the cells [0, n_cells) and the rows [0, n_rows) take together, in the units of the device. 0 if the
// memory is not lazy.
LLAMA_API size_t llama_memory_lazy_bytes(llama_memory_t mem, uint32_t n_cells, uint32_t n_rows);

// Give back the memory above the cells and rows that are in use. The memory does that itself when a sequence is
// removed from its first position on or the memory is cleared; after a removal of the last tokens of a sequence it
// keeps what it has, so that a sequence that rolls back and goes on does not give memory back at every step.
LLAMA_API void llama_memory_lazy_trim(llama_memory_t mem);

LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Give the memory that the context computes with on a device (NULL for the CPU) back to it: the compute buffer of
// the device's backend. The next graph that runs on the device takes what it needs again. The memory module keeps
// its buffers (see llama_memory_stream_release).
// Returns the size of the compute buffer that went back.
LLAMA_API size_t llama_context_release_compute(struct llama_context * ctx, ggml_backend_dev_t dev);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

//
// model/context data extraction
//

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLAMA_API bool llama_model_classifier_free_guidance(const struct llama_model * model);
