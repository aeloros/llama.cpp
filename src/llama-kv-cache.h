#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <string>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
                         bool   no_alloc,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "",
        // lightning indexer keys are rotated with the indexer RoPE parameters
                       bool   is_indexer = false,
        // not null: the cells of the cache take their memory lazily, within this budget (see lazy below). It is
        // honoured for one pool of cells that the cache owns, on one buffer type that can make lazy buffers
          llama_memory_lazy *   lazy = nullptr);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // only for a unified cache that does not share its cells
    // the size is padded to a multiple of 256, the stream holds no cells until a sequence is assigned to it
    int32_t stream_add(ggml_backend_buffer_type_t buft, uint32_t size, bool alloc) override;
    bool stream_alloc(uint32_t strm) override;
    bool stream_release(uint32_t strm) override;
    size_t stream_size(uint32_t strm) const override;
    bool seq_set_stream(llama_seq_id seq_id, uint32_t strm) override;
    bool seq_span_stream(llama_seq_id seq_id, uint32_t strm) override;

    bool     can_span(uint32_t strm) const override;
    uint32_t seq_n_cells(llama_seq_id seq_id, uint32_t strm) const override;

    // An attention read the cells of the current batch through llama_kv_cache_context::get_k, and said whether it
    // reads those of a second stream as well (span_ok): what can_span answers with.
    void note_read(bool span_ok) const;

    // lazy memory
    bool    lazy_info  (llama_memory_lazy_info & info) const override;
    size_t  lazy_bytes (uint32_t n_cells, uint32_t n_rows) const override;
    int64_t lazy_budget(uint32_t n_rows) const override;
    void   lazy_trim () override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_size(uint32_t strm) const;
    uint32_t get_n_seq_max() const;
    uint32_t get_n_stream() const;

    bool get_unified() const;

    // the stream of each sequence id if a ubatch must not hold sequences of different streams, otherwise nullptr
    // pass it to llama_batch_allocr::split_reset
    const std::vector<uint32_t> * get_seq_groups() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // The stream holding seq_id's cells.
    uint32_t get_stream(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
            slot_info_vec_t * sinfos_out,
      const slot_info_vec_t * sinfos_in);

    // undo a state_read() of seq_id (-1 for the whole cache) that another memory module failed to complete
    void state_clear(llama_seq_id seq_id);

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // attention over the cells of each sequence of the ubatch instead of the whole cache (see ggml_flash_attn_ext_rows)
    // the ubatch splits into n_slices groups of tokens with one sequence each
    // if it cannot (f.ex. interleaved or coupled sequences), it is a single slice over the cells of all its sequences
    // TODO: allow slices with coupled sequences
    static uint32_t get_n_kv_slices(const llama_ubatch & ubatch);

    // padded length of the longest cell list of the slices; 0 if kv_rows are not used (multiple streams)
    uint32_t get_n_kv_rows(const llama_ubatch & ubatch) const;

    // A sequence may hold cells in a stream beside the one that takes its new cells (seq_span_stream). The other
    // stream that holds cells of a sequence of the ubatch, -1 if there is none. Attention then runs over the cells of
    // each of the two streams, with a cell list and a mask for each (llm_attn_mask::strm_cells).
    int32_t get_strm_span(const llama_ubatch & ubatch) const;

    // as get_n_kv, get_n_kv_rows, get_k and get_v, over the cells of one stream of a unified cache
    uint32_t get_n_kv_strm(uint32_t strm) const;
    uint32_t get_n_kv_rows(const llama_ubatch & ubatch, uint32_t strm) const;

    ggml_tensor * get_k_strm(ggml_context * ctx, int32_t il, uint32_t n_kv, uint32_t strm) const;
    ggml_tensor * get_v_strm(ggml_context * ctx, int32_t il, uint32_t n_kv, uint32_t strm) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    ggml_tensor * get_k_scale(int32_t il) const;
    ggml_tensor * get_v_scale(int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    std::vector<ggml_tensor *> get_k_rot() const;
    std::vector<ggml_tensor *> get_v_rot() const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_attn_mask (const llm_attn_mask & attn_mask, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_relative_pos(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const;

    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        ggml_tensor * k_scale;
        ggml_tensor * v_scale;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        ggml_tensor * k_rot = nullptr;
        ggml_tensor * v_rot = nullptr;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    uint32_t n_stream = 1;

    // one pool of cells for all sequences, instead of one stream per sequence
    // true for a unified cache and for a cache with a single sequence: the cache is created with one stream
    // n_stream only grows past that with stream_add
    const bool unified = true;

    const std::string name_tag;

    // SWA
    const uint32_t n_swa = 0;

    const bool is_indexer = false;

    // env: LLAMA_ATTN_ROT_DISABLE
    uint32_t n_rot_k = 0;
    uint32_t n_rot_v = 0;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // keep constants separate from mutable KV buffers
    llama_memory_bufs attn_rot_bufs;

    void init_rotation(const char * name_tag);

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    llama_memory_bufs ctxs_bufs;

    // The tensors of a stream that was added with stream_add, and their buffer. The stream can give the buffer back
    // to its device while no sequence takes its cells from it (stream_release), and take it again (stream_alloc).
    struct stream_bufs {
        ggml_context_ptr           ctx;
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_backend_buffer_ptr    buf; // null while the stream is without its memory
    };

    // [n_stream], the entries of the streams that the cache was created with stay empty
    std::vector<stream_bufs> strm_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id: the stream that takes the new cells of the sequence
    std::vector<uint32_t> seq_to_stream;

    // The streams that hold cells of a sequence, the one of seq_to_stream first, with or without cells. In a unified
    // cache a sequence may hold cells in a second stream as well (seq_span_stream).
    std::vector<uint32_t> seq_streams(llama_seq_id seq_id) const;

    // whether the cache itself can hold a sequence in stream strm and another one
    bool can_span_strm(uint32_t strm) const;

    // the attentions that have read the cache's cells so far (note_read)
    mutable bool reads_both = false; // one that reads the cells of both streams of a sequence
    mutable bool reads_one  = false; // one that reads the cells of one stream

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * k,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;

    // the ranges of one entry: of one stream, or of the two streams that hold the cells of one sequence
    void state_write_data(llama_io_write_i & io, const std::vector<cell_ranges_t> & crs) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);

    void state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);

    // true for a stream of a unified cache that was added with stream_add: it has K/V tensors of its own
    bool is_stream_added(uint32_t strm) const;

    // false for an added stream that is without its memory (stream_release)
    bool stream_has_memory(uint32_t strm) const;

    // Lazy memory (llama_memory_lazy), null if the cache holds all of its memory. The tensors of stream 0 are in a
    // lazy buffer, and the cells below lazy_mark have their memory. The mark rises, in steps of LAZY_PAD cells, when a
    // batch or a restored state is placed in cells above it (lazy_take), and it falls to the highest cell in use
    // when cells are removed (lazy_trim_cells). Nothing may read or write a cell at or above the mark: get_n_kv() is
    // the highest cell in use, padded to the same step, and the K-shift, which covers every cell, is off.
    llama_memory_lazy * lazy = nullptr;

    uint32_t lazy_mark = 0;

    static constexpr uint32_t LAZY_PAD      = 256;  // as n_kv is padded
    static constexpr uint32_t LAZY_TRIM_MIN = 8192; // a removal of last tokens gives back no less

    // the memory of the cells [c0, c1) of stream 0, in every tensor, and those tensors
    std::vector<llama_memory_lazy_range> lazy_ranges(uint32_t c0, uint32_t c1) const;
    std::vector<const ggml_tensor *>     lazy_tensors() const;

    // have the memory of the cells [0, n_cells); false if it cannot be had, and the mark stays
    bool lazy_take(uint32_t n_cells);

    // give back the memory above the highest cell in use. exact: all of it; otherwise only if LAZY_TRIM_MIN cells
    // or more go back, so that a sequence that removes its last tokens and goes on keeps what it had
    void lazy_trim_cells(bool exact);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;
    uint32_t get_n_kv_rows() const;

    // the first stream of the current ubatch, get_k / get_v / cpy_k / cpy_v use its tensors
    uint32_t get_strm() const;

    // The second stream that holds cells of a sequence of the current ubatch, -1 if there is none
    // (llama_kv_cache::get_strm_span), and what attention over those cells needs. An attention that reads the cells of
    // one stream only says so by leaving span_ok out of get_k and get_v, which then refuse such a ubatch.
    int32_t  get_strm_span() const;
    uint32_t get_n_kv_span() const;
    uint32_t get_n_kv_rows_span() const;

    ggml_tensor * get_k_span(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_span(ggml_context * ctx, int32_t il) const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, bool span_ok = false) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, bool span_ok = false) const;

    ggml_tensor * get_k_scale(int32_t il) const;
    ggml_tensor * get_v_scale(int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    std::vector<ggml_tensor *> get_k_rot() const;
    std::vector<ggml_tensor *> get_v_rot() const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_attn_mask (const llm_attn_mask & attn_mask, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_relative_pos(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const;

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;

    // see llama_kv_cache::get_n_kv_rows
    uint32_t n_kv_rows = 0;

    // see get_strm_span
    int32_t  strm_span      = -1;
    uint32_t n_kv_span      = 0;
    uint32_t n_kv_rows_span = 0;
};
