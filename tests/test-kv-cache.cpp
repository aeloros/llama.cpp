#include "llama-kv-cache.h"
#include "llama-model.h"

#include <numeric>
#include <string>

static void init_model(llama_model & model, uint32_t n_layer) {
    model.hparams = {};
    model.hparams.n_layer_all = n_layer;
    model.hparams.n_embd_head_k_full = 4;
    model.hparams.n_embd_head_v_full = 4;
    model.hparams.n_head_arr.fill(1);
    model.hparams.n_head_kv_arr.fill(1);
    std::iota(model.hparams.kv_source_impl.begin(), model.hparams.kv_source_impl.end(), 0);
    model.layers.resize(n_layer);
}

// a token-less ubatch: n tokens of seq_id starting at position p0, appended to ubatch if it already has tokens
static void ubatch_add(llama_ubatch & ubatch, llama_seq_id seq_id, llama_pos p0, uint32_t n) {
    if (!ubatch.data) {
        ubatch = {};
        ubatch.data = std::make_shared<llama_ubatch::data_t>();
        ubatch.data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
        ubatch.n_seqs = 1;
        ubatch.n_pos = 1;
    }
    auto & data = *ubatch.data;
    data.seq_idx[seq_id] = (int32_t) data.seq_id_unq.size();
    data.seq_id_unq.push_back(seq_id);
    for (uint32_t i = 0; i < n; ++i) {
        data.pos.push_back(p0 + i);
        data.n_seq_id.push_back(1);
        data.seq_id_data.push_back(seq_id);
        data.output.push_back(0);
    }
    data.seq_id.clear();
    for (auto & id : data.seq_id_data) {
        data.seq_id.push_back(&id);
    }
    ubatch.n_tokens = (uint32_t) data.pos.size();
    ubatch.n_seq_tokens = ubatch.n_tokens;
    ubatch.n_seqs_unq = (uint32_t) data.seq_id_unq.size();
    ubatch.pos = data.pos.data();
    ubatch.n_seq_id = data.n_seq_id.data();
    ubatch.seq_id = data.seq_id.data();
    ubatch.seq_id_unq = data.seq_id_unq.data();
    ubatch.seq_idx = data.seq_idx.data();
    ubatch.output = data.output.data();
}

int main() {
    for (int bit = 0; bit < 64; ++bit) {
        GGML_ASSERT(ctz64(uint64_t{1} << bit) == bit);
        GGML_ASSERT(ctz64(UINT64_MAX << bit) == bit);
    }

    llama_backend_init();

    {
        std::string errors;
        llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
            if (level == GGML_LOG_LEVEL_ERROR) {
                *static_cast<std::string *>(user_data) += text;
            }
        }, &errors);

        for (llm_arch arch : { LLM_ARCH_DEEPSEEK2, LLM_ARCH_DEEPSEEK32, LLM_ARCH_DEEPSEEK4,
                LLM_ARCH_DEEPSEEK41, LLM_ARCH_GLM_DSA, LLM_ARCH_GLM5_NEXT, LLM_ARCH_DOTS3NOTE,
                LLM_ARCH_MISTRAL_LARGE4, LLM_ARCH_HY_V4, LLM_ARCH_MINIMAX_M3, LLM_ARCH_QWEN4EXP,
                LLM_ARCH_DFLASH }) {
            std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
            init_model(*model, 1);
            if (arch == LLM_ARCH_DEEPSEEK2) {
                model->hparams.n_embd_head_k_mla_impl = 4;
                model->hparams.n_embd_head_v_mla_impl = 4;
            }

            for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                    if (type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16) {
                        continue;
                    }
                    auto params = llama_context_default_params();
                    params.type_k = type_k;
                    params.type_v = type_v;
                    errors.clear();
                    llama_context * ctx = llama_init_from_model(model.get(), params);
                    GGML_ASSERT(ctx == nullptr);
                    if (type_k == type_v || (!model->hparams.is_mla() && arch != LLM_ARCH_DEEPSEEK4 && arch != LLM_ARCH_DEEPSEEK41)) {
                        GGML_ASSERT(errors.find("FP8 cache is not supported for architecture") != std::string::npos);
                    } else {
                        GGML_ASSERT(errors.find("model does not support different K") != std::string::npos);
                    }
                }
            }
        }

        llama_log_set(nullptr, nullptr);
    }

    {
        std::unique_ptr<llama_model> target(llama_model_create(LLM_ARCH_GEMMA4, llama_model_default_params()));
        std::unique_ptr<llama_model> assistant(llama_model_create(LLM_ARCH_GEMMA4_ASSISTANT, llama_model_default_params()));
        init_model(*target, 3);
        init_model(*assistant, 2);

        ggml_context_ptr ctx(ggml_init({ 4096, nullptr, false }));
        target->layers[2].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        target->layers[2].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].k_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[0].v_cache_s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        assistant->layers[1].k_cache_s = assistant->layers[0].k_cache_s;
        assistant->layers[1].v_cache_s = assistant->layers[0].v_cache_s;
        *static_cast<float *>(target->layers[2].k_cache_s->data) = 0.5f;
        *static_cast<float *>(target->layers[2].v_cache_s->data) = 0.25f;
        *static_cast<float *>(assistant->layers[0].k_cache_s->data) = 2.0f;
        *static_cast<float *>(assistant->layers[0].v_cache_s->data) = 4.0f;

        for (ggml_type type_k : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
            for (ggml_type type_v : { GGML_TYPE_F16, GGML_TYPE_F8_E4M3 }) {
                llama_kv_cache source(*target, false, target->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr,
                        [](uint32_t il) { return il == 0 ? 2 : -1; }, nullptr);
                llama_kv_cache shared(*assistant, false, assistant->hparams, type_k, type_v,
                        false, false, true, 32, 1, 0, LLAMA_SWA_TYPE_NONE, &source, nullptr, nullptr,
                        [](int32_t il) { return il == 0 ? 2 : 1; });

                GGML_ASSERT(source.get_k_scale(2) == (type_k == GGML_TYPE_F8_E4M3 ? target->layers[2].k_cache_s : nullptr));
                GGML_ASSERT(source.get_v_scale(2) == (type_v == GGML_TYPE_F8_E4M3 ? target->layers[2].v_cache_s : nullptr));
                GGML_ASSERT(source.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(source.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_storage(0) == source.get_k_storage(2));
                GGML_ASSERT(shared.get_k_scale(0) == source.get_k_scale(2));
                GGML_ASSERT(shared.get_v_scale(0) == source.get_v_scale(2));
                GGML_ASSERT(shared.get_k_scale(1) == nullptr);
                GGML_ASSERT(shared.get_v_scale(1) == nullptr);
            }
        }
    }

    // a cache for one sequence has one stream whether it was created unified or not, and takes the same paths:
    // the simple batch split, and the position buckets of the T5 decoder
    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_T5, llama_model_default_params()));
        init_model(*model, 2);
        model->hparams.n_rel_attn_bkts = 32;

        llama_vocab vocab;

        for (bool unified : { false, true }) {
            llama_kv_cache kv(*model, false, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
                    false, false, unified, 32, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);

            GGML_ASSERT(kv.get_n_stream() == 1);

            // 5 embeddings of sequence 0 in ubatches of 8: one ubatch that is not split into equal sequences
            llama_batch_ext batch(64, 2, 2, 1, nullptr, 0, 1);
            const std::vector<float> embd(2, 1.0f);
            for (llama_pos p = 0; p < 5; ++p) {
                const llama_pos pos[GGML_MROPE_SECTIONS] = { p, 0, 0, 0 };
                const int32_t idx = batch.add_token(0);
                GGML_ASSERT(idx >= 0);
                GGML_ASSERT(batch.set_token_embd(idx, { embd.data(), 1, 2 }));
                GGML_ASSERT(batch.set_token_pos(idx, pos));
                GGML_ASSERT(batch.set_output(idx, p == 4));
            }
            llama_batch_allocr balloc(1);
            GGML_ASSERT(balloc.init(batch, vocab, false));

            auto mctx = kv.init_batch(balloc, 8, false);
            GGML_ASSERT(mctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
            GGML_ASSERT(mctx->get_ubatch().n_tokens == 5);
            GGML_ASSERT(!mctx->get_ubatch().equal_seqs());
            GGML_ASSERT(mctx->apply());

            ggml_context_ptr ctx(ggml_init({ 2*ggml_tensor_overhead(), nullptr, true }));
            ggml_tensor * buckets = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 32, 5);
            ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
            GGML_ASSERT(buf);

            kv.set_input_pos_bucket(buckets, &mctx->get_ubatch());
            for (int i = 0; i < 32*5; ++i) {
                const int32_t bucket = ((const int32_t *) buckets->data)[i];
                GGML_ASSERT(bucket >= 0 && bucket < 32);
            }

            GGML_ASSERT(kv.get_unified());
        }
    }

    // a unified cache with a second stream of its own size: a sequence assigned to it keeps its cells there
    {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
        init_model(*model, 2);

        llama_kv_cache kv(*model, false, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
                false, false, true, 32, 2, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);

        const int32_t strm = kv.stream_add(ggml_backend_cpu_buffer_type(), 16);
        GGML_ASSERT(strm == 1);
        GGML_ASSERT(kv.get_n_stream() == 2);
        GGML_ASSERT(kv.get_size() == 32);
        // padded like n_ctx
        GGML_ASSERT(kv.get_size(strm) == 256);
        GGML_ASSERT(kv.memory_breakdown().at(ggml_backend_cpu_buffer_type()) > 0);

        GGML_ASSERT(kv.seq_set_stream(1, strm));
        GGML_ASSERT(kv.get_stream(0) == 0);
        GGML_ASSERT(kv.get_stream(1) == 1);

        llama_ubatch ub1;
        ubatch_add(ub1, 1, 0, 8);
        auto sinfo = kv.find_slot(ub1, false);
        GGML_ASSERT(!sinfo.empty() && sinfo.n_stream() == 1 && sinfo.strm[0] == 1 && sinfo.size() == 8);
        for (uint32_t idx : sinfo.idxs[0]) {
            GGML_ASSERT(idx < 256);
        }
        kv.apply_ubatch(sinfo, ub1);
        GGML_ASSERT(kv.seq_pos_max(1) == 7);
        GGML_ASSERT(kv.seq_pos_max(0) == -1);
        GGML_ASSERT(kv.get_cells(1).get_used() == 8);
        GGML_ASSERT(kv.get_cells(0).get_used() == 0);

        // the second stream has 248 cells left, the first has all 32
        llama_ubatch ub1_big;
        ubatch_add(ub1_big, 1, 8, 249);
        GGML_ASSERT(kv.find_slot(ub1_big, false).empty());

        llama_ubatch ub0;
        ubatch_add(ub0, 0, 0, 20);
        sinfo = kv.find_slot(ub0, false);
        GGML_ASSERT(!sinfo.empty() && sinfo.strm[0] == 0 && sinfo.size() == 20);
        kv.apply_ubatch(sinfo, ub0);
        GGML_ASSERT(kv.get_cells(0).get_used() == 20);
        GGML_ASSERT(kv.get_cells(1).get_used() == 8);

        // sequences of two streams in one ubatch are refused
        llama_ubatch ub_mixed;
        ubatch_add(ub_mixed, 0, 20, 2);
        ubatch_add(ub_mixed, 1, 8, 2);
        GGML_ASSERT(kv.find_slot(ub_mixed, false).empty());

        // a sequence changes stream only while it has no cells
        GGML_ASSERT(!kv.seq_set_stream(1, 0));
        GGML_ASSERT(kv.seq_rm(1, -1, -1));
        GGML_ASSERT(kv.get_cells(0).get_used() == 20);
        GGML_ASSERT(kv.seq_set_stream(1, 0));
        GGML_ASSERT(!kv.seq_set_stream(1, 2));
    }

    llama_backend_free();
    return 0;
}
