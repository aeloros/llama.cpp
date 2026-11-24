// KV cache state cases
//
#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"

#include "../../src/llama-ext.h" // llama_model_set_attn_rot, llama_arch_from_string

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

//
// reference run
//

// - decode all but the last token, saving the state to disk
// - decode the last token
// - generate n_predict tokens
// the logits it produces are the reference of every case with needs_baseline()
struct case_baseline : ctx_test {
    const char * name() const override { return "baseline"; }

    const char * description() const override {
        return "reference generation over the whole prompt, saves the state before the last token";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_seq_max = 2;

        auto ctx  = mr.ctx(opts);
        auto smpl = mr.sampler();
        if (!ctx) {
            return case_result::fail();
        }

        auto n_past = 0;
        if (!common_prompt_batch_decode(ctx.get(), mr.tokens, (int) mr.tokens.size(), n_past, mr.params.n_batch, mr.params.out_file, true)) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }

        generation_result result;
        if (!generate_tokens(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 0, result)) {
            return case_result::fail();
        }
        if (result.empty()) {
            return case_result::fail();
        }

        mr.baseline      = std::move(result);
        mr.have_baseline = true;

        return case_result::pass();
    }
};

//
// sequences
//

// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
struct case_seq_rm_isolated : ctx_test {
    const char * name() const override { return "seq_rm"; }

    const char * description() const override {
        return "removing one sequence leaves the state of the other sequences untouched";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        const size_t n_tokens = std::min<size_t>(mr.tokens.size(), 128);
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            if (!decode_seq(ctx.get(), mr.tokens, 0, (llama_pos) n_tokens, seq_id, true)) {
                TLOG_ERR("%s: failed to decode the prompt for sequence %d\n", __func__, seq_id);
                return case_result::fail();
            }
        }

        std::vector<uint8_t> before;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, before)) {
            return case_result::fail();
        }

        if (!seq_rm(ctx.get(), 0)) {
            return case_result::fail();
        }

        std::vector<uint8_t> after;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, after)) {
            return case_result::fail();
        }

        CTX_REQUIRE(blobs_equal(before, after), "removing seq 0 changed the state of seq 1");

        return case_result::pass();
    }
};

// the device of the added stream of the stream cases: LLAMA_TEST_STREAM_DEV names it (f.ex. CUDA1), the default is the CPU
// the context needs a backend for it, f.ex. --device CUDA0,CUDA1 -ts 1,0 keeps the model on CUDA0
// two different GPUs can round a large ubatch differently, use -ub 64 to compare them
static ggml_backend_dev_t stream_dev() {
    const char * name = getenv("LLAMA_TEST_STREAM_DEV");

    return name ? ggml_backend_dev_by_name(name) : nullptr;
}

static std::string diff_str(const logits_diff & diff) {
    char buf[96];
    snprintf(buf, sizeof(buf), " (nmse %g, max diff %g, nan %zu)", diff.nmse(), (double) diff.diff_max, diff.n_nan);

    return buf;
}

// - add a second stream to the cache and take the cells of sequence 1 from it
// - decode the same prompt into sequence 0 and sequence 1, the last token in a decode of its own
// - verify that both sequences produce the same logits
struct case_streams : ctx_test {
    const char * name() const override { return "streams"; }

    const char * description() const override {
        return "a sequence with its cells in a second stream decodes like one in the first stream";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_memory_t mem = llama_get_memory(ctx.get());

        const int32_t strm = llama_memory_stream_add(mem, stream_dev(), 192, true);
        if (strm < 0) {
            return case_result::skip("the memory has no streams");
        }
        CTX_REQUIRE(llama_memory_seq_set_stream(mem, 1, strm), "failed to assign seq 1 to the new stream");

        const llama_pos n_tokens = (llama_pos) std::min<size_t>(mr.tokens.size(), 128);
        CTX_REQUIRE(n_tokens >= 2, "the prompt is too short");

        std::vector<float> logits[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            if (!decode_seq(ctx.get(), mr.tokens, 0, n_tokens - 1, seq_id, false) ||
                !decode_seq(ctx.get(), mr.tokens, n_tokens - 1, n_tokens, seq_id, true)) {
                TLOG_ERR("%s: failed to decode the prompt for sequence %d\n", __func__, seq_id);
                return case_result::fail();
            }
            if (!get_current_logits(ctx.get(), logits[seq_id])) {
                return case_result::fail();
            }
        }

        CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == n_tokens - 1, "seq 1 does not hold the prompt");

        const logits_diff diff = compare_logits(logits[0].data(), logits[1].data(), logits[0].size());
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq 1 in the second stream produced different logits than seq 0" + diff_str(diff));

        return case_result::pass();
    }
};

// - decode the prompt and four more tokens into sequence 0, whole in the first stream: the reference
// - decode the prompt with another first half into sequence 5: how much another context changes the logits
// - decode the first half of the prompt into sequence 1 in the first stream, let the sequence take its further cells
//   from a second stream, and decode the rest
// - verify that it produces the reference's logits, far closer than the other context does, and that the cache answers
//   for it as for any sequence
// - without a recurrent state: remove its end, across both streams, decode that again and compare again
// - copy it to an empty sequence, remove it, and verify that the copy goes on like the reference
// - decode the prompt into sequence 4, whole in the second stream, and then two tokens of the copy in a batch like
//   its last one: the graph changes for a batch with cells in two streams when nothing else is different
// - save the state of the copy, restore it into another sequence, and verify that this one goes on like the reference
struct case_stream_span : ctx_test {
    const char * name() const override { return "span"; }

    const char * description() const override {
        return "a sequence with cells in two streams decodes like one that is whole in the first stream";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 512;
        opts.n_seq_max  = 6;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_memory_t mem = llama_get_memory(ctx.get());

        const int32_t strm = llama_memory_stream_add(mem, stream_dev(), 192, true);
        if (strm < 0) {
            return case_result::skip("the memory has no streams");
        }

        const llama_pos n_tokens = (llama_pos) std::min<size_t>(mr.tokens.size(), 80);
        CTX_REQUIRE(n_tokens >= 16, "the prompt is too short");

        const llama_pos n_half = n_tokens/2;
        const int       n_more = 4;

        // the prompt, and tokens to go on with
        llama_tokens tokens(mr.tokens.begin(), mr.tokens.begin() + n_tokens);
        for (int i = 0; i < n_more; ++i) {
            tokens.push_back(mr.tokens[(3*i + 1) % n_tokens]);
        }

        // the reference: the logits after the prompt, and after each of the further tokens
        std::vector<float> logits_ref[n_more + 1];
        if (!decode_seq(ctx.get(), tokens, 0, n_tokens - 1, 0, false)) {
            return case_result::fail();
        }
        for (int i = 0; i <= n_more; ++i) {
            if (!decode_seq(ctx.get(), tokens, n_tokens - 1 + i, n_tokens + i, 0, true) || !get_current_logits(ctx.get(), logits_ref[i])) {
                return case_result::fail();
            }
        }

        // What another context does to the logits after the prompt: other tokens in its first half, those of the
        // prompt's second half. A sequence that lost the cells of one stream is off by about as much, and one that has
        // them all by far less.
        double nmse_other = 0.0;
        {
            llama_tokens tokens_other = tokens;
            for (llama_pos i = 0; i < n_half; ++i) {
                tokens_other[i] = tokens[n_half + (i*7 + 3) % (n_tokens - n_half)];
            }

            std::vector<float> logits_other;
            if (!decode_seq(ctx.get(), tokens_other, 0, n_tokens - 1, 5, false) ||
                !decode_seq(ctx.get(), tokens_other, n_tokens - 1, n_tokens, 5, true) ||
                !get_current_logits(ctx.get(), logits_other) ||
                !seq_rm(ctx.get(), 5)) {
                return case_result::fail();
            }
            nmse_other = compare_logits(logits_ref[0].data(), logits_other.data(), logits_ref[0].size()).nmse();
        }

        // the reference once more, its last two tokens and the next two as batches of two: for the batches of two below
        std::vector<float> logits_ref_pair[2];
        if (!decode_seq(ctx.get(), tokens, 0, n_tokens - 2, 5, false) ||
            !decode_seq(ctx.get(), tokens, n_tokens - 2, n_tokens, 5, true) ||
            !get_current_logits(ctx.get(), logits_ref_pair[0]) ||
            !decode_seq(ctx.get(), tokens, n_tokens, n_tokens + 2, 5, true) ||
            !get_current_logits(ctx.get(), logits_ref_pair[1]) ||
            !seq_rm(ctx.get(), 5)) {
            return case_result::fail();
        }

        // a model whose logits do not move with its context cannot tell: its sequences are still checked for what the
        // cache holds of them
        const bool   can_tell   = nmse_other > 1e-10;
        const double nmse_limit = can_tell ? std::min(NMSE_THRESHOLD, nmse_other/20) : NMSE_THRESHOLD;

        const auto same = [&](const std::vector<float> & ref, const char * what) -> std::string {
            std::vector<float> cur;
            if (!get_current_logits(ctx.get(), cur)) {
                return std::string(what) + ": no logits";
            }
            const logits_diff diff = compare_logits(ref.data(), cur.data(), ref.size());
            if (diff.nmse() > nmse_limit) {
                char buf[64];
                snprintf(buf, sizeof(buf), "; another context: nmse %g", nmse_other);
                return std::string(what) + " produced different logits than the sequence in one stream" + diff_str(diff) + buf;
            }
            return "";
        };

        // the first half in the first stream, the rest in the second
        if (!decode_seq(ctx.get(), tokens, 0, n_half, 1, false)) {
            return case_result::fail();
        }

        // what the memory says of it, now that the model's attention has read the cache: it has to be what happens
        const bool can_span = llama_memory_can_span(mem, strm);

        if (!llama_memory_seq_span_stream(mem, 1, strm)) {
            return case_result::fail("the memory does not let a sequence take its further cells from a second stream");
        }
        if (!decode_seq(ctx.get(), tokens, n_half, n_tokens - 1, 1, false)) {
            CTX_REQUIRE(!can_span, "the memory said that a sequence may go on in a second stream, and its batch was refused");

            // An attention that reads the cells of one stream refuses the batch (llama_kv_cache_context::get_k), and so
            // does a model without flash attention over the cells of each sequence: nothing of the batch is left, and
            // the sequence is what it was. (A recurrent state cannot go back, after this failure as after any other.)
            if (!mr.is_recurrent()) {
                CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == n_half - 1, "a batch that was refused left cells behind");
            }
            CTX_REQUIRE(llama_memory_seq_rm(mem, 1, -1, -1) && llama_memory_seq_set_stream(mem, 1, 0), "seq 1 cannot go back to the first stream");
            CTX_REQUIRE(llama_memory_stream_release(mem, strm), "the second stream holds cells of a batch that was refused");

            return case_result::skip("the model's attention reads the cells of one stream: the batch was refused");
        }
        CTX_REQUIRE(can_span, "the memory said that a sequence may not go on in a second stream, and its batch was computed");

        if (!decode_seq(ctx.get(), tokens, n_tokens - 1, n_tokens, 1, true)) {
            TLOG_ERR("%s: failed to decode the last token of sequence 1 in the second stream\n", __func__);
            return case_result::fail();
        }

        // Attention has to run where the cells are: each of the two parts on the device of its stream. An attention
        // that runs elsewhere gives the same result, with the K and V of its stream copied to it for every layer.
        CTX_REQUIRE(llama_context_n_attn_away(ctx.get()) == 0, "an attention ran on another device than the cells it reads");

        CTX_REQUIRE(llama_memory_seq_n_cells(mem, 1, 0)    == n_half,            "seq 1 does not hold the first half of the prompt in the first stream");
        CTX_REQUIRE(llama_memory_seq_n_cells(mem, 1, strm) == n_tokens - n_half, "seq 1 does not hold the rest of the prompt in the second stream");

        // a recurrent state is the state after the last token: such a memory starts there
        if (!mr.is_recurrent()) {
            CTX_REQUIRE(llama_memory_seq_pos_min(mem, 1) == 0, "seq 1 does not start at position 0");
        }
        CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == n_tokens - 1, "seq 1 does not hold the prompt");
        {
            const std::string why = same(logits_ref[0], "seq 1, with cells in two streams,");
            CTX_REQUIRE(why.empty(), why);
        }

        // a removal that reaches into both streams, and the same tokens again (a recurrent state cannot go back so far)
        if (!mr.is_recurrent()) {
            CTX_REQUIRE(llama_memory_seq_rm(mem, 1, n_half - 3, -1), "failed to remove the end of seq 1");
            CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == n_half - 4, "seq 1 holds cells past the removal");
            if (!decode_seq(ctx.get(), tokens, n_half - 3, n_tokens - 1, 1, false) ||
                !decode_seq(ctx.get(), tokens, n_tokens - 1, n_tokens, 1, true)) {
                return case_result::fail();
            }
            CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == n_tokens - 1, "seq 1 does not hold the prompt after the removal");

            const std::string why = same(logits_ref[0], "seq 1, after a removal across both streams,");
            CTX_REQUIRE(why.empty(), why);
        }

        // a copy shares the cells of both streams, and goes on by itself when the sequence it came from is gone
        llama_memory_seq_cp(mem, 1, 2, -1, -1);
        CTX_REQUIRE(llama_memory_seq_pos_max(mem, 2) == n_tokens - 1, "the copy does not hold the prompt");
        CTX_REQUIRE(llama_memory_seq_rm(mem, 1, -1, -1), "failed to remove seq 1");
        CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == -1, "seq 1 holds cells after its removal");
        CTX_REQUIRE(llama_memory_seq_pos_max(mem, 2) == n_tokens - 1, "removing seq 1 took cells of its copy");

        // A sequence that is whole in the second stream takes its new cells from the stream that the copy takes its
        // own from. Its last two tokens are one batch, and so are the copy's next two: all that differs between the
        // two batches is whether their sequence holds cells in the first stream as well. The second one must not run
        // in the graph of the first, which reads one stream.
        CTX_REQUIRE(llama_memory_seq_set_stream(mem, 4, strm), "failed to assign seq 4 to the second stream");
        if (!decode_seq(ctx.get(), tokens, 0, n_tokens - 2, 4, false) ||
            !decode_seq(ctx.get(), tokens, n_tokens - 2, n_tokens, 4, true)) {
            return case_result::fail();
        }
        {
            const std::string why = same(logits_ref_pair[0], "a sequence that is whole in the second stream");
            CTX_REQUIRE(why.empty(), why);
        }

        if (!decode_seq(ctx.get(), tokens, n_tokens, n_tokens + 2, 2, true)) {
            return case_result::fail();
        }
        {
            const std::string why = same(logits_ref_pair[1], "the copy of seq 1, after a batch of its size without cells in the first stream,");
            CTX_REQUIRE(why.empty(), why);
        }

        // a state holds the cells of both streams: restored into another sequence, it goes on like the reference
        std::vector<uint8_t> state;
        if (!get_seq_state(ctx.get(), 2, LLAMA_STATE_SEQ_FLAGS_NONE, state)) {
            return case_result::fail("no state of a sequence with cells in two streams");
        }
        if (!set_seq_state(ctx.get(), 3, LLAMA_STATE_SEQ_FLAGS_NONE, state)) {
            return case_result::fail("failed to restore the state of a sequence with cells in two streams");
        }
        CTX_REQUIRE(llama_memory_seq_pos_max(mem, 3) == n_tokens + 1, "the restored sequence does not hold what was saved");
        if (!decode_seq(ctx.get(), tokens, n_tokens + 2, n_tokens + 3, 3, true)) {
            return case_result::fail();
        }
        {
            const std::string why = same(logits_ref[3], "the restored sequence");
            CTX_REQUIRE(why.empty(), why);
        }

        // nothing of the sequences is left behind in either stream: each can go back to the first one, and the second
        // stream can give its memory back
        for (llama_seq_id seq_id = 0; seq_id < 6; ++seq_id) {
            CTX_REQUIRE(llama_memory_seq_rm(mem, seq_id, -1, -1), "failed to remove seq " + std::to_string(seq_id));
            CTX_REQUIRE(llama_memory_seq_set_stream(mem, seq_id, 0), "seq " + std::to_string(seq_id) + " still holds cells after its removal");
        }
        CTX_REQUIRE(llama_memory_stream_release(mem, strm), "the second stream still holds cells after every sequence was removed");

        if (!can_tell) {
            return case_result::skip("the model's logits do not move with its context: what its sequences hold was checked, their logits could not be");
        }

        return case_result::pass();
    }
};

// - add a second stream to the cache and take the cells of sequence 1 from it
// - decode the prompt and the reversed prompt alone, as the references
// - decode the prompt into sequence 0 and the reversed prompt into sequence 1 in the same batches, their tokens interleaved
// - verify that both sequences produce their reference logits
struct case_streams_batch : ctx_test {
    const char * name() const override { return "streams_b"; }

    const char * description() const override {
        return "sequences of two streams decode in the same batch like a sequence decoded alone";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_batch    = 256;
        opts.n_ubatch   = 64;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_memory_t mem = llama_get_memory(ctx.get());

        const int32_t strm = llama_memory_stream_add(mem, stream_dev(), 192, true);
        if (strm < 0) {
            return case_result::skip("the memory has no streams");
        }
        CTX_REQUIRE(llama_memory_seq_set_stream(mem, 1, strm), "failed to assign seq 1 to the new stream");

        const llama_pos n_tokens = (llama_pos) std::min<size_t>(mr.tokens.size(), 128);
        CTX_REQUIRE(n_tokens >= 2, "the prompt is too short");

        // the sequences hold different tokens, so that the K/V of one cannot stand in for the other
        llama_tokens tokens[2];
        tokens[0].assign(mr.tokens.begin(), mr.tokens.begin() + n_tokens);
        tokens[1].assign(tokens[0].rbegin(), tokens[0].rend());

        std::vector<float> logits_ref[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            if (!decode_seq(ctx.get(), tokens[seq_id], 0, n_tokens - 1, 0, false) ||
                !decode_seq(ctx.get(), tokens[seq_id], n_tokens - 1, n_tokens, 0, true) ||
                !get_current_logits(ctx.get(), logits_ref[seq_id]) ||
                !seq_rm(ctx.get(), 0)) {
                return case_result::fail();
            }
        }

        {
            common_batch batch(ctx.get());
            for (llama_pos pos = 0; pos < n_tokens - 1; ++pos) {
                for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
                    batch.add(tokens[seq_id][pos], pos, seq_id, false);
                }
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                    "failed to decode a batch with sequences of two streams");
        }

        {
            common_batch batch(ctx.get());
            for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
                batch.add(tokens[seq_id][n_tokens - 1], n_tokens - 1, seq_id, true);
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                    "failed to decode the last token of both sequences");
        }

        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            CTX_REQUIRE(llama_memory_seq_pos_max(mem, seq_id) == n_tokens - 1, "seq " + std::to_string(seq_id) + " does not hold the prompt");

            const float * logits = llama_get_logits_ith(ctx.get(), seq_id);
            CTX_REQUIRE(logits != nullptr, "missing logits of seq " + std::to_string(seq_id));

            const logits_diff diff = compare_logits(logits_ref[seq_id].data(), logits, logits_ref[seq_id].size());
            CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq " + std::to_string(seq_id) + " produced different logits in the shared batch" + diff_str(diff));
        }

        return case_result::pass();
    }
};

// - decode three different token lists into three sequences in the same batches, their tokens interleaved
// - do the same in a second context, where sequences 1 and 2 take their cells from an added stream
// - verify that every sequence produces the same logits in both
// note: the reference is the same batches without the stream, that is what a stream must not change
struct case_streams_multi : ctx_test {
    const char * name() const override { return "streams_m"; }

    const char * description() const override {
        return "two sequences that share a second stream decode as in the first stream, in the same batches as a third sequence";
    }

    std::string skip_reason(model_run & mr) const override {
        char arch_name[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch_name, sizeof(arch_name));

        // plamo2 does not decode several sequences in one batch correctly, with or without a stream: the logits of a
        // sequence differ from the same sequence decoded alone, and a ubatch that holds some of the sequences trips an
        // out-of-bounds view of the mamba states (models/plamo2.cpp). The streams make such ubatches
        if (llama_arch_from_string(arch_name) == LLM_ARCH_PLAMO2) {
            return "plamo2 does not decode several sequences in one batch";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const llama_seq_id n_seqs = 3;

        // a batch holds the tokens of the three sequences, and it cannot be larger than the context
        ctx_opts opts;
        opts.n_ctx      = 512;
        opts.n_batch    = 512;
        opts.n_ubatch   = 64;
        opts.n_seq_max  = n_seqs;
        opts.kv_unified = 1;

        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        const llama_pos n_tokens = (llama_pos) std::min<size_t>(mr.tokens.size(), 128);
        CTX_REQUIRE(n_tokens >= 2, "the prompt is too short");

        // the sequences hold different tokens, so that the K/V of one cannot stand in for another
        llama_tokens tokens[n_seqs];
        tokens[0].assign(mr.tokens.begin(), mr.tokens.begin() + n_tokens);
        tokens[1].assign(tokens[0].rbegin(), tokens[0].rend());
        tokens[2].resize(n_tokens);
        std::rotate_copy(tokens[0].begin(), tokens[0].begin() + n_tokens/2, tokens[0].end(), tokens[2].begin());

        // the logits of the last token of every sequence, with and without the stream
        // the context with the stream comes first, so that a memory without streams is skipped before any decode
        std::vector<float> logits[2][n_seqs];
        for (int with_stream = 1; with_stream >= 0; --with_stream) {
            auto ctx = mr.ctx(opts);
            if (!ctx) {
                return case_result::fail();
            }

            llama_memory_t mem = llama_get_memory(ctx.get());

            if (with_stream) {
                // room for the two sequences that share the stream
                const int32_t strm = llama_memory_stream_add(mem, stream_dev(), 256, true);
                if (strm < 0) {
                    return case_result::skip("the memory has no streams");
                }
                for (llama_seq_id seq_id = 1; seq_id < n_seqs; ++seq_id) {
                    CTX_REQUIRE(llama_memory_seq_set_stream(mem, seq_id, strm), "failed to assign seq " + std::to_string(seq_id) + " to the new stream");
                }
            }

            {
                common_batch batch(ctx.get());
                for (llama_pos pos = 0; pos < n_tokens - 1; ++pos) {
                    for (llama_seq_id seq_id = 0; seq_id < n_seqs; ++seq_id) {
                        batch.add(tokens[seq_id][pos], pos, seq_id, false);
                    }
                }
                CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                        "failed to decode a batch with the three sequences");
            }

            {
                common_batch batch(ctx.get());
                for (llama_seq_id seq_id = 0; seq_id < n_seqs; ++seq_id) {
                    batch.add(tokens[seq_id][n_tokens - 1], n_tokens - 1, seq_id, true);
                }
                CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                        "failed to decode the last token of the three sequences");
            }

            for (llama_seq_id seq_id = 0; seq_id < n_seqs; ++seq_id) {
                CTX_REQUIRE(llama_memory_seq_pos_max(mem, seq_id) == n_tokens - 1, "seq " + std::to_string(seq_id) + " does not hold its tokens");

                const float * ptr = llama_get_logits_ith(ctx.get(), seq_id);
                CTX_REQUIRE(ptr != nullptr, "missing logits of seq " + std::to_string(seq_id));

                logits[with_stream][seq_id].assign(ptr, ptr + n_vocab);
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < n_seqs; ++seq_id) {
            const logits_diff diff = compare_logits(logits[0][seq_id].data(), logits[1][seq_id].data(), n_vocab);
            CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq " + std::to_string(seq_id) + " produced different logits with the stream than without" + diff_str(diff));
        }

        return case_result::pass();
    }
};

// - add a second stream to the cache and take the cells of sequence 1 from it
// - decode the prompt into both sequences
// - save sequence 1, remove it, restore it: it continues like sequence 0
// - restore the state of sequence 0 into sequence 1, across the streams: it continues like sequence 0
struct case_streams_state : ctx_test {
    case_streams_state(bool on_device) : on_device(on_device) {}

    const char * name() const override { return on_device ? "streams_sd" : "streams_sh"; }

    const char * description() const override {
        return on_device ?
            "the state of a sequence is saved and restored within and across streams (device buffers)" :
            "the state of a sequence is saved and restored within and across streams (host buffers)";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_memory_t mem = llama_get_memory(ctx.get());

        const int32_t strm = llama_memory_stream_add(mem, stream_dev(), 192, true);
        if (strm < 0) {
            return case_result::skip("the memory has no streams");
        }
        CTX_REQUIRE(llama_memory_seq_set_stream(mem, 1, strm), "failed to assign seq 1 to the new stream");

        const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;

        // keep two tokens of the prompt to continue with
        const llama_pos n_tokens = (llama_pos) std::min<size_t>(mr.tokens.size(), 128) - 2;
        CTX_REQUIRE(n_tokens >= 1, "the prompt is too short");

        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            if (!decode_seq(ctx.get(), mr.tokens, 0, n_tokens, seq_id, false)) {
                TLOG_ERR("%s: failed to decode the prompt for sequence %d\n", __func__, seq_id);
                return case_result::fail();
            }
        }

        // decode the next token of the prompt in both sequences and compare the logits
        auto step_equal = [&](llama_pos pos, logits_diff & diff) {
            std::vector<float> logits[2];
            for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
                if (!decode_one(ctx.get(), mr.tokens[pos], pos, seq_id) || !get_current_logits(ctx.get(), logits[seq_id])) {
                    return false;
                }
            }
            diff = compare_logits(logits[0].data(), logits[1].data(), logits[0].size());
            return true;
        };

        std::vector<uint8_t> state;
        logits_diff diff;

        if (!get_seq_state(ctx.get(), 1, flags, state) || !seq_rm(ctx.get(), 1) || !set_seq_state(ctx.get(), 1, flags, state)) {
            return case_result::fail("failed to save and restore seq 1");
        }
        CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == n_tokens - 1, "the restored seq 1 does not hold the prompt");
        CTX_REQUIRE(step_equal(n_tokens, diff), "failed to continue after the restore of seq 1");
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "the restored seq 1 continued differently than seq 0" + diff_str(diff));

        if (!get_seq_state(ctx.get(), 0, flags, state) || !seq_rm(ctx.get(), 1) || !set_seq_state(ctx.get(), 1, flags, state)) {
            return case_result::fail("failed to restore the state of seq 0 into seq 1");
        }
        CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == n_tokens, "seq 1 does not hold the state of seq 0");
        CTX_REQUIRE(step_equal(n_tokens + 1, diff), "failed to continue after the restore across the streams");
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq 1 with the state of seq 0 continued differently" + diff_str(diff));

        return case_result::pass();
    }

private:
    bool on_device;
};

// the memory that the context holds for its KV cells and states, over all devices
static size_t ctx_mem_bytes(const llama_context * ctx) {
    size_t res = 0;
    for (const auto & [_, data] : llama_get_memory_breakdown(ctx)) {
        res += data.context;
    }

    return res;
}

// - add a second stream to the cache without its memory: it takes no sequence
// - take its memory: sequence 1 decodes in it like sequence 0 in the first stream
// - the memory stays for as long as sequence 1 takes its cells from the stream, then it goes back
// - sequence 1 decodes in the first stream meanwhile
// - take the memory again: sequence 1 decodes in the stream as before
struct case_streams_release : ctx_test {
    const char * name() const override { return "streams_r"; }

    const char * description() const override {
        return "a stream gives its memory back and takes it again, and takes no sequence while it is without";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_memory_t mem = llama_get_memory(ctx.get());

        const size_t n_bytes_before = ctx_mem_bytes(ctx.get());

        const int32_t strm = llama_memory_stream_add(mem, stream_dev(), 192, false);
        if (strm < 0) {
            return case_result::skip("the memory has no streams");
        }

        CTX_REQUIRE(ctx_mem_bytes(ctx.get()) == n_bytes_before, "a stream that was added without its memory took memory");
        CTX_REQUIRE(!llama_memory_seq_set_stream(mem, 1, strm), "a stream without its memory took a sequence");

        // what its memory takes is known before it is taken
        const size_t n_bytes_strm = llama_memory_stream_size(mem, strm);
        CTX_REQUIRE(n_bytes_strm > 0, "a stream without its memory does not say what its memory takes");
        CTX_REQUIRE(llama_memory_stream_size(mem, 0) == 0, "the first stream says it is an added one");
        CTX_REQUIRE(llama_memory_stream_release(mem, strm), "giving back what a stream does not have is not an error");

        const llama_pos n_tokens = (llama_pos) std::min<size_t>(mr.tokens.size(), 128);
        CTX_REQUIRE(n_tokens >= 2, "the prompt is too short");

        // the logits of the last token of the prompt, decoded into a sequence
        auto decode_prompt = [&](llama_seq_id seq_id, std::vector<float> & logits) {
            return decode_seq(ctx.get(), mr.tokens, 0, n_tokens - 1, seq_id, false) &&
                   decode_seq(ctx.get(), mr.tokens, n_tokens - 1, n_tokens, seq_id, true) &&
                   get_current_logits(ctx.get(), logits);
        };

        std::vector<float> logits_ref;
        CTX_REQUIRE(decode_prompt(0, logits_ref), "failed to decode the prompt in seq 0");

        size_t n_bytes_held = 0;

        for (int round = 0; round < 2; ++round) {
            const std::string where = round == 0 ? " (first time)" : " (second time)";

            CTX_REQUIRE(llama_memory_stream_alloc(mem, strm), "the stream failed to take its memory" + where);
            CTX_REQUIRE(llama_memory_stream_alloc(mem, strm), "taking what a stream has already is not an error" + where);

            if (round == 0) {
                n_bytes_held = ctx_mem_bytes(ctx.get());
                CTX_REQUIRE(n_bytes_held > n_bytes_before, "the memory of the stream does not show in the memory of the context");
                CTX_REQUIRE(n_bytes_held - n_bytes_before == n_bytes_strm, "the stream took another amount of memory than it said");
            } else {
                CTX_REQUIRE(ctx_mem_bytes(ctx.get()) == n_bytes_held, "the stream took a different amount of memory" + where);
            }

            CTX_REQUIRE(llama_memory_stream_size(mem, strm) == n_bytes_strm, "the stream says another size with its memory" + where);

            CTX_REQUIRE(llama_memory_seq_set_stream(mem, 1, strm), "failed to assign seq 1 to the stream" + where);

            std::vector<float> logits;
            CTX_REQUIRE(decode_prompt(1, logits), "failed to decode the prompt in the stream" + where);
            CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) == n_tokens - 1, "seq 1 does not hold the prompt" + where);

            logits_diff diff = compare_logits(logits_ref.data(), logits.data(), logits_ref.size());
            CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq 1 in the stream produced different logits than seq 0" + where + diff_str(diff));

            CTX_REQUIRE(!llama_memory_stream_release(mem, strm), "the stream gave its memory back under the cells of seq 1" + where);
            CTX_REQUIRE(seq_rm(ctx.get(), 1), "failed to remove seq 1" + where);
            CTX_REQUIRE(!llama_memory_stream_release(mem, strm), "the stream gave its memory back while seq 1 takes its cells from it" + where);
            CTX_REQUIRE(llama_memory_seq_set_stream(mem, 1, 0), "failed to assign seq 1 to the first stream" + where);
            CTX_REQUIRE(llama_memory_stream_release(mem, strm), "the stream failed to give its memory back" + where);

            CTX_REQUIRE(ctx_mem_bytes(ctx.get()) == n_bytes_before, "the memory of the stream is still counted after it went back" + where);
            CTX_REQUIRE(llama_memory_stream_size(mem, strm) == n_bytes_strm, "the stream says another size after its memory went back" + where);
            CTX_REQUIRE(!llama_memory_seq_set_stream(mem, 1, strm), "a stream without its memory took a sequence" + where);

            // the first stream is as it was
            CTX_REQUIRE(decode_prompt(1, logits), "failed to decode the prompt in the first stream" + where);

            diff = compare_logits(logits_ref.data(), logits.data(), logits_ref.size());
            CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq 1 in the first stream produced different logits than seq 0" + where + diff_str(diff));

            CTX_REQUIRE(seq_rm(ctx.get(), 1), "failed to remove seq 1 from the first stream" + where);
        }

        return case_result::pass();
    }
};

// - a context with lazy memory holds none of its cells before a sequence has any
// - a sequence takes the cells that it needs and decodes to the logits of a context that holds its memory whole
// - removing the sequence that lies above the others gives its cells back; removing one below them does not
// - a sequence restores from its saved state into cells that are taken for it
// - clearing the memory gives everything back, and the memory is usable afterwards
// On the CPU the memory that is not taken faults when it is touched, so a stray read or write ends the test.
struct case_lazy_cells : ctx_test {
    const char * name() const override { return "lazy_c"; }

    const char * description() const override {
        return "lazy memory: cells take their memory when sequences need them, and give it back";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 4096;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        ctx_opts opts_lazy = opts;
        opts_lazy.n_seq_budget = 2;
        opts_lazy.quiet        = true;

        auto ctx = mr.ctx(opts_lazy);
        if (!ctx) {
            // LLAMA_TEST_LAZY_STRICT: for a run over models whose memory has to support it
            if (getenv("LLAMA_TEST_LAZY_STRICT")) {
                return case_result::fail("a context with lazy memory was refused");
            }
            return case_result::skip("the memory of this model cannot be lazy");
        }

        auto ctx_ref = mr.ctx(opts);
        if (!ctx_ref) {
            return case_result::fail();
        }

        llama_memory_t mem = llama_get_memory(ctx.get());

        llama_memory_lazy_info info;
        CTX_REQUIRE(!llama_memory_get_lazy_info(llama_get_memory(ctx_ref.get()), &info), "a context with its memory whole says that it is lazy");
        CTX_REQUIRE(llama_memory_get_lazy_info(mem, &info), "a context that was created with lazy memory says that it is not lazy");
        if (info.n_cells == 0) {
            return case_result::skip("the memory of this model has no KV cells");
        }
        CTX_REQUIRE(info.n_cells >= 4096 && info.cell_bytes > 0, "the lazy memory does not say what its cells are");
        CTX_REQUIRE(info.budget > 0, "the lazy memory has no budget");
        CTX_REQUIRE(info.n_cells_taken == 0, "cells are taken before a sequence has any");
        CTX_REQUIRE(info.taken == llama_memory_lazy_bytes(mem, 0, info.n_rows_taken), "memory is taken before a sequence has cells");

        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        // tokens [p0, p1) of a sequence, and the logits of the last one
        auto decode_n = [&](llama_context * c, llama_seq_id seq_id, llama_pos p0, llama_pos p1, std::vector<float> & logits) {
            const llama_pos n_chunk = 128;
            for (llama_pos p = p0; p < p1 - 1; p += n_chunk) {
                if (!decode_gen(c, n_vocab, seq_id, p, std::min(p + n_chunk, p1 - 1))) {
                    return false;
                }
            }
            return decode_one(c, gen_token(n_vocab, seq_id, p1 - 1), p1 - 1, seq_id) && get_current_logits(c, logits);
        };

        // what the memory says it holds is what the marks take, and it stays within the budget
        auto consistent = [&](uint32_t n_cells_expected, std::string & why) {
            llama_memory_lazy_info cur;
            if (!llama_memory_get_lazy_info(mem, &cur)) {
                why = "the memory no longer says that it is lazy";
                return false;
            }
            if (cur.n_cells_taken != n_cells_expected) {
                why = "the cell mark is at " + std::to_string(cur.n_cells_taken) + ", expected " + std::to_string(n_cells_expected);
                return false;
            }
            if (cur.taken != llama_memory_lazy_bytes(mem, cur.n_cells_taken, cur.n_rows_taken)) {
                why = "the memory holds " + std::to_string(cur.taken) + " bytes, and its marks take " +
                      std::to_string(llama_memory_lazy_bytes(mem, cur.n_cells_taken, cur.n_rows_taken));
                return false;
            }
            if (cur.taken > cur.budget) {
                why = "the memory holds more than its budget";
                return false;
            }
            return true;
        };

        // the positions stay within the context that the model was trained with: a model with learned positions
        // has no more of them
        const llama_pos n_train = llama_model_n_ctx_train(mr.model);
        CTX_REQUIRE(n_train >= 16, "the model was trained with too short a context");

        const llama_pos n1 = std::min<llama_pos>(1500, n_train - 2);
        const llama_pos n0 = std::min<llama_pos>(200, n1/2);

        // the marks: the highest cell in use, in steps of 256
        const uint32_t mark0  = GGML_PAD((uint32_t) n0, 256);
        const uint32_t mark01 = GGML_PAD((uint32_t) (n0 + n1 + 1), 256);

        std::string why;

        std::vector<float> logits0;
        std::vector<float> logits1;
        CTX_REQUIRE(decode_n(ctx.get(), 0, 0, n0, logits0), "failed to decode seq 0");
        CTX_REQUIRE(consistent(mark0, why), "after seq 0: " + why);
        CTX_REQUIRE(decode_n(ctx.get(), 1, 0, n1, logits1), "failed to decode seq 1");
        CTX_REQUIRE(consistent(GGML_PAD((uint32_t) (n0 + n1), 256), why), "after seq 1: " + why);

        std::vector<float> ref0;
        std::vector<float> ref1;
        CTX_REQUIRE(decode_n(ctx_ref.get(), 0, 0, n0, ref0), "failed to decode seq 0 with its memory whole");
        CTX_REQUIRE(decode_n(ctx_ref.get(), 1, 0, n1, ref1), "failed to decode seq 1 with its memory whole");

        logits_diff diff = compare_logits(ref0.data(), logits0.data(), ref0.size());
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq 0 produced different logits in lazy memory" + diff_str(diff));
        diff = compare_logits(ref1.data(), logits1.data(), ref1.size());
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq 1 produced different logits in lazy memory" + diff_str(diff));

        // seq 1 lies above seq 0: without it the mark comes down to seq 0, and its state takes the cells again
        std::vector<uint8_t> state1;
        CTX_REQUIRE(get_seq_state(ctx.get(), 1, 0, state1), "failed to save seq 1");
        CTX_REQUIRE(seq_rm(ctx.get(), 1), "failed to remove seq 1");
        CTX_REQUIRE(consistent(mark0, why), "after seq 1 was removed: " + why);
        CTX_REQUIRE(set_seq_state(ctx.get(), 1, 0, state1), "failed to restore seq 1");
        CTX_REQUIRE(consistent(GGML_PAD((uint32_t) (n0 + n1), 256), why), "after seq 1 was restored: " + why);

        CTX_REQUIRE(decode_n(ctx.get(),     1, n1, n1 + 1, logits1), "failed to go on with the restored seq 1");
        CTX_REQUIRE(decode_n(ctx_ref.get(), 1, n1, n1 + 1, ref1),    "failed to go on with seq 1 with its memory whole");
        diff = compare_logits(ref1.data(), logits1.data(), ref1.size());
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "the restored seq 1 produced different logits in lazy memory" + diff_str(diff));

        // seq 0 lies below seq 1: its cells stay taken when it goes
        CTX_REQUIRE(seq_rm(ctx.get(), 0), "failed to remove seq 0");
        CTX_REQUIRE(consistent(mark01, why), "after seq 0 was removed: " + why);

        llama_memory_clear(mem, true);
        CTX_REQUIRE(consistent(0, why), "after the memory was cleared: " + why);

        CTX_REQUIRE(decode_n(ctx.get(), 0, 0, n0, logits0), "failed to decode seq 0 after the memory was cleared");
        CTX_REQUIRE(consistent(mark0, why), "after seq 0 was decoded again: " + why);
        diff = compare_logits(ref0.data(), logits0.data(), ref0.size());
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq 0 produced different logits after the memory went back and was taken again" + diff_str(diff));

        return case_result::pass();
    }
};

// - load the state saved by the baseline case
// - replay its last token
// - generate again and compare against the baseline logits
struct case_state_load : ctx_test {
    const char * name() const override { return "state_load"; }

    const char * description() const override {
        return "a saved session replays to the same logits as the run that saved it";
    }

    bool needs_baseline() const override { return true; }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_seq_max = 2;

        auto ctx  = mr.ctx(opts);
        auto smpl = mr.sampler();
        if (!ctx) {
            return case_result::fail();
        }

        int n_past = 0;
        if (!load_state_replay_last(ctx.get(), mr.params.out_file, mr.tokens, n_past)) {
            return case_result::fail();
        }

        if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 0, mr.baseline, NMSE_THRESHOLD)) {
            return case_result::fail();
        }

        return case_result::pass();
    }
};

// - load the state saved by the baseline case and replay its last token
// - copy sequence 0 to sequence 1 over the given io path, clearing the cache in between
// - continue on sequence 1 and compare against the baseline logits
struct case_seq_cp : ctx_test {
    explicit case_seq_cp(bool on_device_) : on_device(on_device_) {}

    const char * name() const override { return on_device ? "cp_d" : "cp_h"; }

    const char * description() const override {
        return on_device ? "copying a sequence over the device io path keeps its state"
                         : "copying a sequence over the host io path keeps its state";
    }

    bool needs_baseline() const override { return true; }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_seq_max = 2;

        auto ctx  = mr.ctx(opts);
        auto smpl = mr.sampler();
        if (!ctx) {
            return case_result::fail();
        }

        const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;
        TLOG_INF("io path: %s\n", on_device ? "device" : "host");

        int n_past = 0;
        if (!load_state_replay_last(ctx.get(), mr.params.out_file, mr.tokens, n_past)) {
            return case_result::fail();
        }

        {
            std::vector<uint8_t> seq_store;
            if (!get_seq_state(ctx.get(), 0, flags, seq_store)) {
                return case_result::fail();
            }
            TLOG_TRC("%s: seq 0 copied, %zu bytes\n", __func__, seq_store.size());

            llama_memory_clear(llama_get_memory(ctx.get()), true);
            TLOG_TRC("%s: kv cache cleared\n", __func__);

            if (!set_seq_state(ctx.get(), 1, flags, seq_store)) {
                return case_result::fail();
            }
            TLOG_TRC("%s: seq 1 restored, %zu bytes\n", __func__, seq_store.size());
        }

        if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 1, mr.baseline, NMSE_THRESHOLD)) {
            return case_result::fail();
        }

        return case_result::pass();
    }

private:
    bool on_device;
};

// - decode the same prefix on two sequences, interleaving seq 0 cells between the seq 1 cells
// - save the seq 1 state, free the interleaved seq 0 cells, and restore via the given io path
// - the restore destination is non-contiguous: scatter reads are batched per contiguous run
// - save again on the host and compare the two blobs byte for byte
struct case_seq_cp_scatter : ctx_test {
    explicit case_seq_cp_scatter(bool on_device_) : on_device(on_device_) {}

    const char * name() const override { return on_device ? "cp_d_s" : "cp_h_s"; }

    const char * description() const override {
        return "saving and restoring a sequence with a fragmented (scattered) cell range";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;
        TLOG_INF("io path: %s\n", on_device ? "device" : "host");

        // seq 0 cells 0,1,4 interleave the seq 1 cells 2,3,5
        if (!decode_one(ctx.get(), mr.tokens[0], 0, 0) ||
            !decode_one(ctx.get(), mr.tokens[1], 1, 0) ||
            !decode_one(ctx.get(), mr.tokens[0], 0, 1) ||
            !decode_one(ctx.get(), mr.tokens[1], 1, 1) ||
            !decode_one(ctx.get(), mr.tokens[2], 2, 0) ||
            !decode_one(ctx.get(), mr.tokens[2], 2, 1)) {
            TLOG_ERR("%s: failed to build the interleaved state\n", __func__);
            return case_result::fail();
        }

        // host blob: contains the KV data, used for the byte-for-byte comparison
        std::vector<uint8_t> state_before;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
            return case_result::fail();
        }

        // save via the io path under test
        std::vector<uint8_t> state_save;
        if (!get_seq_state(ctx.get(), 1, flags, state_save)) {
            return case_result::fail();
        }
        TLOG_TRC("%s: seq 1 saved via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

        // free seq 0's cells so the ring is fragmented: the restore destination (seq 1's interleaved cells) stays non-contiguous
        if (!seq_rm(ctx.get(), 0)) {
            return case_result::fail();
        }

        // restore via the io path under test
        if (!set_seq_state(ctx.get(), 1, flags, state_save)) {
            return case_result::fail();
        }
        TLOG_TRC("%s: seq 1 restored via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

        std::vector<uint8_t> state_after;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
            return case_result::fail();
        }

        // the blob is serialized in sequence cell order, so identical bytes iff the restore wrote the same KV
        CTX_REQUIRE(blobs_equal(state_before, state_after), "the restored KV state is not byte-identical to the saved state");

        return case_result::pass();
    }

private:
    bool on_device;
};

//
// sessions
//

// - save a sequence, erase it, restore it, save it again
// - the two blobs must be identical
// compares blobs rather than generated text: a partially restored cell still decodes to plausible tokens
struct case_state_roundtrip : ctx_test {
    const char * name() const override { return "rt"; }

    const char * description() const override {
        return "save, erase, restore, save again - the two blobs must be identical";
    }

    case_result run(model_run & mr) const override {
        auto ctx = mr.ctx();
        if (!ctx) {
            return case_result::fail();
        }

        common_batch batch = common_batch_get_one(ctx.get(), mr.tokens);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }

        std::vector<uint8_t> blob_a;
        if (!get_seq_state(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, blob_a)) {
            return case_result::fail();
        }

        if (!seq_rm(ctx.get(), 0)) {
            return case_result::fail();
        }

        if (!set_seq_state(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, blob_a)) {
            TLOG_ERR("%s: failed to restore seq 0\n", __func__);
            return case_result::fail();
        }

        std::vector<uint8_t> blob_b;
        if (!get_seq_state(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, blob_b)) {
            return case_result::fail();
        }

        CTX_REQUIRE(blobs_equal(blob_a, blob_b), "the state changed across a save / erase / restore cycle");

        return case_result::pass();
    }
};

// the partial state holds what seq_rm cannot roll back: recurrent states and SWA windows
struct case_state_partial_size : ctx_test {
    const char * name() const override { return "ps"; }

    const char * description() const override {
        return "the partial state is not larger than the full state";
    }

    case_result run(model_run & mr) const override {
        auto ctx = mr.ctx();
        if (!ctx) {
            return case_result::fail();
        }

        const llama_pos n_tokens = mr.tokens.size();
        const llama_pos n_half   = n_tokens / 2;

        if (!decode_seq(ctx.get(), mr.tokens, 0, n_half, 0, true)) {
            return case_result::fail();
        }

        const size_t partial_a = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const size_t full_a    = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);

        if (!decode_seq(ctx.get(), mr.tokens, n_half, n_tokens, 0, true)) {
            return case_result::fail();
        }

        const size_t partial_b = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const size_t full_b    = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);

        TLOG_INF("%s: partial %zu -> %zu, full %zu -> %zu\n", __func__, partial_a, partial_b, full_a, full_b);

        CTX_REQUIRE(partial_a <= full_a && partial_b <= full_b, "the partial state is larger than the full state");

        return case_result::pass();
    }
};

// a failed restore must leave the sequence empty and must not change the logits of other sequences
struct case_state_restore_failure : ctx_test {
    const char * name() const override { return "rf"; }

    const char * description() const override {
        return "restoring a corrupted state fails and leaves the other sequences untouched";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 4;
        opts.kv_unified = 1;

        // without flash attention, corrupted data left behind by the restore shows up as NaN logits on the other sequences
        opts.fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_memory_t mem = llama_get_memory(ctx.get());
        if (mem == nullptr) {
            TLOG_INF("no memory to test\n");
            return case_result::pass();
        }

        const auto decode = [&](const llama_tokens & inp, llama_seq_id seq_id, std::vector<float> * logits_out) {
            if (!decode_tokens(ctx.get(), inp, seq_id)) {
                TLOG_ERR("%s: failed to decode on sequence %d\n", __func__, seq_id);
                return false;
            }

            if (logits_out && !get_current_logits(ctx.get(), *logits_out)) {
                TLOG_ERR("%s: failed to get logits\n", __func__);
                return false;
            }

            return true;
        };

        const llama_tokens tokens_save  (mr.tokens.begin(), mr.tokens.begin() + std::min<size_t>(24, mr.tokens.size()));
        const llama_tokens tokens_verify(mr.tokens.end() - std::min<size_t>(8, mr.tokens.size()), mr.tokens.end());

        // the cases share a working directory, so the state file is named after the model
        const std::string path = "state-restore-failure." + mr.name + ".tmp.bin";

        llama_memory_clear(mem, true);

        std::vector<float> baseline;
        if (!decode(tokens_verify, 1, &baseline)) {
            return case_result::fail();
        }

        const std::vector<std::pair<const char *, std::function<bool()>>> sub_cases = {
            { "buffer", [&]() {
                std::vector<uint8_t> state(llama_state_seq_get_size(ctx.get(), 0));
                GGML_ASSERT(llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0) == state.size());
                llama_memory_seq_rm(mem, 0, -1, -1);

                if (!corrupt_state(state)) {
                    return false;
                }

                return llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0) == 0;
            }},
            { "file", [&]() {
                GGML_ASSERT(llama_state_seq_save_file(ctx.get(), path.c_str(), 0, tokens_save.data(), tokens_save.size()) > 0);
                llama_memory_seq_rm(mem, 0, -1, -1);

                std::vector<uint8_t> data;
                {
                    std::ifstream f(path, std::ios::binary);
                    data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                }

                if (!corrupt_state(data)) {
                    std::remove(path.c_str());
                    return false;
                }

                {
                    std::ofstream f(path, std::ios::binary);
                    f.write((const char *) data.data(), data.size());
                }

                llama_tokens tokens_out(tokens_save.size());
                size_t n_token_count = 0;
                const size_t nread = llama_state_seq_load_file(ctx.get(), path.c_str(), 0, tokens_out.data(), tokens_out.size(), &n_token_count);
                std::remove(path.c_str());

                return nread == 0;
            }},
        };

        for (const auto & [name, restore_failed] : sub_cases) {
            llama_memory_clear(mem, true);

            if (!decode(tokens_save, 0, nullptr)) {
                return case_result::fail();
            }

            CTX_REQUIRE(restore_failed(), std::string(name) + ": restoring a corrupted state did not fail");

            CTX_REQUIRE(llama_memory_seq_pos_max(mem, 0) == -1,
                    std::string(name) + ": the sequence is not empty after a failed restore");

            std::vector<float> logits;
            if (!decode(tokens_verify, 1, &logits)) {
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(baseline.data(), logits.data(), logits.size());

            CTX_REQUIRE(diff.n_nan == 0 && diff.diff_max <= 1e-6f,
                    std::string(name) + ": the logits changed after a failed restore (max diff = " +
                        std::to_string(diff.diff_max) + ", nan = " + std::to_string(diff.n_nan) + ")");

            TLOG_TRC("%s: %s: logits match (max diff = %g)\n", __func__, name, (double) diff.diff_max);
        }

        return case_result::pass();
    }
};

// a KV state saved with attention rotation enabled must restore only into a context with the
// same setting
// note: rotation is only active for quantized KV caches with a head size that is a multiple of
//       64, for other models the restore into the rotation-disabled context is valid and the
//       test passes vacuously
struct case_state_rotation : ctx_test {
    const char * name() const override { return "rot"; }

    const char * description() const override {
        return "a state saved with attention rotation restores only into a matching context";
    }

    case_result run(model_run & mr) const override {
        const auto make_context = [&](ggml_type type_k, ggml_type type_v, bool disable_rotation) {
            // a setting of the model, not of the process - a sweep that runs several models at a
            //   time must not let this leak into the contexts of another model
            llama_model_set_attn_rot(mr.model, !disable_rotation);

            ctx_opts opts;
            opts.n_ctx    = 32;
            opts.n_batch  = 1;
            opts.n_ubatch = 1;
            opts.type_k   = type_k;
            opts.type_v   = type_v;
            opts.fa       = LLAMA_FLASH_ATTN_TYPE_ENABLED;

            // probing whether this KV type is supported is expected to fail for some models
            opts.quiet = true;

            return mr.ctx(opts);
        };

        std::vector<std::pair<ggml_type, ggml_type>> type_pairs;
        for (const auto & types : { std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0} }) {
            if (make_context(types.first, types.second, false)) {
                type_pairs.push_back(types);
            }
        }
        if (type_pairs.empty()) {
            return case_result::skip("no supported quantized KV cache type combination");
        }

        bool success = true;
        for (const auto & types : type_pairs) {
            auto src = make_context(types.first, types.second, false);
            if (!src) {
                TLOG_ERR("%s: failed to create the source context\n", __func__);
                success = false;
                break;
            }

            llama_token token = 0;
            if (llama_decode(src.get(), llama_batch_get_one(&token, 1))) {
                TLOG_ERR("%s: failed to decode a token\n", __func__);
                success = false;
                break;
            }

            std::vector<uint8_t> state;
            if (!get_seq_state(src.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, state)) {
                if (state.empty()) {
                    continue; // no KV state to test
                }
                success = false;
                break;
            }

            auto matching = make_context(types.first, types.second, false);
            if (!matching || !set_seq_state(matching.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, state)) {
                TLOG_ERR("%s: failed to restore a matching rotation\n", __func__);
                success = false;
                break;
            }

            auto mismatched = make_context(types.first, types.second, true);
            if (!mismatched) {
                TLOG_ERR("%s: failed to create a mismatched rotation context\n", __func__);
                success = false;
                break;
            }
            // a model that does not use attention rotation accepts the mismatched restore
            if (set_seq_state(mismatched.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE, state)) {
                TLOG_TRC("%s: state restored into rotation-disabled context, model does not use attention rotation\n", __func__);
            }
        }

        llama_model_set_attn_rot(mr.model, true); // the default, the cases that follow rely on it

        if (!success) {
            return case_result::fail();
        }

        return case_result::pass();
    }
};

// restore a sequence into a fragmented cache: the slots freed by the removed sequence are
// scattered between live cells of the other sequences, so the restore cannot claim one
// contiguous block
// the restored state must come back byte for byte, and the sequence must continue to behave
// exactly as the same sequence in a context where it was never removed
// ref: https://github.com/ggml-org/llama.cpp/issues/17527
struct case_state_restore_fragmented : ctx_test {
    const char * name() const override { return "frag"; }

    const char * description() const override {
        return "restore a sequence into a cache fragmented by the removal of another sequence";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the interleaved batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 3;
        opts.kv_unified = 1;
        opts.n_batch    = 3*70;
        opts.n_ubatch   = 3*70; // the interleaved prompt is a single multi-seq batch

        // ctx removes and restores seq 1, ctx_keep keeps it untouched as the reference
        llama_context_ptr ctx      = mr.ctx(opts);
        llama_context_ptr ctx_keep = mr.ctx(opts);
        if (!ctx || !ctx_keep) {
            return case_result::fail();
        }

        const llama_pos n_tokens = (llama_pos) std::min<size_t>(70, mr.tokens.size());

        // interleave the 3 sequences: 01201230123... - seq 1 owns every third cell
        const auto decode_interleaved = [&](llama_context * cur) {
            common_batch batch(cur);
            for (llama_pos i = 0; i < n_tokens; i++) {
                for (llama_seq_id s = 0; s < 3; ++s) {
                    batch.add(mr.tokens[i], i, s, false);
                }
            }
            batch.set_output(batch.size() - 1, true);

            return llama_process(cur, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        if (!decode_interleaved(ctx.get()) || !decode_interleaved(ctx_keep.get())) {
            TLOG_ERR("%s: failed to decode the interleaved prompt\n", __func__);
            return case_result::fail();
        }

        TLOG_INF("%s: processed the prompt on seq 0, 1, 2 (%d tokens each)\n", __func__, n_tokens);

        std::vector<uint8_t> state_before;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
            return case_result::fail();
        }

        // clearing seq 1 leaves holes where its cells were - no contiguous block large enough
        // for the seq 1 state is left in the cache
        llama_memory_seq_rm(llama_get_memory(ctx.get()), 1, -1, -1);

        if (!set_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
            TLOG_ERR("%s: failed to restore the seq state into a fragmented cache\n", __func__);
            return case_result::fail();
        }

        TLOG_INF("%s: restored seq 1 into a fragmented cache, %zu bytes\n", __func__, state_before.size());

        // the restore has to be lossless - the cells the sequence ends up in are irrelevant, its
        //   state has to come back exactly as it was saved
        std::vector<uint8_t> state_after;
        if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
            return case_result::fail();
        }

        CTX_REQUIRE(blobs_equal(state_before, state_after), "restoring into a fragmented cache changed the state of seq 1");

        // the restored sequence must continue exactly as the untouched one - decode the same token
        //   on seq 1 in both contexts and compare the logits they produce
        const llama_token token = gen_token(n_vocab, 1, n_tokens);

        if (!decode_one(ctx.get(), token, n_tokens, 1) || !decode_one(ctx_keep.get(), token, n_tokens, 1)) {
            TLOG_ERR("%s: failed to decode with the restored state\n", __func__);
            return case_result::fail();
        }

        const float * logits      = llama_get_logits(ctx.get());
        const float * logits_keep = llama_get_logits(ctx_keep.get());
        if (!logits || !logits_keep) {
            TLOG_ERR("%s: missing logits after the restore\n", __func__);
            return case_result::fail();
        }

        const logits_diff diff = compare_logits(logits, logits_keep, n_vocab);

        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD,
                "the logits of the restored seq 1 differ from the reference: nmse " + std::to_string(diff.nmse()));

        TLOG_INF("%s: restored seq 1 continued identically to the reference, nmse %g\n", __func__, diff.nmse());

        return case_result::pass();
    }
};

//
// no_alloc
//

// - load a second instance of the model with no_alloc, so the tensor data stays on disk
// - create a context from it, which reserves the compute graphs over the dummy buffers
// - the memory use must still be reported, and nothing real may be allocated for the memory
struct case_noalloc : ctx_test {
    const char * name() const override { return "noalloc"; }

    const char * description() const override {
        return "with no_alloc the memory use is simulated: non-zero breakdown, null buffer bases";
    }

    case_result run(model_run & mr) const override {
        const std::string & path = mr.params.model.path;
        CTX_REQUIRE(!path.empty(), "the model path is not known");

        llama_model_params mparams = llama_model_default_params();
        mparams.no_alloc  = true;
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;

        llama_model_ptr model(llama_model_load_from_file(path.c_str(), mparams));
        CTX_REQUIRE(model, "failed to load the model with no_alloc");

        auto ctx = init_ctx(model.get(), common_context_params_to_llama(mr.params), 0);
        CTX_REQUIRE(ctx, "failed to create the context with no_alloc");

        size_t n_ctx_mem = 0;
        for (const auto & [_, data] : llama_get_memory_breakdown(ctx.get())) {
            n_ctx_mem += data.context;
        }

        // models without a memory module (BERT, diffusion) have a null memory
        if (llama_get_memory(ctx.get()) != nullptr) {
            CTX_REQUIRE(n_ctx_mem > 0, "the simulated memory breakdown is zero");
        } else {
            CTX_REQUIRE(n_ctx_mem == 0, "a model without memory reported context memory");
        }

        for (ggml_backend_buffer_t buf : llama_memory_get_buffers(llama_get_memory(ctx.get()))) {
            CTX_REQUIRE(ggml_backend_buffer_get_base(buf) == nullptr, "a memory buffer was really allocated");
        }

        return case_result::pass();
    }
};

// the memory that the context computes with, over all devices
static size_t ctx_compute_bytes(const llama_context * ctx) {
    size_t res = 0;
    for (const auto & [_, data] : llama_get_memory_breakdown(ctx)) {
        res += data.compute;
    }

    return res;
}

// - decode the prompt into sequence 0
// - give the compute memory of every device of the context back
// - decode the prompt into sequence 1: the context takes what it needs again, and the logits are those of sequence 0
// - sequence 0, decoded before the memory went back, continues like sequence 1
struct case_comp_release : ctx_test {
    const char * name() const override { return "comp_release"; }

    const char * description() const override {
        return "the compute memory of a device goes back, and the next decode takes what it needs again";
    }

    case_result run(model_run & mr) const override {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_seq_max  = 2;
        opts.kv_unified = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        // keep one token of the prompt to continue with
        const llama_pos n_tokens = (llama_pos) std::min<size_t>(mr.tokens.size(), 128) - 1;
        CTX_REQUIRE(n_tokens >= 2, "the prompt is too short");

        auto decode_prompt = [&](llama_seq_id seq_id, std::vector<float> & logits) {
            return decode_seq(ctx.get(), mr.tokens, 0, n_tokens - 1, seq_id, false) &&
                   decode_seq(ctx.get(), mr.tokens, n_tokens - 1, n_tokens, seq_id, true) &&
                   get_current_logits(ctx.get(), logits);
        };

        std::vector<float> logits[2];
        CTX_REQUIRE(decode_prompt(0, logits[0]), "failed to decode the prompt in seq 0");

        const size_t n_bytes_before = ctx_compute_bytes(ctx.get());
        CTX_REQUIRE(n_bytes_before > 0, "the context reports no compute memory");

        // the devices of the model, and the CPU
        size_t n_bytes_back = llama_context_release_compute(ctx.get(), nullptr);
        for (int i = 0; i < llama_model_n_devices(mr.model); ++i) {
            n_bytes_back += llama_context_release_compute(ctx.get(), llama_model_get_device(mr.model, i));
        }

        CTX_REQUIRE(n_bytes_back == n_bytes_before, "the compute memory that went back is not what the context had: " +
                std::to_string(n_bytes_back) + " of " + std::to_string(n_bytes_before) + " bytes");
        CTX_REQUIRE(ctx_compute_bytes(ctx.get()) == 0, "compute memory is still counted after it went back");

        CTX_REQUIRE(decode_prompt(1, logits[1]), "failed to decode the prompt after the compute memory went back");
        CTX_REQUIRE(ctx_compute_bytes(ctx.get()) > 0, "the decode took no compute memory");

        logits_diff diff = compare_logits(logits[0].data(), logits[1].data(), logits[0].size());
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "the decode after the compute memory went back produced different logits" + diff_str(diff));

        // what was decoded before is in the memory module, which kept its buffers
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            CTX_REQUIRE(decode_one(ctx.get(), mr.tokens[n_tokens], n_tokens, seq_id) && get_current_logits(ctx.get(), logits[seq_id]),
                    "failed to continue seq " + std::to_string(seq_id));
        }

        diff = compare_logits(logits[0].data(), logits[1].data(), logits[0].size());
        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, "seq 0, decoded before the compute memory went back, continued differently than seq 1" + diff_str(diff));

        return case_result::pass();
    }
};

struct case_comp_padding : ctx_test {
    const char * name() const override { return "comp_padding"; }

    const char * description() const override {
        return "fixed-sized DeepSeek batches reuse the compressed-cache graph";
    }

    std::string skip_reason(model_run & mr) const override {
        char arch_name[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch_name, sizeof(arch_name));
        const llm_arch arch = llama_arch_from_string(arch_name);
        return arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41 ? "" : "requires DeepSeek V4 or V4.1";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        for (const uint32_t n_seq : { 1u, 2u }) {
            ctx_opts opts;
            opts.n_ctx      = 256*n_seq;
            opts.n_batch    = 64;
            // Reserve the DSV4.1 small-batch attention graph when n_seq == 1.
            opts.n_ubatch   = 32;
            opts.n_seq_max  = n_seq;
            opts.kv_unified = n_seq == 1;

            for (const uint32_t n_tokens : { 3u, 5u, 6u, 7u }) {
                auto ctx = mr.ctx(opts);
                CTX_REQUIRE(ctx, "failed to create the context");
                std::vector<llama_pos> pos(n_seq);
                for (uint32_t s = 0; s < n_seq; ++s) {
                    pos[s] = 32 + s;
                    CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, s, 0, pos[s]), "prefill failed");
                }

                // Stay inside the same attention-cache size while crossing compressor boundaries.
                for (int step = 0; step < 3; ++step) {
                    common_batch batch(ctx.get());
                    // Equal lengths keep multiple sequences in one ubatch.
                    for (uint32_t s = 0; s < n_seq; ++s) {
                        for (uint32_t i = 0; i < n_tokens; ++i, ++pos[s]) {
                            batch.add(gen_token(n_vocab, s, pos[s]), pos[s], s, true);
                        }
                    }
                    const int32_t n_reused = llama_perf_context(ctx.get()).n_reused;
                    CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decode failed");
                    if (step > 0) {
                        CTX_REQUIRE(llama_perf_context(ctx.get()).n_reused == n_reused + 1,
                                "compressed-cache graph was rebuilt for " + std::to_string(n_tokens) + " tokens and " + std::to_string(n_seq) + " sequences");
                    }
                }
            }
        }

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_kv_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_baseline>());
    cases.push_back(std::make_unique<case_seq_rm_isolated>());
    cases.push_back(std::make_unique<case_streams>());
    cases.push_back(std::make_unique<case_streams_batch>());
    cases.push_back(std::make_unique<case_stream_span>());
    cases.push_back(std::make_unique<case_streams_multi>());
    cases.push_back(std::make_unique<case_streams_state>(false));
    cases.push_back(std::make_unique<case_streams_state>(true));
    cases.push_back(std::make_unique<case_streams_release>());
    cases.push_back(std::make_unique<case_lazy_cells>());
    cases.push_back(std::make_unique<case_state_load>());
    cases.push_back(std::make_unique<case_seq_cp>(false));
    cases.push_back(std::make_unique<case_seq_cp>(true));
    cases.push_back(std::make_unique<case_seq_cp_scatter>(false));
    cases.push_back(std::make_unique<case_seq_cp_scatter>(true));
    cases.push_back(std::make_unique<case_state_roundtrip>());
    cases.push_back(std::make_unique<case_state_partial_size>());
    cases.push_back(std::make_unique<case_state_restore_failure>());
    cases.push_back(std::make_unique<case_state_rotation>());
    cases.push_back(std::make_unique<case_state_restore_fragmented>());
    cases.push_back(std::make_unique<case_noalloc>());
    cases.push_back(std::make_unique<case_comp_release>());
    cases.push_back(std::make_unique<case_comp_padding>());

    return cases;
}
