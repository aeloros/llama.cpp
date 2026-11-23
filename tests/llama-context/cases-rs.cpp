// recurrent state cases

#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"

#include "../../src/llama-ext.h" // llama_arch_from_string

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// roll back multiple sequences, then replay them in a single batch whose per-seq token count exceeds n_ubatch: each
// seq's replay spans several ubatches while its rollback restore is still pending. Compared against a reference context
// that never advanced past the rollback point and decodes the identical replay batch. Only the first replayed token of
// each seq is compared.
struct case_multi_seq_split_replay : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "replay"; }

    const char * description() const override {
        return "replay rolled-back sequences in a batch wider than n_ubatch, against a reference context";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t  n_seqs     = 2;
        constexpr uint32_t  n_ubatch   = 16;
        constexpr uint32_t  n_prompt   = 20;
        constexpr uint32_t  n_rollback = 3;
        constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
        constexpr llama_pos p0         = n_prompt - n_rollback;

        // both contexts have the same layout, only what they decode differs
        const auto make_ctx_multi = [&]() {
            ctx_opts opts;
            opts.n_ctx      = 512;
            opts.n_batch    = 256;
            opts.n_ubatch   = n_ubatch;
            opts.n_seq_max  = n_seqs;
            opts.n_rs_seq   = 8;
            opts.kv_unified = 0;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_roll = make_ctx_multi();
        llama_context_ptr ctx_ref  = make_ctx_multi();
        if (!ctx_roll || !ctx_ref) {
            return case_result::fail();
        }

        if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        const auto tok = [&](uint32_t seq, llama_pos pos) {
            return gen_token(n_vocab, (llama_seq_id) seq, pos);
        };

        bool ok = true;

        // keep the anchor at p0 - 1 when rolling back the tail
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);

            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) n_prompt);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) p0);

            ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

            // a second partial removal while one is pending must be refused
            ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
        }
        CTX_REQUIRE(ok, "the multi-seq prefill / rollback failed");

        // all seqs replay in a single batch
        const auto decode_replay = [&](llama_context * ctx) {
            common_batch batch(ctx);
            for (uint32_t s = 0; s < n_seqs; ++s) {
                for (uint32_t i = 0; i < n_replay; ++i) {
                    const llama_pos pos = p0 + (llama_pos) i;
                    batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
                }
            }
            return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        ok = decode_replay(ctx_roll.get()) && decode_replay(ctx_ref.get());
        CTX_REQUIRE(ok, "the multi-seq replay decode failed");

        // the first replayed token of each seq reads the state right after the rollback, so its
        //   logits should match; the later tokens are not compared because small state
        //   differences at the start grow with each replayed token
        // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
        constexpr float nmse_eps = 1e-4f;

        logits_diff diff;
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t i = s*n_replay; // first replayed token of this seq

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
                return case_result::fail();
            }

            diff.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(diff.nmse() <= nmse_eps,
                "multi-seq split replay logits mismatch (max diff " + std::to_string(diff.diff_max) +
                ", nmse " + std::to_string(diff.nmse()) + ", first difference at index " + std::to_string(diff.i_first) + ")");

        TLOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff.diff_max, diff.nmse());

        // the replay above can leave the 2 contexts in slightly different states, and the tail
        //   below assumes they start from the same state - so copy the roll state into ref
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            std::vector<uint8_t> state;

            ok = ok && get_seq_state(ctx_roll.get(), (llama_seq_id) s, 0, state);
            ok = ok && set_seq_state(ctx_ref.get(),  (llama_seq_id) s, 0, state);

            // do a second round of state save/load to negate any SWA-positioning effects
            if (ok) {
                std::vector<uint8_t> restored;
                ok = ok && get_seq_state(ctx_ref.get(), (llama_seq_id) s, 0, restored);

                // what was loaded into ref must come back out bit-identical
                CTX_REQUIRE(ok && state == restored,
                        "the seq " + std::to_string(s) + " state did not round-trip bit-exact (sizes " +
                        std::to_string(state.size()) + "/" + std::to_string(restored.size()) + ")");

                ok = ok && set_seq_state(ctx_roll.get(), (llama_seq_id) s, 0, restored);
            }
        }
        CTX_REQUIRE(ok, "the multi-seq state copy failed");

        // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
        //   in ctx_ref only, then compare identical seq-1-only continuations
        constexpr uint32_t n_tail = 4;

        {
            common_batch batch_tail(ctx_ref.get());
            for (uint32_t i = 0; i < n_tail; ++i) {
                const llama_pos pos = p0 + (llama_pos) (n_replay + i);
                batch_tail.add(tok(0, pos + 7), pos, 0, false);
            }
            ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
        }

        logits_diff diff_tail;
        for (uint32_t i = 0; i < n_tail && ok; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1) &&
                 decode_one(ctx_ref.get(),  tok(1, pos), pos, 1);
            if (!ok) {
                break;
            }

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing tail logits at step %u\n", __func__, i);
                return case_result::fail();
            }

            diff_tail.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(ok && diff_tail.nmse() <= 0.0f,
                "a seq-1-only decode leaked the state of seq 0 (nmse " + std::to_string(diff_tail.nmse()) + ")");

        TLOG_INF("%s: seq-1-only decode is independent of seq 0 (max diff %g, nmse %g)\n",
                __func__, (double) diff_tail.diff_max, diff_tail.nmse());

        return case_result::pass();
    }
};

// save a rolled-back single-seq state, restore it into a fresh and into a dirty context, and
// verify exact logit matches on replay
struct case_rollback : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback"; }

    const char * description() const override {
        return "a rolled-back recurrent state restores into fresh and dirty contexts with identical replay";
    }

    case_result run(model_run & mr) const override {
        const llama_vocab * vocab   = llama_model_get_vocab(mr.model);
        const int           n_vocab = llama_vocab_n_tokens(vocab);

        const auto make_rs_ctx = [&]() {
            ctx_opts opts;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_src = make_rs_ctx();
        llama_context_ptr ctx_dst = make_rs_ctx();
        if (!ctx_src || !ctx_dst) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        constexpr uint32_t n_rollback = 3;
        if (n_rs_seq < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        llama_tokens tokens;
        if (!mr.has_vocab()) {
            tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        } else {
            tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
        }
        if (tokens.empty()) {
            TLOG_ERR("%s: not enough prompt tokens\n", __func__);
            return case_result::fail();
        }
        tokens.resize(n_rs_seq + 1, tokens.back());

        const llama_pos rollback_pos = (llama_pos) tokens.size() - n_rollback;

        // Decode the full prompt on the source, then roll back three positions.
        // Replaying them crosses DSV4's ratio-4 compressor boundary.
        // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
        if (!decode_tokens(ctx_src.get(), tokens)) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: rollback failed\n", __func__);
            return case_result::fail();
        }

        // Save the rolled-back state and restore it into a fresh context.
        common_prompt_checkpoint ckpt;
        ckpt.update_tgt(ctx_src.get(), 0, 0);
        ckpt.load_tgt(ctx_dst.get(), 0, 0);

        // exact match - the same context configuration replaying the same tokens
        constexpr float nmse_eps = 0.0;

        std::vector<std::vector<float>> logits_src_replay(n_rollback);
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                TLOG_ERR("%s: missing replay logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);

            const logits_diff diff = compare_logits(logits_src, logits_dst, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "replay logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        // the replay ran in single-token ubatches, which hold no snapshot that deep
        CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) &&
                    !llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1),
                "a rollback past the snapshots of single-token ubatches was accepted");

        // Repeat the load into a context that already has its own rollback state:
        // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
        // non-zero at load time. The restore must wipe that state and still match.
        llama_context_ptr ctx_dirty = make_rs_ctx();
        if (!ctx_dirty) {
            return case_result::fail();
        }

        llama_tokens noise = tokens;
        for (auto & tok : noise) {
            tok = (tok + 1) % n_vocab;
            if (tok < 0) {
                tok = 0;
            }
        }
        if (!decode_tokens(ctx_dirty.get(), noise)) {
            TLOG_ERR("%s: failed to decode the dirty prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: dirty rollback failed\n", __func__);
            return case_result::fail();
        }

        ckpt.load_tgt(ctx_dirty.get(), 0, 0);

        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
            if (logits_dirty == nullptr) {
                TLOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(logits_src_replay[i].data(), logits_dirty, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "dirty-ctx logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        TLOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);

        return case_result::pass();
    }
};

// after a ubatch of n tokens, probe rollbacks from deepest to shallowest: every rollback past the
// snapshots that ubatch guarantees is refused and the deepest guaranteed one is accepted
struct case_rollback_bound : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback_bound"; }

    const char * description() const override {
        return "seq_rm accepts exactly the rollbacks whose snapshots the last ubatch wrote";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr llama_pos n_prefix = 16;

        char arch_name[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch_name, sizeof(arch_name));
        const llm_arch arch = llama_arch_from_string(arch_name);

        // the DSV4 cache rebuilds snapshot d from the state before the ubatch, so d <= n is exact;
        // the recurrent memory gets snapshots 0..n-1 from ggml_gated_delta_net, only for n <= n_rs_seq + 1
        const bool from_ubatch_start = arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41;

        const auto make_ctx = [&]() {
            ctx_opts opts;
            opts.n_batch   = 64;
            opts.n_ubatch  = 64;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx = make_ctx();
        if (!ctx) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        for (const uint32_t n : { 1u, 2u, n_rs_seq + 1, n_rs_seq + 2 }) {
            const uint32_t r_max = from_ubatch_start ? std::min(n, n_rs_seq) : (n <= n_rs_seq + 1 ? n - 1 : 0);

            ctx = make_ctx();
            if (!ctx) {
                return case_result::fail();
            }

            const llama_pos pos_end = n_prefix + (llama_pos) n;

            CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, 0, 0, n_prefix) &&
                        decode_gen(ctx.get(), n_vocab, 0, n_prefix, pos_end),
                    "the prefix / ubatch decode failed");

            for (uint32_t r = n_rs_seq + 1; r > r_max; --r) {
                CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_end - (llama_pos) r, -1),
                        "rollback " + std::to_string(r) + " after a " + std::to_string(n) + "-token ubatch was accepted");
            }

            if (r_max > 0) {
                CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_end - (llama_pos) r_max, -1),
                        "rollback " + std::to_string(r_max) + " after a " + std::to_string(n) + "-token ubatch was refused");
            }
        }

        TLOG_INF("%s: rollback bounds hold for ubatches of 1, 2, %u and %u tokens\n", __func__, n_rs_seq + 1, n_rs_seq + 2);

        return case_result::pass();
    }
};

// decode a prompt into seq 0, share its cells with a second sequence, then keep decoding both
struct case_shared_seq_reserve : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "shared"; }

    const char * description() const override {
        return "sequences sharing cells keep decoding correctly after a seq_cp";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the shared-seq batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_seqs     = 2;
        constexpr uint32_t n_prompt   = 128;
        constexpr uint32_t n_continue = 32;

        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_batch    = 256;
        opts.n_ubatch   = 64;
        opts.n_seq_max  = n_seqs;
        opts.kv_unified = 1; // only a unified cache shares cells on seq_cp

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        if (!decode_gen(ctx.get(), n_vocab, 0, 0, (llama_pos) n_prompt)) {
            TLOG_ERR("%s: prompt decode failed\n", __func__);
            return case_result::fail();
        }

        // this is what llama-batched-bench does for -pps
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

        for (uint32_t i = 0; i < n_continue; ++i) {
            const llama_pos pos = (llama_pos) (n_prompt + i);

            common_batch batch(ctx.get());
            for (uint32_t s = 0; s < n_seqs; ++s) {
                batch.add(gen_token(n_vocab, (llama_seq_id) s, pos), pos, (llama_seq_id) s, true);
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                    "shared-seq decode failed at step " + std::to_string(i));

            for (uint32_t s = 0; s < n_seqs; ++s) {
                const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
                if (logits == nullptr) {
                    TLOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                    return case_result::fail();
                }

                bool finite = true;
                for (int t_ = 0; t_ < n_vocab; ++t_) {
                    finite = finite && std::isfinite(logits[t_]);
                }

                CTX_REQUIRE(finite, "non-finite shared-seq logit at step " + std::to_string(i) + ", seq " + std::to_string(s));
            }
        }

        TLOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);

        return case_result::pass();
    }
};

// One batch holds the last tokens of a sequence, few enough to be rolled back, and tokens of other sequences that a later
// ubatch decodes. That later ubatch re-orders the state cells so that its own sequences are next to each other, and it
// may move the cell of the first sequence while doing so. The rollback has to survive that: a server rolls back the
// rejected draft tokens of one slot after a batch that also read the prompts of other slots.
// Compared against a context that decoded the sequence up to the rollback point and no further.
struct case_rollback_other_seqs : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback_others"; }

    const char * description() const override {
        return "a rollback is exact after a later ubatch of the same batch moved the sequence's state cell";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr llama_pos n_prefix = 8;

        // the sequence that rolls back, and the two whose cells are on either side of its cell
        constexpr llama_seq_id seq_roll = 3;
        constexpr llama_seq_id seq_a    = 1;
        constexpr llama_seq_id seq_b    = 2;

        const auto make_ctx = [&]() {
            ctx_opts opts;
            opts.n_ctx      = 512;
            opts.n_batch    = 64;
            opts.n_ubatch   = 64;
            opts.n_seq_max  = 4;
            opts.n_rs_seq   = 2; // a sampled token and two draft tokens
            opts.kv_unified = 1;
            return mr.ctx(opts);
        };

        {
            llama_context_ptr ctx = make_ctx();
            if (!ctx) {
                return case_result::fail();
            }
            if (llama_n_rs_seq(ctx.get()) < 2) {
                return case_result::skip("n_rs_seq is disabled");
            }
        }

        // the token that follows the rollback point is not the one that was rolled back
        const llama_token tok_next = (gen_token(n_vocab, seq_roll, n_prefix + 1) + 1) % n_vocab;

        // the reference: the sequence alone, up to the rollback point, then the next token
        std::vector<float> logits_ref;
        {
            llama_context_ptr ctx = make_ctx();
            if (!ctx) {
                return case_result::fail();
            }

            CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, seq_roll, 0, n_prefix) &&
                        decode_gen(ctx.get(), n_vocab, seq_roll, n_prefix, n_prefix + 1) &&
                        decode_one(ctx.get(), tok_next, n_prefix + 1, seq_roll) &&
                        get_current_logits(ctx.get(), logits_ref),
                    "the reference decode failed");
        }

        // stale: the cell that the sequence is moved to still has the rollback count of another sequence's 3-token ubatch
        for (const bool stale : { false, true }) {
            const std::string what = stale ? " (moved to a cell with another sequence's snapshots)" : " (moved to a cell without snapshots)";

            llama_context_ptr ctx = make_ctx();
            if (!ctx) {
                return case_result::fail();
            }

            // the sequences take their state cells in the order of their first decode: the cell of seq_roll is
            // between those of seq_a and seq_b
            llama_pos n_b = n_prefix;

            bool ok = true;
            ok = ok && decode_gen(ctx.get(), n_vocab, seq_a,    0, n_prefix);
            ok = ok && decode_gen(ctx.get(), n_vocab, seq_roll, 0, n_prefix);
            ok = ok && decode_gen(ctx.get(), n_vocab, seq_b,    0, n_prefix);
            if (stale) {
                ok = ok && decode_gen(ctx.get(), n_vocab, seq_b, n_b, n_b + 3);
                n_b += 3;
            }
            CTX_REQUIRE(ok, "the prefill failed" + what);

            // one batch: the three last tokens of seq_roll first, then one token of seq_a and eight of seq_b
            // the split gives seq_roll a ubatch of its own, whose snapshots allow a rollback of two tokens, then
            // seq_a and seq_b go on together
            {
                common_batch batch(ctx.get());
                for (llama_pos i = 0; i < 3; ++i) {
                    batch.add(gen_token(n_vocab, seq_roll, n_prefix + i), n_prefix + i, seq_roll, true);
                }
                batch.add(gen_token(n_vocab, seq_a, n_prefix), n_prefix, seq_a, true);
                for (llama_pos i = 0; i < 8; ++i) {
                    batch.add(gen_token(n_vocab, seq_b, n_b + i), n_b + i, seq_b, true);
                }
                CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                        "the batch of the three sequences failed" + what);
            }

            // both draft tokens are rejected
            CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx.get()), seq_roll, n_prefix + 1, -1),
                    "the rollback of two tokens was refused" + what);

            std::vector<float> logits;
            CTX_REQUIRE(decode_one(ctx.get(), tok_next, n_prefix + 1, seq_roll) && get_current_logits(ctx.get(), logits),
                    "the decode after the rollback failed" + what);

            // as in the replay case: the first token after a rollback reads the restored state, and random dummy
            // models drift up to ~1.7e-5
            constexpr float nmse_eps = 1e-4f;

            const logits_diff diff = compare_logits(logits.data(), logits_ref.data(), logits_ref.size());
            CTX_REQUIRE(logits.size() == logits_ref.size() && diff.nmse() <= nmse_eps,
                    "the logits after the rollback differ from the reference" + what + " (max diff " + std::to_string(diff.diff_max) +
                    ", nmse " + std::to_string(diff.nmse()) + ")");

            TLOG_INF("%s: rollback matched%s (max diff %g, nmse %g)\n", __func__, what.c_str(), (double) diff.diff_max, diff.nmse());
        }

        return case_result::pass();
    }
};

// - a context with lazy memory holds no row of the recurrent state before a sequence has one
// - each sequence takes a row, the lowest that is free, and decodes to the logits of a context with its memory whole
// - a row goes back when the sequences above it are gone, and a new sequence takes the lowest free row
// - a sequence restores from its saved state into a row that is taken for it
// - the rows beyond n_seq_budget take their memory out of what the cells have not taken: a sequence that finds none
//   is refused, nothing has changed, and the other goes on
// On the CPU the memory that is not taken faults when it is touched, so a stray read or write ends the test.
// With n_rs > 0 the rows have rollback snapshots: a row is taken in every rollback group, and a rollback reads a
// snapshot row.
struct case_lazy_rows : ctx_test {
    explicit case_lazy_rows(uint32_t n_rs_) : n_rs(n_rs_) {}

    const uint32_t n_rs;

    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return n_rs > 0 ? "lazy_rr" : "lazy_r"; }

    const char * description() const override {
        return n_rs > 0 ? "lazy memory: the rows of the recurrent state with rollback snapshots"
                        : "lazy memory: the rows of the recurrent state take their memory when sequences need them, within the budget";
    }

    case_result run(model_run & mr) const override {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        // tokens [p0, p1) of a sequence, and the logits of the last one
        auto decode_n = [&](llama_context * c, llama_seq_id seq_id, llama_pos p0, llama_pos p1, std::vector<float> & logits) {
            const llama_pos n_chunk = 64;
            for (llama_pos p = p0; p < p1 - 1; p += n_chunk) {
                if (!decode_gen(c, n_vocab, seq_id, p, std::min(p + n_chunk, p1 - 1))) {
                    return false;
                }
            }
            return decode_one(c, gen_token(n_vocab, seq_id, p1 - 1), p1 - 1, seq_id) && get_current_logits(c, logits);
        };

        auto same = [&](const std::vector<float> & ref, const std::vector<float> & got) {
            return ref.size() == got.size() && compare_logits(ref.data(), got.data(), ref.size()).nmse() <= NMSE_THRESHOLD;
        };

        // what the memory says it holds is what its marks take, and it stays within the budget
        auto consistent = [&](llama_memory_t mem, uint32_t n_rows_expected, std::string & why) {
            llama_memory_lazy_info cur;
            if (!llama_memory_get_lazy_info(mem, &cur)) {
                why = "the memory no longer says that it is lazy";
                return false;
            }
            if (cur.n_rows_taken != n_rows_expected) {
                why = "the row mark is at " + std::to_string(cur.n_rows_taken) + ", expected " + std::to_string(n_rows_expected);
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

        ctx_opts opts;
        opts.n_ctx      = 1024;
        opts.n_seq_max  = 4;
        opts.kv_unified = 1;
        opts.n_rs_seq   = n_rs;

        auto ctx_ref = mr.ctx(opts);
        if (!ctx_ref) {
            return case_result::fail();
        }

        if (n_rs > 0 && llama_n_rs_seq(ctx_ref.get()) == 0) {
            return case_result::skip("the model has no rollback snapshots");
        }

        const llama_pos n = 24;

        // of the memory with a budget of four rows, for the one with a budget of one
        size_t n_whole_4     = 0;
        size_t n_row_lazy    = 0;
        size_t n_row_held    = 0;
        bool   is_least_of_4 = false;

        std::string why;
        std::vector<float> logits;
        std::vector<float> ref;

        //
        // the rows, with a budget that holds all of them
        //
        {
            ctx_opts opts_lazy = opts;
            opts_lazy.n_seq_budget = 4;
            opts_lazy.quiet        = true;

            auto ctx = mr.ctx(opts_lazy);
            if (!ctx) {
                // LLAMA_TEST_LAZY_STRICT: for a run over models whose memory has to support it
                if (getenv("LLAMA_TEST_LAZY_STRICT")) {
                    return case_result::fail("a context with lazy memory was refused");
                }
                return case_result::skip("the memory of this model cannot be lazy");
            }

            llama_memory_t mem = llama_get_memory(ctx.get());

            llama_memory_lazy_info info;
            CTX_REQUIRE(llama_memory_get_lazy_info(mem, &info), "a context that was created with lazy memory says that it is not lazy");
            if (info.n_rows == 0 && !getenv("LLAMA_TEST_LAZY_STRICT")) {
                // a recurrent state that is conv state only: its rows are small and are held whole
                return case_result::skip("the recurrent state of this model has no rows that are lazy");
            }
            CTX_REQUIRE(info.n_rows == 4 && info.row_bytes > 0, "the recurrent state of a lazy memory is not lazy");
            CTX_REQUIRE(info.budget > 0 && info.budget <= llama_memory_lazy_bytes(mem, info.n_cells, 4), "the budget is more than all the cells and four rows take");
            CTX_REQUIRE(consistent(mem, 0, why), "before any sequence: " + why);

            // what a memory of four sequences holds in all: its budget and what it holds from the start
            n_whole_4     = info.budget + info.held;
            n_row_lazy    = info.row_bytes;
            n_row_held    = info.held / 4;
            is_least_of_4 = info.budget == llama_memory_lazy_bytes(mem, info.n_cells > 0 ? 1 : 0, 1);

            for (llama_seq_id s = 0; s < 3; ++s) {
                CTX_REQUIRE(decode_n(ctx.get(),     s, 0, n, logits), "failed to decode seq " + std::to_string(s));
                CTX_REQUIRE(decode_n(ctx_ref.get(), s, 0, n, ref),    "failed to decode seq " + std::to_string(s) + " with its memory whole");
                CTX_REQUIRE(consistent(mem, s + 1, why), "after seq " + std::to_string(s) + ": " + why);
                CTX_REQUIRE(same(ref, logits), "seq " + std::to_string(s) + " produced different logits in lazy memory");
            }

            // a sequence goes on by two tokens in one batch, which leaves a snapshot to roll the last one back to
            auto two = [&](llama_context * c, llama_seq_id s, llama_pos p, std::vector<float> & out) {
                common_batch batch(c);
                batch.add(gen_token(n_vocab, s, p),     p,     s, false);
                batch.add(gen_token(n_vocab, s, p + 1), p + 1, s, true);
                return llama_process(c, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0 && get_current_logits(c, out);
            };
            // it rolls its last token back and decodes it again: that reads a snapshot row of its row
            auto again = [&](llama_context * c, llama_seq_id s, llama_pos p, std::vector<float> & out) {
                return llama_memory_seq_rm(llama_get_memory(c), s, p, -1) &&
                       decode_one(c, gen_token(n_vocab, s, p), p, s) && get_current_logits(c, out);
            };
            // a sequence goes on by one token, in both contexts, and has the logits of the memory that is held whole
            auto goes_on = [&](llama_seq_id s, llama_pos p, const std::string & when) {
                if (!decode_n(ctx.get(), s, p, p + 1, logits) || !decode_n(ctx_ref.get(), s, p, p + 1, ref)) {
                    why = "seq " + std::to_string(s) + " failed to go on " + when;
                    return false;
                }
                if (!same(ref, logits)) {
                    why = "seq " + std::to_string(s) + " produced different logits in lazy memory " + when;
                    return false;
                }
                return true;
            };

            if (n_rs > 0) {
                // seq 0 goes on by two tokens, rolls the last one back and decodes it again, which gives the logits
                // of the first time
                std::vector<float> first;
                CTX_REQUIRE(two(ctx.get(), 0, n, first) && two(ctx_ref.get(), 0, n, ref), "seq 0 failed to go on by two tokens");
                CTX_REQUIRE(same(ref, first), "seq 0 produced different logits in lazy memory when it went on");
                CTX_REQUIRE(again(ctx.get(), 0, n + 1, logits) && again(ctx_ref.get(), 0, n + 1, ref), "seq 0 failed to roll back a token and decode it again");
                CTX_REQUIRE(same(first, logits), "seq 0 produced different logits after a rollback in lazy memory");
                CTX_REQUIRE(same(ref, logits), "seq 0 produced different logits after a rollback than with its memory whole");
                CTX_REQUIRE(consistent(mem, 3, why), "after the rollback of seq 0: " + why);
            }

            // The rows that hold memory are the rows in use. Seq 2 holds the highest row: when seq 1 below it is
            // removed, seq 2 moves into that row, with its rollback snapshots, and the memory of the highest goes back.
            llama_pos n2 = n; // tokens of seq 2
            std::vector<float> first2;
            if (n_rs > 0) {
                // seq 2 leaves a snapshot behind before it moves, and rolls back to it afterwards
                CTX_REQUIRE(two(ctx.get(), 2, n2, first2) && two(ctx_ref.get(), 2, n2, ref), "seq 2 failed to go on by two tokens");
                CTX_REQUIRE(same(ref, first2), "seq 2 produced different logits in lazy memory when it went on");
                n2 += 2;
            }

            CTX_REQUIRE(seq_rm(ctx.get(), 1) && seq_rm(ctx_ref.get(), 1), "failed to remove seq 1");
            CTX_REQUIRE(consistent(mem, 2, why), "after seq 1 was removed from the row below the highest: " + why);

            if (n_rs > 0) {
                CTX_REQUIRE(again(ctx.get(), 2, n2 - 1, logits) && again(ctx_ref.get(), 2, n2 - 1, ref), "seq 2 failed to roll back a token after it moved");
                CTX_REQUIRE(same(first2, logits), "seq 2 produced different logits when it rolled back after it moved to a lower row");
                CTX_REQUIRE(same(ref, logits), "seq 2 produced different logits after it moved than with its memory whole");
            }
            CTX_REQUIRE(goes_on(2, n2, "after it moved to a lower row"), why);
            n2 += 1;
            {
                const llama_pos n0 = llama_memory_seq_pos_max(mem, 0) + 1;
                CTX_REQUIRE(goes_on(0, n0, "after another sequence moved"), why);
            }
            CTX_REQUIRE(consistent(mem, 2, why), "after seq 2 moved and both went on: " + why);

            CTX_REQUIRE(seq_rm(ctx.get(), 2) && seq_rm(ctx_ref.get(), 2), "failed to remove seq 2");
            CTX_REQUIRE(consistent(mem, 1, why), "after seq 2 was removed: " + why);

            // a new sequence takes the lowest free row
            CTX_REQUIRE(decode_n(ctx.get(),     3, 0, n, logits), "failed to decode seq 3");
            CTX_REQUIRE(decode_n(ctx_ref.get(), 3, 0, n, ref),    "failed to decode seq 3 with its memory whole");
            CTX_REQUIRE(consistent(mem, 2, why), "after seq 3: " + why);
            CTX_REQUIRE(same(ref, logits), "seq 3 produced different logits in lazy memory");

            // its state takes the row again
            std::vector<uint8_t> state;
            CTX_REQUIRE(get_seq_state(ctx.get(), 3, 0, state), "failed to save seq 3");
            CTX_REQUIRE(seq_rm(ctx.get(), 3), "failed to remove seq 3");
            CTX_REQUIRE(consistent(mem, 1, why), "after seq 3 was removed: " + why);
            CTX_REQUIRE(set_seq_state(ctx.get(), 3, 0, state), "failed to restore seq 3");
            CTX_REQUIRE(consistent(mem, 2, why), "after seq 3 was restored: " + why);

            CTX_REQUIRE(decode_n(ctx.get(),     3, n, n + 1, logits), "failed to go on with the restored seq 3");
            CTX_REQUIRE(decode_n(ctx_ref.get(), 3, n, n + 1, ref),    "failed to go on with seq 3 with its memory whole");
            CTX_REQUIRE(same(ref, logits), "the restored seq 3 produced different logits in lazy memory");

            // A copy of seq 3 shares its row until it goes on by itself, and then has the next one. When seq 0 leaves
            // the lowest row, the copy, which holds the highest, moves into it.
            llama_memory_seq_cp(mem, 3, 1, -1, -1);
            llama_memory_seq_cp(llama_get_memory(ctx_ref.get()), 3, 1, -1, -1);
            CTX_REQUIRE(consistent(mem, 2, why), "after seq 3 was copied to seq 1: " + why);
            CTX_REQUIRE(goes_on(1, n + 1, "as a copy of seq 3"), why);
            CTX_REQUIRE(consistent(mem, 3, why), "after the copy of seq 3 went on by itself: " + why);

            CTX_REQUIRE(seq_rm(ctx.get(), 0) && seq_rm(ctx_ref.get(), 0), "failed to remove seq 0");
            CTX_REQUIRE(consistent(mem, 2, why), "after seq 0 was removed from the lowest row: " + why);
            CTX_REQUIRE(goes_on(1, n + 2, "after it moved to the lowest row"), why);
            CTX_REQUIRE(goes_on(3, n + 1, "after its copy moved"), why);

            // only seq 1 is kept: one row is left, the lowest
            llama_memory_seq_keep(mem, 1);
            llama_memory_seq_keep(llama_get_memory(ctx_ref.get()), 1);
            CTX_REQUIRE(consistent(mem, 1, why), "after all but seq 1 were removed: " + why);
            CTX_REQUIRE(goes_on(1, n + 3, "as the only sequence that is left"), why);

            // A token of two sequences merges them into the row of the one that is named first, without a removal:
            // the row of the other is free then, below a row in use. The next batch finds the rows in use together.
            // (A memory of states only takes no such batch: its split is sequential.)
            if (!arch_reserves_single_seq(mr.model) && info.n_cells > 0) {
                for (llama_seq_id s : { 0, 2 }) {
                    CTX_REQUIRE(decode_n(ctx.get(),     s, 0, n, logits), "failed to decode seq " + std::to_string(s) + " again");
                    CTX_REQUIRE(decode_n(ctx_ref.get(), s, 0, n, ref),    "failed to decode seq " + std::to_string(s) + " again with its memory whole");
                }
                CTX_REQUIRE(consistent(mem, 3, why), "with three sequences again: " + why);

                auto merged = [&](llama_context * c, std::vector<float> & out) {
                    common_batch batch(c);
                    batch.add(gen_token(n_vocab, 2, n), n, std::vector<llama_seq_id>{ 2, 0 }, true);
                    return llama_process(c, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0 && get_current_logits(c, out);
                };
                CTX_REQUIRE(merged(ctx.get(), logits) && merged(ctx_ref.get(), ref), "failed to decode a token of two sequences");
                CTX_REQUIRE(same(ref, logits), "a token of two sequences produced different logits in lazy memory");

                CTX_REQUIRE(goes_on(1, n + 4, "after two other sequences merged into one row"), why);
                CTX_REQUIRE(consistent(mem, 2, why), "after two sequences merged into one row and another went on: " + why);
                CTX_REQUIRE(goes_on(2, n + 1, "after it moved with the sequence that shares its row"), why);
            }

            llama_memory_clear(mem, true);
            CTX_REQUIRE(consistent(mem, 0, why), "after the memory was cleared: " + why);
            CTX_REQUIRE(llama_memory_get_lazy_info(mem, &info) && info.taken == 0, "memory is still taken after the memory was cleared");
        }

        //
        // a budget of one row: a second sequence takes its row out of what the cells have not taken
        //
        {
            llama_memory_clear(llama_get_memory(ctx_ref.get()), true);

            ctx_opts opts_lazy = opts;
            opts_lazy.n_seq_budget = 1;

            auto ctx = mr.ctx(opts_lazy);
            if (!ctx) {
                return case_result::fail();
            }

            llama_memory_t mem = llama_get_memory(ctx.get());

            llama_memory_lazy_info info;
            CTX_REQUIRE(llama_memory_get_lazy_info(mem, &info), "the context with a budget of one row says that it is not lazy");
            CTX_REQUIRE(info.budget > 0 && info.budget <= llama_memory_lazy_bytes(mem, info.n_cells, 1), "the budget is more than all the cells and one row take");

            // The memory holds no more than a whole one of one sequence would: three rows less than the whole one of
            // four, their lazy parts and what they hold from the start. (Unless one of the two budgets is the least
            // that the device's units allow, as with a small model on a GPU.)
            const bool is_least = info.budget == llama_memory_lazy_bytes(mem, info.n_cells > 0 ? 1 : 0, 1);
            if (!is_least && !is_least_of_4) {
                CTX_REQUIRE(info.budget + info.held == n_whole_4 - 3*(n_row_lazy + n_row_held),
                        "with three slots beyond its budget the memory may hold " + std::to_string(info.budget + info.held) +
                        " bytes, and a whole memory of one sequence holds " + std::to_string(n_whole_4 - 3*(n_row_lazy + n_row_held)));
            }

            // seq 0 grows until a second sequence would not find the memory of its row, or the cells are nearly used up
            const llama_pos n_new = 16;

            llama_pos n0 = n;
            CTX_REQUIRE(decode_n(ctx.get(),     0, 0, n0, logits), "failed to decode seq 0");
            CTX_REQUIRE(decode_n(ctx_ref.get(), 0, 0, n0, ref),    "failed to decode seq 0 with its memory whole");

            auto fits = [&]() {
                return llama_memory_lazy_bytes(mem, info.n_cells == 0 ? 0 : (uint32_t) (n0 + n_new), 2) <= info.budget;
            };

            while (fits() && (uint32_t) (n0 + 64 + n_new + 256) <= info.n_cells) {
                CTX_REQUIRE(decode_n(ctx.get(),     0, n0, n0 + 64, logits), "failed to grow seq 0");
                CTX_REQUIRE(decode_n(ctx_ref.get(), 0, n0, n0 + 64, ref),    "failed to grow seq 0 with its memory whole");
                n0 += 64;
            }
            CTX_REQUIRE(consistent(mem, 1, why), "with seq 0 alone: " + why);

            llama_memory_lazy_info before;
            CTX_REQUIRE(llama_memory_get_lazy_info(mem, &before), "the memory no longer says that it is lazy");

            common_batch batch(ctx.get());
            for (llama_pos pos = 0; pos < n_new; ++pos) {
                batch.add(gen_token(n_vocab, 1, pos), pos, 1, pos + 1 == n_new);
            }
            const int32_t res = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());

            if (fits()) {
                // the row of seq 1 costs less than the cells that are left: it takes them
                CTX_REQUIRE(res == 0, "seq 1 was refused although its row fits beside the cells of seq 0");
                CTX_REQUIRE(consistent(mem, 2, why), "after seq 1 took its row from the cells: " + why);
            } else {
                CTX_REQUIRE(res == 1, "a sequence whose row does not fit in the budget was not refused (result " + std::to_string(res) + ")");

                llama_memory_lazy_info after;
                CTX_REQUIRE(llama_memory_get_lazy_info(mem, &after), "the memory no longer says that it is lazy");
                CTX_REQUIRE(after.taken == before.taken && after.n_rows_taken == before.n_rows_taken && after.n_cells_taken == before.n_cells_taken,
                        "the refused batch left memory taken");
                CTX_REQUIRE(llama_memory_seq_pos_max(mem, 1) < 0, "the refused sequence holds state");
            }

            // seq 0 goes on either way
            CTX_REQUIRE(decode_n(ctx.get(),     0, n0, n0 + 1, logits), "seq 0 failed to go on");
            CTX_REQUIRE(decode_n(ctx_ref.get(), 0, n0, n0 + 1, ref),    "seq 0 failed to go on with its memory whole");
            CTX_REQUIRE(same(ref, logits), "seq 0 produced different logits after a second sequence asked for a row");
        }

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_rs_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_rollback>());
    cases.push_back(std::make_unique<case_rollback_bound>());
    cases.push_back(std::make_unique<case_rollback_other_seqs>());
    cases.push_back(std::make_unique<case_multi_seq_split_replay>());
    cases.push_back(std::make_unique<case_shared_seq_reserve>());
    cases.push_back(std::make_unique<case_lazy_rows>(0));
    cases.push_back(std::make_unique<case_lazy_rows>(2));

    return cases;
}
