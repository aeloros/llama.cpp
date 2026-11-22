// The partial form of ggml_flash_attn_ext_rows and ggml_flash_attn_ext_combine, on the CPU backend.
//
// The K/V rows that a sequence attends to may be held in two places, for example two streams of a KV cache on two
// devices. Each place then runs the partial form of the op over its own rows, and the combine has to give what one
// pass over all the rows gives.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct test_case {
    const char * name;
    ggml_type    type_kv;
    int          n_embd;     // of a head, for K and V alike
    int          n_head;
    int          n_head_kv;
    int          n_batch;    // tokens of a sequence
    int          n_seq;
    int          n_rows_a;   // K/V rows held in place a, and in place b
    int          n_rows_b;
    int          n_kv_a;     // mask columns of a sequence over place a, and over place b
    int          n_kv_b;
    float        keep_b;     // the share of place b's columns that a token sees; 0 = b holds nothing for anybody
};

static std::mt19937 rng(1234);

static float rnd() {
    return std::normal_distribution<float>(0.0f, 1.0f)(rng);
}

static void fill_f32(ggml_tensor * t) {
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        d[i] = rnd();
    }
}

// rows [0, n_rows) of src (F32, [n_embd, n_rows_src, n_head_kv]) starting at row0, as a tensor of the given type
static ggml_tensor * make_kv(ggml_context * ctx, ggml_type type, const ggml_tensor * src, int64_t row0, int64_t n_rows) {
    const int64_t n_embd    = src->ne[0];
    const int64_t n_head_kv = src->ne[2];
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, n_embd, n_rows, n_head_kv);
    for (int64_t h = 0; h < n_head_kv; h++) {
        for (int64_t r = 0; r < n_rows; r++) {
            const float * s = (const float *) ((const char *) src->data + (row0 + r)*src->nb[1] + h*src->nb[2]);
            void        * d = (char *) t->data + r*t->nb[1] + h*t->nb[2];
            if (type == GGML_TYPE_F32) {
                memcpy(d, s, n_embd*sizeof(float));
            } else if (type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(s, (ggml_fp16_t *) d, n_embd);
            } else {
                ggml_quantize_chunk(type, s, d, 0, 1, n_embd, nullptr);
            }
        }
    }
    return t;
}

struct compare_result {
    double nmse;
    double max_abs;
};

static compare_result compare(const ggml_tensor * a, const ggml_tensor * b) {
    const float * x = (const float *) a->data;
    const float * y = (const float *) b->data;
    double err = 0.0, ref = 0.0, max_abs = 0.0;
    for (int64_t i = 0; i < ggml_nelements(a); i++) {
        const double d = (double) x[i] - (double) y[i];
        err += d*d;
        ref += (double) x[i]*(double) x[i];
        max_abs = std::fmax(max_abs, std::fabs(d));
    }
    return { ref > 0.0 ? err/ref : err, max_abs };
}

static bool run(const test_case & tc, int n_threads) {
    ggml_init_params ip = { /*.mem_size =*/ 256u*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);

    const int n_kv   = tc.n_kv_a + tc.n_kv_b;
    const int n_rows = tc.n_rows_a + tc.n_rows_b;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.n_embd, tc.n_batch, tc.n_head, tc.n_seq);
    fill_f32(q);

    // the rows of both places, place a first; K and V of every place are made from these
    ggml_tensor * k_src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.n_embd, n_rows, tc.n_head_kv);
    ggml_tensor * v_src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.n_embd, n_rows, tc.n_head_kv);
    fill_f32(k_src);
    fill_f32(v_src);

    ggml_tensor * k_all = make_kv(ctx, tc.type_kv, k_src, 0, n_rows);
    ggml_tensor * v_all = make_kv(ctx, tc.type_kv, v_src, 0, n_rows);
    ggml_tensor * k_a   = make_kv(ctx, tc.type_kv, k_src, 0, tc.n_rows_a);
    ggml_tensor * v_a   = make_kv(ctx, tc.type_kv, v_src, 0, tc.n_rows_a);
    ggml_tensor * k_b   = make_kv(ctx, tc.type_kv, k_src, tc.n_rows_a, tc.n_rows_b);
    ggml_tensor * v_b   = make_kv(ctx, tc.type_kv, v_src, tc.n_rows_a, tc.n_rows_b);

    // row lists and masks: the columns of a sequence over place a, then those over place b
    ggml_tensor * rows_all = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv,      tc.n_seq);
    ggml_tensor * rows_a   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tc.n_kv_a, tc.n_seq);
    ggml_tensor * rows_b   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tc.n_kv_b, tc.n_seq);
    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv,      tc.n_batch, 1, tc.n_seq);
    ggml_tensor * mask_a   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.n_kv_a, tc.n_batch, 1, tc.n_seq);
    ggml_tensor * mask_b   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.n_kv_b, tc.n_batch, 1, tc.n_seq);

    const ggml_fp16_t zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (int s = 0; s < tc.n_seq; s++) {
        int32_t * ra  = (int32_t *) ((char *) rows_a->data   + s*rows_a->nb[1]);
        int32_t * rb  = (int32_t *) ((char *) rows_b->data   + s*rows_b->nb[1]);
        int32_t * rall = (int32_t *) ((char *) rows_all->data + s*rows_all->nb[1]);
        // a tenth of the columns are padding: a negative row, masked for every token
        for (int i = 0; i < tc.n_kv_a; i++) {
            ra[i]   = uni(rng) < 0.1f ? -1 : (int32_t) (rng() % tc.n_rows_a);
            rall[i] = ra[i];
        }
        for (int i = 0; i < tc.n_kv_b; i++) {
            rb[i]               = uni(rng) < 0.1f ? -1 : (int32_t) (rng() % tc.n_rows_b);
            rall[tc.n_kv_a + i] = rb[i] < 0 ? -1 : tc.n_rows_a + rb[i];
        }
        for (int t = 0; t < tc.n_batch; t++) {
            ggml_fp16_t * ma   = (ggml_fp16_t *) ((char *) mask_a->data   + t*mask_a->nb[1]   + s*mask_a->nb[3]);
            ggml_fp16_t * mb   = (ggml_fp16_t *) ((char *) mask_b->data   + t*mask_b->nb[1]   + s*mask_b->nb[3]);
            ggml_fp16_t * mall = (ggml_fp16_t *) ((char *) mask_all->data + t*mask_all->nb[1] + s*mask_all->nb[3]);
            int n_seen = 0;
            for (int i = 0; i < tc.n_kv_a; i++) {
                const bool seen = ra[i] >= 0 && uni(rng) < 0.7f;
                ma[i]   = seen ? zero : neg_inf;
                mall[i] = ma[i];
                n_seen += seen;
            }
            if (n_seen == 0) {
                // every token sees at least one row of place a
                int i = 0;
                while (ra[i] < 0) { ra[i] = 0; rall[i] = 0; }
                ma[i] = zero;
                mall[i] = zero;
            }
            for (int i = 0; i < tc.n_kv_b; i++) {
                const bool seen = rb[i] >= 0 && uni(rng) < tc.keep_b;
                mb[i]               = seen ? zero : neg_inf;
                mall[tc.n_kv_a + i] = mb[i];
            }
        }
    }

    const float scale = 1.0f/std::sqrt((float) tc.n_embd);

    ggml_tensor * full    = ggml_flash_attn_ext_rows(ctx, q, k_all, v_all, mask_all, rows_all, scale, 0.0f, 0.0f);
    ggml_tensor * part_a  = ggml_flash_attn_ext_rows_partial(ctx, q, k_a, v_a, mask_a, rows_a, scale, 0.0f, 0.0f);
    ggml_tensor * part_b  = ggml_flash_attn_ext_rows_partial(ctx, q, k_b, v_b, mask_b, rows_b, scale, 0.0f, 0.0f);
    ggml_tensor * joined  = ggml_flash_attn_ext_combine(ctx, part_a, part_b);
    ggml_tensor * swapped = ggml_flash_attn_ext_combine(ctx, part_b, part_a);
    // one place alone: its partial result combined with that of a place that holds nothing for anybody
    ggml_tensor * only_a  = ggml_flash_attn_ext_rows(ctx, q, k_a, v_a, mask_a, rows_a, scale, 0.0f, 0.0f);
    ggml_tensor * mask_0  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.n_kv_b, tc.n_batch, 1, tc.n_seq);
    for (int64_t i = 0; i < ggml_nelements(mask_0); i++) {
        ((ggml_fp16_t *) mask_0->data)[i] = neg_inf;
    }
    ggml_tensor * part_0  = ggml_flash_attn_ext_rows_partial(ctx, q, k_b, v_b, mask_0, rows_b, scale, 0.0f, 0.0f);
    ggml_tensor * alone   = ggml_flash_attn_ext_combine(ctx, part_a, part_0);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, joined);
    ggml_build_forward_expand(gf, swapped);
    ggml_build_forward_expand(gf, only_a);
    ggml_build_forward_expand(gf, alone);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    bool ok = true;

    // the shapes: a partial result has two more values per head and token, the combined result has the op's shape
    ok = ok && part_a->ne[0] == tc.n_embd + 2 && part_a->ne[1] == full->ne[1] && part_a->ne[2] == full->ne[2] && part_a->ne[3] == full->ne[3];
    ok = ok && ggml_are_same_shape(joined, full);

    const compare_result c_joined  = compare(full,   joined);
    const compare_result c_swapped = compare(full,   swapped);
    const compare_result c_alone   = compare(only_a, alone);

    // the same rows in another order of summation: the results differ in the last bits only
    const double limit = 1e-9;
    ok = ok && c_joined.nmse < limit && c_swapped.nmse < limit && c_alone.nmse < limit;

    bool finite = true;
    for (int64_t i = 0; i < ggml_nelements(joined); i++) {
        finite = finite && std::isfinite(((const float *) joined->data)[i]) && std::isfinite(((const float *) alone->data)[i]);
    }
    ok = ok && finite;

    printf("%-34s threads %d: two places nmse %.2e (max %.2e), swapped %.2e, one place alone %.2e, %s  %s\n",
            tc.name, n_threads, c_joined.nmse, c_joined.max_abs, c_swapped.nmse, c_alone.nmse,
            finite ? "finite" : "NOT FINITE", ok ? "OK" : "FAIL");

    ggml_free(ctx);
    return ok;
}

int main() {
    const std::vector<test_case> cases = {
        //  name                          type_kv         embd head hkv batch seq rows_a rows_b kv_a kv_b keep_b
        { "f16, one sequence",            GGML_TYPE_F16,   64,   4,  2,    7,  1,    96,    64,  48,  32, 0.7f },
        { "f16, three sequences",         GGML_TYPE_F16,   64,   4,  2,    5,  3,   128,   128,  64,  64, 0.7f },
        { "f16, one token",               GGML_TYPE_F16,   64,   8,  2,    1,  2,   256,   512, 128, 256, 0.9f },
        { "f16, a batch of 64",           GGML_TYPE_F16,   64,   4,  4,   64,  1,   512,   256, 256, 128, 0.5f },
        { "f16, place b holds little",    GGML_TYPE_F16,   64,   4,  2,    9,  2,    96,    64,  48,  32, 0.05f },
        { "f16, place b holds nothing",   GGML_TYPE_F16,   64,   4,  2,    9,  2,    96,    64,  48,  32, 0.0f },
        { "f32",                          GGML_TYPE_F32,   32,   4,  2,    6,  2,    64,    64,  32,  32, 0.7f },
        { "q8_0",                         GGML_TYPE_Q8_0,  64,   4,  2,    6,  2,    64,    96,  32,  48, 0.7f },
        { "q4_0",                         GGML_TYPE_Q4_0,  64,   4,  2,    6,  2,    64,    96,  32,  48, 0.7f },
        { "f16, heads of 256",            GGML_TYPE_F16,  256,  24,  4,    3,  2,   128,   128,  64,  64, 0.7f },
    };

    int n_fail = 0;
    for (const test_case & tc : cases) {
        for (int n_threads : { 1, 4 }) {
            n_fail += run(tc, n_threads) ? 0 : 1;
        }
    }

    printf("%s: %d of %zu runs failed\n", n_fail == 0 ? "OK" : "FAILED", n_fail, cases.size()*2);
    return n_fail == 0 ? 0 : 1;
}
