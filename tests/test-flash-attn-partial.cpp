// The partial form of ggml_flash_attn_ext_rows and ggml_flash_attn_ext_combine, on every device of the build.
//
// The K/V rows that a sequence attends to may be held in two places, for example two streams of a KV cache on two
// devices. Each place then runs the partial form of the op over its own rows, and the combine has to give what one
// pass over all the rows gives. The two places may be two different devices, so a partial result of one device has
// to combine with one of another.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

struct test_case {
    const char * name;
    ggml_type    type_kv;
    bool         cache_layout; // K and V as a KV cache holds them, [n_embd, n_head_kv, n_rows], permuted for the op
    int          n_embd;       // of a head, for K and V alike
    int          n_head;
    int          n_head_kv;
    int          n_batch;      // tokens of a sequence
    int          n_seq;
    int          n_rows_a;     // K/V rows held in place a, and in place b
    int          n_rows_b;
    int          n_kv_a;       // mask columns of a sequence over place a, and over place b
    int          n_kv_b;
    float        keep_b;       // the share of place b's columns that a token sees; 0 = b holds nothing for anybody
    float        pad_tail;     // the share at the end of a sequence's lists that is padding, more for each further sequence
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

// rows [row0, row0 + n_rows) of src (F32, [n_embd, n_rows_src, n_head_kv]) as a tensor of the given type:
// [n_embd, n_rows, n_head_kv], or [n_embd, n_head_kv, n_rows] in the layout of a KV cache
static ggml_tensor * make_kv(ggml_context * ctx, ggml_type type, bool cache_layout, const ggml_tensor * src, int64_t row0, int64_t n_rows) {
    const int64_t n_embd    = src->ne[0];
    const int64_t n_head_kv = src->ne[2];
    ggml_tensor * t = cache_layout ? ggml_new_tensor_3d(ctx, type, n_embd, n_head_kv, n_rows) : ggml_new_tensor_3d(ctx, type, n_embd, n_rows, n_head_kv);
    for (int64_t h = 0; h < n_head_kv; h++) {
        for (int64_t r = 0; r < n_rows; r++) {
            const float * s = (const float *) ((const char *) src->data + (row0 + r)*src->nb[1] + h*src->nb[2]);
            void        * d = cache_layout ? (char *) t->data + h*t->nb[1] + r*t->nb[2] : (char *) t->data + r*t->nb[1] + h*t->nb[2];
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

typedef std::vector<float> values;

struct compare_result {
    double nmse;     // over all values
    double nmse_row; // the largest of the rows: one head of one token
    double max_abs;
};

static compare_result compare(const values & x, const values & y, int64_t n_row) {
    double err = 0.0, ref = 0.0, max_abs = 0.0, nmse_row = 0.0;
    if (x.size() != y.size()) {
        return { INFINITY, INFINITY, INFINITY };
    }
    for (size_t i0 = 0; i0 < x.size(); i0 += n_row) {
        double err_row = 0.0, ref_row = 0.0;
        for (size_t i = i0; i < i0 + n_row; i++) {
            const double d = (double) x[i] - (double) y[i];
            err_row += d*d;
            ref_row += (double) x[i]*(double) x[i];
            max_abs = std::fmax(max_abs, std::fabs(d));
        }
        err += err_row;
        ref += ref_row;
        nmse_row = std::fmax(nmse_row, ref_row > 0.0 ? err_row/ref_row : err_row);
    }
    return { ref > 0.0 ? err/ref : err, nmse_row, max_abs };
}

static bool all_finite(const values & x) {
    for (const float f : x) {
        if (!std::isfinite(f)) {
            return false;
        }
    }
    return true;
}

// a partial result holds, for one head of one token, n_row values, then the maximum of the scores and their sum:
// the values and the sum are finite, and the maximum is, wherever the place holds anything for the token
static bool partial_is_sound(const values & x, int64_t n_row) {
    for (size_t i0 = 0; i0 + n_row + 2 <= x.size(); i0 += n_row + 2) {
        for (size_t i = i0; i < i0 + n_row; i++) {
            if (!std::isfinite(x[i])) {
                return false;
            }
        }
        const float M = x[i0 + n_row];
        const float S = x[i0 + n_row + 1];
        if (!std::isfinite(S) || S < 0.0f || std::isnan(M) || M == INFINITY || (S != 0.0f && !std::isfinite(M))) {
            return false;
        }
    }
    return x.size() % (n_row + 2) == 0;
}

typedef std::array<int64_t, 4> shape;

typedef std::function<std::vector<ggml_tensor *>(ggml_context *, const std::vector<ggml_tensor *> &)> build_fn;

// Computes on the backend what build makes of copies of the host tensors, and reads the values back.
// Returns false and says which op it is when the backend does not support an op of the graph.
static bool compute(ggml_backend_t backend, const std::vector<ggml_tensor *> & host, const build_fn & build,
        std::vector<values> & out, std::vector<shape> * out_shapes, std::string & unsupported) {
    ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead()*128 + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);

    std::vector<ggml_tensor *> leaves;
    for (const ggml_tensor * h : host) {
        leaves.push_back(ggml_dup_tensor(ctx, h));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (size_t i = 0; i < host.size(); i++) {
        ggml_backend_tensor_set(leaves[i], host[i]->data, 0, ggml_nbytes(host[i]));
    }

    const std::vector<ggml_tensor *> outs = build(ctx, leaves);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * o : outs) {
        ggml_build_forward_expand(gf, o);
    }

    bool ok = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf) && ok; i++) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, node)) {
            unsupported = ggml_op_name(node->op);
            if (node->op == GGML_OP_FLASH_ATTN_EXT && ggml_flash_attn_ext_is_partial(node)) {
                unsupported += " in its partial form";
            }
            ok = false;
        }
    }

    out.clear();
    if (ok) {
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ok = ggml_gallocr_alloc_graph(galloc, gf) && ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        if (!ok) {
            unsupported = "(the graph did not compute)";
        }
        for (const ggml_tensor * o : outs) {
            values v(ggml_nelements(o));
            if (ok) {
                ggml_backend_tensor_get(o, v.data(), 0, ggml_nbytes(o));
            }
            out.push_back(std::move(v));
        }
        ggml_gallocr_free(galloc);
    }
    if (out_shapes) {
        out_shapes->clear();
        for (const ggml_tensor * o : outs) {
            out_shapes->push_back({ o->ne[0], o->ne[1], o->ne[2], o->ne[3] });
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

struct device {
    std::string    name;
    ggml_backend_t backend;
    bool           is_cpu;
};

// the inputs of a case, in system memory
struct inputs {
    ggml_context * ctx;
    ggml_tensor  * q;
    ggml_tensor  * k_all, * v_all, * k_a, * v_a, * k_b, * v_b;
    ggml_tensor  * rows_all, * rows_a, * rows_b;
    ggml_tensor  * mask_all, * mask_a, * mask_b, * mask_0;
    float          scale;
};

static inputs make_inputs(const test_case & tc) {
    ggml_init_params ip = { /*.mem_size =*/ 1024u*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    inputs in = {};
    in.ctx = ggml_init(ip);
    ggml_context * ctx = in.ctx;

    const int n_kv   = tc.n_kv_a + tc.n_kv_b;
    const int n_rows = tc.n_rows_a + tc.n_rows_b;

    in.q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.n_embd, tc.n_batch, tc.n_head, tc.n_seq);
    fill_f32(in.q);

    // the rows of both places, place a first; K and V of every place are made from these
    ggml_tensor * k_src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.n_embd, n_rows, tc.n_head_kv);
    ggml_tensor * v_src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.n_embd, n_rows, tc.n_head_kv);
    fill_f32(k_src);
    fill_f32(v_src);

    in.k_all = make_kv(ctx, tc.type_kv, tc.cache_layout, k_src, 0, n_rows);
    in.v_all = make_kv(ctx, tc.type_kv, tc.cache_layout, v_src, 0, n_rows);
    in.k_a   = make_kv(ctx, tc.type_kv, tc.cache_layout, k_src, 0, tc.n_rows_a);
    in.v_a   = make_kv(ctx, tc.type_kv, tc.cache_layout, v_src, 0, tc.n_rows_a);
    in.k_b   = make_kv(ctx, tc.type_kv, tc.cache_layout, k_src, tc.n_rows_a, tc.n_rows_b);
    in.v_b   = make_kv(ctx, tc.type_kv, tc.cache_layout, v_src, tc.n_rows_a, tc.n_rows_b);

    // row lists and masks: the columns of a sequence over place a, then those over place b
    in.rows_all = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv,      tc.n_seq);
    in.rows_a   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tc.n_kv_a, tc.n_seq);
    in.rows_b   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tc.n_kv_b, tc.n_seq);
    in.mask_all = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv,      tc.n_batch, 1, tc.n_seq);
    in.mask_a   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.n_kv_a, tc.n_batch, 1, tc.n_seq);
    in.mask_b   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.n_kv_b, tc.n_batch, 1, tc.n_seq);
    in.mask_0   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.n_kv_b, tc.n_batch, 1, tc.n_seq);

    const ggml_fp16_t zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (int s = 0; s < tc.n_seq; s++) {
        int32_t * ra   = (int32_t *) ((char *) in.rows_a->data   + s*in.rows_a->nb[1]);
        int32_t * rb   = (int32_t *) ((char *) in.rows_b->data   + s*in.rows_b->nb[1]);
        int32_t * rall = (int32_t *) ((char *) in.rows_all->data + s*in.rows_all->nb[1]);
        // the lists of a sequence end with padding, as those of a KV cache do, and a tenth of the other columns are
        // padding too: a negative row, masked for every token
        const float share_used = 1.0f - std::fmin(0.9f, tc.pad_tail*(s + 1));
        const int   n_used_a   = std::max(1, (int) (tc.n_kv_a*share_used));
        const int   n_used_b   = (int) (tc.n_kv_b*share_used);
        for (int i = 0; i < tc.n_kv_a; i++) {
            ra[i]   = i >= n_used_a || uni(rng) < 0.1f ? -1 : (int32_t) (rng() % tc.n_rows_a);
            rall[i] = ra[i];
        }
        for (int i = 0; i < tc.n_kv_b; i++) {
            rb[i]               = i >= n_used_b || uni(rng) < 0.1f ? -1 : (int32_t) (rng() % tc.n_rows_b);
            rall[tc.n_kv_a + i] = rb[i] < 0 ? -1 : tc.n_rows_a + rb[i];
        }
        for (int t = 0; t < tc.n_batch; t++) {
            ggml_fp16_t * ma   = (ggml_fp16_t *) ((char *) in.mask_a->data   + t*in.mask_a->nb[1]   + s*in.mask_a->nb[3]);
            ggml_fp16_t * mb   = (ggml_fp16_t *) ((char *) in.mask_b->data   + t*in.mask_b->nb[1]   + s*in.mask_b->nb[3]);
            ggml_fp16_t * mall = (ggml_fp16_t *) ((char *) in.mask_all->data + t*in.mask_all->nb[1] + s*in.mask_all->nb[3]);
            int n_seen = 0;
            for (int i = 0; i < tc.n_kv_a; i++) {
                const bool seen = ra[i] >= 0 && uni(rng) < 0.7f;
                ma[i]   = seen ? zero : neg_inf;
                mall[i] = ma[i];
                n_seen += seen;
            }
            if (n_seen == 0) {
                // every token sees at least one row of place a
                if (ra[0] < 0) {
                    ra[0]   = 0;
                    rall[0] = 0;
                }
                ma[0]   = zero;
                mall[0] = zero;
            }
            for (int i = 0; i < tc.n_kv_b; i++) {
                const bool seen = rb[i] >= 0 && uni(rng) < tc.keep_b;
                mb[i]               = seen ? zero : neg_inf;
                mall[tc.n_kv_a + i] = mb[i];
            }
        }
    }
    for (int64_t i = 0; i < ggml_nelements(in.mask_0); i++) {
        ((ggml_fp16_t *) in.mask_0->data)[i] = neg_inf;
    }

    in.scale = 1.0f/std::sqrt((float) tc.n_embd);
    return in;
}

// a partial result as a tensor in system memory, to hand it to another device
static ggml_tensor * host_tensor(ggml_context * ctx, const values & v, const int64_t * ne) {
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    memcpy(t->data, v.data(), v.size()*sizeof(float));
    return t;
}

struct outputs {
    values  full, only_a, part_a, part_b, joined, swapped, alone;
    int64_t ne_part[4];
    bool    shapes_ok;
};

enum run_status { RUN_OK, RUN_SKIPPED, RUN_FAILED };

// everything of a case on one device: the op over all rows, the partial form over each place, and the combines
static run_status run_on(const device & dev, const test_case & tc, const inputs & in, outputs & out, std::string & why) {
    const bool  cl    = tc.cache_layout;
    const float scale = in.scale;

    const auto kv = [cl](ggml_context * ctx, ggml_tensor * t) {
        return cl ? ggml_permute(ctx, t, 0, 2, 1, 3) : t;
    };

    // the op itself decides whether the device takes part: a shape that the device does not run is not a failure
    {
        std::vector<values> res;
        const build_fn build = [&](ggml_context * ctx, const std::vector<ggml_tensor *> & t) {
            return std::vector<ggml_tensor *> { ggml_flash_attn_ext_rows(ctx, t[0], kv(ctx, t[1]), kv(ctx, t[2]), t[3], t[4], scale, 0.0f, 0.0f) };
        };
        if (!compute(dev.backend, { in.q, in.k_all, in.v_all, in.mask_all, in.rows_all }, build, res, nullptr, why)) {
            return RUN_SKIPPED;
        }
        out.full = res[0];
    }

    std::vector<values> res;
    std::vector<shape>  res_ne;
    const build_fn build = [&](ggml_context * ctx, const std::vector<ggml_tensor *> & t) {
        ggml_tensor * q = t[0];
        ggml_tensor * only_a  = ggml_flash_attn_ext_rows        (ctx, q, kv(ctx, t[1]), kv(ctx, t[2]), t[5], t[8], scale, 0.0f, 0.0f);
        ggml_tensor * part_a  = ggml_flash_attn_ext_rows_partial(ctx, q, kv(ctx, t[1]), kv(ctx, t[2]), t[5], t[8], scale, 0.0f, 0.0f);
        ggml_tensor * part_b  = ggml_flash_attn_ext_rows_partial(ctx, q, kv(ctx, t[3]), kv(ctx, t[4]), t[6], t[9], scale, 0.0f, 0.0f);
        // one place alone: its partial result combined with that of a place that holds nothing for anybody
        ggml_tensor * part_0  = ggml_flash_attn_ext_rows_partial(ctx, q, kv(ctx, t[3]), kv(ctx, t[4]), t[7], t[9], scale, 0.0f, 0.0f);
        ggml_tensor * joined  = ggml_flash_attn_ext_combine(ctx, part_a, part_b);
        ggml_tensor * swapped = ggml_flash_attn_ext_combine(ctx, part_b, part_a);
        ggml_tensor * alone   = ggml_flash_attn_ext_combine(ctx, part_a, part_0);
        // the partial results are read back, so they must not share their memory with a later node
        ggml_set_output(part_a);
        ggml_set_output(part_b);
        return std::vector<ggml_tensor *> { only_a, part_a, part_b, joined, swapped, alone };
    };
    if (!compute(dev.backend, { in.q, in.k_a, in.v_a, in.k_b, in.v_b, in.mask_a, in.mask_b, in.mask_0, in.rows_a, in.rows_b }, build, res, &res_ne, why)) {
        return RUN_FAILED;
    }
    out.only_a  = res[0];
    out.part_a  = res[1];
    out.part_b  = res[2];
    out.joined  = res[3];
    out.swapped = res[4];
    out.alone   = res[5];

    // the shapes: a partial result has two more values per head and token, the combined result has the op's shape
    const int64_t ne_full[4] = { tc.n_embd, tc.n_head, tc.n_batch, tc.n_seq };
    out.shapes_ok = true;
    for (int i = 0; i < 4; i++) {
        out.ne_part[i] = res_ne[1][i];
        out.shapes_ok  = out.shapes_ok && res_ne[1][i] == ne_full[i] + (i == 0 ? 2 : 0) && res_ne[3][i] == ne_full[i];
    }
    return RUN_OK;
}

// the combine of two partial results, on a device that computed neither or only one of them
static bool combine_on(const device & dev, const inputs & in, const values & a, const values & b, const int64_t * ne_part, values & out, std::string & why) {
    std::vector<values> res;
    const build_fn build = [&](ggml_context * ctx, const std::vector<ggml_tensor *> & t) {
        return std::vector<ggml_tensor *> { ggml_flash_attn_ext_combine(ctx, t[0], t[1]) };
    };
    if (!compute(dev.backend, { host_tensor(in.ctx, a, ne_part), host_tensor(in.ctx, b, ne_part) }, build, res, nullptr, why)) {
        return false;
    }
    out = res[0];
    return true;
}

static int n_fail = 0;
static int n_run  = 0;
static int n_skip = 0;

static void check(bool ok, const char * what, const std::string & dev, const compare_result & c, double limit, double limit_row) {
    const bool pass = ok && c.nmse < limit && c.nmse_row < limit_row;
    n_run++;
    n_fail += pass ? 0 : 1;
    printf("  %-8s %-46s nmse %.2e, worst row %.2e, max diff %.2e  %s\n", dev.c_str(), what, c.nmse, c.nmse_row, c.max_abs, pass ? "OK" : "FAIL");
}

static void fail(const char * what, const std::string & dev, const std::string & why) {
    n_run++;
    n_fail++;
    printf("  %-8s %-46s %s  FAIL\n", dev.c_str(), what, why.c_str());
}

static void run(const test_case & tc, const std::vector<device> & devices) {
    printf("%s\n", tc.name);

    const inputs in = make_inputs(tc);

    // the same rows in another order of summation: on the CPU the results differ in the last bits only
    const double limit_cpu     = 1e-9;
    const double limit_cpu_row = 1e-8;
    // a device may compute in half precision
    const double limit_dev     = 2e-5;
    const double limit_dev_row = 2e-4;

    const device * cpu = nullptr;
    outputs        out_cpu;

    for (const device & dev : devices) {
        const double limit     = dev.is_cpu ? limit_cpu     : limit_dev;
        const double limit_row = dev.is_cpu ? limit_cpu_row : limit_dev_row;

        outputs     out;
        std::string why;
        const run_status status = run_on(dev, tc, in, out, why);
        if (status == RUN_SKIPPED) {
            n_skip++;
            printf("  %-8s skipped: it does not run %s for these shapes\n", dev.name.c_str(), why.c_str());
            continue;
        }
        if (status == RUN_FAILED) {
            fail("the partial form and the combine", dev.name, "not supported: " + why);
            continue;
        }

        const bool finite = all_finite(out.joined) && all_finite(out.swapped) && all_finite(out.alone);
        const bool sound  = out.shapes_ok && partial_is_sound(out.part_a, tc.n_embd) && partial_is_sound(out.part_b, tc.n_embd);
        const bool ok     = finite && sound && out.shapes_ok;
        if (!finite) {
            fail("finite values", dev.name, "a combined result is not finite");
        }
        if (out.shapes_ok && !sound) {
            fail("sound partial results", dev.name, "a partial result has a value, a sum or a maximum that is not finite where it has to be");
        }
        if (!out.shapes_ok) {
            fail("the shapes", dev.name, "a partial result has two more values per head and token than the op");
        }

        check(ok, "two places, against one pass here",        dev.name, compare(out.full,   out.joined,  tc.n_embd), limit, limit_row);
        check(ok, "two places in the other order",            dev.name, compare(out.full,   out.swapped, tc.n_embd), limit, limit_row);
        check(ok, "one place and one that holds nothing",     dev.name, compare(out.only_a, out.alone,   tc.n_embd), limit, limit_row);

        if (dev.is_cpu) {
            if (!cpu) {
                cpu     = &dev;
                out_cpu = out;
            }
            continue;
        }
        if (!cpu) {
            continue;
        }

        // Against the CPU. A device may compute with K and V in half precision, also from a quantized cache, and then
        // its one pass differs from the CPU's already (5e-4 is what the op's own test allows it, test-backend-ops).
        // Two places together may differ from the CPU's pass by as much as that pass of the device does, and a half
        // more, whichever device holds a place and whichever combines.
        const compare_result base = compare(out_cpu.full, out.full, tc.n_embd);
        check(ok, "one pass here, against one pass on the CPU", dev.name, base, 5e-4, 5e-3);

        const double limit_x     = std::max(limit_dev,     1.5*base.nmse);
        const double limit_x_row = std::max(limit_dev_row, 1.5*base.nmse_row);

        check(ok, "two places, against one pass on the CPU",  dev.name, compare(out_cpu.full, out.joined, tc.n_embd), limit_x, limit_x_row);

        values mixed;
        if (combine_on(dev, in, out_cpu.part_a, out.part_b, out.ne_part, mixed, why)) {
            check(all_finite(mixed), "place a on the CPU, b and the combine here", dev.name, compare(out_cpu.full, mixed, tc.n_embd), limit_x, limit_x_row);
        } else {
            fail("place a on the CPU, b and the combine here", dev.name, "not supported: " + why);
        }
        if (combine_on(*cpu, in, out.part_a, out_cpu.part_b, out.ne_part, mixed, why)) {
            check(all_finite(mixed), "place a here, b and the combine on the CPU", dev.name, compare(out_cpu.full, mixed, tc.n_embd), limit_x, limit_x_row);
        } else {
            fail("place a here, b and the combine on the CPU", dev.name, "not supported: " + why);
        }
    }

    ggml_free(in.ctx);
}

int main() {
    ggml_backend_load_all();

    // the CPU with one thread and with four, then every other device of the build
    std::vector<device> devices;
    for (const int n_threads : { 1, 4 }) {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        devices.push_back({ "CPU x" + std::to_string(n_threads), cpu, true });
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend) {
            devices.push_back({ ggml_backend_dev_name(dev), backend, false });
        }
    }
    printf("devices:");
    for (const device & dev : devices) {
        printf(" %s", dev.name.c_str());
    }
    printf("\n");

    const std::vector<test_case> cases = {
        //  name                                   type_kv        cache embd head hkv batch seq rows_a rows_b  kv_a  kv_b keep_b pad_tail
        { "f16, one sequence",                     GGML_TYPE_F16,  false,  64,   4,  2,    7,  1,    96,    64,   48,   32, 0.7f,  0.0f },
        { "f16, three sequences",                  GGML_TYPE_F16,  false,  64,   4,  2,    5,  3,   128,   128,   64,   64, 0.7f,  0.0f },
        { "f16, one token",                        GGML_TYPE_F16,  false,  64,   8,  2,    1,  2,   256,   512,  128,  256, 0.9f,  0.0f },
        { "f16, a batch of 64",                    GGML_TYPE_F16,  false,  64,   4,  4,   64,  1,   512,   256,  256,  128, 0.5f,  0.0f },
        { "f16, place b holds little",             GGML_TYPE_F16,  false,  64,   4,  2,    9,  2,    96,    64,   48,   32, 0.05f, 0.0f },
        { "f16, place b holds nothing",            GGML_TYPE_F16,  false,  64,   4,  2,    9,  2,    96,    64,   48,   32, 0.0f,  0.0f },
        { "f32",                                   GGML_TYPE_F32,  false,  32,   4,  2,    6,  2,    64,    64,   32,   32, 0.7f,  0.0f },
        { "q8_0",                                  GGML_TYPE_Q8_0, false,  64,   4,  2,    6,  2,    64,    96,   32,   48, 0.7f,  0.0f },
        { "q4_0",                                  GGML_TYPE_Q4_0, false,  64,   4,  2,    6,  2,    64,    96,   32,   48, 0.7f,  0.0f },
        { "f16, heads of 256",                     GGML_TYPE_F16,  false, 256,  24,  4,    3,  2,   128,   128,   64,   64, 0.7f,  0.0f },
        // as a KV cache holds its cells and as llama.cpp pads the lists, for the kernels of a GPU: one token of one
        // sequence over long lists, batches of many tokens and sequences, a last tile of tokens that is not full
        { "cache, f16, one token, long lists",     GGML_TYPE_F16,  true,  256,  32,  4,    1,  1,  8192, 16384, 4096, 8192, 0.7f,  0.1f },
        { "cache, q4_0, one token, long lists",    GGML_TYPE_Q4_0, true,  256,  32,  4,    1,  1,  8192, 16384, 4096, 8192, 0.7f,  0.1f },
        { "cache, q8_0, two tokens, heads of 128", GGML_TYPE_Q8_0, true,  128,  16,  4,    2,  1,  8192,  8192, 2048, 4096, 0.7f,  0.1f },
        { "cache, f16, a batch of 8",              GGML_TYPE_F16,  true,  256,  16,  2,    8,  2,  4096,  4096, 1024, 2048, 0.7f,  0.2f },
        { "cache, q4_0, a batch of 75, 3 seqs",    GGML_TYPE_Q4_0, true,  256,  24,  4,   75,  3,  4096,  8192, 1024, 1280, 0.7f,  0.2f },
        { "cache, f16, a batch of 512",            GGML_TYPE_F16,  true,  128,   8,  2,  512,  1,  2048,  4096,  768, 1024, 0.5f,  0.1f },
        { "cache, q8_0, no shared heads",          GGML_TYPE_Q8_0, true,  128,   4,  4,   33,  2,  2048,  2048,  512,  768, 0.7f,  0.2f },
        { "cache, f16, heads of 64, 5 seqs",       GGML_TYPE_F16,  true,   64,   8,  4,   17,  5,  3000,  3000,  768,  512, 0.7f,  0.1f },
        { "cache, f16, heads of 512",              GGML_TYPE_F16,  true,  512,  16,  2,    4,  2,  4096,  4096, 1024, 1024, 0.7f,  0.1f },
        { "cache, q4_0, lists of 256",             GGML_TYPE_Q4_0, true,  256,  32,  4,    3,  4,  4096,  4096,  256,  256, 0.7f,  0.3f },
        { "cache, q4_0, place b holds nothing",    GGML_TYPE_Q4_0, true,  256,  32,  4,    5,  2,  4096,  4096, 1024, 1024, 0.0f,  0.2f },
        { "cache, f16, place b holds little",      GGML_TYPE_F16,  true,  256,  32,  4,    1,  3,  4096,  4096, 2048, 2048, 0.01f, 0.2f },
    };

    for (const test_case & tc : cases) {
        run(tc, devices);
    }

    printf("%s: %d of %d checks failed (%d times a device did not run a case)\n", n_fail == 0 ? "OK" : "FAILED", n_fail, n_run, n_skip);
    return n_fail == 0 ? 0 : 1;
}
