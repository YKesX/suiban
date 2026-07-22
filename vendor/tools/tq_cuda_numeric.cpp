// tq_cuda_numeric — compiled-CUDA numeric validation for the TurboQuant
// TQ3_0/TQ4_0 patchset (suiban/vendor/patches), run ON a GPU.
//
// Exercises the actual compiled CUDA kernels against the CPU reference:
//   1. dequantize-rows (convert.cu): contiguous to_fp32/to_fp16 and the
//      non-contiguous (strided) variants flash attention's prefill path uses.
//   2. SET_ROWS f32 -> TQ (set-rows.cu): the KV-write quantizer, as a real
//      ggml graph on the CUDA backend vs the CPU backend.
//   3. FLASH_ATTN_EXT with V = TQ3_0/TQ4_0 (fattn-common.cuh dequantize_V +
//      the fattn-vec instances): nb=1 hits the vec kernel (decode), nb=512
//      hits the prefill (MMA/tile) route that reads V through the fp16
//      convert path. Graph construction mirrors the fork's own
//      tests/test-backend-ops.cpp test_flash_attn_ext.
//
// Compiled and driven by vendor/run_kernel_tests.py stage 7 (host-only C++,
// links the build-cuda shared libs + cudart; no nvcc needed at this point).
// Exit codes: 0 pass, 1 fail, 3 no CUDA device (harness reports SKIP).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Internal libggml-cuda entry points (ggml/src/ggml-cuda/convert.cuh). The
// symbols are exported from the shared lib; mangling covers parameters only,
// so these declarations bind as long as the parameter list matches.
typedef void (*to_fp32_cuda_t)(const void * x, float * y, int64_t k, cudaStream_t stream);
typedef void (*to_fp16_cuda_t)(const void * x, half * y, int64_t k, cudaStream_t stream);
typedef void (*to_fp32_nc_cuda_t)(const void * x, float * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);
typedef void (*to_fp16_nc_cuda_t)(const void * x, half * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);
to_fp32_cuda_t    ggml_get_to_fp32_cuda(ggml_type type);
to_fp16_cuda_t    ggml_get_to_fp16_cuda(ggml_type type);
to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);

static bool g_fail = false;

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),    \
                    __FILE__, __LINE__);                                             \
            return 1;                                                                \
        }                                                                            \
    } while (0)

static void check(bool ok, const char * what) {
    if (!ok) {
        printf("    FAILED: %s\n", what);
        g_fail = true;
    }
}

static std::vector<float> random_floats(size_t n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> out(n);
    for (auto & v : out) {
        v = dist(rng);
    }
    return out;
}

static std::vector<uint8_t> quantize_host(ggml_type type, const std::vector<float> & src) {
    const size_t n_blocks = src.size() / ggml_blck_size(type);
    std::vector<uint8_t> out(n_blocks * ggml_type_size(type));
    const size_t written =
        ggml_quantize_chunk(type, src.data(), out.data(), 0, 1, (int64_t) src.size(), nullptr);
    if (written != out.size()) {
        fprintf(stderr, "quantize_host: wrote %zu B, expected %zu B\n", written, out.size());
        exit(1);
    }
    return out;
}

static std::vector<float> dequantize_host(ggml_type type, const uint8_t * data, int64_t n) {
    std::vector<float> out(n);
    ggml_get_type_traits(type)->to_float(data, out.data(), n);
    return out;
}

static double max_abs_dev(const float * a, const float * b, size_t n) {
    double dev = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dev = std::max(dev, (double) std::fabs(a[i] - b[i]));
    }
    return dev;
}

static double nmse(const float * ref, const float * got, size_t n) {
    double err = 0.0, sig = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = got[i] - ref[i];
        err += d * d;
        sig += (double) ref[i] * ref[i];
    }
    return sig > 0.0 ? err / sig : err;
}

// ---------------------------------------------------------------------------
// section 1: dequantize-rows kernels (convert.cu), direct calls
// ---------------------------------------------------------------------------
static int section_convert(ggml_type type) {
    const char * tname = ggml_type_name(type);
    const int64_t k = 4096 * 32; // 4096 blocks
    const auto src_f = random_floats(k, 0x7451);
    const auto q     = quantize_host(type, src_f);
    const auto ref   = dequantize_host(type, q.data(), k); // CPU reference kernel

    void * d_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, q.size()));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q.size(), cudaMemcpyHostToDevice));

    // contiguous to_fp32
    {
        float * d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_y, k * sizeof(float)));
        ggml_get_to_fp32_cuda(type)(d_q, d_y, k, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> y(k);
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, k * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_y));
        const double dev = max_abs_dev(ref.data(), y.data(), k);
        printf("  %s dequantize rows (contiguous, f32, %" PRId64 " values): max dev vs CPU %.3e\n",
               tname, k, dev);
        check(dev <= 1e-4, "contiguous f32 dequant deviation");
    }
    // contiguous to_fp16 (fp16 output adds rounding: compare vs fp16-rounded ref)
    {
        half * d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_y, k * sizeof(half)));
        ggml_get_to_fp16_cuda(type)(d_q, d_y, k, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<half> y(k);
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, k * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_y));
        double dev = 0.0;
        for (int64_t i = 0; i < k; ++i) {
            dev = std::max(dev, (double) std::fabs(__half2float(y[i]) - ref[i]));
        }
        printf("  %s dequantize rows (contiguous, f16): max dev vs CPU f32 ref %.3e\n",
               tname, dev);
        check(dev <= 2e-3, "contiguous f16 dequant deviation (fp16 rounding included)");
    }
    CUDA_CHECK(cudaFree(d_q));

    // non-contiguous (strided) variant — the layout fattn's prefill conversion
    // sees: ne00=128 (4 blocks/row), 64 rows, 8 "heads", row stride padded by
    // one extra block, head stride padded by one extra row.
    {
        const int64_t ne00 = 128, ne01 = 64, ne02 = 8, ne03 = 1;
        const int64_t nblk0 = ne00 / 32;
        const int64_t s01 = nblk0 + 1;          // blocks per row incl. padding
        const int64_t s02 = s01 * (ne01 + 1);   // blocks per head incl. padding
        const int64_t s03 = s02 * ne02;
        const size_t ts = ggml_type_size(type);
        const size_t total_blocks = (size_t) (s03 * ne03);
        std::vector<uint8_t> padded(total_blocks * ts, 0xA5); // poison padding
        const auto dense_f = random_floats((size_t) (ne00 * ne01 * ne02 * ne03), 0x7452);
        std::vector<float> expect(dense_f.size());
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = 0; i01 < ne01; ++i01) {
                const int64_t row = i02 * ne01 + i01;
                const auto qrow = quantize_host(
                    type, std::vector<float>(dense_f.begin() + row * ne00,
                                             dense_f.begin() + (row + 1) * ne00));
                memcpy(padded.data() + (i02 * s02 + i01 * s01) * ts, qrow.data(), qrow.size());
                const auto drow = dequantize_host(type, qrow.data(), ne00);
                memcpy(expect.data() + row * ne00, drow.data(), ne00 * sizeof(float));
            }
        }
        void * d_x = nullptr;
        float * d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, padded.size()));
        CUDA_CHECK(cudaMalloc(&d_y, expect.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, padded.data(), padded.size(), cudaMemcpyHostToDevice));
        ggml_get_to_fp32_nc_cuda(type)(d_x, d_y, ne00, ne01, ne02, ne03, s01, s02, s03, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> y(expect.size());
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        const double dev = max_abs_dev(expect.data(), y.data(), expect.size());
        printf("  %s dequantize rows (strided nc, f32, %zu values): max dev vs CPU %.3e\n",
               tname, expect.size(), dev);
        check(dev <= 1e-4, "strided nc dequant deviation");
    }
    return 0;
}

// ---------------------------------------------------------------------------
// graph helper (mirrors test-backend-ops' single-backend compute)
// ---------------------------------------------------------------------------
struct GraphRun {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;

    ggml_context * begin() {
        ggml_init_params params = {
            /*mem_size  =*/ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
            /*mem_buffer=*/ nullptr,
            /*no_alloc  =*/ true,
        };
        ctx = ggml_init(params);
        return ctx;
    }

    bool alloc_and_check(ggml_backend_t backend, ggml_tensor * out) {
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                printf("    (op %s not supported by %s)\n", ggml_op_name(node->op),
                       ggml_backend_name(backend));
                return false;
            }
        }
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(galloc, graph);
        return true;
    }

    void compute(ggml_backend_t backend) {
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
    }

    ~GraphRun() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
    }
};

// ---------------------------------------------------------------------------
// section 2: SET_ROWS f32 -> TQ (KV-write quantizer) on CUDA vs CPU backend
// ---------------------------------------------------------------------------
static int section_set_rows(ggml_backend_t cuda, ggml_backend_t cpu, ggml_type type) {
    const char * tname = ggml_type_name(type);
    const int64_t ne0 = 256, n_dst = 64, n_src = 32;
    const auto src_f = random_floats((size_t) (ne0 * n_src), 0x5e70);
    std::vector<int64_t> ids(n_src);
    for (int64_t i = 0; i < n_src; ++i) {
        ids[i] = (i * 2 + 1) % n_dst; // distinct target rows
    }

    std::vector<uint8_t> out_bytes[2];
    const size_t dst_bytes = (size_t) (ne0 / 32 * n_dst) * ggml_type_size(type);
    ggml_backend_t backends[2] = {cpu, cuda};
    for (int b = 0; b < 2; ++b) {
        GraphRun run;
        ggml_context * ctx = run.begin();
        ggml_tensor * dst = ggml_new_tensor_2d(ctx, type, ne0, n_dst);
        ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, n_src);
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_src);
        ggml_tensor * out = ggml_set_rows(ctx, dst, src, idx);
        if (!run.alloc_and_check(backends[b], out)) {
            return 1;
        }
        std::vector<uint8_t> zeros(dst_bytes, 0);
        ggml_backend_tensor_set(dst, zeros.data(), 0, dst_bytes);
        ggml_backend_tensor_set(src, src_f.data(), 0, src_f.size() * sizeof(float));
        ggml_backend_tensor_set(idx, ids.data(), 0, ids.size() * sizeof(int64_t));
        run.compute(backends[b]);
        out_bytes[b].resize(dst_bytes);
        ggml_backend_tensor_get(out, out_bytes[b].data(), 0, dst_bytes);
    }

    size_t diff_bytes = 0;
    for (size_t i = 0; i < dst_bytes; ++i) {
        diff_bytes += out_bytes[0][i] != out_bytes[1][i];
    }
    const auto deq_cpu = dequantize_host(type, out_bytes[0].data(), ne0 * n_dst);
    const auto deq_gpu = dequantize_host(type, out_bytes[1].data(), ne0 * n_dst);
    const double dev = max_abs_dev(deq_cpu.data(), deq_gpu.data(), deq_cpu.size());
    printf("  %s SET_ROWS quantize (%" PRId64 " rows x %" PRId64 "): byte mismatch "
           "%zu/%zu (%.4f%%), dequantized max dev %.3e\n",
           tname, n_src, ne0, diff_bytes, dst_bytes, 100.0 * diff_bytes / dst_bytes, dev);
    // Codes are NOT guaranteed bit-identical (nvcc -use_fast_math can flip
    // codes right at Lloyd-Max boundaries); the reconstruction must stay
    // within one level step and mismatches must be rare.
    check(100.0 * diff_bytes / dst_bytes <= 1.0, "SET_ROWS byte mismatch rate <= 1%");
    check(dev <= 0.2, "SET_ROWS dequantized deviation within one level step");
    return 0;
}

// ---------------------------------------------------------------------------
// section 3: FLASH_ATTN_EXT with TurboQuant V, CUDA vs CPU backend
// (construction mirrors tests/test-backend-ops.cpp test_flash_attn_ext)
// ---------------------------------------------------------------------------
static int section_fattn(ggml_backend_t cuda, ggml_backend_t cpu,
                         ggml_type type_K, ggml_type type_V,
                         int64_t hs, int64_t kv, int64_t nb) {
    const int64_t nh = 4;
    const auto q_f = random_floats((size_t) (hs * nb * nh), 0xFA77 + (uint32_t) (hs + nb));
    const auto k_f = random_floats((size_t) (hs * kv * nh), 0xFA78 + (uint32_t) hs);
    const auto v_f = random_floats((size_t) (hs * kv * nh), 0xFA79 + (uint32_t) hs);

    // one shared quantization on the host: both backends read identical bytes
    std::vector<uint8_t> k_data;
    if (type_K == GGML_TYPE_F16) {
        k_data.resize(k_f.size() * sizeof(ggml_fp16_t));
        ggml_fp32_to_fp16_row(k_f.data(), (ggml_fp16_t *) k_data.data(), (int64_t) k_f.size());
    } else {
        k_data = quantize_host(type_K, k_f);
    }
    const auto v_data = quantize_host(type_V, v_f);
    const std::vector<ggml_fp16_t> mask_zero((size_t) (kv * nb), ggml_fp32_to_fp16(0.0f));

    std::vector<float> out_f[2];
    ggml_backend_t backends[2] = {cpu, cuda};
    for (int b = 0; b < 2; ++b) {
        GraphRun run;
        ggml_context * ctx = run.begin();
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, nb, nh, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, type_K, hs, kv, nh, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, type_V, hs, kv, nh, 1);
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb, 1, 1);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) hs), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (!run.alloc_and_check(backends[b], out)) {
            return 1;
        }
        ggml_backend_tensor_set(q, q_f.data(), 0, q_f.size() * sizeof(float));
        ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size());
        ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size());
        ggml_backend_tensor_set(m, mask_zero.data(), 0, mask_zero.size() * sizeof(ggml_fp16_t));
        run.compute(backends[b]);
        out_f[b].resize((size_t) (hs * nb * nh));
        ggml_backend_tensor_get(out, out_f[b].data(), 0, out_f[b].size() * sizeof(float));
    }

    const double dev = max_abs_dev(out_f[0].data(), out_f[1].data(), out_f[0].size());
    const double err = nmse(out_f[0].data(), out_f[1].data(), out_f[0].size());
    printf("  FLASH_ATTN_EXT K=%s V=%s D=%" PRId64 " kv=%" PRId64 " nb=%" PRId64
           " (%s path): max dev %.3e, NMSE %.3e\n",
           ggml_type_name(type_K), ggml_type_name(type_V), hs, kv, nb,
           nb == 1 ? "vec" : "prefill", dev, err);
    // 5e-4 is the fork's own test-backend-ops NMSE threshold for FLASH_ATTN_EXT
    check(err <= 5e-4, "FLASH_ATTN_EXT NMSE vs CPU");
    return 0;
}

int main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("no CUDA device available\n");
        return 3;
    }
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cuda == nullptr || cpu == nullptr) {
        fprintf(stderr, "backend init failed\n");
        return 3;
    }
    char desc[256];
    ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    printf("device: %s\n", desc);

    const ggml_type tq_types[2] = {GGML_TYPE_TQ3_0, GGML_TYPE_TQ4_0};

    printf("section 1: dequantize-rows kernels (convert.cu), on-device vs CPU reference\n");
    for (ggml_type t : tq_types) {
        if (section_convert(t) != 0) return 1;
    }

    printf("section 2: SET_ROWS KV-write quantizer (set-rows.cu), CUDA vs CPU backend\n");
    for (ggml_type t : tq_types) {
        if (section_set_rows(cuda, cpu, t) != 0) return 1;
    }

    printf("section 3: FLASH_ATTN_EXT V-dequant (fattn), CUDA vs CPU backend\n");
    for (ggml_type t : tq_types) {
        for (int64_t hs : {64, 128, 256}) {
            for (int64_t nb : {1, 512}) {
                if (section_fattn(cuda, cpu, GGML_TYPE_Q8_0, t, hs, 1024, nb) != 0) return 1;
            }
        }
        // the K=f16 pairing at the Bonsai head dims, vec path
        if (section_fattn(cuda, cpu, GGML_TYPE_F16, t, 128, 1024, 1) != 0) return 1;
        if (section_fattn(cuda, cpu, GGML_TYPE_F16, t, 256, 1024, 1) != 0) return 1;
    }

    ggml_backend_free(cuda);
    ggml_backend_free(cpu);

    if (g_fail) {
        printf("TQ CUDA NUMERIC: FAILED\n");
        return 1;
    }
    printf("TQ CUDA NUMERIC: PASS\n");
    return 0;
}
