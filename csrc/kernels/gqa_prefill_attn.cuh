// gqa_prefill_attn.cuh — header-only kernel definition (no torch dependency)
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>

using bf16 = __nv_bfloat16;

static constexpr int Br = 32;
static constexpr int Bc = 64;

__device__ inline float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__global__ void gqa_prefill_attn_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bool* __restrict__ mask,
    bf16* __restrict__ O,
    int B, int Hq, int Hk, int q_len, int kv_len, int D,
    int is_causal, int causal_offset, int use_mask
) {
    int q_tile = blockIdx.x;
    int q_head = blockIdx.y;
    int batch  = blockIdx.z;
    int q_row  = q_tile * Br + threadIdx.y;
    int d_part = threadIdx.x;
    int dpw    = D >> 5;

    int kv_head = q_head / (Hq / Hk);

    float qs[8] = {0};
    if (q_row < q_len) {
        float sc = rsqrtf((float)D);
        int q_off = (((batch * Hq + q_head) * q_len + q_row) * D) + d_part * dpw;
        for (int i = 0; i < dpw; i++)
            qs[i] = __bfloat162float(Q[q_off + i]) * sc;
    }

    int kv_base = ((batch * Hk + kv_head) * kv_len) * D;

    extern __shared__ __align__(16) bf16 smem[];
    bf16* sK = smem;
    bf16* sV = smem + Bc * D;

    float m = -FLT_MAX, l = 0.0f, acc[8] = {0};

    int tiles = (kv_len + Bc - 1) / Bc;
    int tt = blockDim.x * blockDim.y;

    for (int ti = 0; ti < tiles; ti++) {
        int kv0  = ti * Bc;
        int tlen = min(Bc, kv_len - kv0);

        for (int i = threadIdx.y * blockDim.x + threadIdx.x;
             i < tlen * D; i += tt) {
            int r = i / D, c = i % D, idx = r * D + c;
            int g_off = kv_base + (kv0 + r) * D + c;
            sK[idx] = K[g_off];
            sV[idx] = V[g_off];
        }
        __syncthreads();

        int lim = tlen;
        if (is_causal && q_row < q_len) {
            int ep = q_row + causal_offset + 1;
            if (kv0 >= ep)        lim = 0;
            else if (kv0 + tlen > ep) lim = ep - kv0;
        }

        for (int s = 0; s < lim; s++) {
            float dot = 0.0f;
            for (int i = 0; i < dpw; i++)
                dot += qs[i] * __bfloat162float(sK[s * D + d_part * dpw + i]);
            dot = warp_sum(dot);

            if (use_mask && !mask[batch * q_len * kv_len + q_row * kv_len + kv0 + s])
                dot = -FLT_MAX;

            float nm = fmaxf(m, dot);
            float al = expf(m - nm);
            float be = expf(dot - nm);
            l = l * al + be;

            for (int i = 0; i < dpw; i++)
                acc[i] = acc[i] * al + __bfloat162float(sV[s * D + d_part * dpw + i]) * be;
            m = nm;
        }
        __syncthreads();
    }

    if (q_row < q_len) {
        int o_off = (((batch * Hq + q_head) * q_len + q_row) * D) + d_part * dpw;
        float rl = (l > 1e-10f) ? (1.0f / l) : 0.0f;
        for (int i = 0; i < dpw; i++)
            O[o_off + i] = __float2bfloat16(acc[i] * rl);
    }
}
