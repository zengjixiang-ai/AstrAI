#pragma once
#include "gqa_common.cuh"

__global__ void gqa_prefill_attn_kernel(GQAParams p) {
    int q_tile = blockIdx.x;
    int q_head = blockIdx.y;
    int batch  = blockIdx.z;
    int q_row  = q_tile * Br + threadIdx.y;
    int d_part = threadIdx.x;
    int dpw    = p.head_dim >> 5;

    int kv_head = q_head / (p.q_head / p.kv_head);

    float qs[8] = {0};
    if (q_row < p.q_len) {
        int q_off = (((batch * p.q_head + q_head) * p.q_len + q_row) * p.head_dim) + d_part * dpw;
        for (int i = 0; i < dpw; i++)
            qs[i] = __bfloat162float(p.q[q_off + i]) * p.scale;
    }

    int kv_base = ((batch * p.kv_head + kv_head) * p.kv_len) * p.head_dim;

    extern __shared__ __align__(16) bf16 smem[];
    bf16* sK = smem;
    bf16* sV = smem + Bc * p.head_dim;

    float m = -FLT_MAX, l = 0.0f, acc[8] = {0};

    int tiles = (p.kv_len + Bc - 1) / Bc;
    int tt = blockDim.x * blockDim.y;

    for (int ti = 0; ti < tiles; ti++) {
        int kv0  = ti * Bc;
        int tlen = min(Bc, p.kv_len - kv0);

        for (int i = threadIdx.y * blockDim.x + threadIdx.x;
             i < tlen * p.head_dim; i += tt) {
            int r = i / p.head_dim, c = i % p.head_dim, idx = r * p.head_dim + c;
            int g_off = kv_base + (kv0 + r) * p.head_dim + c;
            sK[idx] = p.k[g_off];
            sV[idx] = p.v[g_off];
        }
        __syncthreads();

        int lim = tlen;
        if (p.is_causal && q_row < p.q_len) {
            int ep = q_row + p.causal_offset + 1;
            if (kv0 >= ep)
                lim = 0;
            else if (kv0 + tlen > ep)
                lim = ep - kv0;
        }

        for (int s = 0; s < lim; s++) {
            float dot = 0.0f;
            for (int i = 0; i < dpw; i++)
                dot += qs[i] * __bfloat162float(sK[s * p.head_dim + d_part * dpw + i]);
            dot = warp_reduce_sum(dot);

            if (p.use_mask && p.mask && !p.mask[batch * p.kv_len + kv0 + s])
                dot = -FLT_MAX;

            float nm = fmaxf(m, dot);
            float al = expf(m - nm);
            float be = expf(dot - nm);
            l = l * al + be;

            for (int i = 0; i < dpw; i++)
                acc[i] = acc[i] * al + __bfloat162float(sV[s * p.head_dim + d_part * dpw + i]) * be;
            m = nm;
        }
        __syncthreads();
    }

    if (q_row < p.q_len) {
        int o_off = (((batch * p.q_head + q_head) * p.q_len + q_row) * p.head_dim) + d_part * dpw;
        float rl = (l > 1e-10f) ? (1.0f / l) : 0.0f;
        for (int i = 0; i < dpw; i++)
            p.o[o_off + i] = __float2bfloat16(acc[i] * rl);
    }
}
