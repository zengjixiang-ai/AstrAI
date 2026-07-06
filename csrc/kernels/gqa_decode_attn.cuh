// gqa_decode_attn.cuh — header-only decode kernel
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>
using std::min;
using bf16 = __nv_bfloat16;

constexpr int DC_CHUNK = 64;

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void gqa_decode_attn_kernel(
    const bf16* __restrict__ q_ptr,
    const bf16* __restrict__ k_ptr,
    const bf16* __restrict__ v_ptr,
    const bool*  __restrict__ mask_ptr,
    bf16* __restrict__ out_ptr,
    int B, int n_heads, int n_kv_heads, int seq_len, int hd
) {
    int batch = blockIdx.x / n_kv_heads;
    int kv_head = blockIdx.x % n_kv_heads;
    int group_size = blockDim.y;
    int q_head = kv_head * group_size + threadIdx.y;
    int lane = threadIdx.x;
    int hd_per_thread = hd / 32;

    float q_reg[8];
    int q_off = ((batch * n_heads + q_head) * 1) * hd + lane * hd_per_thread;
    for (int i = 0; i < hd_per_thread; i++)
        q_reg[i] = __bfloat162float(q_ptr[q_off + i]);

    int kv_base = ((batch * n_kv_heads + kv_head) * seq_len) * hd;
    int mask_base = batch * seq_len;

    float m = -FLT_MAX, d = 0.0f, acc_reg[8] = {0.0f};
    float scale = rsqrtf((float)hd);

    extern __shared__ __align__(16) bf16 k_smem[];

    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += DC_CHUNK) {
        int this_chunk = min(DC_CHUNK, seq_len - chunk_start);

        int total = this_chunk * hd;
        for (int i = threadIdx.y * 32 + lane; i < total; i += blockDim.x * blockDim.y)
            k_smem[i] = k_ptr[kv_base + chunk_start * hd + i];
        __syncthreads();

        for (int s = 0; s < this_chunk; s++) {
            float partial = 0.0f;
            for (int i = 0; i < hd_per_thread; i++)
                partial += q_reg[i] * __bfloat162float(k_smem[s * hd + lane * hd_per_thread + i]);
            partial = warp_reduce_sum(partial) * scale;

            if (!mask_ptr[mask_base + chunk_start + s]) partial = -FLT_MAX;

            float new_m = fmaxf(m, partial);
            float alpha = expf(m - new_m);
            float beta  = expf(partial - new_m);
            d = d * alpha + beta;

            int v_off = kv_base + (chunk_start + s) * hd + lane * hd_per_thread;
            for (int i = 0; i < hd_per_thread; i++)
                acc_reg[i] = acc_reg[i] * alpha + __bfloat162float(v_ptr[v_off + i]) * beta;
            m = new_m;
        }
        __syncthreads();
    }

    int out_off = ((batch * n_heads + q_head) * 1) * hd + lane * hd_per_thread;
    for (int i = 0; i < hd_per_thread; i++)
        out_ptr[out_off + i] = __float2bfloat16(acc_reg[i] / d);
}
