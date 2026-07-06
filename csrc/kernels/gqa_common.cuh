#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

using bf16 = __nv_bfloat16;
using std::min;

constexpr int DC_CHUNK = 64;
constexpr int Br = 32, Bc = 64;

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

struct GQAParams {
    int batch;
    int q_head;
    int kv_head;
    int q_len;
    int kv_len;
    int head_dim;
    int use_mask;
    int is_causal;
    int causal_offset;
    float scale;
    const bf16* __restrict__ q;
    const bf16* __restrict__ k;
    const bf16* __restrict__ v;
    const bool* __restrict__ mask;
    bf16* __restrict__ o;
};
