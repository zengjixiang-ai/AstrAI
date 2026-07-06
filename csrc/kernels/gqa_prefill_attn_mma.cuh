#pragma once
#include "gqa_common.cuh"

// Tensor-core prefill, register-resident flash attention (raw mma.sync PTX).
// One warp owns BR=16 query rows. S = Q@K^T and O = P@V run on bf16 tensor
// cores via mma.sync.m16n8k16 (f32 accumulate). Q stays resident in registers;
// S, O, and the online-softmax stats (m, l) live in registers too — nothing is
// staged through shared memory except the cooperatively-loaded K/V tiles. The
// mma fragment layout is used directly: the S accumulator (f32) maps element-
// for-element onto the P matrix_a (bf16) operand, so softmax needs no shuffle
// repack; row reductions fold across the 4-lane thread group. Templated on
// <HEAD_DIM, WARPS, BC> with BC a multiple of 16.

// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// (only compiled when ASTRAI_HAS_MMA is set, i.e. built for sm_80+)
__device__ __forceinline__ void mma16816(float* d, const unsigned* a,
                                          const unsigned* b, const float* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// read two adjacent bf16 from smem as one packed .b32 (elem0 low, elem1 high)
__device__ __forceinline__ unsigned ld2(const bf16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}
__device__ __forceinline__ unsigned pk2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&v);
}
// pack two (non-contiguous) bf16 into one .b32
__device__ __forceinline__ unsigned pkb(bf16 a, bf16 b) {
    __nv_bfloat162 v;
    v.x = a;
    v.y = b;
    return *reinterpret_cast<unsigned*>(&v);
}

template <int HEAD_DIM, int WARPS, int BC>
__global__ void gqa_prefill_attn_mma_kernel(GQAParams p) {
    constexpr int BR = 16;
    constexpr int KD = HEAD_DIM / 16;  // Q/K k-tiles
    constexpr int NC8 = BC / 8;        // S n-tiles (N=8 each)
    constexpr int KT2 = BC / 16;       // P k-tiles (K=16 each)
    constexpr int DN8 = HEAD_DIM / 8;  // O n-tiles (N=8 each)

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = lane >> 2;   // 0..7  → rows gid, gid+8
    const int tid4 = lane & 3;   // 0..3
    const int nthreads = WARPS * 32;

    const int q_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_head = q_head / (p.q_head / p.kv_head);
    const int qrow0 = (blockIdx.x * WARPS + warp) * BR;

    extern __shared__ __align__(16) bf16 smem[];
    bf16* sK = smem;                    // [BC][HEAD_DIM]
    bf16* sV = sK + BC * HEAD_DIM;       // [BC][HEAD_DIM]
    bf16* sQ = sV + BC * HEAD_DIM + warp * (BR * HEAD_DIM);  // per-warp [BR][HEAD_DIM]

    // stage Q into smem (zero-padded past q_len)
    const int q_base = ((batch * p.q_head + q_head) * p.q_len) * HEAD_DIM;
    for (int i = lane; i < BR * HEAD_DIM; i += 32) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        int qr = qrow0 + r;
        sQ[i] = (qr < p.q_len) ? p.q[q_base + qr * HEAD_DIM + d] : __float2bfloat16(0.0f);
    }
    __syncwarp();

    // Q resident A-fragments: Qa[kt][0..3]
    unsigned Qa[KD][4];
#pragma unroll
    for (int kt = 0; kt < KD; kt++) {
        int c0 = kt * 16 + 2 * tid4;
        Qa[kt][0] = ld2(&sQ[gid * HEAD_DIM + c0]);
        Qa[kt][1] = ld2(&sQ[(gid + 8) * HEAD_DIM + c0]);
        Qa[kt][2] = ld2(&sQ[gid * HEAD_DIM + c0 + 8]);
        Qa[kt][3] = ld2(&sQ[(gid + 8) * HEAD_DIM + c0 + 8]);
    }

    float Oacc[DN8][4];
#pragma unroll
    for (int j = 0; j < DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int kv_base = ((batch * p.kv_head + kv_head) * p.kv_len) * HEAD_DIM;
    const int tiles = (p.kv_len + BC - 1) / BC;
    const int qr0 = qrow0 + gid;      // row for c0/c1
    const int qr1 = qrow0 + gid + 8;  // row for c2/c3

    for (int ti = 0; ti < tiles; ti++) {
        int kv0 = ti * BC;

        for (int i = threadIdx.x; i < BC * HEAD_DIM; i += nthreads) {
            int r = i / HEAD_DIM, d = i % HEAD_DIM;
            int kc = kv0 + r;
            bf16 z = __float2bfloat16(0.0f);
            sK[i] = (kc < p.kv_len) ? p.k[kv_base + kc * HEAD_DIM + d] : z;
            sV[i] = (kc < p.kv_len) ? p.v[kv_base + kc * HEAD_DIM + d] : z;
        }
        __syncthreads();

        // S = Q @ K^T  → Sacc[n8][0..3]   (n8: 8 kv cols each)
        float Sacc[NC8][4];
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            Sacc[n8][0] = Sacc[n8][1] = Sacc[n8][2] = Sacc[n8][3] = 0.0f;
            int kv = kv0 + n8 * 8 + gid;
#pragma unroll
            for (int kt = 0; kt < KD; kt++) {
                unsigned b[2];
                int kr = kt * 16 + 2 * tid4;
                b[0] = ld2(&sK[(n8 * 8 + gid) * HEAD_DIM + kr]);
                b[1] = ld2(&sK[(n8 * 8 + gid) * HEAD_DIM + kr + 8]);
                mma16816(Sacc[n8], Qa[kt], b, Sacc[n8]);
            }
            (void)kv;
        }

        // ---- online softmax (in registers) ----
        // scale + mask, then per-row (gid, gid+8) max over held cols
        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            int cc = kv0 + n8 * 8 + 2 * tid4;  // col for c0/c2
            bool bc0 = (cc >= p.kv_len) ||
                       (p.use_mask && p.mask && !p.mask[batch * p.kv_len + cc]);
            bool bc1 = (cc + 1 >= p.kv_len) ||
                       (p.use_mask && p.mask && !p.mask[batch * p.kv_len + cc + 1]);
            bool cz = p.is_causal;
            int off = p.causal_offset;
            bool bad0 = bc0 || (cz && cc > qr0 + off);
            bool bad1 = bc1 || (cz && (cc + 1) > qr0 + off);
            bool bad2 = bc0 || (cz && cc > qr1 + off);
            bool bad3 = bc1 || (cz && (cc + 1) > qr1 + off);
            float s0 = bad0 ? -FLT_MAX : Sacc[n8][0] * p.scale;
            float s1 = bad1 ? -FLT_MAX : Sacc[n8][1] * p.scale;
            float s2 = bad2 ? -FLT_MAX : Sacc[n8][2] * p.scale;
            float s3 = bad3 ? -FLT_MAX : Sacc[n8][3] * p.scale;
            Sacc[n8][0] = s0; Sacc[n8][1] = s1; Sacc[n8][2] = s2; Sacc[n8][3] = s3;
            rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
            rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
        }
        // reduce max across the 4-lane group (tid4)
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

        float nm0 = fmaxf(m0, rmax0), nm1 = fmaxf(m1, rmax1);
        float corr0 = (nm0 == -FLT_MAX) ? 1.0f : __expf(m0 - nm0);
        float corr1 = (nm1 == -FLT_MAX) ? 1.0f : __expf(m1 - nm1);

        float rsum0 = 0.0f, rsum1 = 0.0f;
#pragma unroll
        for (int n8 = 0; n8 < NC8; n8++) {
            float p0 = (Sacc[n8][0] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][0] - nm0);
            float p1 = (Sacc[n8][1] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][1] - nm0);
            float p2 = (Sacc[n8][2] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][2] - nm1);
            float p3 = (Sacc[n8][3] == -FLT_MAX) ? 0.0f : __expf(Sacc[n8][3] - nm1);
            Sacc[n8][0] = p0; Sacc[n8][1] = p1; Sacc[n8][2] = p2; Sacc[n8][3] = p3;
            rsum0 += p0 + p1;
            rsum1 += p2 + p3;
        }
        rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 1);
        rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 2);
        rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 1);
        rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 2);
        l0 = l0 * corr0 + rsum0;
        l1 = l1 * corr1 + rsum1;
        m0 = nm0; m1 = nm1;

        // rescale O accumulator by per-row correction
#pragma unroll
        for (int j = 0; j < DN8; j++) {
            Oacc[j][0] *= corr0; Oacc[j][1] *= corr0;
            Oacc[j][2] *= corr1; Oacc[j][3] *= corr1;
        }

        // O += P @ V
#pragma unroll
        for (int kt2 = 0; kt2 < KT2; kt2++) {
            unsigned Pa[4];
            Pa[0] = pk2(Sacc[kt2 * 2][0], Sacc[kt2 * 2][1]);
            Pa[1] = pk2(Sacc[kt2 * 2][2], Sacc[kt2 * 2][3]);
            Pa[2] = pk2(Sacc[kt2 * 2 + 1][0], Sacc[kt2 * 2 + 1][1]);
            Pa[3] = pk2(Sacc[kt2 * 2 + 1][2], Sacc[kt2 * 2 + 1][3]);
#pragma unroll
            for (int dn8 = 0; dn8 < DN8; dn8++) {
                unsigned b[2];
                int kr = kt2 * 16 + 2 * tid4;
                int d = dn8 * 8 + gid;
                b[0] = pkb(sV[kr * HEAD_DIM + d], sV[(kr + 1) * HEAD_DIM + d]);
                b[1] = pkb(sV[(kr + 8) * HEAD_DIM + d], sV[(kr + 9) * HEAD_DIM + d]);
                mma16816(Oacc[dn8], Pa, b, Oacc[dn8]);
            }
        }
        __syncthreads();  // sK/sV reused next tile
    }

    // ---- write output ----
    float rl0 = (l0 > 1e-20f) ? (1.0f / l0) : 0.0f;
    float rl1 = (l1 > 1e-20f) ? (1.0f / l1) : 0.0f;
    const int o_base = ((batch * p.q_head + q_head) * p.q_len) * HEAD_DIM;
#pragma unroll
    for (int dn8 = 0; dn8 < DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        if (qr0 < p.q_len) {
            p.o[o_base + qr0 * HEAD_DIM + d] = __float2bfloat16(Oacc[dn8][0] * rl0);
            p.o[o_base + qr0 * HEAD_DIM + d + 1] = __float2bfloat16(Oacc[dn8][1] * rl0);
        }
        if (qr1 < p.q_len) {
            p.o[o_base + qr1 * HEAD_DIM + d] = __float2bfloat16(Oacc[dn8][2] * rl1);
            p.o[o_base + qr1 * HEAD_DIM + d + 1] = __float2bfloat16(Oacc[dn8][3] * rl1);
        }
    }
}
