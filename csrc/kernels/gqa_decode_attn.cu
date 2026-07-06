#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <torch/extension.h>

using bf16 = __nv_bfloat16;

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void gqa_decode_attn_kernel(
    const bf16* q_ptr, const bf16* k_ptr, const bf16* v_ptr,
    const bool*  mask_ptr, bf16* out_ptr,
    int B, int n_heads, int n_kv_heads, int seq_len, int hd
) {
    int batch = blockIdx.x / n_heads;
    int q_head = blockIdx.x % n_heads;
    int kv_head = q_head / (n_heads / n_kv_heads);
    int tid = threadIdx.x;

    float q_val = __bfloat162float(
        q_ptr[((batch * n_heads + q_head) * 1) * hd + tid]);
    int kv_base = ((batch * n_kv_heads + kv_head) * seq_len) * hd;
    int mask_base = batch * seq_len;

    float m = -FLT_MAX, d = 0.0f, acc = 0.0f;
    __shared__ float smem[2];
    float scale = 1.0f / sqrtf((float)hd);

    for (int s = 0; s < seq_len; s++) {
        int off = kv_base + s * hd + tid;
        float partial = q_val * __bfloat162float(k_ptr[off]);
        partial = warp_reduce_sum(partial) * scale;

        if (tid % 32 == 0) smem[tid / 32] = partial;
        __syncthreads();
        if (tid == 0) smem[0] = smem[0] + smem[1];
        __syncthreads();

        float score = smem[0];
        if (!mask_ptr[mask_base + s]) score = -FLT_MAX;

        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta  = expf(score - new_m);
        d = d * alpha + beta;
        acc = acc * alpha + __bfloat162float(v_ptr[off]) * beta;
        m = new_m;
    }

    int out_off = ((batch * n_heads + q_head) * 1) * hd + tid;
    out_ptr[out_off] = __float2bfloat16(acc / d);
}

torch::Tensor gqa_decode_attn(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && mask.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);
    TORCH_CHECK(mask.dtype() == torch::kBool);
    TORCH_CHECK(q.size(2) == 1, "Q seq_len must be 1");

    int B = q.size(0), n_heads = q.size(1), n_kv = k.size(1);
    int seq_len = k.size(2), hd = q.size(3);
    auto out = torch::empty_like(q);

    gqa_decode_attn_kernel<<<dim3(B * n_heads), dim3(hd)>>>(
        reinterpret_cast<const bf16*>(q.data_ptr()),
        reinterpret_cast<const bf16*>(k.data_ptr()),
        reinterpret_cast<const bf16*>(v.data_ptr()),
        mask.data_ptr<bool>(),
        reinterpret_cast<bf16*>(out.data_ptr()),
        B, n_heads, n_kv, seq_len, hd
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_decode_attn", &gqa_decode_attn, "GQA decode attention (fused)");
}
