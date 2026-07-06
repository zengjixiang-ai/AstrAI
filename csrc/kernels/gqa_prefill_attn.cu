// torch binding for gqa_prefill_attn
// kernel defined in gqa_prefill_attn.cuh
#include "gqa_prefill_attn.cuh"
#include <torch/extension.h>

torch::Tensor gqa_prefill_attn(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    bool is_causal = false, int64_t causal_offset = 0,
    c10::optional<double> scale = c10::nullopt
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);

    int B = q.size(0), Hq = q.size(1), q_len = q.size(2), D = q.size(3);
    int Hk = k.size(1), kv_len = k.size(2);
    TORCH_CHECK(D % 32 == 0, "head_dim must be multiple of 32");

    bool use_mask = mask.has_value();
    const bool* mask_ptr = nullptr;
    if (use_mask) {
        TORCH_CHECK(mask.value().dtype() == torch::kBool);
        TORCH_CHECK(mask.value().dim() == 2);
        TORCH_CHECK(mask.value().size(0) == B);
        TORCH_CHECK(mask.value().size(1) == kv_len);
        mask_ptr = mask.value().data_ptr<bool>();
    }

    auto O = torch::empty_like(q);

    dim3 grid((q_len + Br - 1) / Br, Hq, B);
    dim3 block(32, Br, 1);
    size_t smem = 2 * Bc * D * sizeof(bf16);

    gqa_prefill_attn_kernel<<<grid, block, smem>>>(
        (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(),
        (const bf16*)v.data_ptr(), mask_ptr, (bf16*)O.data_ptr(),
        B, Hq, Hk, q_len, kv_len, D, (int)is_causal, (int)causal_offset, (int)use_mask
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_prefill_attn", &gqa_prefill_attn, "GQA prefill v3 (compute-opt)");
}
