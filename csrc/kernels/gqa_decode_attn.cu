// torch binding for gqa_decode_attn
// kernel defined in gqa_decode_attn.cuh
#include "gqa_decode_attn.cuh"
#include <torch/extension.h>

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
    TORCH_CHECK(hd % 32 == 0, "head_dim must be multiple of 32");
    int group_size = n_heads / n_kv;
    auto out = torch::empty_like(q);

    size_t smem = DC_CHUNK * hd * sizeof(bf16);
    dim3 block(32, group_size);
    dim3 grid(B * n_kv);

    gqa_decode_attn_kernel<<<grid, block, smem>>>(
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
    m.def("gqa_decode_attn", &gqa_decode_attn, "GQA decode v2 (per-KV-head, shared K)");
}
