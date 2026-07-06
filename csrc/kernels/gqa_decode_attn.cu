#include "gqa_decode_attn.cuh"
#include <torch/extension.h>

torch::Tensor gqa_decode_attn(
    torch::Tensor q,
    torch::Tensor k, 
    torch::Tensor v,
    c10::optional<torch::Tensor> mask,
    bool is_causal = false, 
    int64_t causal_offset = 0,
    c10::optional<double> scale = c10::nullopt
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == torch::kBFloat16);
    TORCH_CHECK(v.dtype() == torch::kBFloat16);
    TORCH_CHECK(q.size(2) == 1, "Q seq_len must be 1");

    GQAParams p;
    p.batch = q.size(0);
    p.q_head = q.size(1);
    p.kv_head = k.size(1);
    p.q_len = 1;
    p.kv_len = k.size(2);
    p.head_dim = q.size(3);
    TORCH_CHECK(p.head_dim % 32 == 0, "head_dim must be multiple of 32");
    p.use_mask = mask.has_value();
    p.is_causal = (int)is_causal;
    p.causal_offset = (int)causal_offset;
    p.scale = scale.has_value() ? (float)scale.value() : 1.0f / sqrtf((float)p.head_dim);
    p.q = (const bf16*)q.data_ptr();
    p.k = (const bf16*)k.data_ptr();
    p.v = (const bf16*)v.data_ptr();
    if (p.use_mask) {
        TORCH_CHECK(mask.value().dtype() == torch::kBool);
        TORCH_CHECK(mask.value().dim() == 2);
        TORCH_CHECK(mask.value().size(0) == p.batch);
        TORCH_CHECK(mask.value().size(1) == p.kv_len);
        p.mask = mask.value().data_ptr<bool>();
    } else {
        p.mask = nullptr;
    }

    auto O = torch::empty_like(q);
    p.o = (bf16*)O.data_ptr();

    int group_size = p.q_head / p.kv_head;
    size_t smem = DC_CHUNK * p.head_dim * sizeof(bf16);
    dim3 block(32, group_size);
    dim3 grid(p.batch * p.kv_head);

    gqa_decode_attn_kernel<<<grid, block, smem>>>(p);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gqa_decode_attn", &gqa_decode_attn,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask") = py::none(),
        py::arg("is_causal") = false,
        py::arg("causal_offset") = 0,
        py::arg("scale") = py::none(),
        "GQA decode (per-KV-head, shared K)");
}
