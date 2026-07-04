"""IFD (Instruction Following Difficulty) data quality scoring.

IFD = conditional_NLL / unconditional_NLL

- Messages format: plain text concatenation (no chat template)
- Plain format: raw instr_key + resp_key fields
"""

import argparse
import json
import statistics

import torch
import torch.nn.functional as F
import tqdm

from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


def _pack_bins(pairs, max_len):
    """BFD bin packing: pack (c+r) into bins of max total length."""
    indexed = sorted(enumerate(pairs), key=lambda x: -(len(x[1][0]) + len(x[1][1])))
    bins = []  # each bin: list of (orig_idx, ctx_ids, resp_ids)
    lengths = []
    for orig_idx, (c, r) in indexed:
        size = len(c) + len(r)
        best_bin = -1
        for bi, rem in enumerate(lengths):
            if rem >= size:
                if best_bin < 0 or rem < lengths[best_bin]:
                    best_bin = bi
        if best_bin >= 0:
            bins[best_bin].append((orig_idx, c, r))
            lengths[best_bin] -= size
        else:
            bins.append([(orig_idx, c, r)])
            lengths.append(max_len - size)
    return bins


@torch.inference_mode()
def _score_batch(pairs, model, device, max_len=2048):
    """BFD-packed IFD: pack items into bins, one forward pass per bin."""
    if not pairs:
        return []
    bins = _pack_bins(pairs, max_len)

    result = [None] * len(pairs)

    for bin_items in bins:
        seq_ids = []
        global_pos = []  # doc-reset position IDs for RoPE
        doc_ids = []  # document index for attention mask
        doc_offsets = []

        for di, (orig_idx, c, r) in enumerate(bin_items):
            ctx_len = len(c)
            start = len(seq_ids)
            item_len = len(c) + len(r)
            seq_ids.extend(c)
            seq_ids.extend(r)
            end = len(seq_ids)
            global_pos.extend(range(item_len))
            doc_ids.extend([di] * item_len)
            doc_offsets.append((start, end, orig_idx, ctx_len))

        full_ids = torch.tensor([seq_ids], device=device, dtype=torch.long)
        pos_ids = torch.tensor([global_pos], device=device, dtype=torch.long)
        T = len(seq_ids)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        doc_t = torch.tensor([doc_ids], device=device)
        doc_mask = doc_t.unsqueeze(-1) == doc_t.unsqueeze(-2)
        attn_mask = (causal & doc_mask[0]).unsqueeze(0).unsqueeze(0)
        logits_full = model(full_ids, position_ids=pos_ids, input_mask=attn_mask)[
            "logits"
        ][0]

        for start, end, orig_idx, ctx_len in doc_offsets:
            rl = end - start - ctx_len
            if rl < 2:
                continue
            resp_start = start + ctx_len - 1
            resp_logits = logits_full[resp_start : end - 1]
            resp_targets = torch.tensor(
                seq_ids[start + ctx_len : end], device=device, dtype=torch.long
            )
            L_cond = F.cross_entropy(resp_logits, resp_targets, reduction="mean").item()
            result[orig_idx] = (L_cond, rl)

    # unconditional pass: batch all responses separately (sorted by length)
    resp_seqs = [
        (i, result[i][1], pairs[i][1])
        for i in range(len(pairs))
        if result[i] is not None
    ]
    if resp_seqs:
        resp_seqs.sort(key=lambda x: -x[1])
        r_batch = torch.zeros(
            len(resp_seqs),
            max(len(r) for _, _, r in resp_seqs),
            dtype=torch.long,
            device=device,
        )
        for ri, (_, rl, r_ids) in enumerate(resp_seqs):
            r_batch[ri, :rl] = torch.tensor(r_ids, dtype=torch.long)
        logits_resp = model(r_batch)["logits"]

        for ri, (orig_idx, rl, _) in enumerate(resp_seqs):
            L_cond = result[orig_idx][0]
            unp_logits = logits_resp[ri, : rl - 1]
            unp_targets = r_batch[ri, 1:rl]
            L_uncond = F.cross_entropy(unp_logits, unp_targets, reduction="mean").item()
            ifd = L_cond / L_uncond if L_uncond > 0 else None
            result[orig_idx] = {
                "L_cond": round(L_cond, 6),
                "L_uncond": round(L_uncond, 6),
                "ifd": round(ifd, 6) if ifd is not None else None,
                "resp_len": rl,
            }

    return result


def _trim(context_ids, resp_ids, max_len):
    """Truncate to fit max_len, keeping response intact if possible."""
    if len(resp_ids) > max_len // 2:
        resp_ids = resp_ids[: max_len // 2]
    full_ids = context_ids + resp_ids
    if len(full_ids) <= max_len:
        return context_ids, resp_ids
    overflow = len(full_ids) - max_len
    if overflow >= len(context_ids):
        return [], resp_ids[:max_len]
    return context_ids[overflow:], resp_ids


def score_plain(model, tokenizer, instruction, response, device, max_len=2048):
    """Compute IFD for a single instruction-response pair (plain format)."""
    ctx_ids = tokenizer.encode(instruction, add_special_tokens=False)
    resp_ids = tokenizer.encode(response, add_special_tokens=False)
    ctx_ids, resp_ids = _trim(ctx_ids, resp_ids, max_len)
    if not ctx_ids or not resp_ids:
        return {"L_cond": None, "L_uncond": None, "ifd": None, "error": "empty"}
    return _score_batch([(ctx_ids, resp_ids)], model, device, max_len)[0]


def score_messages(model, tokenizer, messages, device, max_len=2048):
    """Compute IFD for each assistant turn in a messages array."""
    turns = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        ctx_text = "\n\n".join(m["content"] for m in messages[:i])
        ctx_ids = tokenizer.encode(ctx_text)
        resp_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        ctx_ids, resp_ids = _trim(ctx_ids, resp_ids, max_len)
        if ctx_ids and resp_ids:
            turns.append((ctx_ids, resp_ids))
    if not turns:
        return None
    raw_scores = _score_batch(turns, model, device, max_len)
    valid = [s for s in raw_scores if s is not None and s["ifd"] is not None]
    if not valid:
        return {"ifd": None, "ifd_turns": raw_scores}
    avg = sum(s["ifd"] for s in valid) / len(valid)
    return {
        "ifd": avg,
        "ifd_detail": valid[0] if len(valid) == 1 else None,
        "ifd_turns": raw_scores,
    }


def process_file(
    param_path,
    input_file,
    output_file,
    instr_key,
    resp_key,
    max_len=2048,
    data_format="plain",
    batch_size=1,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if "cuda" in device else torch.float32

    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device=device, dtype=dtype)
    model.eval()

    with open(input_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    results = []
    all_ifds = []
    buffer = []

    for item in tqdm.tqdm(data, desc="Computing IFD", unit="sample"):
        if data_format == "messages":
            turns = []
            for i, msg in enumerate(item.get("messages", [])):
                if msg.get("role") != "assistant":
                    continue
                ctx_text = "\n\n".join(m["content"] for m in item["messages"][:i])
                ctx_ids = tokenizer.encode(ctx_text)
                resp_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
                ctx_ids, resp_ids = _trim(ctx_ids, resp_ids, max_len)
                if ctx_ids and resp_ids:
                    turns.append((ctx_ids, resp_ids))
            if not turns:
                results.append({**item, "ifd": None, "ifd_turns": []})
                continue
            buffer.append((item, turns, "messages"))
        else:
            ctx_ids = tokenizer.encode(item[instr_key], add_special_tokens=False)
            resp_ids = tokenizer.encode(item[resp_key], add_special_tokens=False)
            ctx_ids, resp_ids = _trim(ctx_ids, resp_ids, max_len)
            if not ctx_ids or not resp_ids:
                results.append({**item, "ifd": None, "ifd_detail": {"error": "empty"}})
                continue
            buffer.append((item, [(ctx_ids, resp_ids)], "plain"))

        if len(buffer) >= batch_size:
            _flush_buffer(buffer, results, all_ifds, model, device, max_len)

    if buffer:
        _flush_buffer(buffer, results, all_ifds, model, device, max_len)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    valid_ifd = [v for v in all_ifds if v is not None]
    if valid_ifd:
        print(f"\n{'=' * 50}")
        print(f"  Samples:    {len(data)}")
        print(f"  Valid IFD:  {len(valid_ifd)}")
        print(f"  Mean IFD:   {statistics.mean(valid_ifd):.4f}")
        print(f"  Median IFD: {statistics.median(valid_ifd):.4f}")
        print(f"  Stdev IFD:  {statistics.stdev(valid_ifd):.4f}")
        print(f"  Min IFD:    {min(valid_ifd):.4f}")
        print(f"  Max IFD:    {max(valid_ifd):.4f}")
        print(f"{'=' * 50}")

    print(f"Results saved to {output_file}")


def _flush_buffer(buffer, results, all_ifds, model, device, max_len=2048):
    all_pairs = []
    indices = []
    for item, turns, fmt in buffer:
        start = len(all_pairs)
        all_pairs.extend(turns)
        indices.append((item, turns, fmt, start, len(all_pairs)))

    raw = _score_batch(all_pairs, model, device, max_len)

    for item, turns, fmt, start, end in indices:
        turn_scores = raw[start:end]
        if fmt == "messages":
            valid = [s for s in turn_scores if s is not None and s["ifd"] is not None]
            if not valid:
                results.append({**item, "ifd": None, "ifd_turns": turn_scores})
            else:
                avg = sum(s["ifd"] for s in valid) / len(valid)
                all_ifds.append(avg)
                results.append(
                    {
                        **item,
                        "ifd": avg,
                        "ifd_detail": valid[0] if len(valid) == 1 else None,
                        "ifd_turns": turn_scores,
                    }
                )
        else:
            score = turn_scores[0]
            all_ifds.append(score["ifd"])
            results.append({**item, "ifd": score["ifd"], "ifd_detail": score})

    buffer.clear()


def main():
    parser = argparse.ArgumentParser(
        description="Compute IFD scores for instruction-response data"
    )
    parser.add_argument("--param_path", type=str, required=True, help="Model directory")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--max_len", type=int, default=2048, help="Max token length")
    parser.add_argument(
        "--format",
        type=str,
        default="plain",
        choices=["plain", "messages"],
        help="Input format",
    )
    parser.add_argument(
        "--instr_key", type=str, default="instruction", help="Key for instruction field"
    )
    parser.add_argument(
        "--resp_key", type=str, default="response", help="Key for response field"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for model forward passes"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda:0)")
    args = parser.parse_args()

    process_file(
        args.param_path,
        args.input,
        args.output,
        args.instr_key,
        args.resp_key,
        args.max_len,
        data_format=args.format,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
