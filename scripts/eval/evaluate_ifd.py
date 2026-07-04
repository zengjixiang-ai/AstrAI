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


def _score(context_ids, resp_ids, model, device):
    """Core IFD computation: context → L_cond, response alone → L_uncond."""
    if not resp_ids:
        return None
    full_ids = context_ids + resp_ids
    inp_full = torch.tensor([full_ids], device=device, dtype=torch.long)
    inp_resp = torch.tensor([resp_ids], device=device, dtype=torch.long)
    logits_full = model(inp_full)["logits"][0]
    logits_resp = model(inp_resp)["logits"][0]
    ctx_len = len(context_ids)
    resp_logits = logits_full[ctx_len - 1 : -1]
    resp_targets = torch.tensor(resp_ids, device=device, dtype=torch.long)
    L_cond = F.cross_entropy(resp_logits, resp_targets, reduction="mean").item()
    unp_logits = logits_resp[:-1]
    unp_targets = torch.tensor(resp_ids[1:], device=device, dtype=torch.long)
    L_uncond = F.cross_entropy(unp_logits, unp_targets, reduction="mean").item()
    ifd = L_cond / L_uncond if L_uncond > 0 else None
    return {
        "L_cond": round(L_cond, 6),
        "L_uncond": round(L_uncond, 6),
        "ifd": round(ifd, 6) if ifd is not None else None,
        "resp_len": len(resp_ids),
    }


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
    return _score(ctx_ids, resp_ids, model, device)


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
            turns.append(_score(ctx_ids, resp_ids, model, device))
    if not turns:
        return None
    valid = [t for t in turns if t is not None and t["ifd"] is not None]
    if not valid:
        return {"ifd": None, "ifd_turns": turns}
    avg = sum(t["ifd"] for t in valid) / len(valid)
    return {
        "ifd": avg,
        "ifd_detail": valid[0] if len(valid) == 1 else None,
        "ifd_turns": turns,
    }


@torch.inference_mode()
def process_file(
    param_path,
    input_file,
    output_file,
    instr_key,
    resp_key,
    max_len=2048,
    data_format="plain",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device=device, dtype=dtype)
    model.eval()

    with open(input_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    results = []
    all_ifds = []

    for item in tqdm.tqdm(data, desc="Computing IFD", unit="sample"):
        if data_format == "messages":
            scores = score_messages(
                model, tokenizer, item.get("messages", []), device, max_len
            )
            if scores is None:
                results.append({**item, "ifd": None, "ifd_turns": []})
            else:
                all_ifds.append(scores["ifd"])
                results.append({**item, **scores})
        else:
            scores = score_plain(
                model, tokenizer, item[instr_key], item[resp_key], device, max_len
            )
            all_ifds.append(scores["ifd"])
            results.append({**item, "ifd": scores["ifd"], "ifd_detail": scores})

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
        help="Input format: 'plain' for instr_key+resp_key, 'messages' for messages array",
    )
    parser.add_argument(
        "--instr_key", type=str, default="instruction", help="Key for instruction field"
    )
    parser.add_argument(
        "--resp_key", type=str, default="response", help="Key for response field"
    )
    args = parser.parse_args()

    process_file(
        args.param_path,
        args.input,
        args.output,
        args.instr_key,
        args.resp_key,
        args.max_len,
        data_format=args.format,
    )


if __name__ == "__main__":
    main()
