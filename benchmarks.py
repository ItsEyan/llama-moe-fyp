from typing import List, Tuple, Dict, Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
import math
from dataclasses import dataclass
from torch.profiler import profile, ProfilerActivity

@dataclass
class RunStats:
    total: int = 0
    correct: int = 0
    processed_tokens: int = 0     # tokens model processed in forwards (prefix + suffix, all options)
    supervised_tokens: int = 0    # tokens we actually supervised/scored (e.g., only suffix)
    generated_tokens: int = 0     # for generative tasks (BoolQ)

def _report_energy(
    task_name: str,
    res: RunStats,
    energy_J: float,
    avg_W: float,
    sec: float,
    flops_info: dict | None = None,
):
    Wh = energy_J / 3600.0
    acc = (res.correct / max(1, res.total)) if res.total else float("nan")
    tokps_proc = (res.processed_tokens / sec) if sec > 0 else float("nan")
    tokps_gen  = (res.generated_tokens / sec) if sec > 0 else float("nan")

    J_per_token_proc = energy_J / max(1, res.processed_tokens) if res.processed_tokens else float("nan")
    J_per_token_sup  = energy_J / max(1, res.supervised_tokens) if res.supervised_tokens else float("nan")
    Wh_per_1k_proc   = (Wh / max(1, res.processed_tokens)) * 1000.0 if res.processed_tokens else float("nan")
    Wh_per_1k_sup    = (Wh / max(1, res.supervised_tokens)) * 1000.0 if res.supervised_tokens else float("nan")

    print("\n=== Evaluation Summary ===")
    print(f"{task_name} — N={res.total} | Acc: {acc*100:.2f}%")
    print(f"Time: {sec:.3f}s | Avg GPU Power: {avg_W:.1f} W | Energy: {energy_J:.1f} J ({Wh:.4f} Wh)")
    if res.generated_tokens:
        print(f"Generated tokens: {res.generated_tokens} | gen tok/s: {tokps_gen:.1f}")
    print(f"Processed tokens: {res.processed_tokens} | proc tok/s: {tokps_proc:.1f}")
    if res.supervised_tokens:
        print(f"J/token (proc): {J_per_token_proc:.3f} | J/token (sup): {J_per_token_sup:.3f}")
        print(f"Wh/1k tok (proc): {Wh_per_1k_proc:.3f} | Wh/1k tok (sup): {Wh_per_1k_sup:.3f}")
    else:
        print(f"J/token (proc): {J_per_token_proc:.3f} | Wh/1k tok (proc): {Wh_per_1k_proc:.3f}")

    # ---- FLOPs summary ----
    if flops_info and flops_info.get("total_flops_forward") is not None:
        fi = flops_info
        print(
            f"[FLOPs] Forward (first batch): {fi['total_flops_forward']/1e12:.3f} TFLOPs "
            f"(~{fi['flops_per_token_forward']:.0f} FLOPs/token over {fi['profiled_batch_tokens']} tokens)"
        )
        approx_total_flops_eval = fi["flops_per_token_forward"] * max(1, res.generated_tokens or res.supervised_tokens)
        print(
            f"[FLOPs] Approx energy per 10¹² FLOPs (using new/supervised tokens): "
            f"{energy_J / max(1, approx_total_flops_eval / 1e12):.3f} J / TFLOP"
        )

    # ---- MoE expert activation histogram ----
    try:
        from smoe.modules.moe.moe_gates import global_hist
    except Exception:
        global_hist = {}

    sorted_hist = dict(sorted(global_hist.items()))
    total_h = sum(sorted_hist.values())
    if sorted_hist:
        print("\n=== Accumulated Expert Activation Histogram ===")
        for k, count in sorted_hist.items():
            pct = (count / total_h * 100.0) if total_h else 0.0
            print(f"k={k:>2d}: {count:>8d}  ({pct:.2f}%)")



def _sequence_token_count(input_ids: torch.Tensor) -> int:
    # counts tokens “seen” by the model (we predict for T-1 positions, but energy is for all)
    return int(input_ids.numel())

# ----------------------------
# Public API
# ----------------------------
__all__ = [
    "eval_boolq",
    "eval_piqa",
    "eval_hellaswag",
    "eval_arc",
    "eval_lambada",
]

# ----------------------------
# Shared helpers (no model state)
# ----------------------------
def _pad_batch_texts(tokenizer, device, texts: List[str], max_len: int = 1024):
    return tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    ).to(device)

@torch.no_grad()
def _sum_logprobs_for_suffix(
    model,
    tokenizer,
    device,
    prefix_ids: torch.Tensor,
    suffix_ids_list: List[torch.Tensor],
    attention_mask_prefix: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, int, int]:
    """
    Returns:
      scores [B, K],
      processed_tokens (int): total tokens passed through model across all K forwards,
      supervised_tokens (int): total suffix tokens scored across all K forwards.
    """
    B = prefix_ids.size(0)
    K = len(suffix_ids_list)
    scores = []
    processed_tokens = 0
    supervised_tokens = 0

    for k in range(K):
        suffix_ids = suffix_ids_list[k]
        assert suffix_ids.size(0) == B
        inp = torch.cat([prefix_ids, suffix_ids], dim=1)  # [B, Tp+Ts]
        attn = None
        if attention_mask_prefix is not None:
            attn = torch.cat(
                [attention_mask_prefix, torch.ones_like(suffix_ids, device=device)],
                dim=1
            )

        out = model(
            input_ids=inp,
            attention_mask=attn,
            use_cache=False,
            return_dict=True
        )

        # token accounting
        processed_tokens += _sequence_token_count(inp)
        Ts = suffix_ids.size(1)
        supervised_tokens += int(Ts * B)

        logits = out.logits[:, :-1, :]
        target = inp[:, 1:]

        Tp = prefix_ids.size(1)
        mask = torch.zeros_like(target, dtype=torch.bool, device=device)
        mask[:, Tp-1:] = True

        logprobs = torch.log_softmax(logits, dim=-1)
        tgt_logp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        scores.append(tgt_logp.masked_select(mask).view(B, -1).sum(dim=1))

    return torch.stack(scores, dim=1), processed_tokens, supervised_tokens


# =========================================================
# Evaluations (each takes: model, tokenizer, device, energy_fn)
# energy_fn must be integrate_gpu_energy_joules(callable)
# =========================================================

# ---------- BoolQ ----------
@torch.no_grad()
def eval_boolq(model, tokenizer, device, integrate_gpu_energy_joules,
               max_eval: int = 1024, batch_size: int = 8, max_new_tokens: int = 3, collect_flops: bool = True):
    """
    Mirrors your existing BoolQ flow but without MoE diagnostics to keep it generic.
    If you want MoE probes, keep those in your main file where your gate internals live.
    """
    ds = load_dataset("google/boolq")
    val_ds = ds["validation"]
    subset = val_ds if (max_eval == -1 or max_eval >= len(val_ds)) else val_ds.select(
        range(min(max_eval, len(val_ds)))
    )

    PROMPT_TMPL = (
        "You are a precise assistant answering Yes/No using the passage.\n"
        "Passage: {passage}\n"
        "Question: {question}\n"
        "Answer with a single word: Yes or No.\n"
        "Answer: "
    )

    def make_batch_prompts(batch):
        return [
            PROMPT_TMPL.format(passage=p, question=q)
            for p, q in zip(batch["passage"], batch["question"])
        ]

    def parse_yes_no(text: str) -> str:
        t = text.strip().lower()
        t = t.split()[0] if t else ""
        if t.startswith("yes"): return "yes"
        if t.startswith("no"):  return "no"
        if "yes" in text.lower(): return "yes"
        if "no"  in text.lower(): return "no"
        return "unknown"

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}
    def _run():
        total = 0; correct = 0; total_new_tokens = 0; processed_tokens = 0
        pbar = tqdm(range(0, len(subset), batch_size), desc="BoolQ", unit="batch")
        for bi, i in enumerate(pbar):
            batch = subset[i:i+batch_size]
            prompts = make_batch_prompts(batch)
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

            # ---- FLOPs on first batch (forward-only, no generate) ----
            if collect_flops and bi == 0:
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             with_flops=True, record_shapes=False) as prof:
                    _ = model(**enc, use_cache=False, return_dict=True)
                    torch.cuda.synchronize()
                total_flops = sum(e.flops for e in prof.key_averages())
                num_tokens = enc["input_ids"].numel()
                flops_info["total_flops_forward"] = total_flops
                flops_info["flops_per_token_forward"] = total_flops / max(1, num_tokens)
                flops_info["profiled_batch_tokens"] = int(num_tokens)

            in_len = enc["input_ids"].shape[1]
            processed_tokens += _sequence_token_count(enc["input_ids"])

            torch.cuda.synchronize()
            out_ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
            torch.cuda.synchronize()

            new_tok_batch = out_ids.shape[1] - in_len
            total_new_tokens += new_tok_batch * out_ids.shape[0]

            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for j, full_text in enumerate(decoded):
                gold_bool = bool(batch["answer"][j])
                gen_suffix = full_text[-max(0, max_new_tokens * 8):]
                pred = parse_yes_no(gen_suffix)
                gold = "yes" if gold_bool else "no"
                total += 1
                correct += int(pred == gold)

            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(acc=f"{acc*100:.2f}%", gen_tokens=total_new_tokens)

        res_stats.total = total
        res_stats.correct = correct
        res_stats.generated_tokens = total_new_tokens
        res_stats.processed_tokens = processed_tokens
        return {"ok": True}

    _, energy_J, avg_W, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy("BoolQ", res_stats, energy_J, avg_W, sec, flops_info)

# ---------- PIQA ----------
@torch.no_grad()
def eval_piqa(model, tokenizer, device, integrate_gpu_energy_joules,
              batch_size: int = 16, max_eval: int = -1, collect_flops: bool = True):
    ds = load_dataset("piqa")
    val = ds["validation"]
    if max_eval != -1:
        val = val.select(range(min(max_eval, len(val))))

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}
    
    def _run():
        total = 0; correct = 0; proc_tok = 0; sup_tok = 0

        for bi in tqdm(range(0, len(val), batch_size), desc="PIQA", unit="batch"):
            batch = val[bi:bi+batch_size]
            goals = batch["goal"]
            sol1  = batch["sol1"]
            sol2  = batch["sol2"]
            labels = batch["label"]

            prefixes = [f"Goal: {g}\nSelect the best solution.\nSolution:" for g in goals]
            enc_prefix = _pad_batch_texts(tokenizer, device, prefixes)
            sfx1 = _pad_batch_texts(tokenizer, device, [" " + s for s in sol1])
            sfx2 = _pad_batch_texts(tokenizer, device, [" " + s for s in sol2])

            # ---- FLOPs on first batch: prefix + option-0 as representative ----
            if collect_flops and bi == 0:
                inp0  = torch.cat([enc_prefix["input_ids"], sfx1["input_ids"]], dim=1)
                attn0 = torch.cat([enc_prefix["attention_mask"],
                                   torch.ones_like(sfx1["input_ids"], device=device)], dim=1)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             with_flops=True, record_shapes=False) as prof:
                    _ = model(input_ids=inp0, attention_mask=attn0, use_cache=False, return_dict=True)
                    torch.cuda.synchronize()
                total_flops = sum(e.flops for e in prof.key_averages())
                flops_info["total_flops_forward"] = total_flops
                flops_info["flops_per_token_forward"] = total_flops / max(1, inp0.numel())
                flops_info["profiled_batch_tokens"] = int(inp0.numel())

            scores, pt, st = _sum_logprobs_for_suffix(
                model, tokenizer, device,
                enc_prefix["input_ids"],
                [sfx1["input_ids"], sfx2["input_ids"]],
                attention_mask_prefix=enc_prefix["attention_mask"]
            )
            proc_tok += pt; sup_tok += st

            pred_idx = scores.argmax(dim=1).tolist()
            total += len(labels)
            for j, lab in enumerate(labels):
                correct += int(pred_idx[j] == lab)

        res_stats.total = total
        res_stats.correct = correct
        res_stats.processed_tokens = proc_tok
        res_stats.supervised_tokens = sup_tok
        return {"ok": True}

    _, E, Pavg, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy("PIQA", res_stats, E, Pavg, sec, flops_info)


# ---------- HellaSwag ----------
@torch.no_grad()
def eval_hellaswag(
    model, tokenizer, device, integrate_gpu_energy_joules,
    batch_size: int = 8, max_eval: int = -1, collect_flops: bool = True
):
    ds = load_dataset("hellaswag")
    val = ds["validation"]
    if max_eval != -1:
        val = val.select(range(min(max_eval, len(val))))

    def _ctx(x):
        if "ctx" in x and x["ctx"]:
            return x["ctx"]
        return (x.get("ctx_a","") + " " + x.get("ctx_b","")).strip()

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}

    def _run():
        total = 0; correct = 0; proc_tok = 0; sup_tok = 0
        for bi in tqdm(range(0, len(val), batch_size), desc="HellaSwag", unit="batch"):
            batch = val[bi:bi+batch_size]
            ctxs = [_ctx(x) for x in batch]
            ends = batch["endings"]
            endings = list(zip(*ends)) if isinstance(ends[0], list) else [[e[k] for e in ends] for k in range(4)]
            gold = batch["label"]

            enc_ctx = _pad_batch_texts(tokenizer, device, ctxs)
            suffix_batches = [_pad_batch_texts(tokenizer, device, [" " + e for e in endings[k]]) for k in range(4)]

            # FLOPs on first batch: profile forward on ctx + option-0 as a representative
            if collect_flops and bi == 0:
                inp0 = torch.cat([enc_ctx["input_ids"], suffix_batches[0]["input_ids"]], dim=1)
                attn0 = torch.cat([enc_ctx["attention_mask"], torch.ones_like(suffix_batches[0]["input_ids"], device=device)], dim=1)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             with_flops=True, record_shapes=False) as prof:
                    _ = model(input_ids=inp0, attention_mask=attn0, use_cache=False, return_dict=True)
                    torch.cuda.synchronize()
                total_flops = sum(e.flops for e in prof.key_averages())
                flops_info["total_flops_forward"] = total_flops
                flops_info["flops_per_token_forward"] = total_flops / max(1, inp0.numel())
                flops_info["profiled_batch_tokens"] = int(inp0.numel())

            scores, pt, st = _sum_logprobs_for_suffix(
                model, tokenizer, device,
                enc_ctx["input_ids"],
                [s["input_ids"] for s in suffix_batches],
                attention_mask_prefix=enc_ctx["attention_mask"]
            )
            proc_tok += pt; sup_tok += st

            pred = scores.argmax(dim=1).tolist()
            total += len(gold)
            for j, lab in enumerate(gold):
                correct += int(pred[j] == lab)

        res_stats.total = total
        res_stats.correct = correct
        res_stats.processed_tokens = proc_tok
        res_stats.supervised_tokens = sup_tok
        return {"ok": True}

    _, E, Pavg, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy("HellaSwag", res_stats, E, Pavg, sec, flops_info)


# ---------- ARC (Easy/Challenge) ----------
@torch.no_grad()
def eval_arc(
    model, tokenizer, device, integrate_gpu_energy_joules,
    subset: str = "ARC-Challenge", batch_size: int = 8, max_eval: int = -1, collect_flops: bool = True
):
    ds = load_dataset("ai2_arc", subset)
    val = ds["validation"]
    if max_eval != -1:
        val = val.select(range(min(max_eval, len(val))))

    def _format_q(q) -> tuple[str, list[str], int]:
        stem = q["question"]
        texts = q["choices"]["text"]
        labels = q["choices"]["label"]  # e.g., ["A","B","C","D"]
        key = q["answerKey"]           # e.g., "C"
        gold_idx = labels.index(key)
        opt_lines = [f"{labels[i]}. {texts[i]}" for i in range(len(texts))]
        prompt = "Question: " + stem + "\n" + "\n".join(opt_lines) + "\nAnswer:"
        return prompt, texts, gold_idx

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}

    def _run():
        total = 0; correct = 0; proc_tok = 0; sup_tok = 0
        for bi in tqdm(range(0, len(val), batch_size), desc=f"ARC-{subset.split('-')[-1]}", unit="batch"):
            batch = val[bi:bi+batch_size]
            formatted = [_format_q(x) for x in batch]
            prompts = [f for (f, _, _) in formatted]
            enc = _pad_batch_texts(tokenizer, device, prompts)

            K = max(len(x["choices"]["text"]) for x in batch)
            option_tensors = []
            for k in range(K):
                opts_k = []
                for b in range(len(batch)):
                    opt_list = batch[b]["choices"]["text"]
                    opts_k.append(" " + opt_list[k] if k < len(opt_list) else " [N/A]")
                option_tensors.append(_pad_batch_texts(tokenizer, device, opts_k)["input_ids"])

            # FLOPs on first batch: prompt + first option as representative
            if collect_flops and bi == 0:
                inp0 = torch.cat([enc["input_ids"], option_tensors[0]], dim=1)
                attn0 = torch.cat([enc["attention_mask"], torch.ones_like(option_tensors[0], device=device)], dim=1)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             with_flops=True, record_shapes=False) as prof:
                    _ = model(input_ids=inp0, attention_mask=attn0, use_cache=False, return_dict=True)
                    torch.cuda.synchronize()
                total_flops = sum(e.flops for e in prof.key_averages())
                flops_info["total_flops_forward"] = total_flops
                flops_info["flops_per_token_forward"] = total_flops / max(1, inp0.numel())
                flops_info["profiled_batch_tokens"] = int(inp0.numel())

            scores, pt, st = _sum_logprobs_for_suffix(
                model, tokenizer, device,
                enc["input_ids"],
                option_tensors,
                attention_mask_prefix=enc["attention_mask"]
            )
            proc_tok += pt; sup_tok += st

            gold = [g for (_, _, g) in formatted]
            pred = scores.argmax(dim=1).tolist()
            total += len(gold)
            for j in range(len(gold)):
                correct += int(pred[j] == gold[j])

        res_stats.total = total
        res_stats.correct = correct
        res_stats.processed_tokens = proc_tok
        res_stats.supervised_tokens = sup_tok
        return {"ok": True}

    _, E, Pavg, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy(subset, res_stats, E, Pavg, sec, flops_info)


# ---------- LAMBADA ----------
@torch.no_grad()
def eval_lambada(model, tokenizer, device, integrate_gpu_energy_joules,
                 batch_size: int = 16, max_eval: int = 5000, max_len: int = 1024, collect_flops: bool = True):
    # Try common sources
    try:
        ds = load_dataset("EleutherAI/lambada_open")
        split = "test"
    except Exception:
        ds = load_dataset("lambada")
        split = "test"

    data = ds[split]
    if max_eval != -1:
        data = data.select(range(min(max_eval, len(data))))

    def _split_ctx_target(text: str) -> Tuple[str, str]:
        text = text.strip()
        parts = text.rsplit(" ", 1)
        if len(parts) == 1:
            return "", parts[0]
        return parts[0], parts[1]

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}
    
    def _run():
        total = 0; correct = 0; proc_tok = 0

        for bi in tqdm(range(0, len(data), batch_size), desc="LAMBADA", unit="batch"):
            batch = data[bi:bi+batch_size]
            texts = batch["text"] if "text" in batch.features else batch["sentence"]
            ctxs, targets = zip(*[_split_ctx_target(t) for t in texts])

            enc = tokenizer(list(ctxs), return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            proc_tok += _sequence_token_count(enc["input_ids"])

            # ---- FLOPs on first batch: plain forward over the context ----
            if collect_flops and bi == 0:
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             with_flops=True, record_shapes=False) as prof:
                    _ = model(**enc, use_cache=False, return_dict=True)
                    torch.cuda.synchronize()
                total_flops = sum(e.flops for e in prof.key_averages())
                num_tokens = enc["input_ids"].numel()
                flops_info["total_flops_forward"] = total_flops
                flops_info["flops_per_token_forward"] = total_flops / max(1, num_tokens)
                flops_info["profiled_batch_tokens"] = int(num_tokens)

            out = model(**enc, use_cache=False, return_dict=True)
            logits = out.logits[:, -1, :]
            pred_ids = logits.argmax(dim=-1)

            tgt_ids = tokenizer(list(targets), add_special_tokens=False).input_ids
            tgt_first_ids = [ids[0] if len(ids) > 0 else tokenizer.eos_token_id for ids in tgt_ids]
            tgt_first_ids = torch.tensor(tgt_first_ids, device=device)

            total += len(texts)
            correct += (pred_ids == tgt_first_ids).sum().item()

        res_stats.total = total
        res_stats.correct = correct
        res_stats.processed_tokens = proc_tok
        return {"ok": True}

    _, E, Pavg, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy("LAMBADA", res_stats, E, Pavg, sec, flops_info)