from typing import List, Optional
import torch
import time
import threading
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass
from torch.profiler import profile, ProfilerActivity
from pynvml import (nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetPowerUsage)
from torch.profiler import profile, ProfilerActivity

@dataclass
class RunStats:
    total: int = 0
    correct: int = 0
    processed_tokens: int = 0     # tokens model processed in forwards (prefix + suffix, all options)
    supervised_tokens: int = 0    # tokens we actually supervised/scored (e.g., only suffix)
    generated_tokens: int = 0     # for generative tasks (BoolQ)

def integrate_gpu_energy_joules(fn, poll_ms=10, device_index=0):
    """
    Run `fn()` on the MAIN thread (so prints/logs show), and sample GPU power
    in a background daemon thread. Returns (res, energy_J, avg_power_W, duration_s).
    """
    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(device_index)
        samples = []
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                try:
                    samples.append((time.perf_counter(), nvmlDeviceGetPowerUsage(handle)))
                except Exception:
                    pass
                time.sleep(poll_ms / 1000.0)

        t0 = time.perf_counter()
        th = threading.Thread(target=sampler, daemon=True)
        th.start()

        res = fn()  # run eval on main thread

        stop.set()
        th.join(timeout=1.0)

        t1 = time.perf_counter()
        try:
            samples.append((t1, nvmlDeviceGetPowerUsage(handle)))
        except Exception:
            pass

        # trapezoidal integration (power in mW -> W)
        energy_J = 0.0
        for (t_prev, p_prev), (t_cur, p_cur) in zip(samples[:-1], samples[1:]):
            dt = t_cur - t_prev
            energy_J += 0.5 * ((p_prev + p_cur) / 1000.0) * dt

        duration_s = t1 - t0
        avg_power_W = (energy_J / duration_s) if duration_s > 0 else float("nan")
        return res, energy_J, avg_power_W, duration_s
    finally:
        nvmlShutdown()

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
    
    J_per_example = energy_J / max(1, res.total) if res.total else float("nan")

    print("\n=== Evaluation Summary ===")
    print(f"{task_name} — N={res.total} | Acc: {acc*100:.2f}%")
    print(f"Time: {sec:.3f}s | Avg GPU Power: {avg_W:.1f} W | Energy: {energy_J:.1f} J ({Wh:.4f} Wh)")
    print(f"Energy/example: {J_per_example:.3f} J")
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
        approx_total_flops_eval = fi["flops_per_token_forward"] * max(1, res.processed_tokens)
        print(
            f"[FLOPs] Approx energy per 10¹² FLOPs (using new/supervised tokens): "
            f"{energy_J / max(1, approx_total_flops_eval / 1e12):.3f} J / TFLOP"
        )

    # ---- MoE expert activation histogram ----
    try:
        from smoe.modules.moe.moe_gates import global_hist
    except Exception:
        global_hist = {}
    
    try:
        from smoe.modules.moe.moe_gates import global_expert_hist
    except Exception:
        global_expert_hist = {}

    sorted_hist = dict(sorted(global_hist.items()))
    total_h = sum(sorted_hist.values())
    if sorted_hist:
        print("\n=== Accumulated k-per-token Histogram ===")
        for k, count in sorted_hist.items():
            pct = (count / total_h * 100.0) if total_h else 0.0
            print(f"k={k:>2d}: {count:>8d}  ({pct:.2f}%)")

    sorted_ehist = dict(sorted(global_expert_hist.items()))
    total_e = sum(sorted_ehist.values())
    if sorted_ehist:
        print("\n=== Accumulated Expert Activation Histogram ===")
        for e, count in sorted_ehist.items():
            pct = (count / total_e * 100.0) if total_e else 0.0
            print(f"expert={e:>2d}: {count:>8d}  ({pct:.2f}%)")



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
    Compute log-prob scores for multiple suffixes given a common prefix.
    
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

def _collect_flops_once(
    task_name: str,
    model,
    forward_kwargs: dict,
    flops_info: dict,
    sort_by: str = "self_cuda_time_total",
    row_limit: int = 60,
):
    """
    Profiles a single forward pass, prints a CPU/CUDA table, and fills `flops_info` with:
      - total_flops_forward
      - flops_per_token_forward
      - profiled_batch_tokens
    Works on CPU-only and on torch builds without `with_flops` (falls back gracefully).
    """

    # Warmup to reduce kernel-init noise
    _ = model(**forward_kwargs, use_cache=False, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])

    # Try with FLOPs; if not supported, retry without.
    try:
        with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
            _ = model(**forward_kwargs, use_cache=False, return_dict=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        with_flops_ok = True
    except TypeError:
        with profile(activities=activities, record_shapes=True) as prof:
            _ = model(**forward_kwargs, use_cache=False, return_dict=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        with_flops_ok = False

    # Robust table print
    print(f"\n=== PyTorch Profiler (first batch) — {task_name} ===")
    table_str = None
    try:
        table_str = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    except Exception:
        # Try common alternates; then default
        for key in ("cuda_time_total", "self_cpu_time_total", "cpu_time_total"):
            try:
                table_str = prof.key_averages().table(sort_by=key, row_limit=row_limit)
                break
            except Exception:
                continue
        if table_str is None:
            try:
                table_str = prof.key_averages().table(row_limit=row_limit)
            except Exception as e:
                table_str = f"[Profiler table unavailable: {e}]"
    print(table_str)

    # Aggregate FLOPs summary (if available)
    total_flops = 0
    if with_flops_ok:
        try:
            for e in prof.key_averages():
                total_flops += (getattr(e, "flops", 0) or 0)
        except Exception:
            total_flops = 0

    # Normalize by tokens if present
    inp = forward_kwargs.get("input_ids")
    num_tokens = int(inp.numel()) if torch.is_tensor(inp) else 0

    flops_info["total_flops_forward"] = total_flops or None
    flops_info["flops_per_token_forward"] = (total_flops / max(1, num_tokens)) if (total_flops and num_tokens) else None
    flops_info["profiled_batch_tokens"] = num_tokens or None

# =========================================================
# Evaluations (each takes: model, tokenizer, device, energy_fn)
# energy_fn must be integrate_gpu_energy_joules(callable)
# =========================================================

# ---------- BoolQ ----------
@torch.no_grad()
def eval_boolq(model, tokenizer, device, max_eval: int = 1024, batch_size: int = 8, collect_flops: bool = True):
    """
    BoolQ evaluation via conditional log-likelihood:
      score(" Yes" | prefix) vs score(" No" | prefix), choose higher.

    Tracks processed_tokens (all prefix+suffix tokens fed to the model)
    and supervised_tokens (suffix tokens actually scored).
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

    def make_batch_prefixes(batch):
        return [
            PROMPT_TMPL.format(passage=p, question=q)
            for p, q in zip(batch["passage"], batch["question"])
        ]

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}

    def _run():
        total = 0
        correct = 0
        processed_tokens = 0
        supervised_tokens = 0

        pbar = tqdm(range(0, len(subset), batch_size), desc="BoolQ", unit="batch")
        for bi, i in enumerate(pbar):
            batch = subset[i:i+batch_size]

            # ----- Build prefixes -----
            prefixes = make_batch_prefixes(batch)
            enc_prefix = _pad_batch_texts(tokenizer, device, prefixes)  # dict with input_ids, attention_mask

            # ----- Two fixed verbalizers with leading space -----
            B = enc_prefix["input_ids"].size(0)
            sfx_yes = _pad_batch_texts(tokenizer, device, [" Yes"] * B)
            sfx_no  = _pad_batch_texts(tokenizer, device, [" No"]  * B)
            
            if collect_flops and bi == 0:
                inp_full = torch.cat([enc_prefix["input_ids"], sfx_yes["input_ids"]], dim=1)
                attn_full = torch.cat([enc_prefix["attention_mask"], torch.ones_like(sfx_yes["input_ids"], device=device)], dim=1)
                _collect_flops_once(
                    task_name="BoolQ (prefix+suffix)",
                    model=model,
                    forward_kwargs={"input_ids": inp_full, "attention_mask": attn_full},
                    flops_info=flops_info,
                    sort_by="self_cuda_time_total",
                )


            # ----- Score suffixes: log p(" Yes" | prefix) vs log p(" No" | prefix) -----
            scores, pt, st = _sum_logprobs_for_suffix(
                model, tokenizer, device,
                enc_prefix["input_ids"],
                [sfx_yes["input_ids"], sfx_no["input_ids"]],
                attention_mask_prefix=enc_prefix["attention_mask"]
            )
            processed_tokens += pt
            supervised_tokens += st

            # ----- Predictions -----
            pred_idx = scores.argmax(dim=1).tolist()  # 0 -> " Yes", 1 -> " No"
            gold = [("yes" if bool(a) else "no") for a in batch["answer"]]
            for j, p in enumerate(pred_idx):
                pred = "yes" if p == 0 else "no"
                correct += int(pred == gold[j])
            total += len(gold)

            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(acc=f"{acc*100:.2f}%")

        res_stats.total = total
        res_stats.correct = correct
        res_stats.generated_tokens = 0
        res_stats.processed_tokens = processed_tokens
        res_stats.supervised_tokens = supervised_tokens
        return {"ok": True}

    _, energy_J, avg_W, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy("BoolQ", res_stats, energy_J, avg_W, sec, flops_info)

# ---------- PIQA ----------
@torch.no_grad()
def eval_piqa(model, tokenizer, device, batch_size: int = 16, max_eval: int = -1, collect_flops: bool = True):
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
                _collect_flops_once(
                    task_name="PIQA",
                    model=model,
                    forward_kwargs={"input_ids": inp0, "attention_mask": attn0},
                    flops_info=flops_info,
                    sort_by="self_cuda_time_total",
                )

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
    model, tokenizer, device, batch_size: int = 8, max_eval: int = -1, collect_flops: bool = True
):
    ds = load_dataset("hellaswag")
    val = ds["validation"]
    if max_eval != -1:
        val = val.select(range(min(max_eval, len(val))))

    # small helpers so we work with either a Dataset slice or a dict-of-lists
    def _batch_len(batch) -> int:
        if hasattr(batch, "num_rows"):
            return int(batch.num_rows)
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, (list, tuple)):
                    return len(v)
            return 0
        return len(batch)

    def _col(batch, name, default, n: int):
        if hasattr(batch, "column_names") and name in batch.column_names:
            return batch[name]
        if isinstance(batch, dict) and name in batch:
            return batch[name]
        return default if not callable(default) else default(n)

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}

    def _run():
        total = 0
        correct = 0
        proc_tok = 0
        sup_tok = 0

        for bi in tqdm(range(0, len(val), batch_size), desc="HellaSwag", unit="batch"):
            batch = val[bi:bi + batch_size]
            n = _batch_len(batch)
            if n == 0:
                continue

            # ---- contexts (multiple schema variants across dumps) ----
            ctxs = _col(batch, "ctx", None, n)
            if ctxs is None:
                ctxs = _col(batch, "context", None, n)
            if ctxs is None:
                a = _col(batch, "ctx_a", [""] * n, n)
                b = _col(batch, "ctx_b", [""] * n, n)
                ctxs = [(aa + " " + bb).strip() for aa, bb in zip(a, b)]

            # ---- endings: [n, K] -> [K, n] ----
            ends = _col(batch, "endings", lambda _n: [["", "", "", ""] for _ in range(_n)], n)
            # normalize to python lists then transpose
            ends = [list(e) if not isinstance(e, list) else e for e in ends]
            endings_by_k = list(map(list, zip(*ends)))
            K = len(endings_by_k)

            # ---- labels (force int in [0..K-1]) ----
            gold_raw = _col(batch, "label", [0] * n, n)
            try:
                gold = [int(x) for x in gold_raw]
            except Exception:
                gold = [int(x.item() if hasattr(x, "item") else x) for x in gold_raw]

            # ---- tokenize ----
            enc_ctx = _pad_batch_texts(tokenizer, device, ctxs)
            suffix_batches = [
                _pad_batch_texts(tokenizer, device, [" " + e for e in endings_by_k[k]])
                for k in range(K)
            ]

            # ---- FLOPs on first batch: ctx + option-0 as representative ----
            if collect_flops and bi == 0:
                inp0  = torch.cat([enc_ctx["input_ids"], suffix_batches[0]["input_ids"]], dim=1)
                attn0 = torch.cat(
                    [enc_ctx["attention_mask"], torch.ones_like(suffix_batches[0]["input_ids"], device=device)],
                    dim=1
                )
                _collect_flops_once(
                    task_name="HellaSwag",
                    model=model,
                    forward_kwargs={"input_ids": inp0, "attention_mask": attn0},
                    flops_info=flops_info,
                    sort_by="self_cuda_time_total",
                )

            # ---- score options ----
            scores, pt, st = _sum_logprobs_for_suffix(
                model, tokenizer, device,
                enc_ctx["input_ids"],
                [s["input_ids"] for s in suffix_batches],
                attention_mask_prefix=enc_ctx["attention_mask"]
            )
            proc_tok += pt
            sup_tok  += st

            pred = scores.argmax(dim=1).tolist()
            total += n
            for j, lab in enumerate(gold):
                correct += int(pred[j] == lab)

        res_stats.total = total
        res_stats.correct = correct
        res_stats.processed_tokens = proc_tok
        res_stats.supervised_tokens = sup_tok
        return {"ok": True}

    _, E, Pavg, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy("HellaSwag", res_stats, E, Pavg, sec, flops_info)


# ---------- ARC (Easy/Challenge) — robust batch handling + FLOPs ----------
@torch.no_grad()
def eval_arc(
    model, tokenizer, device, subset: str = "ARC-Challenge", batch_size: int = 8, max_eval: int = -1, collect_flops: bool = True
):
    ds = load_dataset("ai2_arc", subset)
    val = ds["validation"]
    if max_eval != -1:
        val = val.select(range(min(max_eval, len(val))))

    # helpers to handle Dataset slices OR dict-of-lists
    def _batch_len(batch) -> int:
        if hasattr(batch, "num_rows"):
            return int(batch.num_rows)
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, (list, tuple)):
                    return len(v)
            return 0
        return len(batch)

    def _col(batch, name, default, n: int):
        if hasattr(batch, "column_names") and name in batch.column_names:
            return batch[name]
        if isinstance(batch, dict) and name in batch:
            return batch[name]
        return default if not callable(default) else default(n)

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}

    def _run():
        total = 0; correct = 0; proc_tok = 0; sup_tok = 0

        for bi in tqdm(range(0, len(val), batch_size), desc=f"ARC-{subset.split('-')[-1]}", unit="batch"):
            batch = val[bi:bi+batch_size]
            n = _batch_len(batch)
            if n == 0:
                continue

            # pull columns
            questions   = _col(batch, "question", [""] * n, n)
            answer_keys = _col(batch, "answerKey", [""] * n, n)
            choices     = _col(batch, "choices", None, n)

            # choices can be dict-of-lists OR list-of-dicts; normalize per-row
            row_texts, row_labels = [], []
            if isinstance(choices, dict):
                # dict-of-lists: {"text": [list per row], "label": [list per row]}
                text_cols  = choices.get("text", [[]] * n)
                label_cols = choices.get("label", [[]] * n)
                for i in range(n):
                    row_texts.append(list(text_cols[i]))
                    row_labels.append(list(label_cols[i]))
            elif isinstance(choices, list):
                # list-of-dicts: [{"text":[...], "label":[...]} ...]
                for i in range(n):
                    ch_i = choices[i] or {}
                    row_texts.append(list(ch_i.get("text", [])))
                    row_labels.append(list(ch_i.get("label", [])))
            else:
                # fallback: no choices column; make empty
                row_texts = [[] for _ in range(n)]
                row_labels = [[] for _ in range(n)]

            # build prompts + gold indices
            prompts = []
            gold_idx = []
            Kmax = 0
            for i in range(n):
                texts_i  = row_texts[i]
                labels_i = row_labels[i]
                Kmax = max(Kmax, len(texts_i))

                # prompt with lettered options
                opt_lines = [f"{labels_i[j]}. {texts_i[j]}" for j in range(len(texts_i))]
                prompt = "Question: " + questions[i] + "\n" + "\n".join(opt_lines) + "\nAnswer:"
                prompts.append(prompt)

                # gold index: map answerKey (e.g., "C") to index in labels
                ak = str(answer_keys[i])
                try:
                    gi = labels_i.index(ak)
                except ValueError:
                    # if labels_i are 0/1/2 etc, try int cast
                    try:
                        gi = labels_i.index(int(ak))
                    except Exception:
                        gi = -1
                gold_idx.append(gi)

            # tokenize prompts
            enc = _pad_batch_texts(tokenizer, device, prompts)
            proc_tok += _sequence_token_count(enc["input_ids"])

            # per-option suffix tensors (pad missing options with N/A)
            option_tensors = []
            for k in range(Kmax):
                opts_k = []
                for i in range(n):
                    if k < len(row_texts[i]):
                        opts_k.append(" " + row_texts[i][k])
                    else:
                        opts_k.append(" [N/A]")
                option_tensors.append(_pad_batch_texts(tokenizer, device, opts_k)["input_ids"])

            # FLOPs on first batch: prompt + first option as representative
            if collect_flops and bi == 0 and Kmax > 0:
                inp0  = torch.cat([enc["input_ids"], option_tensors[0]], dim=1)
                attn0 = torch.cat(
                    [enc["attention_mask"], torch.ones_like(option_tensors[0], device=device)],
                    dim=1
                )
                _collect_flops_once(
                    task_name=subset,
                    model=model,
                    forward_kwargs={"input_ids": inp0, "attention_mask": attn0},
                    flops_info=flops_info,
                    sort_by="self_cuda_time_total",
                )

            # score options
            scores, pt, st = _sum_logprobs_for_suffix(
                model, tokenizer, device,
                enc["input_ids"],
                option_tensors,
                attention_mask_prefix=enc["attention_mask"]
            )
            proc_tok += pt; sup_tok += st

            # predictions
            pred = scores.argmax(dim=1).tolist()
            total += n
            for i in range(n):
                if 0 <= gold_idx[i] < scores.size(1):
                    correct += int(pred[i] == gold_idx[i])
                else:
                    # unknown/malformed gold → skip counting as correct
                    pass

        res_stats.total = total
        res_stats.correct = correct
        res_stats.processed_tokens = proc_tok
        res_stats.supervised_tokens = sup_tok
        return {"ok": True}

    _, E, Pavg, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy(subset, res_stats, E, Pavg, sec, flops_info)


# ---------- LAMBADA (robust batch handling + FLOPs) ----------
@torch.no_grad()
def eval_lambada(
    model, tokenizer, device, batch_size: int = 16, max_eval: int = 5000, max_len: int = 1024, collect_flops: bool = True
):
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

    # Helpers to handle Dataset slices OR dict-of-lists
    def _batch_len(batch) -> int:
        if hasattr(batch, "num_rows"):
            return int(batch.num_rows)
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, (list, tuple)):
                    return len(v)
            return 0
        return len(batch)

    def _col(batch, name, default, n: int):
        if hasattr(batch, "column_names") and name in batch.column_names:
            return batch[name]
        if isinstance(batch, dict) and name in batch:
            return batch[name]
        return default if not callable(default) else default(n)

    def _split_ctx_target(text: str) -> tuple[str, str]:
        s = (text or "").strip()
        parts = s.rsplit(" ", 1)
        if len(parts) == 1:
            return "", parts[0]
        return parts[0], parts[1]

    res_stats = RunStats()
    flops_info = {"total_flops_forward": None}

    def _run():
        total = 0
        correct = 0
        proc_tok = 0

        for bi in tqdm(range(0, len(data), batch_size), desc="LAMBADA", unit="batch"):
            batch = data[bi:bi + batch_size]
            n = _batch_len(batch)
            if n == 0:
                continue

            # Text column name can vary; prefer "text", fallback to "sentence"
            texts = _col(batch, "text", None, n)
            if texts is None:
                texts = _col(batch, "sentence", [""] * n, n)

            ctxs, targets = zip(*[_split_ctx_target(t) for t in texts])

            enc = tokenizer(list(ctxs), return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(device)
            proc_tok += _sequence_token_count(enc["input_ids"])

            # ---- FLOPs on first batch: plain forward over the context ----
            if collect_flops and bi == 0:
                _collect_flops_once(
                    task_name="LAMBADA",
                    model=model,
                    forward_kwargs={"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
                    flops_info=flops_info,
                    sort_by="self_cuda_time_total",
                )

            out = model(**enc, use_cache=False, return_dict=True)
            logits = out.logits[:, -1, :]
            pred_ids = logits.argmax(dim=-1)

            # Tokenize targets; take **first token** of the last word
            tgt_ids_list = tokenizer(list(targets), add_special_tokens=False).input_ids
            tgt_first_ids = [
                ids[0] if (isinstance(ids, list) and len(ids) > 0) else tokenizer.eos_token_id
                for ids in tgt_ids_list
            ]
            tgt_first = torch.tensor(tgt_first_ids, device=device)

            total += n
            correct += (pred_ids == tgt_first).sum().item()

        res_stats.total = total
        res_stats.correct = correct
        res_stats.processed_tokens = proc_tok
        return {"ok": True}

    _, E, Pavg, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)
    _report_energy("LAMBADA", res_stats, E, Pavg, sec, flops_info)
