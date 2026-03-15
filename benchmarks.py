from tkinter.filedialog import test
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import time
import threading
from datasets import load_dataset
from dataclasses import asdict
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

@dataclass
class RouterStats:
    mean_k: float = float("nan")
    mean_entropy_full: float = float("nan")
    mean_entropy_selected: float = float("nan")
    total_tokens: int = 0
    total_gate_calls: int = 0
    k_hist: Dict[str, int] = None

class GateStatsCollector:
    def __init__(self, model: nn.Module, gate_class_name: str = "DynamicTopGate"):
        self.model = model
        self.gate_class_name = gate_class_name
        self.hooks = []
        self.reset()

    def reset(self):
        self._sum_k = 0.0
        self._sum_ent_full = 0.0
        self._sum_ent_sel = 0.0
        self._tok_full = 0
        self._tok_sel = 0
        self._total_tokens = 0
        self._gate_calls = 0
        self._k_hist = {}

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(1e-12)
        return -(p * p.log()).sum(dim=-1)

    def _hook_fn(self, module: nn.Module, inp, out):
        if not isinstance(out, dict):
            return

        k_vec = out.get("k_vec", None)
        if k_vec is not None:
            B = int(k_vec.numel())
            self._sum_k += float(k_vec.float().sum().item())
            self._total_tokens += B

            kv = k_vec.detach().to(torch.int64).cpu()
            binc = torch.bincount(kv, minlength=128)
            for k, c in enumerate(binc.tolist()):
                if c:
                    self._k_hist[k] = self._k_hist.get(k, 0) + int(c)

        p_full = out.get("probs_full", None)
        if p_full is not None:
            H = self._entropy(p_full.float())
            self._sum_ent_full += float(H.sum().item())
            self._tok_full += int(H.numel())

        top_scores = out.get("topK_scores", None)
        top_mask = out.get("topK_mask", None)
        if top_scores is not None and top_mask is not None:
            s = top_scores.float() * top_mask.float()
            z = s.sum(dim=1, keepdim=True).clamp_min(1e-12)
            p_sel = s / z
            Hs = self._entropy(p_sel)
            self._sum_ent_sel += float(Hs.sum().item())
            self._tok_sel += int(Hs.numel())

        self._gate_calls += 1

    def attach(self):
        self.detach()
        for m in self.model.modules():
            if m.__class__.__name__ == self.gate_class_name:
                self.hooks.append(m.register_forward_hook(self._hook_fn))

    def detach(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def summarize(self) -> RouterStats:
        stats = RouterStats()
        stats.total_tokens = int(self._total_tokens)
        stats.total_gate_calls = int(self._gate_calls)

        if self._total_tokens > 0:
            stats.mean_k = float(self._sum_k / self._total_tokens)

        if self._tok_full > 0:
            stats.mean_entropy_full = float(self._sum_ent_full / self._tok_full)

        if self._tok_sel > 0:
            stats.mean_entropy_selected = float(self._sum_ent_sel / self._tok_sel)

        stats.k_hist = {str(k): int(v) for k, v in sorted(self._k_hist.items()) if v > 0}
        return stats
    
def run_single_eval_with_stats(model, eval_fn):
    collector = GateStatsCollector(model)
    collector.attach()
    collector.reset()

    with torch.no_grad():
        metrics = eval_fn()

    router_stats = asdict(collector.summarize())
    collector.detach()

    return {
        "router_stats": router_stats,
        "metrics": metrics,
    }

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

    if flops_info and flops_info.get("total_flops_forward") is not None:
        fi = flops_info
        print(
            f"[FLOPs] Forward (profiled batches): {fi['total_flops_forward']/1e12:.3f} TFLOPs "
            f"(~{fi['flops_per_token_forward']:.0f} FLOPs/token over {fi['profiled_batch_tokens']} tokens)"
        )
        approx_total_flops_eval = fi["flops_per_token_forward"] * max(1, res.processed_tokens)
        print(
            f"[FLOPs] Approx energy per 10¹² FLOPs (using new/supervised tokens): "
            f"{energy_J / max(1, approx_total_flops_eval / 1e12):.3f} J / TFLOP"
        )

    return {
        "task_name": task_name,
        "num_examples": res.total,
        "accuracy": acc,
        "time_s": sec,
        "avg_gpu_power_W": avg_W,
        "energy_J": energy_J,
        "energy_Wh": Wh,
        "energy_per_example_J": J_per_example,
        "processed_tokens": res.processed_tokens,
        "supervised_tokens": res.supervised_tokens,
        "generated_tokens": res.generated_tokens,
        "proc_tokens_per_s": tokps_proc,
        "gen_tokens_per_s": tokps_gen,
        "J_per_token_proc": J_per_token_proc,
        "J_per_token_sup": J_per_token_sup,
        "Wh_per_1k_proc": Wh_per_1k_proc,
        "Wh_per_1k_sup": Wh_per_1k_sup,
        "flops_info": flops_info if flops_info else None,
    }



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
def _pad_batch_texts(tokenizer, device, texts: List[str], max_len: int = 1024, add_special_tokens: bool = True):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=add_special_tokens,
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
    
def collect_flops_over_batches(
    task_name: str,
    model,
    batches: list[dict],
    sort_by: str = "self_cuda_time_total",
    row_limit: int = 40,
):
    model.eval()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    total_flops_all = 0
    total_tokens_all = 0
    per_batch_flops_per_token = []
    with_flops_ok = True

    # warmup
    with torch.inference_mode():
        for warmup_batch in batches[:2]:
            _ = model(**warmup_batch, use_cache=False, return_dict=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    for i, forward_kwargs in enumerate(batches):
        with torch.inference_mode():
            try:
                with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
                    _ = model(**forward_kwargs, use_cache=False, return_dict=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            except TypeError:
                with_flops_ok = False
                with profile(activities=activities, record_shapes=True) as prof:
                    _ = model(**forward_kwargs, use_cache=False, return_dict=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

        # if i == 0:
        #     print(f"\n=== PyTorch Profiler sample — {task_name} ===")
        #     try:
        #         print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))
        #     except Exception:
        #         print(prof.key_averages().table(row_limit=row_limit))

        batch_flops = 0
        if with_flops_ok:
            for e in prof.key_averages():
                batch_flops += (getattr(e, "flops", 0) or 0)

        input_ids = forward_kwargs.get("input_ids")
        batch_tokens = int(input_ids.numel()) if torch.is_tensor(input_ids) else 0

        if batch_flops and batch_tokens:
            total_flops_all += batch_flops
            total_tokens_all += batch_tokens
            per_batch_flops_per_token.append(batch_flops / batch_tokens)

    return {
        "total_flops_forward": total_flops_all or None,
        "profiled_batch_tokens": total_tokens_all or None,
        "flops_per_token_forward": (
            total_flops_all / total_tokens_all if total_flops_all and total_tokens_all else None
        ),
        "flops_per_token_batch_mean": (
            sum(per_batch_flops_per_token) / len(per_batch_flops_per_token)
            if per_batch_flops_per_token else None
        ),
        "num_profiled_batches": len(per_batch_flops_per_token),
    }

# =========================================================
# Evaluations (each takes: model, tokenizer, device, energy_fn)
# energy_fn must be integrate_gpu_energy_joules(callable)
# =========================================================

# ---------- BoolQ ----------
def profile_boolq_flops(model, tokenizer, device, subset, batch_size=8, num_profile_batches=5):
    model.eval()

    flops_batches = []

    old_trunc_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"

    PROMPT_TMPL = (
        "Passage: {passage}\n"
        "Question: {question}\n"
        "Answer: "
    )

    def make_batch_prefixes(batch):
        return [
            PROMPT_TMPL.format(passage=p, question=q)
            for p, q in zip(batch["passage"], batch["question"])
        ]

    try:
        for i in range(0, min(len(subset), batch_size * num_profile_batches), batch_size):
            batch = subset[i:i+batch_size]
            prefixes = make_batch_prefixes(batch)

            enc = tokenizer(
                prefixes,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=True,
            ).to(device)

            flops_batches.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            })

        return collect_flops_over_batches(
            task_name="BoolQ (prompt only)",
            model=model,
            batches=flops_batches,
            sort_by="self_cuda_time_total",
        )
    finally:
        tokenizer.truncation_side = old_trunc_side

@torch.no_grad()
def eval_boolq(
    model,
    tokenizer,
    device,
    max_eval: int = 1024,
    batch_size: int = 8,
    collect_flops: bool = True,
):
    """
    BoolQ evaluation via next-token logits:
      compare logit("Yes") vs logit("No") after the prompt.

    This avoids suffix concatenation issues.
    Assumes "Yes" and "No" are single tokens for the tokenizer.
    """

    ds = load_dataset("google/boolq")
    val_ds = ds["validation"]
    subset = val_ds if (max_eval == -1 or max_eval >= len(val_ds)) else val_ds.select(
        range(min(max_eval, len(val_ds)))
    )

    # Important for long BoolQ passages:
    # keep the END of the prompt (question + answer cue) if truncation happens
    old_trunc_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"

    # We already checked these are single-token for your tokenizer
    yes_ids = tokenizer("Yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer("No", add_special_tokens=False)["input_ids"]

    assert len(yes_ids) == 1, f"'Yes' is not single-token: {yes_ids}"
    assert len(no_ids) == 1, f"'No' is not single-token: {no_ids}"

    yes_id = yes_ids[0]
    no_id = no_ids[0]

    flops_info = {}

    # Optional FLOPs profiling on prompt-only forward pass
    if collect_flops:
        print("\n=== Profiling FLOPs (separate pass, no energy integration) ===")
        flops_info = profile_boolq_flops(
            model=model,
            tokenizer=tokenizer,
            device=device,
            subset=subset,
            batch_size=batch_size,
            num_profile_batches=5,
        )

    res_stats = RunStats()

    def _run():
        total = 0
        correct = 0
        processed_tokens = 0
        supervised_tokens = 0
        num_yes = 0
        num_no = 0

        pbar = tqdm(range(0, len(subset), batch_size), desc="BoolQ", unit="batch")
        for i in pbar:
            batch = subset[i:i+batch_size]

            prefixes = [
                f"Passage: {p}\nQuestion: {q}\nAnswer: "
                for p, q in zip(batch["passage"], batch["question"])
            ]

            enc = tokenizer(
                prefixes,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=True,
            ).to(device)

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                use_cache=False,
                return_dict=True,
            )

            # Since we use LEFT padding, the final real token is at index -1
            next_logits = out.logits[:, -1, :]   # [B, vocab]

            yes_scores = next_logits[:, yes_id]
            no_scores = next_logits[:, no_id]

            pred_is_yes = (yes_scores > no_scores).tolist()
            gold_is_yes = [bool(a) for a in batch["answer"]]

            processed_tokens += int(enc["attention_mask"].sum().item())
            supervised_tokens += len(gold_is_yes)   # 1 supervised decision per example

            for py, gy in zip(pred_is_yes, gold_is_yes):
                pred = "yes" if py else "no"
                gold = "yes" if gy else "no"
                correct += int(pred == gold)

                if py:
                    num_yes += 1
                else:
                    num_no += 1

            total += len(gold_is_yes)
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(acc=f"{acc*100:.2f}%")
            # print(f"Batch {i//batch_size+1}: Acc={acc*100:.2f}% | Yes: {num_yes} | No: {num_no}")

        res_stats.total = total
        res_stats.correct = correct
        res_stats.processed_tokens = processed_tokens
        res_stats.supervised_tokens = supervised_tokens
        res_stats.generated_tokens = 0

        return {"ok": True}

    try:
        _, energy_J, avg_W, sec = integrate_gpu_energy_joules(
            _run, poll_ms=10, device_index=0
        )
    finally:
        tokenizer.truncation_side = old_trunc_side

    energy_metrics = _report_energy("BoolQ", res_stats, energy_J, avg_W, sec, flops_info)

    return {
        "accuracy": res_stats.correct / max(1, res_stats.total),
        "run_stats": {
            "total": res_stats.total,
            "correct": res_stats.correct,
            "processed_tokens": res_stats.processed_tokens,
            "supervised_tokens": res_stats.supervised_tokens,
            "generated_tokens": res_stats.generated_tokens,
        },
        "energy": energy_metrics,
    }

# ---------- PIQA ----------

def profile_piqa_flops(model, tokenizer, device, val, batch_size=16, num_profile_batches=5):
    model.eval()
    flops_batches = []

    for i in range(0, min(len(val), batch_size * num_profile_batches), batch_size):
        batch = val[i:i+batch_size]
        goals = batch["goal"]
        sol1 = batch["sol1"]

        prefixes = [f"Goal: {g}\nSelect the best solution.\nSolution:" for g in goals]
        enc_prefix = _pad_batch_texts(tokenizer, device, prefixes)
        sfx1 = _pad_batch_texts(tokenizer, device, [" " + s for s in sol1])

        inp0 = torch.cat([enc_prefix["input_ids"], sfx1["input_ids"]], dim=1)
        attn0 = torch.cat(
            [enc_prefix["attention_mask"], torch.ones_like(sfx1["input_ids"], device=device)],
            dim=1
        )

        flops_batches.append({
            "input_ids": inp0,
            "attention_mask": attn0,
        })

    return collect_flops_over_batches(
        task_name="PIQA",
        model=model,
        batches=flops_batches,
        sort_by="self_cuda_time_total",
    )

@torch.no_grad()
def eval_piqa(model, tokenizer, device, batch_size: int = 16, max_eval: int = -1, collect_flops: bool = True):
    ds = load_dataset("piqa")
    val = ds["validation"]
    if max_eval != -1:
        val = val.select(range(min(max_eval, len(val))))

    res_stats = RunStats()
    flops_info = {}

    if collect_flops:
        print("\n=== Profiling FLOPs (PIQA) ===")
        flops_info = profile_piqa_flops(
            model=model,
            tokenizer=tokenizer,
            device=device,
            val=val,
            batch_size=batch_size,
            num_profile_batches=5,
        )
        
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

    _, energy_J, avg_W, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)

    energy_metrics = _report_energy("PIQA", res_stats, energy_J, avg_W, sec, flops_info)

    return {
        "accuracy": res_stats.correct / max(1, res_stats.total),
        "run_stats": {
            "total": res_stats.total,
            "correct": res_stats.correct,
            "processed_tokens": res_stats.processed_tokens,
            "supervised_tokens": res_stats.supervised_tokens,
            "generated_tokens": res_stats.generated_tokens,
        },
        "energy": energy_metrics,
    }


# ---------- HellaSwag ----------
def profile_hellaswag_flops(model, tokenizer, device, val, batch_size=8, num_profile_batches=5):
    model.eval()
    flops_batches = []

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

    max_items = min(len(val), batch_size * num_profile_batches)
    for i in range(0, max_items, batch_size):
        batch = val[i:i + batch_size]
        n = _batch_len(batch)
        if n == 0:
            continue

        ctxs = _col(batch, "ctx", None, n)
        if ctxs is None:
            ctxs = _col(batch, "context", None, n)
        if ctxs is None:
            a = _col(batch, "ctx_a", [""] * n, n)
            b = _col(batch, "ctx_b", [""] * n, n)
            ctxs = [(aa + " " + bb).strip() for aa, bb in zip(a, b)]

        ends = _col(batch, "endings", lambda _n: [["", "", "", ""] for _ in range(_n)], n)
        ends = [list(e) if not isinstance(e, list) else e for e in ends]
        endings_by_k = list(map(list, zip(*ends)))

        enc_ctx = _pad_batch_texts(tokenizer, device, ctxs)
        suffix0 = _pad_batch_texts(tokenizer, device, [" " + e for e in endings_by_k[0]])

        inp0 = torch.cat([enc_ctx["input_ids"], suffix0["input_ids"]], dim=1)
        attn0 = torch.cat(
            [enc_ctx["attention_mask"], torch.ones_like(suffix0["input_ids"], device=device)],
            dim=1
        )

        flops_batches.append({
            "input_ids": inp0,
            "attention_mask": attn0,
        })

    return collect_flops_over_batches(
        task_name="HellaSwag",
        model=model,
        batches=flops_batches,
        sort_by="self_cuda_time_total",
    )

@torch.no_grad()
def eval_hellaswag(
    model, tokenizer, device, batch_size: int = 8, max_eval: int = -1, collect_flops: bool = True
):
    ds = load_dataset("hellaswag")
    val = ds["validation"]
    if max_eval != -1:
        val = val.select(range(min(max_eval, len(val))))
        
    res_stats = RunStats()
    flops_info = {}

    if collect_flops:
        print("\n=== Profiling FLOPs (HellaSwag) ===")
        flops_info = profile_hellaswag_flops(
            model=model,
            tokenizer=tokenizer,
            device=device,
            val=val,
            batch_size=batch_size,
            num_profile_batches=5,
        )

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
    
    _, energy_J, avg_W, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)

    energy_metrics = _report_energy("HellaSwag", res_stats, energy_J, avg_W, sec, flops_info)

    return {
        "accuracy": res_stats.correct / max(1, res_stats.total),
        "run_stats": {
            "total": res_stats.total,
            "correct": res_stats.correct,
            "processed_tokens": res_stats.processed_tokens,
            "supervised_tokens": res_stats.supervised_tokens,
            "generated_tokens": res_stats.generated_tokens,
        },
        "energy": energy_metrics,
    }


# ---------- ARC (Easy/Challenge) — robust batch handling ----------
def profile_arc_flops(model, tokenizer, device, val, subset_name="ARC-Challenge", batch_size=8, num_profile_batches=5):
    model.eval()
    flops_batches = []

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

    max_items = min(len(val), batch_size * num_profile_batches)
    for bi in range(0, max_items, batch_size):
        batch = val[bi:bi+batch_size]
        n = _batch_len(batch)
        if n == 0:
            continue

        questions = _col(batch, "question", [""] * n, n)
        choices = _col(batch, "choices", None, n)

        row_texts, row_labels = [], []
        if isinstance(choices, dict):
            text_cols = choices.get("text", [[]] * n)
            label_cols = choices.get("label", [[]] * n)
            for i in range(n):
                row_texts.append(list(text_cols[i]))
                row_labels.append(list(label_cols[i]))
        elif isinstance(choices, list):
            for i in range(n):
                ch_i = choices[i] or {}
                row_texts.append(list(ch_i.get("text", [])))
                row_labels.append(list(ch_i.get("label", [])))
        else:
            row_texts = [[] for _ in range(n)]
            row_labels = [[] for _ in range(n)]

        prompts = []
        Kmax = 0
        for i in range(n):
            texts_i = row_texts[i]
            labels_i = row_labels[i]
            Kmax = max(Kmax, len(texts_i))
            opt_lines = [f"{labels_i[j]}. {texts_i[j]}" for j in range(len(texts_i))]
            prompt = "Question: " + questions[i] + "\n" + "\n".join(opt_lines) + "\nAnswer:"
            prompts.append(prompt)

        if Kmax == 0:
            continue

        enc = _pad_batch_texts(tokenizer, device, prompts)

        opts0 = []
        for i in range(n):
            if len(row_texts[i]) > 0:
                opts0.append(" " + row_texts[i][0])
            else:
                opts0.append(" [N/A]")
        option0 = _pad_batch_texts(tokenizer, device, opts0)["input_ids"]

        inp0 = torch.cat([enc["input_ids"], option0], dim=1)
        attn0 = torch.cat(
            [enc["attention_mask"], torch.ones_like(option0, device=device)],
            dim=1
        )

        flops_batches.append({
            "input_ids": inp0,
            "attention_mask": attn0,
        })

    return collect_flops_over_batches(
        task_name=subset_name,
        model=model,
        batches=flops_batches,
        sort_by="self_cuda_time_total",
    )
    
@torch.no_grad()
def eval_arc(
    model, tokenizer, device, subset: str = "ARC-Challenge", batch_size: int = 8, max_eval: int = -1, collect_flops: bool = True
):
    ds = load_dataset("ai2_arc", subset)
    val = ds["validation"]
    if max_eval != -1:
        val = val.select(range(min(max_eval, len(val))))
        
    res_stats = RunStats()
    flops_info = {}

    if collect_flops:
        print(f"\n=== Profiling FLOPs ({subset}) ===")
        flops_info = profile_arc_flops(
            model=model,
            tokenizer=tokenizer,
            device=device,
            val=val,
            subset_name=subset,
            batch_size=batch_size,
            num_profile_batches=5,
        )

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

    _, energy_J, avg_W, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)

    energy_metrics = _report_energy(subset, res_stats, energy_J, avg_W, sec, flops_info)

    return {
        "accuracy": res_stats.correct / max(1, res_stats.total),
        "run_stats": {
            "total": res_stats.total,
            "correct": res_stats.correct,
            "processed_tokens": res_stats.processed_tokens,
            "supervised_tokens": res_stats.supervised_tokens,
            "generated_tokens": res_stats.generated_tokens,
        },
        "energy": energy_metrics,
    }


# ---------- LAMBADA (robust batch handling) ----------
def profile_lambada_flops(model, tokenizer, device, data, batch_size=16, max_len=1024, num_profile_batches=5):
    model.eval()
    flops_batches = []

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

    max_items = min(len(data), batch_size * num_profile_batches)
    for bi in range(0, max_items, batch_size):
        batch = data[bi:bi + batch_size]
        n = _batch_len(batch)
        if n == 0:
            continue

        texts = _col(batch, "text", None, n)
        if texts is None:
            texts = _col(batch, "sentence", [""] * n, n)

        ctxs, _ = zip(*[_split_ctx_target(t) for t in texts])

        enc = tokenizer(
            list(ctxs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(device)

        flops_batches.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })

    return collect_flops_over_batches(
        task_name="LAMBADA",
        model=model,
        batches=flops_batches,
        sort_by="self_cuda_time_total",
    )
    
@torch.no_grad()
def eval_lambada(
    model, tokenizer, device, batch_size: int = 16, max_eval: int = 5000, max_len: int = 1024, collect_flops: bool = True
):
    ds = load_dataset("lambada")
    split = "test"

    data = ds[split]
    if max_eval != -1:
        data = data.select(range(min(max_eval, len(data))))
        
    res_stats = RunStats()
    flops_info = {}

    if collect_flops:
        print("\n=== Profiling FLOPs (LAMBADA) ===")
        flops_info = profile_lambada_flops(
            model=model,
            tokenizer=tokenizer,
            device=device,
            data=data,
            batch_size=batch_size,
            max_len=max_len,
            num_profile_batches=5,
        )

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

    _, energy_J, avg_W, sec = integrate_gpu_energy_joules(_run, poll_ms=10, device_index=0)

    energy_metrics = _report_energy("LAMBADA", res_stats, energy_J, avg_W, sec, flops_info)

    return {
        "accuracy": res_stats.correct / max(1, res_stats.total),
        "run_stats": {
            "total": res_stats.total,
            "correct": res_stats.correct,
            "processed_tokens": res_stats.processed_tokens,
            "supervised_tokens": res_stats.supervised_tokens,
            "generated_tokens": res_stats.generated_tokens,
        },
        "energy": energy_metrics,
    }
