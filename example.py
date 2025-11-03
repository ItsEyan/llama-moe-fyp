# python>=3.10
import os, time, threading, sys
from tqdm import tqdm
from smoe.modules.moe.moe_gates import global_hist
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pynvml import (nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetPowerUsage)

########################################
# Energy integration (unchanged logic)
########################################
def integrate_gpu_energy_joules(fn, poll_ms=10, device_index=0):
    """
    Jupyter-friendly version: run `fn()` on the MAIN thread (so prints/logs show),
    and sample GPU power in a background daemon thread.
    """
    from pynvml import (nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
                        nvmlDeviceGetPowerUsage)
    import threading, time

    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(device_index)
        samples = []
        stop = threading.Event()

        def sampler():
            # sample until told to stop
            while not stop.is_set():
                try:
                    samples.append((time.perf_counter(), nvmlDeviceGetPowerUsage(handle)))
                except Exception:
                    pass
                time.sleep(poll_ms / 1000.0)

        t0 = time.perf_counter()
        th = threading.Thread(target=sampler, daemon=True)
        th.start()

        # RUN THE MODEL / EVAL ON MAIN THREAD → prints visible in Jupyter
        res = fn()

        stop.set()
        th.join(timeout=1.0)

        t1 = time.perf_counter()
        try:
            samples.append((t1, nvmlDeviceGetPowerUsage(handle)))
        except Exception:
            pass

        # trapezoidal integration
        energy_J = 0.0
        for (t_prev, p_prev), (t_cur, p_cur) in zip(samples[:-1], samples[1:]):
            dt = t_cur - t_prev
            energy_J += 0.5 * ((p_prev + p_cur) / 1000.0) * dt  # mW→W

        duration_s = t1 - t0
        avg_power_W = (energy_J / duration_s) if duration_s > 0 else float("nan")
        return res, energy_J, avg_power_W, duration_s
    finally:
        nvmlShutdown()


########################################
# Model & tokenizer
########################################
device = "cuda:0"
model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# for causal LMs, ensure a pad token (use EOS if missing)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # better for generation with batching

model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

########################################
# BoolQ dataset & prompt
########################################
# Each example: {"question": str, "passage": str, "answer": bool}
ds = load_dataset("google/boolq")
# Use validation set for evaluation
val_ds = ds["validation"]

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
    # take first token-ish piece
    t = t.split()[0] if t else ""
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    # fallback heuristic: search
    if "yes" in text.lower():
        return "yes"
    if "no" in text.lower():
        return "no"
    return "unknown"

########################################
# Evaluation loop (batched)
########################################
@torch.no_grad()
def eval_boolq(max_eval: int = 1024, batch_size: int = 8, max_new_tokens: int = 3):
    # Slice the validation set
    if max_eval == -1 or max_eval >= len(val_ds):
        subset = val_ds
    else:
        subset = val_ds.select(range(min(max_eval, len(val_ds))))

    total = 0
    correct = 0
    total_new_tokens = 0

    def run_all():
        nonlocal total, correct, total_new_tokens

        # tqdm progress bar
        pbar = tqdm(range(0, len(subset), batch_size), desc="Evaluating", unit="batch")

        for i in pbar:
            batch = subset[i : i + batch_size]
            prompts = make_batch_prompts(batch)

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)

            in_len = enc["input_ids"].shape[1]

            torch.cuda.synchronize()
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            torch.cuda.synchronize()

            new_tok_batch = out_ids.shape[1] - in_len
            total_new_tokens += new_tok_batch * out_ids.shape[0]

            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            # compare with labels
            for j, full_text in enumerate(decoded):
                gold_bool = bool(batch["answer"][j])
                gen_suffix = full_text[-max(0, max_new_tokens * 8):]
                pred = parse_yes_no(gen_suffix)
                gold = "yes" if gold_bool else "no"
                total += 1
                correct += int(pred == gold)

            # update progress bar info
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(acc=f"{acc*100:.2f}%", tokens=total_new_tokens)

        pbar.close()
        return {"total": total, "correct": correct, "total_new_tokens": total_new_tokens}

    # measure energy for the whole eval
    res, energy_J, avg_W, sec = integrate_gpu_energy_joules(run_all, poll_ms=10, device_index=0)

    acc = res["correct"] / max(1, res["total"])
    new_tok = res["total_new_tokens"]
    tokens_per_s = new_tok / sec if sec > 0 else float("nan")
    J_per_token = energy_J / max(1, new_tok)
    Wh = energy_J / 3600.0
    Wh_per_1k_tokens = (Wh / max(1, new_tok)) * 1000.0

    print(f"BoolQ eval — N={res['total']} | Acc: {acc*100:.2f}%")
    print(f"Time: {sec:.3f}s | Avg GPU Power: {avg_W:.1f} W | Energy: {energy_J:.1f} J ({Wh:.4f} Wh)")
    print(f"New tokens: {new_tok} | tokens/s: {tokens_per_s:.1f} | J/token: {J_per_token:.2f} | Wh/1k tok: {Wh_per_1k_tokens:.3f}")

    # convert to sorted dict for readability
    sorted_hist = dict(sorted(global_hist.items()))
    total = sum(sorted_hist.values())
    print("=== Accumulated Expert Activation Histogram ===")
    for k, count in sorted_hist.items():
        print(f"k={k:>2d}: {count:>8d}  ({count/total*100:.2f}%)")


########################################
# Run
########################################
if __name__ == "__main__":
    # quick smoke test on 64 examples; bump to 3-5k for a fuller run
    eval_boolq(max_eval=5000, batch_size=8, max_new_tokens=3)
