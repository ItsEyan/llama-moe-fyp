# python>=3.10
import os, time, threading, sys
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pynvml import (nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetPowerUsage)
from torch.profiler import profile, ProfilerActivity

from benchmarks import (
    eval_boolq,
    eval_piqa,
    eval_hellaswag,
    eval_arc,
    eval_lambada,
)

########################################
# Energy integration (unchanged logic)
########################################
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

########################################
# Model & tokenizer
########################################
device = "cuda:0"
model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-4_16"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# for causal LMs, ensure a pad token (use EOS if missing)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # better for generation with batching

model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

print("Gate Type:", model.config.gate_type)
print("Num Experts:", model.config.num_experts)
print("Num Selects:", model.config.num_selects)
if getattr(model.config, "gate_type", "") == "DynamicTopGate":
    print("k_min:", model.config.k_min)
    print("k_max:", model.config.k_max)
    print("k_band:", model.config.k_band)
    print("select_strategy:", model.config.select_strategy)
    print("p_min:", model.config.p_min)
    print("logit_temperature:", model.config.logit_temperature)
    
def evaluate(datasets):
    collect_flops = False
    if "boolq" in datasets:
        eval_boolq(
            model, tokenizer, device, integrate_gpu_energy_joules,
            max_eval=1024, batch_size=8, collect_flops=collect_flops
        )

    if "piqa" in datasets:
        # 2️⃣ PIQA (2-choice physical commonsense)
        eval_piqa(
            model, tokenizer, device, integrate_gpu_energy_joules,
            max_eval=2000, batch_size=32, collect_flops=collect_flops
        )

    if "hellaswag" in datasets:
        # 3️⃣ HellaSwag (4-choice adversarial completion)
        eval_hellaswag(
            model, tokenizer, device, integrate_gpu_energy_joules,
            max_eval=2000, batch_size=8, collect_flops=collect_flops
        )

    if "arc" in datasets:
        # 4️⃣ ARC-Challenge (multi-choice science)
        eval_arc(
            model, tokenizer, device, integrate_gpu_energy_joules,
            subset="ARC-Challenge", max_eval=1000, batch_size=8, collect_flops=collect_flops
        )

    if "lambada" in datasets:
        # 5️⃣ LAMBADA (long-range next-word prediction)
        eval_lambada(
            model, tokenizer, device, integrate_gpu_energy_joules,
            max_eval=5000, batch_size=32, collect_flops=collect_flops
        )

########################################
# Run
########################################
if __name__ == "__main__":
    eval_datasets = [
        # "boolq",
        # "piqa",
        # "hellaswag",
        "arc",
        # "lambada",
    ]
    evaluate(eval_datasets)