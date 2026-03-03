import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import (
    eval_boolq,
    eval_piqa,
    eval_hellaswag,
    eval_arc,
    eval_lambada,
)

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
            model, tokenizer, device,
            max_eval=1024, batch_size=8, collect_flops=collect_flops
        )

    if "piqa" in datasets:
        # 2️⃣ PIQA (2-choice physical commonsense)
        eval_piqa(
            model, tokenizer, device,
            max_eval=2000, batch_size=32, collect_flops=collect_flops
        )

    if "hellaswag" in datasets:
        # 3️⃣ HellaSwag (4-choice adversarial completion)
        eval_hellaswag(
            model, tokenizer, device,
            max_eval=2000, batch_size=8, collect_flops=collect_flops
        )

    if "arc" in datasets:
        # 4️⃣ ARC-Challenge (multi-choice science)
        eval_arc(
            model, tokenizer, device,
            subset="ARC-Challenge", max_eval=1000, batch_size=8, collect_flops=collect_flops
        )

    if "lambada" in datasets:
        # 5️⃣ LAMBADA (long-range next-word prediction)
        eval_lambada(
            model, tokenizer, device,
            max_eval=5000, batch_size=32, collect_flops=collect_flops
        )

########################################
# Run
########################################
if __name__ == "__main__":
    eval_datasets = [
        "boolq",
        "piqa",
        "hellaswag",
        "arc",
        "lambada",
    ]
    evaluate(eval_datasets)