import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import (
    eval_boolq,
    eval_piqa,
    eval_hellaswag,
    eval_arc,
    eval_lambada,
    run_single_eval_with_stats,
)

def load_model_and_tokenizer(model_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # causal LM padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    return model, tokenizer


def print_model_info(model):
    cfg = model.config
    print("=" * 60)
    print("Model:", getattr(cfg, "_name_or_path", "unknown"))
    print("Model type:", getattr(cfg, "model_type", "unknown"))

    # Only MoE models will have these
    if hasattr(cfg, "gate_type"):
        print("Gate Type:", cfg.gate_type)
        print("Num Experts:", getattr(cfg, "num_experts", "N/A"))
        print("Num Selects:", getattr(cfg, "num_selects", "N/A"))

        if getattr(cfg, "gate_type", "") == "DynamicTopGate":
            print("k_min:", getattr(cfg, "k_min", "N/A"))
            print("k_max:", getattr(cfg, "k_max", "N/A"))
            print("k_band:", getattr(cfg, "k_band", "N/A"))
            print("select_strategy:", getattr(cfg, "select_strategy", "N/A"))
            print("p_min:", getattr(cfg, "p_min", "N/A"))
            print("logit_temperature:", getattr(cfg, "logit_temperature", "N/A"))
    else:
        print("Dense model detected (no MoE gate config).")
    print("=" * 60)

def print_results(results):
    for ds_name, payload in results.items():
        print(f"\n{'='*20} {ds_name.upper()} {'='*20}")

        router = payload.get("router_stats", {})
        metrics = payload.get("metrics", {})
        energy = metrics.get("energy", {}) if isinstance(metrics, dict) else {}

        print("\n[Router Stats]")
        print(f"mean_k                 : {router.get('mean_k', float('nan')):.4f}")
        print(f"mean_entropy_full      : {router.get('mean_entropy_full', float('nan')):.4f}")
        print(f"mean_entropy_selected  : {router.get('mean_entropy_selected', float('nan')):.4f}")
        print(f"total_tokens           : {router.get('total_tokens', 0)}")
        print(f"total_gate_calls       : {router.get('total_gate_calls', 0)}")

        k_hist = router.get("k_hist", {})
        if k_hist:
            print("k_hist                 : " + ", ".join([f"k={k}:{v}" for k, v in k_hist.items()]))

        print("\n[Task Metrics]")
        if isinstance(metrics, dict):
            acc = metrics.get("accuracy", None)
            if acc is not None:
                print(f"accuracy               : {acc*100:.2f}%")

            run_stats = metrics.get("run_stats", {})
            if run_stats:
                print(f"examples               : {run_stats.get('total', 0)}")
                print(f"processed_tokens       : {run_stats.get('processed_tokens', 0)}")
                print(f"supervised_tokens      : {run_stats.get('supervised_tokens', 0)}")
                print(f"generated_tokens       : {run_stats.get('generated_tokens', 0)}")

        if energy:
            print("\n[Energy]")
            print(f"time_s                 : {energy.get('time_s', float('nan')):.3f}")
            print(f"avg_gpu_power_W        : {energy.get('avg_gpu_power_W', float('nan')):.2f}")
            print(f"energy_J               : {energy.get('energy_J', float('nan')):.3f}")
            print(f"energy_Wh              : {energy.get('energy_Wh', float('nan')):.6f}")
            print(f"energy/example_J       : {energy.get('energy_per_example_J', float('nan')):.3f}")
            print(f"J/token(proc)          : {energy.get('J_per_token_proc', float('nan')):.6f}")
            print(f"J/token(sup)           : {energy.get('J_per_token_sup', float('nan')):.6f}")
            print(f"Wh/1k tok(proc)        : {energy.get('Wh_per_1k_proc', float('nan')):.6f}")
            print(f"Wh/1k tok(sup)         : {energy.get('Wh_per_1k_sup', float('nan')):.6f}")    


def evaluate_per_dataset(model, tokenizer, device, datasets, collect_flops=False):
    results = {}

    if "boolq" in datasets:
        results["boolq"] = run_single_eval_with_stats(
            model,
            lambda: eval_boolq(
                model, tokenizer, device,
                max_eval=1024, batch_size=8, collect_flops=collect_flops
            )
        )

    if "piqa" in datasets:
        results["piqa"] = run_single_eval_with_stats(
            model,
            lambda: eval_piqa(
                model, tokenizer, device,
                max_eval=2000, batch_size=32, collect_flops=collect_flops
            )
        )

    if "hellaswag" in datasets:
        results["hellaswag"] = run_single_eval_with_stats(
            model,
            lambda: eval_hellaswag(
                model, tokenizer, device,
                max_eval=2000, batch_size=8, collect_flops=collect_flops
            )
        )

    if "arc" in datasets:
        results["arc"] = run_single_eval_with_stats(
            model,
            lambda: eval_arc(
                model, tokenizer, device,
                subset="ARC-Challenge", max_eval=1000, batch_size=8, collect_flops=collect_flops
            )
        )

    if "lambada" in datasets:
        results["lambada"] = run_single_eval_with_stats(
            model,
            lambda: eval_lambada(
                model, tokenizer, device,
                max_eval=5000, batch_size=32, collect_flops=collect_flops
            )
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="llama-moe/LLaMA-MoE-v1-3_5B-4_16",
        help="Path or HF model name, e.g. llama-moe/... or meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["boolq", "piqa", "hellaswag", "arc", "lambada"]
    )
    parser.add_argument("--flops", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    print_model_info(model)
    results = evaluate_per_dataset(model, tokenizer, args.device, args.datasets, collect_flops=args.flops)
    print_results(results)