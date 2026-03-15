"""
Compare BASE vs LoRA on:
- mean router entropy
- mean-k (avg selected experts per token)
- (optional) accuracy/metric returned by your eval_* functions

This works WITHOUT changing your gate code if:
- your DynamicTopGate forward returns a dict containing "k_vec"
Entropy options:
(A) If gate forward returns "probs_full": we compute true entropy over all experts.
(B) Otherwise we approximate entropy using selected experts only ("topK_scores") renormalized.

Usage:
  python eval_router_stats.py \
    --model llama-moe/LLaMA-MoE-v1-3_5B-4_16 \
    --adapter /path/to/router_lora  \
    --dataset lambada

If you don't pass --adapter, it runs base only.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List

from sympy import comp, pretty_print
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from smoe.modules.lora.lora_gate import inject_lora_into_moe_gates

from benchmarks import (
    eval_boolq,
    eval_piqa,
    eval_hellaswag,
    eval_arc,
    eval_lambada,
)

# If your adapter is PEFT, this will work. If not, we fall back to a simple state_dict load.
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# ----------------------------
# Gate stats collection
# ----------------------------
@dataclass
class RouterStats:
    # weighted means across ALL gate calls
    mean_k: float = float("nan")
    mean_entropy_full: float = float("nan")      # if probs_full available
    mean_entropy_selected: float = float("nan")  # always available if topK_scores exists
    total_tokens: int = 0
    total_gate_calls: int = 0
    k_hist: Dict[str, int] = None  # {"1":123, "2":456, ...}

class GateStatsCollector:
    def __init__(self, model: nn.Module, gate_class_name: str = "DynamicTopGate"):
        self.model = model
        self.gate_class_name = gate_class_name
        self.hooks: List[Any] = []
        self.reset()

    def reset(self):
        self._sum_k = 0.0
        self._sum_ent_full = 0.0
        self._sum_ent_sel = 0.0
        self._tok_full = 0
        self._tok_sel = 0
        self._total_tokens = 0
        self._gate_calls = 0
        self._k_hist: Dict[int, int] = {}

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        # p: [B, E] probability distribution
        p = p.clamp_min(1e-12)
        return -(p * p.log()).sum(dim=-1)  # [B]

    def _hook_fn(self, module: nn.Module, inp, out):
        # out from DynamicTopGate forward is expected to be a dict.
        if not isinstance(out, dict):
            return

        k_vec = out.get("k_vec", None)  # [B]
        if k_vec is not None:
            # k_vec is per-token for that gate call
            B = int(k_vec.numel())
            self._sum_k += float(k_vec.float().sum().item())
            self._total_tokens += B

            # histogram
            # (do on CPU cheaply)
            kv = k_vec.detach().to(torch.int64).cpu()
            binc = torch.bincount(kv, minlength=128)  # enough headroom
            for k, c in enumerate(binc.tolist()):
                if c:
                    self._k_hist[k] = self._k_hist.get(k, 0) + int(c)

        # Full entropy if probs_full exists
        p_full = out.get("probs_full", None)  # [B, E]
        if p_full is not None:
            H = self._entropy(p_full.float())  # [B]
            self._sum_ent_full += float(H.sum().item())
            self._tok_full += int(H.numel())

        # Selected-only entropy (approx) if topK_scores exists
        # topK_scores might already be probs masked; we renormalize within selected experts.
        top_scores = out.get("topK_scores", None)  # [B, K]
        top_mask = out.get("topK_mask", None)      # [B, K]
        if top_scores is not None and top_mask is not None:
            s = top_scores.float() * top_mask.float()
            z = s.sum(dim=1, keepdim=True).clamp_min(1e-12)
            p_sel = s / z
            Hs = self._entropy(p_sel)  # [B]
            self._sum_ent_sel += float(Hs.sum().item())
            self._tok_sel += int(Hs.numel())

        self._gate_calls += 1

    def attach(self):
        self.detach()  # prevent double-hooking
        for m in self.model.modules():
            if m.__class__.__name__ == self.gate_class_name:
                self.hooks.append(m.register_forward_hook(self._hook_fn))
        # print(f"[GateStatsCollector] hooked {len(self.hooks)} {self.gate_class_name} modules")

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


# ----------------------------
# Adapter loading
# ----------------------------
def load_router_adapter(model: nn.Module, adapter_path: str, *, r=8, alpha=16, dropout=0.0) -> nn.Module:
    if adapter_path is None:
        return model

    # If user passes a directory, try find router_lora.pt inside
    ckpt_file = adapter_path
    if os.path.isdir(adapter_path):
        p = os.path.join(adapter_path, "router_lora.pt")
        if os.path.exists(p):
            ckpt_file = p

    payload = torch.load(ckpt_file, map_location="cpu")

    # Your save_router_lora saves a payload dict with "state_dict"
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
    else:
        # fallback: older raw dict
        sd = payload

    # 1) Inject LoRA modules so keys exist
    inject_lora_into_moe_gates(
        model,
        r=r,
        alpha=alpha,
        dropout=dropout,
        gate_class_names=("DynamicTopGate", "TopKBalancedNoisyGate", "SwitchBalancedGate", "UniformLearnableGate"),
        only_gate_network=True,
    )

    # 2) Load LoRA weights
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Adapter] loaded from {ckpt_file}")
    print(f"[Adapter] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")

    # Optional sanity: ensure we actually loaded LoRA params
    loaded_lora = [k for k in sd.keys() if ("lora_A" in k or "lora_B" in k)]
    print(f"[Adapter] lora tensors in ckpt: {len(loaded_lora)}")
    return model


# ----------------------------
# Evaluation wrapper
# ----------------------------
def run_single_eval_with_stats(
    model: nn.Module,
    eval_fn,
) -> Dict[str, Any]:
    collector = GateStatsCollector(model)
    collector.attach()
    collector.reset()

    with torch.no_grad():
        maybe_metrics = eval_fn()

    stats = collector.summarize()
    collector.detach()

    return {
        "router_stats": asdict(stats),
        "metrics": maybe_metrics if maybe_metrics is not None else None,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base model dir or HF id")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir or .pt file (optional)")
    parser.add_argument("--dataset", default="lambada", help="comma-separated: boolq,piqa,hellaswag,arc,lambada")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="If set, evaluate both BASE and LoRA models. "
            "Default: only LoRA (if adapter provided)."
    )
    parser.add_argument("--flops", action="store_true")
    args = parser.parse_args()
    collect_flops = args.flops

    device = args.device
    model_dir = args.model
    datasets_list = [x.strip() for x in args.dataset.split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    def evaluate_per_dataset(model_ref, datasets, collect_flops=False):
        results = {}

        if "boolq" in datasets:
            results["boolq"] = run_single_eval_with_stats(
                model_ref,
                lambda: eval_boolq(
                    model_ref, tokenizer, device,
                    max_eval=1024, batch_size=8, collect_flops=collect_flops
                )
            )

        if "piqa" in datasets:
            results["piqa"] = run_single_eval_with_stats(
                model_ref,
                lambda: eval_piqa(
                    model_ref, tokenizer, device,
                    max_eval=2000, batch_size=32, collect_flops=collect_flops
                )
            )

        if "hellaswag" in datasets:
            results["hellaswag"] = run_single_eval_with_stats(
                model_ref,
                lambda: eval_hellaswag(
                    model_ref, tokenizer, device,
                    max_eval=2000, batch_size=8, collect_flops=collect_flops
                )
            )

        if "arc" in datasets:
            results["arc"] = run_single_eval_with_stats(
                model_ref,
                lambda: eval_arc(
                    model_ref, tokenizer, device,
                    subset="ARC-Challenge", max_eval=1000, batch_size=8, collect_flops=collect_flops
                )
            )

        if "lambada" in datasets:
            results["lambada"] = run_single_eval_with_stats(
                model_ref,
                lambda: eval_lambada(
                    model_ref, tokenizer, device,
                    max_eval=5000, batch_size=32, collect_flops=collect_flops
                )
            )

        return results

    def pretty_print(label: str, payload: Dict[str, Any]):
        print(f"\n===== {label} =====")
        for ds_name, ds_payload in payload.items():
            print(f"\n--- {ds_name} ---")
            rs = ds_payload["router_stats"]
            print(json.dumps(rs, indent=2))
            if ds_payload["metrics"] is not None:
                print("\nmetrics:")
                print(json.dumps(ds_payload["metrics"], indent=2))

    # ============================================================
    # Evaluation logic
    # ============================================================

    if args.compare_base:
        # --------------------------------------------------------
        # BASE run
        # --------------------------------------------------------
        print("\nRunning BASE model...")
        base_model_ref = base_model
        base_out = evaluate_per_dataset(base_model_ref, datasets_list, collect_flops=collect_flops)
        pretty_print("BASE", base_out)

        # --------------------------------------------------------
        # LoRA run (requires adapter)
        # --------------------------------------------------------
        if args.adapter is None:
            raise ValueError("--compare_base requires --adapter to be provided.")

        print("\nRunning LORA model...")
        lora_model = load_router_adapter(
            AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(device).eval(),
            args.adapter
        )

        base_model_ref = lora_model
        lora_out = evaluate_per_dataset(base_model_ref, datasets_list, collect_flops=collect_flops    )
        pretty_print("LORA", lora_out)

        # --------------------------------------------------------
        # Comparison summary
        # --------------------------------------------------------

        print("\n===== COMPARISON =====")
        comp = {}

        for ds in datasets_list:
            if ds not in base_out or ds not in lora_out:
                continue

            b = base_out[ds]["router_stats"]
            l = lora_out[ds]["router_stats"]

            def _get(x, k):
                v = x.get(k, None)
                return float(v) if (v is not None and v == v) else None

            comp[ds] = {
                "mean_k": {
                    "base": _get(b, "mean_k"),
                    "lora": _get(l, "mean_k"),
                },
                "mean_entropy_full": {
                    "base": _get(b, "mean_entropy_full"),
                    "lora": _get(l, "mean_entropy_full"),
                },
                "mean_entropy_selected": {
                    "base": _get(b, "mean_entropy_selected"),
                    "lora": _get(l, "mean_entropy_selected"),
                },
            }

        print(json.dumps(comp, indent=2))

    else:
        # --------------------------------------------------------
        # Default behaviour: LoRA only
        # --------------------------------------------------------
        if args.adapter is None:
            raise ValueError("Default mode expects --adapter to be provided.")

        print("\nRunning LORA model only...")
        lora_model = load_router_adapter(base_model, args.adapter)

        base_model_ref = lora_model
        lora_out = evaluate_per_dataset(base_model_ref, datasets_list, collect_flops=collect_flops)
        pretty_print("LORA", lora_out)

if __name__ == "__main__":
    main()
