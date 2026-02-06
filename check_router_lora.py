# scripts/check_router_lora.py
# Usage:
#   python scripts/check_router_lora.py \
#     --base_model llama-moe/LLaMA-MoE-v1-3_5B-4_16 \
#     --router_lora llama-moe-fyp/outputs/gate_lora_smoke/gate_lora_smoke-48548/checkpoint-50/router_lora.pt \
#     --device cuda \
#     --dtype float16
#
# It will:
#   1) run base model and collect per-layer expert routing (top-k indices)
#   2) load router_lora.pt into the gate LoRA modules
#   3) run again and compare routing + logits

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from smoe.models.llama_moe.modeling_llama_moe import LlamaMoEForCausalLM
from smoe.modules.flash_attn import replace_xformers


def parse_dtype(s: str):
    s = s.lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


@torch.no_grad()
def collect_routing(model: LlamaMoEForCausalLM, input_ids: torch.Tensor) -> Dict[str, Any]:
    """
    Returns routing stats:
      - per_layer_topk: list[L] of tensors [T, K] (expert indices per token, taking first batch)
      - per_layer_scores: list[L] of tensors [T, K] (optional, if available)
      - logits_last: tensor [V] (last token logits for first batch)
    """
    model.eval()

    # We want gate outputs; easiest is to monkeypatch the gate forward to stash last results.
    # Your gate class likely returns a dict with keys including:
    #   "topk_idx" or "indices", and "topk_weight" or "scores"
    # We'll try common keys robustly.
    stashes: List[Dict[str, torch.Tensor]] = []

    def wrap_gate_forward(gate_module):
        orig_forward = gate_module.forward

        def new_forward(x):
            out = orig_forward(x)
            # store only CPU tensors to avoid holding GPU memory
            stash = {}
            for k in ("topk_idx", "indices", "expert_indices", "topk_indices"):
                if isinstance(out, dict) and k in out:
                    stash["idx"] = out[k].detach().cpu()
                    break
            for k in ("topk_weight", "scores", "expert_scores", "topk_scores", "gates"):
                if isinstance(out, dict) and k in out:
                    stash["scores"] = out[k].detach().cpu()
                    break
            stashes.append(stash)
            return out

        gate_module.forward = new_forward
        return orig_forward

    # Patch all MoE gates in all layers
    originals = []
    for layer in model.model.layers:
        gate = layer.mlp.gate
        originals.append((gate, wrap_gate_forward(gate)))

    # Forward
    out = model(input_ids=input_ids)
    logits = out.logits  # [B, T, V]
    logits_last = logits[0, -1].detach().cpu()

    # Restore
    for gate, orig in originals:
        gate.forward = orig

    # Now decode stashes per layer. Expect one stash per layer forward call.
    # Each stash["idx"] is typically [B*T, K] because gate gets flattened tokens.
    per_layer_topk = []
    per_layer_scores = []
    B, T = input_ids.shape
    for li, stash in enumerate(stashes):
        if "idx" not in stash:
            per_layer_topk.append(None)
            per_layer_scores.append(None)
            continue
        idx = stash["idx"]
        # reshape: gate input is flattened (B*T, H) -> idx (B*T, K)
        if idx.dim() == 2 and idx.shape[0] == B * T:
            idx = idx.view(B, T, -1)[0]  # [T, K] for batch0
        elif idx.dim() == 3:
            idx = idx[0]
        per_layer_topk.append(idx)

        sc = stash.get("scores", None)
        if sc is None:
            per_layer_scores.append(None)
        else:
            if sc.dim() == 2 and sc.shape[0] == B * T:
                sc = sc.view(B, T, -1)[0]
            elif sc.dim() == 3:
                sc = sc[0]
            per_layer_scores.append(sc)

    return {
        "per_layer_topk": per_layer_topk,
        "per_layer_scores": per_layer_scores,
        "logits_last": logits_last,
    }


def load_router_lora_into_model(model: torch.nn.Module, router_lora_path: str, device: torch.device):
    """
    Assumes router_lora.pt is a plain torch state_dict saved by your save_router_lora().
    Loads it with strict=False so only matching LoRA params update.
    """
    sd = torch.load(router_lora_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def summarize_changes(
    base: Dict[str, Any], lora: Dict[str, Any], top_layers: int = 4
) -> Dict[str, Any]:
    per_layer_changes = []
    L = len(base["per_layer_topk"])
    for i in range(L):
        a = base["per_layer_topk"][i]
        b = lora["per_layer_topk"][i]
        if a is None or b is None:
            per_layer_changes.append({"layer": i, "changed_frac": None})
            continue
        # fraction of tokens whose top-1 expert changed
        changed = (a[:, 0] != b[:, 0]).float().mean().item()
        per_layer_changes.append({"layer": i, "changed_frac_top1": changed})

    # logit change for last token
    la = base["logits_last"].float()
    lb = lora["logits_last"].float()
    cos = torch.nn.functional.cosine_similarity(la, lb, dim=0).item()
    l2 = torch.norm(la - lb).item()

    # show top differing layers
    ranked = sorted(
        [x for x in per_layer_changes if x.get("changed_frac_top1") is not None],
        key=lambda x: x["changed_frac_top1"],
        reverse=True,
    )
    return {
        "logits_last_cosine": cos,
        "logits_last_l2": l2,
        "top_layers_by_routing_change": ranked[:top_layers],
        "all_layers": per_layer_changes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="HF model id or local path")
    ap.add_argument("--router_lora", required=True, help="Path to router_lora.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "Explain why the sky is blue in one paragraph.",
            "Write a Python function that computes fibonacci numbers iteratively.",
            "Singapore is a country in Southeast Asia. Summarize its economy in 5 bullet points.",
        ],
    )
    ap.add_argument("--max_new_tokens", type=int, default=1, help="Only used to pick last-token logits")
    ap.add_argument("--out_json", default=None, help="Optional path to save results as JSON")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    # Model
    model = LlamaMoEForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)
    replace_xformers(model)

    results = {"base_model": args.base_model, "router_lora": args.router_lora, "runs": []}

    for prompt in args.prompts:
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        # Base routing
        base_stats = collect_routing(model, input_ids)

        # Load router LoRA (in-place), then routing again
        missing, unexpected = load_router_lora_into_model(model, args.router_lora, device)
        lora_stats = collect_routing(model, input_ids)

        summary = summarize_changes(base_stats, lora_stats)

        results["runs"].append(
            {
                "prompt": prompt,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "summary": summary,
            }
        )

    # Print a readable summary
    for r in results["runs"]:
        print("=" * 80)
        print("PROMPT:", r["prompt"])
        print("logits_last_cosine:", r["summary"]["logits_last_cosine"])
        print("logits_last_l2:", r["summary"]["logits_last_l2"])
        print("Top layers by routing-change (top-1 expert):")
        for x in r["summary"]["top_layers_by_routing_change"]:
            print(f"  layer {x['layer']:>2}: changed_frac_top1={x['changed_frac_top1']:.3f}")
        # If nothing changes, that's important to know.
        maxchg = max(
            [x["changed_frac_top1"] for x in r["summary"]["all_layers"] if x.get("changed_frac_top1") is not None],
            default=0.0,
        )
        print("Max routing change across layers:", maxchg)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        # Convert tensors already to numbers; we only saved floats/ints.
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
