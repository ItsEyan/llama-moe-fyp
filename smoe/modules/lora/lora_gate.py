import math
import torch
from torch import nn
import torch.nn.functional as F
import os
from typing import Dict, Any, Optional, Callable

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        assert base.bias is None  # your gates use bias=False

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = F.linear(x, self.weight)
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * delta


def _replace_linears(module: nn.Module, make):
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.bias is None:
            setattr(module, name, make(child))
            replaced += 1
        else:
            replaced += _replace_linears(child, make)
    return replaced


def inject_lora_into_moe_gates(
    model: nn.Module,
    gate_class_names=("TopKBalancedNoisyGate", "DynamicTopGate", "SwitchBalancedGate", "UniformLearnableGate"),
    r=8,
    alpha=16,
    dropout=0.0,
    only_gate_network=True,
):
    total_replaced = 0
    total_gates = 0

    def make(base):
        return LoRALinear(base, r=r, alpha=alpha, dropout=dropout)

    for m in model.modules():
        if m.__class__.__name__ in gate_class_names:
            total_gates += 1
            scope = getattr(m, "gate_network", None) if only_gate_network else m
            if scope is None:
                continue
            total_replaced += _replace_linears(scope, make)

    return {"gates_found": total_gates, "linears_replaced": total_replaced}


def freeze_all_but_router_lora(model: nn.Module, train_gate_noise: bool = False):
    for p in model.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
        if train_gate_noise and "weight_noise" in n:
            p.requires_grad = True

def _unwrap_model(model):
    # DDP / FSDP wrappers often expose `.module`
    return model.module if hasattr(model, "module") else model


def save_router_lora(
    model,
    save_dir: str,
    filename: str = "router_lora.pt",
    *,
    only_trainable: bool = True,
    name_filter: Optional[Callable[[str], bool]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save router LoRA weights to disk.

    By default, saves ONLY parameters with requires_grad=True.
    This is the cleanest approach when you've called freeze_all_but_router_lora().

    DeepSpeed ZeRO-3 note:
      - If parameters are partitioned, we gather them on rank0 via GatheredParameters.

    Returns:
        The path to the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    model_to_save = _unwrap_model(model)

    # Try to support DeepSpeed ZeRO-3 parameter gathering if available
    gathered_ctx = None
    try:
        from deepspeed.zero import GatheredParameters  # type: ignore

        def _gather_param(p):
            # `ds_id` exists for ZeRO params; if not, treat as normal param
            if hasattr(p, "ds_id"):
                return GatheredParameters([p], modifier_rank=0)
            return None

        gathered_ctx = _gather_param
    except Exception:
        gathered_ctx = None

    state: Dict[str, torch.Tensor] = {}
    skipped = 0

    for name, p in model_to_save.named_parameters():
        if only_trainable and not p.requires_grad:
            skipped += 1
            continue
        if name_filter is not None and not name_filter(name):
            skipped += 1
            continue

        # Gather ZeRO-3 shards if needed
        ctx = gathered_ctx(p) if gathered_ctx is not None else None
        if ctx is not None:
            with ctx:
                # After gathering, param is materialized on rank0
                # (only safe to save when args.should_save / rank0)
                state[name] = p.detach().cpu().clone()
        else:
            state[name] = p.detach().cpu().clone()

    payload: Dict[str, Any] = {
        "format_version": 1,
        "state_dict": state,
        "meta": {
            "only_trainable": only_trainable,
            "num_tensors": len(state),
            "skipped": skipped,
        },
    }
    if extra_meta:
        payload["meta"].update(extra_meta)

    torch.save(payload, path)
    return path