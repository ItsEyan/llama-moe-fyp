import argparse, torch

import argparse, os, torch
from transformers.trainer_utils import get_last_checkpoint

def resolve_path(p: str) -> str:
    # If user passes a directory, try to resolve to latest checkpoint file
    if os.path.isdir(p):
        last = get_last_checkpoint(p)
        if last is not None:
            cand = os.path.join(last, "router_lora.pt")
            if os.path.exists(cand):
                return cand
        # fallback: maybe callback saved directly in output dir
        cand = os.path.join(p, "router_lora.pt")
        if os.path.exists(cand):
            return cand
        raise SystemExit(f"Could not find router_lora.pt under {p} (or its latest checkpoint).")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True, help="file path or output_dir")
    args = ap.parse_args()

    a_path = resolve_path(args.a)
    b_path = resolve_path(args.b)

    A = torch.load(a_path, map_location="cpu")
    B = torch.load(b_path, map_location="cpu")

    # allow either raw state_dict or {"state_dict":...}
    if isinstance(A, dict) and "state_dict" in A: A = A["state_dict"]
    if isinstance(B, dict) and "state_dict" in B: B = B["state_dict"]

    keys = sorted(set(A.keys()) & set(B.keys()))
    if not keys:
        raise SystemExit("No overlapping keys between checkpoints.")

    total_l2 = 0.0
    max_l2 = 0.0
    max_k = None

    for k in keys:
        da = A[k].float()
        db = B[k].float()
        d = (db - da).reshape(-1)
        l2 = torch.norm(d, p=2).item()
        total_l2 += l2
        if l2 > max_l2:
            max_l2 = l2
            max_k = k

    print(f"Num tensors compared: {len(keys)}")
    print(f"Sum L2 over tensors: {total_l2:.6f}")
    print(f"Max tensor L2: {max_l2:.6f} @ {max_k}")
    
    flat = []
    for k in keys:
        d = (B[k].float() - A[k].float()).reshape(-1)
        flat.append(d)
    flat = torch.cat(flat)
    print(f"Global max|Δ|: {flat.abs().max().item():.6e}")
    print(f"Global RMS(Δ): {flat.pow(2).mean().sqrt().item():.6e}")

    if total_l2 < 1e-6:
        print("WARNING: LoRA weights did not change (or changes are extremely tiny).")
    else:
        print("OK: LoRA weights changed.")

if __name__ == "__main__":
    main()
