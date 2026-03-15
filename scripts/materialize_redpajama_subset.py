import os, json
from datasets import load_dataset
from tqdm import tqdm

# Choose ONE:
# HF_DATASET = "cerebras/SlimPajama-627B"
HF_DATASET = "MBZUAI-LLM/SlimPajama-627B-DC"  # already split-by-source metadata
SPLIT = "train"

OUT_ROOT = "data/redpajama"
MAX_DOCS_TOTAL = 1_000_000          # start small (fast). Increase later.
SHARD_SIZE = 10_000               # docs per jsonl shard

# map setname -> folder names that your prob_map expects
# (we normalize a few common variants)
CANON = {
    "commoncrawl": "en_cc",
    "cc": "en_cc",
    "c4": "en_c4",
    "github": "github",
    "wikipedia": "en_wikipedia",
    "book": "en_book",
    "books": "en_book",
    "arxiv": "en_arxiv",
    "stackexchange": "en_stack",
    "stack": "en_stack",
}

def get_setname(ex):
    # MBZUAI split often has redpajama_setname directly
    for k in ("redpajama_setname", "redpajama_set_name", "rp_setname"):
        if k in ex and ex[k]:
            return str(ex[k])

    # SlimPajama typically stores metadata inside "meta"
    meta = ex.get("meta", None)
    if isinstance(meta, dict):
        for k in ("redpajama_set_name", "redpajama_setname"):
            if k in meta and meta[k]:
                return str(meta[k])

    return "commoncrawl"  # fallback

def canon_folder(setname: str) -> str:
    s = setname.strip().lower()
    return CANON.get(s, s)  # if unknown, keep original name

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    ds = load_dataset(HF_DATASET, split=SPLIT, streaming=True)

    counters = {}   # folder -> count
    writers = {}    # folder -> (file, shard_idx)

    def open_shard(folder):
        ensure_dir(os.path.join(OUT_ROOT, folder))
        shard_idx = counters.get(folder, 0) // SHARD_SIZE
        path = os.path.join(OUT_ROOT, folder, f"{SPLIT}_{shard_idx:05d}.jsonl")
        f = open(path, "a", encoding="utf-8")
        writers[folder] = (f, shard_idx)

    total = 0
    for ex in tqdm(ds, total=MAX_DOCS_TOTAL):
        text = ex.get("text", None)
        if not text or not isinstance(text, str):
            continue

        folder = canon_folder(get_setname(ex))
        counters.setdefault(folder, 0)

        # rotate shard
        if folder not in writers:
            open_shard(folder)
        if counters[folder] % SHARD_SIZE == 0 and counters[folder] > 0:
            writers[folder][0].close()
            open_shard(folder)

        writers[folder][0].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        counters[folder] += 1
        total += 1
        if total >= MAX_DOCS_TOTAL:
            break

    for folder, (f, _) in writers.items():
        f.close()

    print("Done. Wrote:")
    for k in sorted(counters.keys()):
        print(f"  {k}: {counters[k]} docs -> {OUT_ROOT}/{k}/")

if __name__ == "__main__":
    main()
