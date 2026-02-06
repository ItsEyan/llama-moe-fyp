import logging
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional

from datasets import IterableDataset, load_dataset
from datasets.combine import interleave_datasets
from tqdm import tqdm

from smoe.data.aggregation import group_texts

logger = logging.getLogger(__name__)


def load_streaming_datasets(
    data_dir: str,
    tokenizer,
    prob_map: Optional[dict[str, float]] = None,
    text_field: str = "text",
    num_proc: int = None,
    debug_mode: bool = False,
    block_size: int = 1024,
    split: str = "train",
    verbose: bool = True,
) -> IterableDataset:
    """
    Stream JSONL files under `data_dir`, tokenize (expects `text_field` in JSON),
    group into fixed-length blocks (`input_ids`/`attention_mask`), and interleave
    datasets from different subfolders by `prob_map`.

    Expected directory layout:
      data_dir/
        en_cc/*.jsonl
        en_c4/*.jsonl
        ...

    Expected JSONL example:
      {"text": "..."}   (or change `text_field`)
    """
    dataset_dir = Path(data_dir)
    files = list(dataset_dir.glob("**/*.jsonl"))
    if len(files) == 0:
        raise ValueError(f"No .jsonl files found under: {data_dir}")

    if debug_mode:
        files = [files[0]]

    # --- collect filepaths by "data_type" (parent folder name) ---
    fbar = tqdm(files, desc="Loading files") if verbose else files
    data_type_to_filepaths = defaultdict(list)
    for filepath in fbar:
        data_type = filepath.parent.stem  # e.g. "en_cc"
        if prob_map is not None and data_type not in prob_map:
            raise ValueError(
                f"{data_type} not in prob_map keys: {list(prob_map.keys())}"
            )
        data_type_to_filepaths[data_type].append(str(filepath))

    # --- tokenize then group into blocks ---
    def _tokenize(examples):
        if text_field not in examples:
            # This error message is intentionally explicit to speed up debugging.
            raise KeyError(
                f"Missing field '{text_field}' in dataset example. "
                f"Available fields: {list(examples.keys())}"
            )
        # tokenizer returns dict with input_ids/attention_mask (and maybe others)
        return tokenizer(
            examples[text_field],
            add_special_tokens=True,
            truncation=False,  # grouping happens later
        )

    tokenize_fn = _tokenize
    grouping_fn = partial(group_texts, block_size=block_size)

    data_type_to_dataset = {}
    if verbose:
        pbar = tqdm(total=len(data_type_to_filepaths), desc="Indexing files")
    else:
        pbar = None

    for data_type, filepaths in data_type_to_filepaths.items():
        ds = load_dataset(
            "json",
            data_files=filepaths,
            streaming=True,
            split=split,
        )

        # 1) tokenize raw text -> input_ids/attention_mask
        ds = ds.map(tokenize_fn, batched=True)

        # 2) group tokens into block_size -> batched chunks for LM training
        ds = ds.map(grouping_fn, batched=True)

        data_type_to_dataset[data_type] = ds
        if pbar:
            pbar.update(1)

    # --- build interleaving list + probs ---
    datasets_in_diff_types = []
    probs = []

    dbar = tqdm(total=len(data_type_to_dataset), desc="Mapping datasets with probs") if verbose else None
    for data_type, ds in data_type_to_dataset.items():
        datasets_in_diff_types.append(ds)
        if prob_map is not None:
            probs.append(prob_map[data_type])
            if dbar:
                dbar.update(1)
                dbar.set_postfix({data_type: f"{prob_map[data_type]:.3%}%"})
        else:
            if dbar:
                dbar.update(1)
                dbar.set_postfix({data_type: "uniform"})

    if prob_map is None:
        probs = None
    else:
        s = sum(probs)
        if s <= 0:
            raise ValueError(f"Sum of prob_map must be > 0, got {s}")
        if abs(s - 1.0) > 1e-6:
            logger.warning(f"Summation of prob_map is {s}, scaling to 1.0")
            probs = [p / s for p in probs]

    if len(datasets_in_diff_types) == 0:
        raise ValueError("Unable to interleave an empty list of datasets.")

    if verbose:
        logger.info("Interleaving datasets")
    lm_datasets = interleave_datasets(datasets_in_diff_types, probs)

    # attach metadata for custom trainer logging
    try:
        lm_datasets.prob_map = prob_map
        lm_datasets.data_dir = str(dataset_dir)
        lm_datasets.text_field = text_field
        lm_datasets.block_size = block_size
    except Exception:
        pass


    return lm_datasets
