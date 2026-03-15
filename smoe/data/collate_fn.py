from __future__ import annotations
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

@dataclass
class FaultTolerantMultipleChoiceCollator:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Expects each feature to have:
          input_ids: List[List[int]]      # num_choices x seq_len_i
          attention_mask: List[List[int]]
          labels: List[List[int]]
          correct_idx: int
        """
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        max_len = 0
        for feat in features:
            for choice_ids in feat["input_ids"]:
                max_len = max(max_len, len(choice_ids))

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_correct_idx = []

        for feat in features:
            ex_input_ids = []
            ex_attention_mask = []
            ex_labels = []

            for ids, mask, labels in zip(
                feat["input_ids"],
                feat["attention_mask"],
                feat["labels"],
            ):
                pad_len = max_len - len(ids)

                ex_input_ids.append(ids + [self.pad_token_id] * pad_len)
                ex_attention_mask.append(mask + [0] * pad_len)
                ex_labels.append(labels + [self.label_pad_token_id] * pad_len)

            batch_input_ids.append(ex_input_ids)
            batch_attention_mask.append(ex_attention_mask)
            batch_labels.append(ex_labels)
            batch_correct_idx.append(feat["correct_idx"])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),           # [B, C, L]
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long), # [B, C, L]
            "labels": torch.tensor(batch_labels, dtype=torch.long),                 # [B, C, L]
            "correct_idx": torch.tensor(batch_correct_idx, dtype=torch.long),       # [B]
        }

@dataclass
class FaultTolerantCausalLMCollator:
    pad_token_id: int
    label_pad_id: int = -100
    padding_position: str = "right"  # "left" if you really want

    def __call__(self, features: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        # unwrap objects
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]

        # Convert to tensors (1D) first
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]

        # attention_mask optional; if missing, create it
        if "attention_mask" in features[0] and features[0]["attention_mask"] is not None:
            attention_mask = [
                torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
            ]
        else:
            attention_mask = [torch.ones_like(x) for x in input_ids]

        # labels optional; if missing, default to input_ids (standard causal LM)
        if "labels" in features[0] and features[0]["labels"] is not None:
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        else:
            labels = [x.clone() for x in input_ids]

        # Pad
        if self.padding_position == "right":
            input_ids = rnn_utils.pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            )
            attention_mask = rnn_utils.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
            labels = rnn_utils.pad_sequence(
                labels, batch_first=True, padding_value=self.label_pad_id
            )
        elif self.padding_position == "left":
            # left pad by reversing then padding then reversing back
            input_ids = [torch.flip(x, dims=[0]) for x in input_ids]
            attention_mask = [torch.flip(x, dims=[0]) for x in attention_mask]
            labels = [torch.flip(x, dims=[0]) for x in labels]

            input_ids = torch.flip(
                rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
                dims=[1],
            )
            attention_mask = torch.flip(
                rnn_utils.pad_sequence(attention_mask, batch_first=True, padding_value=0),
                dims=[1],
            )
            labels = torch.flip(
                rnn_utils.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_id),
                dims=[1],
            )
        else:
            raise ValueError("padding_position must be 'left' or 'right'")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def fault_tolerance_data_collator(features: list) -> dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


def identity_collator(examples):  # 不对数据进行处理
    return examples


def tensor_cat_collator(examples):  # 拼接tensor
    return torch.cat(examples, dim=0)


class tensor_cat_padding_collater:  # 拼接tensor，并padding到最大长度
    def __init__(self, padding_id, padding_position="right", return_padding_mask=True):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

    def __call__(self, examples):
        if self.padding_position == "right":
            padded_examples = rnn_utils.pad_sequence(
                examples, batch_first=True, padding_value=self.padding_id
            )
        elif (
            self.padding_position == "left"
        ):  # This will take about twice the time compared to right padding
            flipped_examples = [torch.flip(tensor, dims=[0]) for tensor in examples]
            padded_examples_flip = rnn_utils.pad_sequence(
                flipped_examples, batch_first=True, padding_value=self.padding_id
            )
            padded_examples = torch.flip(padded_examples_flip, dims=[1])
        else:
            raise NotImplementedError

        if self.return_padding_mask:
            padding_mask = padded_examples != self.padding_id
            return padded_examples, padding_mask
        else:
            return padded_examples


def tensor_list_cat_collator(examples):  # 拼接list中对应位置的tensor，返回list
    return [
        torch.cat([tensor[i] for tensor in examples], dim=0)
        for i in range(len(examples[0]))
    ]


class tensor_list_cat_padding_collater:  # 拼接list中对应位置的tensor，并padding到最大长度，返回list
    def __init__(self, padding_id, padding_position="right", return_padding_mask=True):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

    def __call__(self, examples):
        num_tensors = len(examples[0])
        padded_tensors = []
        padding_masks = []

        for i in range(num_tensors):
            tensor_list = [example[i] for example in examples]

            if self.padding_position == "right":
                padded_tensor = rnn_utils.pad_sequence(
                    tensor_list, batch_first=True, padding_value=self.padding_id
                )
            elif (
                self.padding_position == "left"
            ):  # This will take about twice the time compared to right padding
                flipped_tensors = [
                    torch.flip(tensor, dims=[0]) for tensor in tensor_list
                ]
                padded_tensors_flip = rnn_utils.pad_sequence(
                    flipped_tensors, batch_first=True, padding_value=self.padding_id
                )
                padded_tensor = torch.flip(padded_tensors_flip, dims=[1])
            else:
                raise NotImplementedError

            padded_tensors.append(padded_tensor)
            if self.return_padding_mask:
                padding_masks.append(padded_tensors[i] != self.padding_id)

        if self.return_padding_mask:
            return padded_tensors, padding_masks
        else:
            return padded_tensors


def tensor_dict_cat_collator(examples):  # 拼接dict中对应位置的tensor，返回dict
    return {
        key: torch.cat([example[key] for example in examples], dim=0)
        for key in examples[0].keys()
    }
