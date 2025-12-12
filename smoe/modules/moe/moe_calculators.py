from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.utils import ModelOutput

from smoe.modules.moe.moe_experts import LinearGLUExperts
from smoe.modules.norm import WeightNorm
from smoe.utils.debugging import remote_breakpoint


@dataclass
class CalculatorOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None


class BaseCalculator(nn.Module):
    def __init__(self):
        super(BaseCalculator, self).__init__()

    def reset_experts(self):
        self.experts.reset_parameters()


class UniformCalculator(BaseCalculator):
    # efficient calculator for all-select-all gates

    def __init__(self, experts, multiply_gate_scores=True, score_scale_factor=1.0):
        super(UniformCalculator, self).__init__()
        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts

    def forward(self, x, topK_scores, **kwargs) -> CalculatorOutput:
        # num_selects * (bsz*seq_len, hidden_size)
        expert_outputs = [self.experts(x, i) for i in range(self.num_experts)]

        # (num_selects, bsz*seq_len, hidden_size)
        stack_expert_outputs = torch.stack(expert_outputs, 0)  # 拼接专家输出
        if self.multiply_gate_scores:
            expanded_socre = (
                topK_scores.transpose(0, 1)
                .unsqueeze(2)
                .expand(stack_expert_outputs.shape)
            )
            stack_expert_outputs = stack_expert_outputs * (
                expanded_socre * self.score_scale_factor
            )
        y = torch.sum(stack_expert_outputs, dim=0)

        return CalculatorOutput(hidden_states=y, num_dropped_tokens=torch.tensor(-1.0))


class UniversalCalculator(BaseCalculator):
    # traditional calculation mode, forward $num_experts$ times with re-batch optimization
    """
    https://github.com/YeonwooSung/Pytorch_mixture-of-experts
    接收topK scores的DisPatcher，相比原版的SparseDispatcher进行了计算上的优化
    原理依旧是重新分配各个专家的batch。
    """

    def __init__(
        self,
        experts: LinearGLUExperts,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        add_weight_norm: bool = False,
    ):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        # TODO (zhutong): use vmap to boost the training efficiency
        # self.experts_vmap = torch.vmap(self.experts)
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts
        self.mlp_norm = None
        if multiply_gate_scores and add_weight_norm:
            # self.mlp_norm = LlamaRMSNorm(self.experts.out_features, eps=1e-5)
            # self.mlp_norm = WeightNorm(self.experts.out_features, scale=score_scale_factor)
            self.mlp_norm = WeightNorm(1, scale=score_scale_factor)
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        x,
        topK_indices,
        topK_scores,
        expert_batch_size=None,
        topK_mask=None,
        **kwargs,
    ) -> CalculatorOutput:
        # fmt: off
        """
        Forward pass for top-k / dynamic-k MoE.

        Args:
            x:             (num_tokens, hidden_size)
            topK_indices:  (num_tokens, Kmax) expert indices per token-slot
            topK_scores:   (num_tokens, Kmax) gate scores per token-slot
            topK_mask:     (num_tokens, Kmax) mask in {0,1} or bool; 1 = active slot.
                            - If None: assume all slots active (backwards-compatible).
        """
        device = x.device
        batch_size, num_selects = topK_indices.shape  # num_tokens, Kmax

        # ---- Build flat lists of (expert_idx, token_idx, score) ----
        if topK_mask is not None:
            # Use only masked (active) slots
            # topK_mask can be float or bool; treat >0 as True.
            active = topK_mask > 0
            # (N_active,)
            flat_expert_idx = topK_indices[active].reshape(-1)
            flat_scores     = topK_scores[active].reshape(-1)

            # Build a (batch_size, num_selects) grid of token indices and mask it
            token_grid = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_selects)
            flat_token_idx = token_grid[active].reshape(-1)
        else:
            # Original behavior: every slot is active
            flat_expert_idx = topK_indices.reshape(-1)   # (batch_size * num_selects,)
            flat_scores     = topK_scores.reshape(-1)    # (batch_size * num_selects,)
            flat_token_idx  = torch.arange(batch_size, device=device).repeat_interleave(num_selects)

        # Sanity: there should always be at least one active slot per token
        # given k_min >= 1 in the gate; so flat_expert_idx should be non-empty.
        # ---- Sort by expert index so that each expert sees a contiguous chunk ----
        _, index_sorted = flat_expert_idx.sort(0)
        sorted_expert_idx  = flat_expert_idx.index_select(0, index_sorted)   # (N_active,)
        sorted_scores      = flat_scores.index_select(0, index_sorted)       # (N_active,)
        sorted_token_idx   = flat_token_idx.index_select(0, index_sorted)    # (N_active,)

        # ---- Compute per-expert batch sizes ----
        if expert_batch_size is None:
            # bincount over expert index; length = num_experts
            expert_batch_size = sorted_expert_idx.bincount(minlength=self.num_experts).tolist()

        # ---- Rebatch inputs by expert ----
        # Gather tokens in sorted (by expert) order
        sorted_x = x.index_select(0, sorted_token_idx)           # (N_active, hidden_size)
        # Split them into per-expert chunks according to expert_batch_size
        split_x = torch.split(sorted_x, expert_batch_size, dim=0)

        # ---- Run each expert on its chunk ----
        expert_outputs = []
        for i in range(self.num_experts):
            chunk = split_x[i]
            if chunk.shape[0] == 0:
                continue
            out_i = self.experts(chunk, i)
            expert_outputs.append(out_i)

        # Concatenate outputs back in the same order as sorted_x
        cat_expert_outputs = torch.cat(expert_outputs, dim=0)    # (N_active, hidden_dim)
        output_dim = cat_expert_outputs.size(1)

        # ---- Apply gate scores ----
        if self.multiply_gate_scores:
            if self.mlp_norm is None:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs,
                    sorted_scores.reshape(-1, 1) * self.score_scale_factor,
                )
            else:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs,
                    sorted_scores.reshape(-1, 1),
                )
                cat_expert_outputs = self.mlp_norm(cat_expert_outputs)

        # ---- Scatter-add back to token positions ----
        zeros = torch.zeros(
            (batch_size, output_dim),
            device=cat_expert_outputs.device,
            dtype=cat_expert_outputs.dtype,
        )
        y = zeros.index_add(0, sorted_token_idx, cat_expert_outputs)

        return CalculatorOutput(
            hidden_states=y,
            num_dropped_tokens=torch.tensor(-1.0, device=device),
        )
        # fmt: on



class SwitchDropTokenCalculator(BaseCalculator):
    """
    https://arxiv.org/pdf/2101.03961.pdf
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/__init__.py
    https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py
    带有capacity_factor的计算器，自动丢弃超出容量的token
    """

    def __init__(
        self,
        experts,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
        drop_tokens=True,
        dropped_padding="zero",  # zero input
        capacity_factor=1.25,
        add_weight_norm: bool = False,
    ):
        super(SwitchDropTokenCalculator, self).__init__()
        self.available_dropped_padding_choices = ("zero", "input")
        assert dropped_padding in self.available_dropped_padding_choices
        # 如果丢弃token，则必须保证输入输出维度相同
        if drop_tokens and dropped_padding != "zero":
            assert experts.in_features == experts.out_features

        self.experts = experts
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts
        self.out_features = experts.out_features
        self.mlp_norm = None
        if multiply_gate_scores and add_weight_norm:
            self.mlp_norm = WeightNorm(
                self.experts.out_features, scale=score_scale_factor
            )
            self.mlp_norm.reset_parameters()

        # capacity
        self.drop_tokens = drop_tokens
        self.dropped_padding = dropped_padding
        self.capacity_factor = capacity_factor

    def forward(self, x, topK_indices, topK_scores, **kwargs) -> CalculatorOutput:
        """
        Args:
            x: (bsz*seq_len, hidden_size) bsz*seq_len is the total number of tokens in this batch
            topK_indices: (bsz*seq_len,) each element represents the expert idx to consume the token
                e.g. topK_indices[1] = 3 means the token-1 is assigned to expert-3
        """
        batch_size = topK_indices.size(0)
        capacity = int(self.capacity_factor * batch_size / self.num_experts)
        dropped_indices = []
        y = torch.zeros((batch_size, self.out_features), device=x.device, dtype=x.dtype)

        # 各专家分别正向传播，此处应该有并行优化的空间 (如果单次forward不足以占满显卡利用率)
        num_dropped_tokens = -1
        for i in range(self.num_experts):
            # token_indices is a tensor of (num_tokens_in_this_expert,)
            #   where each element denotes the token position idx
            token_indices = (topK_indices == i).nonzero(as_tuple=True)[0]
            num_assigned_tokens = token_indices.numel()
            # Ignore if the expert is not over capacity
            if self.drop_tokens and num_assigned_tokens > capacity:
                shuffled_indices = torch.randperm(num_assigned_tokens, device=x.device)
                # Shuffle indexes before dropping
                token_indices = token_indices[shuffled_indices]
                # Collect the tokens over capacity as dropped tokens
                dropped_indices.append(token_indices[capacity:])
                # Keep only the tokens upto the capacity of the expert
                token_indices = token_indices[:capacity]
                num_dropped_tokens = num_assigned_tokens - capacity

            if num_assigned_tokens > 0:
                expert_output = self.experts(x[token_indices, :], i)
                y[token_indices, :] = expert_output

        if self.dropped_padding == "input" and len(dropped_indices) > 0:
            dropped_indices = torch.cat(dropped_indices, dim=0)
            y[dropped_indices, :] = x[dropped_indices, :]

        if self.multiply_gate_scores:
            # 乘权重
            y = torch.mul(y, topK_scores.reshape(-1, 1) * self.score_scale_factor)
            if self.mlp_norm is not None:
                y = self.mlp_norm(y)

        return CalculatorOutput(
            hidden_states=y, num_dropped_tokens=torch.tensor(num_dropped_tokens)
        )
