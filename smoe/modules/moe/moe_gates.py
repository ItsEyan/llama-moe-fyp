import warnings, logging

import torch
from deepspeed.moe.sharded_moe import gumbel_rsample
from torch import nn
from torch.distributions.normal import Normal

logging.basicConfig(filename="topp_debug.txt", level=logging.INFO, filemode="w")

valid_gate_type = ("linear", "mlp")

from collections import Counter
global_hist = Counter()

global_k_hist = {}
global_expert_hist = {}

def get_gate_network(gate_type, input_size, num_experts):
    gate_type = gate_type.lower()

    if gate_type == "linear":
        gate_network = nn.Linear(input_size, num_experts, bias=False)
        nn.init.zeros_(gate_network.weight)
    elif gate_type == "mlp":
        gate_network = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_experts, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(num_experts, num_experts, bias=False),
        )
    else:
        raise ValueError(f'Expected "gate_type" in {valid_gate_type}, got {gate_type}.')

    return gate_network


class BaseGate(nn.Module):
    def __init__(self):
        super(BaseGate, self).__init__()

    def reset_gate_network(self):
        if "gate_network_type" not in vars(self):
            raise KeyError(f"{type(self)} does not have a gate network.")
        else:
            self.gate_network = get_gate_network(
                self.gate_network_type, self.input_size, self.num_experts
            )


class UniformPlainGate(BaseGate):
    """
    Select all experts with the same score.
    If use_softmax=True, then score=1/num_experts.
    If use_softmax=False, then score=1.
    """

    def __init__(
        self,
        input_size,
        num_experts,
        use_softmax=True,
    ):
        super(UniformPlainGate, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.use_softmax = use_softmax

    def forward(self, x):
        batch_size = x.shape[0]

        scores = torch.ones((batch_size, self.num_experts), device=x.device)
        if self.use_softmax:
            scores /= self.num_experts
        indices = (
            torch.arange(0, self.num_experts, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, self.num_experts)
        )

        return {
            "topK_indices": indices,
            "topK_scores": scores,
            "balance_loss": torch.tensor(0, device=x.device),
            "load": torch.tensor(-1, device=x.device),
            "importance": torch.tensor(-1, device=x.device),
        }


class UniformLearnableGate(BaseGate):
    """
    Select all experts with the same score, with a learnable gate_network controlling expert scores.
    """

    def __init__(
        self,
        input_size,
        num_experts,
        gate_network="mlp",
        use_softmax=True,
    ):
        super(UniformLearnableGate, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        batch_size = x.shape[0]

        logits = self.gate_network(x)  # gate计算出的权重
        scores = self.softmax(logits) if self.use_softmax else logits
        indices = (
            torch.arange(0, self.num_experts, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, self.num_experts)
        )

        return {
            "topK_indices": indices,
            "topK_scores": scores,
            "balance_loss": torch.tensor(0, device=x.device),
            "load": torch.tensor(-1, device=x.device),
            "importance": torch.tensor(-1, device=x.device),
        }


class RandomPlainGate(BaseGate):
    """
    Randomly select k experts each time.
    If use_softmax=True, then score=1/num_selects.
    If use_softmax=False, then score=1.
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        use_softmax=True,
    ):
        super(RandomPlainGate, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.use_softmax = use_softmax

    def forward(self, x):
        batch_size = x.shape[0]

        top_k_scores = torch.ones((batch_size, self.num_experts), device=x.device)
        if self.use_softmax:
            top_k_scores /= self.num_experts
        _, top_k_indices = torch.rand_like(top_k_scores).topk(self.num_selects, dim=1)

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": torch.tensor(0, device=x.device),
            "load": torch.tensor(-1, device=x.device),
            "importance": torch.tensor(-1, device=x.device),
        }


class RandomLearnableGate(BaseGate):
    """
    Randomly select k experts each time, with a learnable gate_network controlling expert scores.
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        add_noise=True,
        noise_epsilon=1e-2,
    ):
        super(RandomLearnableGate, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon

    def forward(self, x):
        logits = self.gate_network(x)  # gate计算出的权重
        gumbel_rsample(logits.shape, device=logits.device).to(
            logits
        ) * self.noise_epsilon

        _, top_k_indices = torch.rand_like(logits).topk(self.num_selects, dim=1)
        top_k_logits = torch.gather(logits, dim=1, index=top_k_indices)
        top_k_scores = self.softmax(top_k_logits) if self.use_softmax else top_k_logits

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": torch.tensor(0, device=x.device),
            "load": torch.tensor(-1, device=x.device),
            "importance": torch.tensor(-1, device=x.device),
        }


class TopKBalancedNoisyGate(BaseGate):
    """
    Select the top-k experts each time, with a learnable gate_network controlling expert scores.
    https://arxiv.org/abs/1701.06538.
    https://github.com/YeonwooSung/Pytorch_mixture-of-experts
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
        noise_epsilon=1e-2,
    ):
        super(TopKBalancedNoisyGate, self).__init__()
        assert num_selects <= num_experts  # 选择数量大于专家数量，报错
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)
        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight

        # add_noise
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.warned = False
        if self.add_noise:
            self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
            # self.weight_noise = nn.Parameter(torch.empty(input_size, num_experts))
            self.weight_noise.weight.data = torch.zeros(
                (num_experts, input_size),
                requires_grad=True,
                device=self.weight_noise.weight.data.device,
                dtype=self.weight_noise.weight.data.dtype,
            )
            self.mean = 0.0
            self.std = 1.0
            self.normal = Normal(self.mean, self.std)
            self.softplus = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self):
        if self.add_noise:
            nn.init.zeros_(self.weight_noise.weight)
            # nn.init.zeros_(self.weight_noise)

    def cv_squared(self, x, eps=1e-10):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.s
        """
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    # fmt: off
    def forward(self, x):
        """先计算所有专家的权重值"""
        logits_gate = self.gate_network(x)  # gate计算出的权重
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)  # 噪声矩阵计算结果
            noise_control = self.softplus(noise_mm) + self.noise_epsilon  # 控制器得到的噪声增加量
            logits_noise = torch.randn_like(logits_gate) * noise_control  # noise附加的权重
            logits = logits_gate + logits_noise  # 最终权重
        else:
            logits = logits_gate  # 最终权重，shape(batch_size, num_experts)

        """选出前k个权重，并计算各个专家的分数scores"""
        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)  # 选择并排序前k+1个权重
        top_k_logits = top_logits[:, :self.num_selects]
        top_k_indices = top_indices[:, :self.num_selects]
        top_k_scores = self.softmax(top_k_logits.to(torch.float32)) if self.use_softmax else top_k_logits
        top_k_scores = top_k_scores.to(logits.dtype)

        """计算importance"""
        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        """计算load"""
        # zhutong: 不要把`self.training`写在里面的if语句中，否则会导致eval模式下balance_loss输出值设备不匹配的错误
        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()
                threshold_positions_if_in = torch.arange(batch_size, device=x.device) * m + self.num_selects
                threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
                # is each value currently in the top k.
                prob_if_in = self.normal.cdf((logits_gate - threshold_if_in) / noise_control)
                prob_if_out = self.normal.cdf((logits_gate - threshold_if_out) / noise_control)
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)
                if not self.add_noise and not self.warned:
                    warnings.warn('Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                                  'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.')
                    self.warned = True
        else:
            load = (scores_filtered > 0).sum(0)

        """计算balance loss"""
        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(-100.0, device=x.device)

        # print("weight", self.gate_network.weight, sep="\n")
        # print("logits_gate", logits_gate, sep="\n")
        # print("importance", importance, sep="\n")
        # print("load", load, sep="\n")
        # print("balance_loss", balance_loss, sep="\n")

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }

    def forward_return_scores(self, x):
        """先计算所有专家的权重值"""
        logits_gate = self.gate_network(x)  # gate计算出的权重
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)  # 噪声矩阵计算结果
            noise_control = self.softplus(noise_mm) + self.noise_epsilon  # 控制器得到的噪声增加量
            logits_noise = torch.randn_like(logits_gate) * noise_control  # noise附加的权重
            logits = logits_gate + logits_noise  # 最终权重
        else:
            logits = logits_gate  # 最终权重，shape(batch_size, num_experts)

        """计算各个专家的分数scores"""
        scores = self.softmax(logits) if self.use_softmax else logits

        """选出前k个权重，并计算各个专家的分数scores"""
        top_logits, top_indices = logits.topk(min(self.num_selects + 1, self.num_experts), dim=1)  # 选择并排序前k+1个权重
        top_k_logits = top_logits[:, :self.num_selects]
        top_k_indices = top_indices[:, :self.num_selects]
        top_k_scores = self.softmax(top_k_logits) if self.use_softmax else top_k_logits

        """计算importance"""
        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(dim=1, index=top_k_indices, src=top_k_scores)  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        """计算load"""
        # zhutong: 不要把`self.training`写在里面的if语句中，否则会导致eval模式下balance_loss输出值设备不匹配的错误
        if self.training:
            if self.add_noise and self.num_selects != self.num_experts:
                batch_size = top_logits.size(0)
                m = top_logits.size(1)
                top_values_flat = top_logits.flatten()
                threshold_positions_if_in = torch.arange(batch_size, device=x.device) * m + self.num_selects
                threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
                is_in = torch.gt(logits_noise, threshold_if_in)
                threshold_positions_if_out = threshold_positions_if_in - 1
                threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
                # is each value currently in the top k.
                prob_if_in = self.normal.cdf((logits_gate - threshold_if_in) / noise_control)
                prob_if_out = self.normal.cdf((logits_gate - threshold_if_out) / noise_control)
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                load = prob.sum(0)
            else:
                load = (scores_filtered > 0).sum(0)
                if not self.add_noise and not self.warned:
                    warnings.warn('Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                                  'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.')
                    self.warned = True
        else:
            load = (scores_filtered > 0).sum(0)

        """计算balance loss"""
        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(0.0, device=x.device)

        return {
            "scores": scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }

    # fmt: on


class SwitchBalancedGate(BaseGate):
    """
    Select 1 expert each time, with a learnable gate_network controlling expert scores.
    https://arxiv.org/pdf/2101.03961.pdf
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/__init__.py
    https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py
    """

    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
    ):
        super(SwitchBalancedGate, self).__init__()
        assert num_selects in (1, 2)
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight
        self.add_noise = add_noise

    # fmt: off
    def forward(self, x):
        batch_size = x.shape[0]
        logits = self.gate_network(x)  # shape(batch_size, num_experts)
        scores = self.softmax(logits) if self.use_softmax else logits
        if self.add_noise:
            # .to(logits) to make sure the noise is in the same dtype as logits
            #   (e.g. bfloat16) while the default type is float32
            logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device).to(logits)
        else:
            logits_w_noise = logits
        top1_scores, top1_indices = torch.max(logits_w_noise, dim=1)

        """balance loss"""
        importance_mean = scores.mean(0)  # shape(num_experts)

        load = top1_indices.bincount(minlength=self.num_experts)  # 不传递梯度，与原论文保持一致
        assert load.shape[0] == self.num_experts
        # print(f"ZHUTONG (RANK: {os.environ['RANK']}): GATE FORWARD LOAD: {load=}")
        load_mean = load / batch_size  # shape(num_experts)

        balance_loss = self.num_experts * torch.sum(importance_mean * load_mean)
        balance_loss *= self.balance_loss_weight

        return {
            "topK_indices": top1_indices,
            "topK_scores": top1_scores,
            "expert_batch_size": load.tolist(),
            "balance_loss": balance_loss,
            "load": load_mean,
            "importance": importance_mean,
        }
    

class DynamicTopGate(nn.Module):
    """
    Dynamic expert selection with a selectable strategy and a band clamp
    around `num_selects`, i.e., k in [num_selects - k_band, num_selects + k_band]
    intersected with [k_min, k_max].

    Strategies: "topk" | "topp" | "threshold" | "entropy_k" | "budget"
    Always returns fixed shapes for dispatcher compatibility.
    """

    def __init__(
        self,
        input_size: int,
        num_experts: int,
        num_selects: int = 2,                 # nominal k
        select_strategy: str = "topp",        # topp | entropy_k | budget | topk | margin | ratio | elbow
        k_min: int = 1,
        k_max: int = 8,
        p_min: float = 0.92,                  # nucleus threshold (topp)
        gate_network: str = "mlp",
        use_softmax: bool = True,
        use_balance: bool = True,
        balance_loss_weight: float = 1e-2,
        add_noise: bool = True,
        noise_epsilon: float = 1e-2,
        # NEW: shaping & banding
        logit_temperature: float = 0.7,       # <1 => sharper => tends to reduce k
        k_band: int = 1,                      # bind k to [num_selects - k_band, num_selects + k_band]
        # optional overuse regularizer (k > num_selects)
        overuse_penalty_weight: float = 1e-2,
        overuse_penalty_p: float = 1.0,
    ):
        super().__init__()
        assert 1 <= k_min <= k_max <= num_experts
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.select_strategy = select_strategy.lower()
        self.k_min = k_min
        self.k_max = k_max
        self.k_band = max(0, int(k_band))
        self.debug_samples = 5  # number of samples to show in debug printouts

        self.p_min = p_min

        self.gate_network_type = gate_network
        self.gate_network = get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=1)
        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight

        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.warned = False

        self.logit_temperature = logit_temperature
        self.overuse_penalty_weight = overuse_penalty_weight
        self.overuse_penalty_p = overuse_penalty_p

        self.register_buffer("p_min_buf", torch.tensor(self.p_min), persistent=False)
        self.pmin_lr = 1e-3
        self.pmin_bounds = (0.45, 0.85)

        if self.add_noise:
            self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
            self.normal = Normal(0.0, 1.0)
            self.softplus = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self):
        if self.add_noise:
            nn.init.zeros_(self.weight_noise.weight)

    @staticmethod
    def cv_squared(x, eps=1e-10):
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _apply_noise(self, x, logits_gate):
        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)
            noise_control = self.softplus(noise_mm) + self.noise_epsilon
            logits_noise = torch.randn_like(logits_gate) * noise_control
            logits = logits_gate + logits_noise
        else:
            noise_control, logits_noise, logits = None, None, logits_gate
        return logits, logits_noise, noise_control

    def _temper(self, logits):
        T = self.logit_temperature
        return logits if (T is None or T == 1.0) else (logits / T)

    def _band_clamp(self, k_vec: torch.Tensor) -> torch.Tensor:
        # clamp k to intersection of [k_min, k_max] and [num_selects - k_band, num_selects + k_band]
        lower = max(self.k_min, self.num_selects - self.k_band)
        upper = min(self.k_max, self.num_selects + self.k_band)
        return torch.clamp(k_vec, min=lower, max=upper)

    def _print_generic_debug(
        self,
        top_probs: torch.Tensor,   # [B, Kband_hi]
        k_vec: torch.Tensor,       # [B]
        top_indices: torch.Tensor, # [B, Kmax]
        top_mask: torch.Tensor,    # [B, Kmax] (0/1)
    ):
        """
        Strategy-agnostic router diagnostics.

        - Logs basic confidence stats (p1, gaps)
        - Accumulates global histogram of selected k per token (global_hist)
        - Prints accumulated histogram (counts + percentages)

        Assumes `global_hist` is defined at module scope in moe_gates.py:
            global_hist = {}
        """
        B, Kband_hi = top_probs.shape

        # --------------------------
        # (A) k-per-token histogram
        # --------------------------
        hist = torch.bincount(
            k_vec.to(torch.long),
            minlength=int(self.k_max) + 1
        )[: int(self.k_max) + 1]

        global global_hist
        for k, c in enumerate(hist.tolist()):
            if c > 0:
                global_hist[k] = global_hist.get(k, 0) + c

        # --------------------------
        # (B) expert activation histogram
        # --------------------------
        # active experts across all tokens (each selected expert counts once)
        active_experts = top_indices[top_mask.bool()].to(torch.long)  # [N_active]
        expert_counts = torch.bincount(active_experts, minlength=self.num_experts)

        global global_expert_hist
        for e, c in enumerate(expert_counts.tolist()):
            if c > 0:
                global_expert_hist[e] = global_expert_hist.get(e, 0) + c

        # --------------------------
        # (C) logging
        # --------------------------
        logging.info(
            f"[DynamicTopGate][{self.select_strategy}] "
            f"T={self.logit_temperature} "
            f"band=[{self.num_selects-self.k_band},{self.num_selects+self.k_band}] "
            f"k_range=[{self.k_min},{self.k_max}]"
        )

        logging.info(f"  batch k hist (0..{self.k_max}): {hist.tolist()}")

        # accumulated k hist
        total_k = sum(global_hist.values())
        if total_k > 0:
            items = sorted(global_hist.items())
            logging.info("  accumulated k hist:")
            for k, count in items:
                logging.info(f"    k={k:>2d}: {count:>8d} ({100.0*count/total_k:5.2f}%)")

        # accumulated expert hist
        total_e = sum(global_expert_hist.values())
        if total_e > 0:
            items = sorted(global_expert_hist.items())
            logging.info("  accumulated expert activations:")
            for e, count in items:
                logging.info(f"    expert={e:>2d}: {count:>8d} ({100.0*count/total_e:5.2f}%)")


    def _print_topp_extra(self, top_probs):
        cum = torch.cumsum(top_probs, dim=1)
        p_at_2 = cum[:, 1] if top_probs.size(1) > 1 else top_probs[:, 0]
        logging.info(
            f"  topp p_min={self.p_min:.3f} "
            f"p@2 mean={float(p_at_2.mean()):.4f}"
        )

    def _print_entropy_extra(self, top_probs):
        p = top_probs.clamp_min(1e-12)
        H = -(p * p.log()).sum(dim=1)
        logging.info(
            f"  entropy mean={float(H.mean()):.4f} "
            f"min={float(H.min()):.4f} max={float(H.max()):.4f}"
        )

    def _print_margin_extra(self, top_probs):
        p1 = top_probs[:, 0]
        p2 = top_probs[:, 1]
        logging.info(
            f"  p1-p2 mean={float((p1-p2).mean()):.4f}"
        )

    def _dynamic_select(self, logits):
        """
        Returns:
            top_indices [B, Kmax], top_scores [B, Kmax], top_mask [B, Kmax], k_vec [B]
        """
        B, E = logits.shape
        device = logits.device

        # Upper bound of the band around num_selects
        Kband_hi = min(self.k_max, self.num_selects + self.k_band)
        Kband_hi = max(1, Kband_hi)  # safety

        # Get only the top-Kband_hi experts once (cheaper than full sort)
        # top_vals, top_idx are sorted in descending order by construction
        top_vals, top_idx = logits.topk(Kband_hi, dim=1)  # [B, Kband_hi]
        top_probs = self.softmax(top_vals) if self.use_softmax else top_vals

        # ---- Choose k based on strategy ----
        if self.select_strategy == "topk":
            # fixed-k
            k_vec = torch.full((B,), self.num_selects, device=device, dtype=torch.long)

        elif self.select_strategy == "topp":
            # nucleus-like selection within the band
            cum = torch.cumsum(top_probs, dim=1)  # [B, Kband_hi]
            reached = (cum >= self.p_min)

            # Fallback index if row never reaches p_min (shouldn't really happen with softmax)
            default_idx = torch.full(
                (B,), Kband_hi - 1, device=device, dtype=torch.long
            )

            idx_first_true = torch.where(
                reached.any(dim=1),
                reached.float().argmax(dim=1),
                default_idx
            )
            # k is 1-based
            k_vec = (idx_first_true + 1).to(torch.long)  # [B]

            # --- confidence heuristics (still within band) ---
            p1 = top_probs[:, 0]
            p2 = top_probs[:, 1] if Kband_hi > 1 else torch.zeros_like(p1)
            p3 = top_probs[:, 2] if Kband_hi > 2 else torch.zeros_like(p1)
            gap12 = p1 - p2
            gap23 = p2 - p3

            # confident top-1 → force k=1
            confident_1 = (p1 >= 0.46) & (gap12 >= 0.10)
            k_vec = torch.where(confident_1, torch.ones_like(k_vec), k_vec)

            # demote 3→2 when marginal benefit of 3rd expert is low
            if Kband_hi > 2:
                cum2 = cum[:, 1]  # cumulative mass of top-2
            else:
                cum2 = p1

            k3 = (k_vec > 2)
            close_delta = 0.10
            k2_close = (cum2 >= (self.p_min - close_delta))
            p3_small = (p3 <= 0.12)
            low_gain = (gap23 <= 0.03)
            demote_3_to_2 = k3 & (k2_close | p3_small | low_gain)
            k_vec = torch.where(demote_3_to_2, torch.full_like(k_vec, 2), k_vec)

            # Online adaptation of p_min only during training
            if self.training:
                with torch.no_grad():
                    mean_k = k_vec.float().mean()
                    self.p_min_buf -= self.pmin_lr * (mean_k - self.target_k)
                    self.p_min_buf.clamp_(*self.pmin_bounds)
                self.p_min = float(self.p_min_buf.item())

        elif self.select_strategy == "entropy_k":
            # entropy over top-Kband_hi probs (cheap)
            p = top_probs.clamp_min(1e-12)
            H = -(p * p.log()).sum(dim=1)

            # normalize by log(Kband_hi) so it's in [0,1]
            H_norm = H / torch.log(torch.tensor(Kband_hi, device=device, dtype=p.dtype))

            alpha = 2.0
            H_scaled = H_norm.pow(alpha)

            # IMPORTANT: keep original mapping range
            k_low = self.k_min
            k_high = min(self.num_selects, Kband_hi)
            k_float = k_low + (k_high - k_low) * H_scaled
            k_vec = k_float.round().to(torch.long)

        elif self.select_strategy == "entropy_full":
            p = self.softmax(logits) if self.use_softmax else torch.softmax(logits, dim=1)
            H = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
            H_norm = H / torch.log(torch.tensor(self.num_experts, device=device, dtype=p.dtype))

            alpha = 2.0   # >1 → biases toward low k
            H_scaled = H_norm.pow(alpha)

            k_low = self.k_min
            k_high = self.num_selects  # 4

            k_float = k_low + (k_high - k_low) * H_scaled
            k_vec = k_float.round().to(torch.long)


        elif self.select_strategy == "margin":
            # --- logit margin ---
            z1 = top_vals[:, 0]
            z2 = top_vals[:, 1] if Kband_hi > 1 else z1.new_zeros(z1.shape)
            m = (z1 - z2).clamp_min(0.0)  # [B]

            # --- match entropy range ---
            k_low  = int(self.k_min)                           # e.g. 2
            k_high = int(min(self.num_selects, Kband_hi))      # e.g. 4

            # If range collapses, fall back
            if k_high <= k_low:
                k_vec = torch.full((B,), k_low, device=device, dtype=torch.long)
            else:
                # ---------------------------------------------------------
                # 3-bin calibrated rule:
                #   small margin  -> k=4
                #   medium margin -> k=3
                #   large margin  -> k=2
                #
                # Set these from the printed quantiles of m:
                #   t43 around ~ median (q50)
                #   t32 around ~ upper quartile / q75–q85
                # ---------------------------------------------------------
                t43 = float(getattr(self, "margin_t43", 0.10))
                t32 = float(getattr(self, "margin_t32", 0.40))
                # Ensure ordering
                if t32 <= t43:
                    t32 = t43 + 1e-6

                k_vec = torch.full((B,), k_high, device=device, dtype=torch.long)  # start at 4
                k_vec = torch.where(m >= t43, torch.full_like(k_vec, max(k_low, 3)), k_vec)  # >=t43 => 3
                k_vec = torch.where(m >= t32, torch.full_like(k_vec, max(k_low, 2)), k_vec)  # >=t32 => 2

                # If your range isn't exactly {2,3,4}, clamp safely:
                k_vec = k_vec.clamp(min=k_low, max=k_high)
                
            # if (not hasattr(self, "_printed_margin_stats")) or (not self._printed_margin_stats):
            #     with torch.no_grad():
            #         m32 = m.detach().float()
            #         qs = m32.new_tensor([0.05, 0.25, 0.50, 0.75, 0.90])
            #         qv = torch.quantile(m32, qs).cpu().tolist()
            #         print("margin (z1-z2) quantiles 5/25/50/75/90:", [round(x, 4) for x in qv])
            #     self._printed_margin_stats = True

        elif self.select_strategy == "ratio":
            """
            Ratio-based confidence using p1/p2 computed from softmax(top_vals),
            with per-layer auto-calibrated thresholds and 3-bin mapping.
            Goal: mostly k=3–4, small k=2 tail, no drift to k=6.
            """

            # --- 0) Always compute probs from logits for ratio (don't trust top_probs) ---
            p = torch.softmax(top_vals.float(), dim=1).to(top_vals.dtype)  # [B, Kband_hi]
            p1 = p[:, 0]
            p2 = p[:, 1] if Kband_hi > 1 else p1.new_full(p1.shape, 1e-6)
            ratio = (p1 / p2.clamp_min(1e-6))  # [B], >= 1

            # --- 1) Match entropy_k range ---
            k_low  = int(self.k_min)
            k_high = int(min(self.num_selects, Kband_hi))  # IMPORTANT: prevents k=5/6

            if k_high <= k_low:
                k_vec = torch.full((B,), k_low, device=device, dtype=torch.long)
            else:
                # --- 2) Auto-calibrate thresholds per gate instance ---
                # Interpretation:
                #   ratio small  => uncertain => k_high (4)
                #   ratio medium =>           => 3
                #   ratio large  => confident => 2
                if getattr(self, "ratio_auto", True):
                    with torch.no_grad():
                        r32 = ratio.detach().float()
                        # choose quantiles to control histogram:
                        # q50: half the tokens become <=3 (instead of 4)
                        # q90: top 10% become 2
                        q_med = torch.quantile(r32, r32.new_tensor(0.50))
                        q_hi  = torch.quantile(r32, r32.new_tensor(0.90))

                        ema = float(getattr(self, "ratio_ema", 0.05))
                        if not hasattr(self, "ratio_t43_buf"):
                            self.ratio_t43_buf = q_med
                            self.ratio_t32_buf = q_hi
                        else:
                            self.ratio_t43_buf = (1 - ema) * self.ratio_t43_buf + ema * q_med
                            self.ratio_t32_buf = (1 - ema) * self.ratio_t32_buf + ema * q_hi

                        r43 = float(self.ratio_t43_buf.item())
                        r32 = float(self.ratio_t32_buf.item())
                else:
                    # manual fallback
                    r43 = float(getattr(self, "ratio_r43", 1.5))
                    r32 = float(getattr(self, "ratio_r32", 2.5))

                # ensure ordering / minimal separation
                min_gap = float(getattr(self, "ratio_min_gap", 0.10))
                if r32 <= r43 + min_gap:
                    r32 = r43 + min_gap

                # --- 3) 3-bin decision ---
                k_vec = torch.full((B,), k_high, device=device, dtype=torch.long)  # start at 4
                k3 = max(k_low, min(3, k_high))
                k2 = max(k_low, min(2, k_high))

                k_vec = torch.where(ratio >= r43, torch.full_like(k_vec, k3), k_vec)  # >=r43 => 3
                k_vec = torch.where(ratio >= r32, torch.full_like(k_vec, k2), k_vec)  # >=r32 => 2

                k_vec = k_vec.clamp(min=k_low, max=k_high)
                
            # if (not hasattr(self, "_printed_ratio_stats")) or (not self._printed_ratio_stats):
            #     with torch.no_grad():
            #         r32 = ratio.detach().float()
            #         qs = r32.new_tensor([0.05, 0.25, 0.50, 0.75, 0.90])
            #         qv = torch.quantile(r32, qs).cpu().tolist()
            #         print("ratio p1/p2 quantiles 5/25/50/75/90:", [round(x, 4) for x in qv])
            #     self._printed_ratio_stats = True



        elif self.select_strategy == "elbow":
            # Pick smallest k such that adding one more expert doesn't add much mass.
            # Uses per-step increments of cumulative probability.
            cum = torch.cumsum(top_probs, dim=1)              # [B, Kband_hi]
            inc = torch.cat([cum[:, :1], cum[:, 1:] - cum[:, :-1]], dim=1)  # [B, Kband_hi]

            # "diminishing return" threshold: if next expert adds < eps mass, stop
            eps = 0.06  # tune: smaller => picks larger k
            # want first position j where inc[j] < eps AFTER at least 1 expert
            # build mask for positions 1..Kband_hi-1
            stop_mask = (inc[:, 1:] < eps)                    # [B, Kband_hi-1]
            default_idx = torch.full((B,), Kband_hi - 1, device=device, dtype=torch.long)
            idx_first_true = torch.where(
                stop_mask.any(dim=1),
                stop_mask.float().argmax(dim=1) + 1,          # +1 to account for skipping inc[:,0]
                default_idx
            )
            k_vec = (idx_first_true + 1).to(torch.long)       # 1-based k

        else:
            raise ValueError(f"Unknown strategy {self.select_strategy}")

        # ---- Band clamp k around num_selects and within [k_min, k_max] ----
        k_vec = self._band_clamp(k_vec)  # [B]

        # ---- Build fixed-size outputs up to Kmax = Kband_hi ----
        Kmax = Kband_hi
        pos = torch.arange(Kmax, device=device).unsqueeze(0).expand(B, Kmax)  # [B,Kmax]
        k_exp = k_vec.unsqueeze(1)                                            # [B,1]

        top_mask = (pos < k_exp).to(logits.dtype)     # [B,Kmax] {0,1}
        top_indices = top_idx[:, :Kmax]               # [B,Kmax]
        top_scores = top_probs[:, :Kmax] * top_mask   # [B,Kmax]

        # For masked-out positions, pad indices with the last valid index in that row
        last_valid = torch.clamp(k_vec - 1, min=0)    # [B]
        pad_fill = top_indices.gather(
            1, last_valid.unsqueeze(1).expand(B, Kmax)
        )                                             # [B,Kmax]
        top_indices = torch.where(top_mask.bool(), top_indices, pad_fill)

        # self._print_generic_debug(
        #     top_probs.detach(),
        #     k_vec.detach(),
        #     top_indices.detach(),
        #     top_mask.detach(),
        # )

        # if self.select_strategy == "topp":
        #     self._print_topp_extra(top_probs.detach())
        # elif self.select_strategy == "entropy_k":
        #     self._print_entropy_extra(top_probs.detach())
        # elif self.select_strategy in {"margin", "ratio"}:
        #     self._print_margin_extra(top_probs.detach())

        return top_indices, top_scores, top_mask, k_vec


    def forward(self, x):
        logits_gate = self.gate_network(x)                       # [B,E]
        logits, logits_noise, noise_control = self._apply_noise(x, logits_gate)
        logits_t = self._temper(logits)                          # temperature before selection

        top_indices, top_scores, top_mask, k_vec = self._dynamic_select(logits_t)

        # importance/load
        B, E = logits.shape
        zeros = torch.zeros_like(logits, requires_grad=False, device=logits.device)
        scores_filtered = zeros.scatter(dim=1, index=top_indices, src=top_scores)
        importance = scores_filtered.sum(0)

        if self.training and self.add_noise and self.select_strategy == "topk" and self.num_selects != self.num_experts:
            # differentiable load only for noisy fixed-k case
            batch_size = logits.shape[0]
            m = min(self.num_selects + 1, self.num_experts)
            top_logits, _ = logits.topk(m, dim=1)
            top_values_flat = top_logits.flatten()
            tpos_in = torch.arange(batch_size, device=x.device) * m + self.num_selects
            thr_in = torch.gather(top_values_flat, 0, tpos_in).unsqueeze(1)
            tpos_out = tpos_in - 1
            thr_out = torch.gather(top_values_flat, 0, tpos_out).unsqueeze(1)
            is_in = torch.gt(logits_noise, thr_in)
            prob_if_in = self.normal.cdf((logits_gate - thr_in) / noise_control)
            prob_if_out = self.normal.cdf((logits_gate - thr_out) / noise_control)
            prob = torch.where(is_in, prob_if_in, prob_if_out)
            load = prob.sum(0)
        else:
            load = (scores_filtered > 0).sum(0)

        # balance loss
        if self.use_balance:
            balance_loss = (self.cv_squared(importance) + self.cv_squared(load)) * self.balance_loss_weight
        else:
            balance_loss = torch.tensor(0.0, device=x.device)

        out = {
            "topK_indices": top_indices,     # [B, Kmax]
            "topK_scores": top_scores,       # [B, Kmax]
            "topK_mask": top_mask,           # [B, Kmax]
            "importance": importance,        # [E]
            "load": load,                    # [E]
            "balance_loss": balance_loss,    # scalar
            "k_vec": k_vec                   # [B]
        }

        # optional budget loss
        if self.select_strategy == "budget":
            mean_k = k_vec.float().mean()
            out["budget_loss"] = (mean_k - self.target_k).pow(2) * self.budget_weight

        # optional overuse regularizer (only penalize k > num_selects)
        if self.overuse_penalty_weight > 0 and self.select_strategy in {"topp","threshold","entropy_k","budget"}:
            over = (k_vec.float() - float(self.num_selects)).clamp_min(0.0)
            out["overuse_loss"] = over.pow(self.overuse_penalty_p).mean() * self.overuse_penalty_weight

        return out
