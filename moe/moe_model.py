import torch
import torch.nn as nn

from moe.experts import ExpertHead
from moe.gating import GatingNetwork


class MoERegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gating = GatingNetwork(input_dim=input_dim, num_experts=num_experts)
        self.experts = nn.ModuleList(
            [ExpertHead(input_dim=input_dim, output_dim=output_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor):
        logits = self.gating(x)
        probs = torch.softmax(logits, dim=-1)

        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(probs)
        sparse_weights.scatter_(1, top_indices, top_values)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=1, keepdim=True) + 1e-8)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        pred = (sparse_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return pred, logits, sparse_weights, top_indices, top_values
