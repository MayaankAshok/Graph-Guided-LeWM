"""Amortized terminal continuation cost for LeWM planning."""

import torch
from torch import nn


class ContinuationHead(nn.Module):
    """Predict normalized log(1 + inverse-CEM residual) from final latent history and goal."""

    def __init__(self, z_dim=192, hidden=(512, 256)):
        super().__init__()
        layers, width = [], 5 * z_dim
        for next_width in hidden:
            layers += [nn.Linear(width, next_width), nn.SiLU()]
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.z_dim, self.hidden = z_dim, tuple(hidden)
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))
        self.register_buffer("target_mean", torch.zeros(()))
        self.register_buffer("target_std", torch.ones(()))
        self.register_buffer("l2_median", torch.zeros(()))
        self.register_buffer("l2_iqr", torch.ones(()))

    def set_stats(self, z_mean, z_std, target_mean, target_std, l2_median, l2_iqr):
        self.z_mean.copy_(torch.as_tensor(z_mean))
        self.z_std.copy_(torch.as_tensor(z_std).clamp_min(1e-6))
        self.target_mean.copy_(torch.as_tensor(target_mean))
        self.target_std.copy_(torch.as_tensor(target_std).clamp_min(1e-6))
        self.l2_median.copy_(torch.as_tensor(l2_median))
        self.l2_iqr.copy_(torch.as_tensor(l2_iqr).clamp_min(1e-6))

    def forward(self, history, goal):
        h = (history - self.z_mean) / self.z_std
        g = (goal - self.z_mean) / self.z_std
        features = torch.cat([h.flatten(start_dim=-2), g, h[..., -1, :] - g], dim=-1)
        return self.net(features).squeeze(-1)

    def residual(self, history, goal):
        return torch.expm1(self.forward(history, goal) * self.target_std + self.target_mean).clamp_min(0)


class ContinuationEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    @property
    def l2_median(self):
        return self.members[0].l2_median

    @property
    def l2_iqr(self):
        return self.members[0].l2_iqr

    def predictions(self, history, goal):
        return torch.stack([member(history, goal) for member in self.members])

    def score(self, history, goal, kappa=1.0):
        predictions = self.predictions(history, goal)
        return predictions.mean(0) + kappa * predictions.std(0, unbiased=False)


def load_continuation_head(path, device="cuda"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "members" in payload:
        members = []
        for state in payload["members"]:
            member = ContinuationHead(payload["z_dim"], tuple(payload["hidden"]))
            member.load_state_dict(state)
            members.append(member)
        head = ContinuationEnsemble(members)
    else:
        head = ContinuationHead(payload["z_dim"], tuple(payload["hidden"]))
        head.load_state_dict(payload["state_dict"])
    return head.to(device).eval(), payload


@torch.no_grad()
def continuation_criterion(head, kind="continuation", weight=1.0, kappa=1.0):
    """Drop-in `jepa.JEPA.criterion`; lower predicted continuation residual is better."""
    if kind not in ("continuation", "continuation_hybrid"):
        raise ValueError(kind)

    def criterion(info):
        predicted = info["predicted_emb"]
        history = predicted[..., -3:, :]
        goal = info["goal_emb"][..., -1, :]
        while goal.ndim < history.ndim - 1:
            goal = goal.unsqueeze(-2)
        goal = goal.expand(*history.shape[:-2], history.shape[-1]).detach()
        score = (head.score(history, goal, kappa)
                 if isinstance(head, ContinuationEnsemble) else head(history, goal))
        if kind == "continuation":
            return score
        l2 = ((history[..., -1, :] - goal) ** 2).sum(-1)
        return (l2 - head.l2_median) / head.l2_iqr + weight * score

    return criterion
