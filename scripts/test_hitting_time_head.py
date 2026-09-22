"""Small invariance check for every one-pass hitting-time head."""

import torch

from common.viability import SKIP, HittingTimeHead


def main():
    torch.manual_seed(0)
    z, zg = torch.randn(7, 12), torch.randn(7, 12)
    for kind in ("softmax", "hazard", "weibull"):
        head = HittingTimeHead(z_dim=12, b_max=9, hidden=(16,), head=kind)
        lp = head.log_pmf(z, zg)
        assert torch.allclose(lp.exp().sum(-1), torch.ones(7), atol=2e-5), kind
        hs = torch.arange(0, head.h_max + 1, SKIP)
        cdf = torch.stack([head.cdf(z, zg, torch.full((7,), h)) for h in hs])
        assert bool((cdf[1:] >= cdf[:-1] - 1e-6).all()), kind
        for h in (0, 15, head.h_max):
            loop = SKIP * sum(1 - head.cdf(z, zg, torch.full((7,), j))
                              for j in range(0, h + 1, SKIP))
            one = head.restricted_expected_steps(z, zg, h)
            assert torch.allclose(one, loop, atol=2e-4), (kind, h, (one - loop).abs().max())
        head.restricted_expected_steps(z, zg, head.h_max).sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()), kind
    print("hitting-time heads: pmf, monotonic CDF, one-pass ET, and gradients OK")


if __name__ == "__main__":
    main()
