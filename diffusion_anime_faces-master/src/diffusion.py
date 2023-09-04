from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


def _diffusion_loss(model: nn.Module, x: Tensor, alpha_prod_sqrt: Tensor, one_minus_alpha_prod_sqrt: Tensor,
                    steps: int, device: torch.device) -> Tensor:
    batch = x.shape[0]
    t = torch.randint(0, steps, size=(batch,)).to(device)

    a = alpha_prod_sqrt[t]
    am1 = one_minus_alpha_prod_sqrt[t]

    a = a.view(-1, *((1,) * (len(x.shape) - 1)))
    am1 = am1.view(-1, *((1,) * (len(x.shape) - 1)))

    epsilon = torch.randn_like(x).to(device)

    x = x * a + am1 * epsilon

    output = model(x, t)
    return F.mse_loss(output, epsilon)


def _diffusion_back(model: nn.Module, x: Tensor, tm: int,
                    beta: torch.Tensor, one_minus_alpha_prod_sqrt: torch.Tensor,
                    post_variance: torch.Tensor,
                    device: torch.device, end: bool = False) -> Tensor:
    t = torch.tensor([tm]).to(device)
    eps_theta = model(x, t)
    coeff = beta[t] / one_minus_alpha_prod_sqrt[t]

    mean = 1 / torch.sqrt(1 - beta[t]) * (x - coeff * eps_theta)
    sigma = torch.sqrt(post_variance[t])

    z = torch.randn_like(x).to(device)

    if not end:
        return mean + sigma * z
    else:
        return mean


class Diffusion:
    device: torch.device

    model: nn.Module
    optim: torch.optim.Optimizer

    alpha: Tensor
    beta: Tensor
    steps: int

    alpha_prod: Tensor
    alpha_prod_prev: Tensor
    alpha_prod_sqrt: Tensor

    posterior_variance: Tensor

    one_minus_alpha_prod_sqrt: Tensor

    def __init__(self, model: nn.Module, optim: torch.optim.Optimizer, beta: Tensor,
                 device: torch.device = torch.device("cpu:0")):
        super(Diffusion, self).__init__()
        self.device = device

        self.model = model
        self.optim = optim

        beta = beta.to(device)
        self.steps = beta.shape[0]
        self.beta = beta
        self.alpha = 1 - beta
        self.alpha_prod = torch.cumprod(self.alpha, 0)
        self.alpha_prod_prev = torch.cat([torch.tensor([1]).to(device), self.alpha_prod[:-1]], dim=0)

        self.alpha_prod_sqrt = torch.sqrt(self.alpha_prod)
        self.one_minus_alpha_prod_sqrt = torch.sqrt(1 - self.alpha_prod)

        self.posterior_variance = self.beta * (1. - self.alpha_prod_prev) / (1. - self.alpha_prod)

    def to(self, device: torch.device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_prod = self.alpha_prod.to(device)
        self.alpha_prod_sqrt = self.alpha_prod_sqrt.to(device)
        self.one_minus_alpha_prod_sqrt = self.one_minus_alpha_prod_sqrt.to(device)

    def train(self, x: Tensor) -> float:
        x = x.to(self.device)
        self.model.zero_grad()
        loss = _diffusion_loss(self.model, x, self.alpha_prod_sqrt, self.one_minus_alpha_prod_sqrt, self.steps,
                               self.device)
        loss.backward()
        self.optim.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, shape: Tuple[int], prog: bool = False) -> List[Tensor]:
        cur_x = torch.randn(shape).to(self.device)
        seq = [cur_x]
        cnt = 0
        for i in reversed(range(self.steps)):
            cur_x = _diffusion_back(self.model, cur_x, i, self.beta, self.one_minus_alpha_prod_sqrt,
                                    self.posterior_variance, self.device, i == 0)
            seq.append(cur_x)
            cnt += 1
            if prog:
                print(f"\r{cnt}/{self.steps}", end="")
        print()
        return seq

    @torch.no_grad()
    def generate_from(self, x: Tensor, t: int) -> List[Tensor]:
        z = torch.randn_like(x).to(self.device)
        cur_x = self.alpha_prod_sqrt[t] * x + self.one_minus_alpha_prod_sqrt[t] * z
        seq = [cur_x]
        for i in reversed(range(t)):
            cur_x = _diffusion_back(self.model, cur_x, i, self.beta, self.one_minus_alpha_prod_sqrt,
                                    self.posterior_variance, self.device, i == 0)
            seq.append(cur_x)
        return seq

    def switch_optimizer(self, optim: torch.optim.Optimizer):
        self.optim = optim
