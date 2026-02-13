from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCAWidetableVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, cat_sizes: list[int], num_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cat_sizes = cat_sizes
        self.num_dim = num_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(512, latent_dim)
        self.logvar_layer = nn.Linear(512, latent_dim)
        self.cat_decoders = nn.ModuleList([nn.Linear(latent_dim, s) for s in cat_sizes])
        self.num_decoder = nn.Linear(latent_dim, num_dim)
        self.logvar_min = -10.0
        self.logvar_max = 10.0

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.mean_layer(h)
        logvar = torch.clamp(self.logvar_layer(h), min=self.logvar_min, max=self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        cat_logits = [head(z) for head in self.cat_decoders]
        num_out = self.num_decoder(z)
        return cat_logits, num_out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        cat_logits, num_out = self.decode(z)
        return cat_logits, num_out, mu, logvar


def vae_loss(
    cat_logits: list[torch.Tensor],
    num_out: torch.Tensor,
    x_cat_targets: list[torch.Tensor],
    x_num_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
):
    ce = 0.0
    for logits, target in zip(cat_logits, x_cat_targets):
        ce = ce + F.cross_entropy(logits, target)
    mse = F.mse_loss(num_out, x_num_target)
    safe_logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl = -0.5 * torch.mean(1 + safe_logvar - mu.pow(2) - safe_logvar.exp())
    total = ce + mse + beta * kl
    return total, ce.detach(), mse.detach(), kl.detach()

