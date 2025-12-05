
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = int(T)
                
        # YOUR IMPLEMENTATION HERE!
        betas = torch.linspace(beta_1, beta_T, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.cond_drop_prob = 0
        
        # Register buffers so they move with .to(device) and are not trained
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        #self.register_buffer("cond_drop_prob", cond_drop_prob)

        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "sqrt_recipm1_alphas",
            torch.sqrt(1.0 / alphas - 1.0),
        )

        # Posterior q(x_{t-1} | x_t, x_0) coefficients
        alphas_bar_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_bar[:-1]], dim=0
        )

        posterior_variance = (
            betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar),
        )

    def forward(self, x_0, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        Inputs  - Original images (batch_size x 3 x 32 x 32), class labels[1 to 10] (batch_size dimension) 
        Outputs - Loss value (mse works for this application)
        
        """
        
        device = x_0.device
        B = x_0.shape[0]

        # Sample random time step t ~ Uniform({0, ..., T-1})
        t = torch.randint(low=0, high=self.T, size=(B,), device=device)

        # Sample noise ε
        eps = torch.randn_like(x_0)

        # Compute x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
        sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, x_0.shape)
        sqrt_one_minus_alphas_bar_t = extract(
            self.sqrt_one_minus_alphas_bar, t, x_0.shape
        )
        x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * eps

        # Classifier-free guidance training: randomly drop condition
        if self.cond_drop_prob > 0.0:
            drop_mask = (
                torch.rand(B, device=device) < self.cond_drop_prob
            )  # True -> use null label
            labels = labels.clone()
            labels[drop_mask] = 0  # 0 reserved for "unconditional"

        eps_theta = self.model(x_t, t, labels)
        loss = F.mse_loss(eps_theta, eps)
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = int(T)
                
        # YOUR IMPLEMENTATION HERE!
        betas = torch.linspace(beta_1, beta_T, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        # Register buffers so they move with .to(device) and are not trained
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "sqrt_recipm1_alphas",
            torch.sqrt(1.0 / alphas - 1.0),
        )

        # Posterior q(x_{t-1} | x_t, x_0) coefficients
        alphas_bar_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_bar[:-1]], dim=0
        )

        posterior_variance = (
            betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar),
        )
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w


    def forward(self, x_T, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        """
        x_t = x_T

        for time_step in reversed(range(self.T)):
            t = torch.full(
                (x_t.shape[0],),
                time_step,
                device=x_t.device,
                dtype=torch.long,
            )

            # Predict noise with (optional) classifier-free guidance
            if self.w == 0.0:
                # Plain conditional prediction ε_θ(x_t, t, y)
                eps_theta = self.model(x_t, t, labels)
            else:
                # Conditional branch
                eps_cond = self.model(x_t, t, labels)
                # Unconditional branch: use null label 0
                null_labels = torch.zeros_like(labels)
                eps_uncond = self.model(x_t, t, null_labels)
                # Classifier-free combination
                eps_theta = (1.0 + self.w) * eps_cond - self.w * eps_uncond

            # Estimate x_0 from predicted noise:
            #   x̂_0 = (x_t - sqrt(1-ᾱ_t) ε_θ) / sqrt(ᾱ_t)
            sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, x_t.shape)
            sqrt_one_minus_alphas_bar_t = extract(
                self.sqrt_one_minus_alphas_bar, t, x_t.shape
            )
            x0_pred = (x_t - sqrt_one_minus_alphas_bar_t * eps_theta) / (
                sqrt_alphas_bar_t + 1e-8
            )

            # Compute posterior mean μ̃_t(x_t, x̂_0)
            posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )

            if time_step > 0:
                noise = torch.randn_like(x_t)
                var = extract(self.posterior_variance, t, x_t.shape)
                x_t = posterior_mean + torch.sqrt(var) * noise
            else:
                # Last step: no extra noise
                x_t = posterior_mean

            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_0 = x_t
        return torch.clamp(x_0, -1.0, 1.0)
           


