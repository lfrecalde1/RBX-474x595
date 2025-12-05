import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 1e-8, 0.999)

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, schedule="linear"):
        super().__init__()

        self.model = model
        self.T = T
        
        if schedule == "linear":
            print("Using a linear schedule")
            self.register_buffer('betas',torch.linspace(beta_1, beta_T, T) )
        else:
            print("Using a cosine schedule")
            betas = cosine_beta_schedule(T)              # [T], float64
            self.register_buffer('betas',  betas.float()) # cast to float32 if you like
        
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
       
        self.register_buffer('sqrt_alphas_bar',torch.sqrt(alphas_bar) )
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
 
        batch_size = x_0.shape[0]
        
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device, dtype=torch.long)
        
        noise = torch.randn_like(x_0)
        
        sqrt_alphas_bar_t = self.sqrt_alphas_bar[t]
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t]

        sqrt_alphas_bar_t = sqrt_alphas_bar_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_bar_t = sqrt_one_minus_alphas_bar_t.view(-1, 1, 1, 1)
        
        x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise       
        
        predicted_noise = self.model(x_t, t, labels)
        
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0., schedule="linear"):
        super().__init__()

        self.model = model
        self.T = T
 
        self.w = w

        if schedule == "linear":
            print("Using a linear schedule")
            self.register_buffer('betas',torch.linspace(beta_1, beta_T, T) )
        else:
            print("Using a cosine schedule")
            betas = cosine_beta_schedule(T)              # [T], float64
            self.register_buffer('betas',  betas.float()) # cast to float32 if you like

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.register_buffer(
            'sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1.0 / alphas_bar))
        self.register_buffer(
            'sqrt_recip_m1_alphas_bar', torch.sqrt(1.0 / alphas_bar - 1))
        
        posterior_variance = (
            self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )

        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer(
            'posterior_mean_coef1',
            self.betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar))
        self.register_buffer(
            'sqrt_posterior_variance', torch.sqrt(posterior_variance))



    def forward(self, x_T, labels):

        x_t = x_T
        for time_step in reversed(range(self.T)):

            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            
            if self.w > 0:

                eps_cond = self.model(x_t, t, labels)

                eps_uncond = self.model(x_t, t, torch.zeros_like(labels))
                eps = (1 + self.w) * eps_cond - self.w * eps_uncond
            else:
                eps = self.model(x_t, t, labels)

            timestep_idx = time_step

            sqrt_recip_alphas_t = self.sqrt_recip_alphas[timestep_idx]
            betas_t = self.betas[timestep_idx]
            sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[timestep_idx]

            mean = sqrt_recip_alphas_t * (x_t - (betas_t * eps) / sqrt_one_minus_alphas_bar_t)
            if time_step > 0:
                noise = torch.randn_like(x_t)
                sigma = self.sqrt_posterior_variance[timestep_idx]
                x_t = mean + sigma * noise
            else:
                x_t = mean
        
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   
