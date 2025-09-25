import torch
import torch.nn as nn
from torch import Tensor

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2), # 3x32x32 -> 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), # 32x16x16 -> 64x8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), # 128x4x4
            nn.ReLU()
        )
        self.enc_dim = 128 * 4 * 4

        self.enc_mean = nn.Linear(self.enc_dim, latent_dim)
        self.enc_logvar = nn.Linear(self.enc_dim, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(self.latent_dim, self.enc_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2), # 128x4x4 -> 64x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2), # 64x8x8 -> 32x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, padding=1, stride=2), # 32x16x16 -> 3x32x32
        )

    def reparameterise(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)

        return mu + sigma * epsilon
    
    def encode(self, x):
        h = self.conv(x).view(x.size(0), -1) # Flatten to [B, 128*4*4]
        mu_z = self.enc_mean(h)
        logvar_z = self.enc_logvar(h)

        return mu_z, logvar_z
    
    def decode(self, z):
        h = self.dec_fc(z).view(z.size(0), 128, 4, 4)
        mu_x = torch.sigmoid(self.deconv(h))

        return mu_x
    
    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterise(mu_z, logvar_z)

        x_mu = self.decode(z)

        return x_mu, mu_z, logvar_z

    @torch.no_grad()
    def sample(self, n_samples=1):
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    @staticmethod
    def elbo_loss(x: Tensor, mu_x: Tensor, logvar_z: Tensor, mu_z: Tensor, sigma2_x: float=0.01, beta: float=1.0):
        def recon(x: Tensor, mu_x: Tensor, sigma2_x: float) -> torch.Tensor:
            # x, mu_x: [B, C, H, W]
            # Flatten to [B, D] where D=C*H*W
            # Sum across D to yield scalar, shape [B]
            diff2 = ((x - mu_x)**2).flatten(1).sum(dim=1)

            # shape [B]
            ll = -0.5 * diff2 / sigma2_x

            # Take mean across batch
            return ll.mean()
        
        def kl_func(mu_z: Tensor, logvar_z: Tensor) -> torch.Tensor:
            # mu_z, logvar_z: [B, latent_dim]
            kl = 0.5 * (mu_z**2 + torch.exp(logvar_z) - logvar_z - 1)

            # Sum across latent dims then take mean across batch
            return kl.sum(dim=1).mean()
        
        ll = recon(x, mu_x, sigma2_x)
        kl = kl_func(mu_z, logvar_z)

        return -(ll - beta * kl), ll, kl