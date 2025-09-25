from vae import VAE
import torch
from torchvision.utils import make_grid, save_image

if __name__ == "__main__":
    vae = VAE(64)
    vae.load_state_dict(torch.load('vae.pt', map_location='cpu'))

    samples = vae.sample(8)
    grid = make_grid(samples, nrow=4)
    save_image(grid, "samples.png")
