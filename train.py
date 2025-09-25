import torch
import time
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from vae import VAE

def train(model, optimiser, dataloader, n_epochs, device):
    model.to(device)
    model.train()

    for epoch in range(n_epochs):
        start = time.time()
        epoch_loss_total = 0
        ll_total = 0
        kl_total = 0

        batch_count = 0

        beta_final, warmup_epochs = 1.0, 1
        beta = min(beta_final, (epoch+1)/warmup_epochs*beta_final)

        for x, _ in dataloader:
            x = x.to(device)
            optimiser.zero_grad(set_to_none=True)

            mu_x, mu_z, logvar_z = model(x)

            loss, ll, kl = VAE.elbo_loss(x=x, mu_x=mu_x, logvar_z=logvar_z, mu_z=mu_z, sigma2_x=0.01, beta=beta) 
            loss.backward()
            optimiser.step()

            epoch_loss_total += loss.item()
            ll_total += ll.item()
            kl_total += kl.item()

            batch_count += 1 
        
        duration = time.time() - start
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs} || Loss: {epoch_loss_total / batch_count:.4f} || Recon: {ll_total / batch_count:.4f} || "
                f"KL: {kl_total / batch_count:.4f} || Time taken: {duration:.2f}s"
            )

    torch.save(model.state_dict(), 'vae.pt')
        
if __name__ == "__main__":
    train_tf = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = datasets.CIFAR10("./data", train=True, transform=train_tf, download=True)
    dog_idxs = [i for i, target in enumerate(train_ds.targets) if target == 5] # Select only dogs
    dog_ds = Subset(train_ds, dog_idxs)

    train_loader = DataLoader(dog_ds, batch_size=8, shuffle=True)

    model = VAE(64)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load('vae.pt', map_location=device))
    train(model, optimiser, train_loader, n_epochs, device)