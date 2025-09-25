import time
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from vae import VAE

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def is_primary():
    return get_rank() == 0

def train(model, optimiser, dataloader, sampler, n_epochs, device):
    model.train()

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)

        start = time.time()
        epoch_loss_total = 0
        ll_total = 0
        kl_total = 0

        batch_count = 0

        beta_final, warmup_epochs = 1.0, 20
        beta = min(beta_final, (epoch+1)/warmup_epochs*beta_final)

        for x, _ in dataloader:
            x = x.to(device, non_blocking=True)
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

        process_loss_mean = torch.tensor((epoch_loss_total / batch_count), device=device)
        dist.all_reduce(process_loss_mean, op=dist.ReduceOp.AVG)

        process_ll_mean = torch.tensor((ll_total / batch_count), device=device)
        dist.all_reduce(process_ll_mean, op=dist.ReduceOp.AVG)

        process_kl_mean = torch.tensor((kl_total / batch_count), device=device)
        dist.all_reduce(process_kl_mean, op=dist.ReduceOp.AVG)

        if is_primary():
            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs} || Loss: {process_loss_mean.item():.4f} || Recon: {process_ll_mean.item():.4f} || "
                    f"KL: {process_kl_mean.item():.4f} || Time taken: {duration:.2f}s"
                )
        
if __name__ == "__main__":
    print("Setting up ...")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        pin = True
    else:
        device = torch.device("cpu")
        pin = False

    train_tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10("./data", train=True, transform=train_tf, download=is_primary())

    dist.barrier()

    sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds, 
        batch_size=8, 
        shuffle=False,
        sampler=sampler,
        pin_memory=pin,
        drop_last=True,
        num_workers=2,
        persistent_workers=True
        )

    model = VAE(64).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 1

    if device.type == "cuda":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(model)

    print("Starting training")
    train(model=model, optimiser=optimiser, dataloader=train_loader, sampler=sampler, n_epochs=n_epochs,device=device)

    print("Training finished. Saving weights.")
    if is_primary():
        torch.save(model.module.state_dict(), "vae_ddp.pt")

    dist.destroy_process_group()
