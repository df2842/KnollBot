import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import warnings
from transformers import ViTModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

BATCH_SIZE = 12
NUM_EPOCHS = 48
NUM_WORKERS = 12

IMAGE_SIZE = 518
TOOL_WEIGHT = 50
LR = 1e-5
WEIGHT_DECAY = 1e-2

LATENT_DIM = 256
KL_WEIGHT = 0.00025

MESSY_DIR_NAME = 'messy'
NEAT_DIR_NAME = 'neat'
VIT_NAME = 'knolling_vit.pth'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESSY_DIR = os.path.join(SCRIPT_DIR, MESSY_DIR_NAME)
NEAT_DIR = os.path.join(SCRIPT_DIR, NEAT_DIR_NAME)
VIT_PATH = os.path.join(SCRIPT_DIR, VIT_NAME)

warnings.filterwarnings("ignore", category=FutureWarning)

class ImagePairDataset(Dataset):
    def __init__(self, messy_dir, neat_dir, image_files, transform=None):
        self.messy_dir = messy_dir
        self.neat_dir = neat_dir
        self.transform = transform
        self.samples = []

        for neat_file in image_files:
            parts = neat_file.split('_')
            img_id = parts[1]

            messy_file = f"messy_{img_id}.png"

            if os.path.exists(os.path.join(messy_dir, messy_file)):
                self.samples.append((messy_file, neat_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messy_img_name, neat_img_name = self.samples[idx]

        messy_path = os.path.join(self.messy_dir, messy_img_name)
        neat_path = os.path.join(self.neat_dir, neat_img_name)

        messy_image = Image.open(messy_path).convert("RGB")
        neat_image = Image.open(neat_path).convert("RGB")

        if self.transform:
            messy_image = self.transform(messy_image)
            neat_image = self.transform(neat_image)

        return messy_image, neat_image

class StochasticViT(nn.Module):
    def __init__(self, vit_model_name='facebook/dinov2-large', latent_dim=LATENT_DIM):
        super().__init__()

        vit_model = ViTModel.from_pretrained(vit_model_name)
        self.encoder = vit_model.encoder
        self.embeddings = vit_model.embeddings
        hidden_size = vit_model.config.hidden_size

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_var = nn.Linear(hidden_size, latent_dim)

        self.z_project = nn.Linear(latent_dim, hidden_size)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=7, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def extract_features(self, pixels):
        embedding_output = self.embeddings(pixels)
        encoder_output = self.encoder(embedding_output)
        return encoder_output.last_hidden_state

    def forward(self, messy_pixels, neat_pixels=None):
        messy_feats = self.extract_features(messy_pixels)

        mu, logvar = None, None

        if self.training and neat_pixels is not None:
            neat_feats = self.extract_features(neat_pixels)

            neat_pooled = neat_feats.mean(dim=1)

            mu = self.fc_mu(neat_pooled)
            logvar = self.fc_var(neat_pooled)

            z = self.reparameterize(mu, logvar)
        else:
            batch_size = messy_pixels.shape[0]
            z = torch.randn(batch_size, LATENT_DIM).to(messy_pixels.device)

        z_proj = self.z_project(z)

        combined_feats = messy_feats + z_proj.unsqueeze(1)

        batch_size = combined_feats.shape[0]
        num_patches_side = int((combined_feats.shape[1] - 1) ** 0.5)
        hidden_size = combined_feats.shape[2]

        decoder_input = combined_feats[:, 1:].permute(0, 2, 1).reshape(batch_size, hidden_size, num_patches_side, num_patches_side)

        return self.decoder(decoder_input), mu, logvar

class StochasticLoss(nn.Module):
    def __init__(self, weight=TOOL_WEIGHT, kl_weight=KL_WEIGHT):
        super().__init__()
        self.weight = weight
        self.kl_weight = kl_weight

    def forward(self, outputs, targets, mu, logvar):
        background_mask = (targets.mean(dim=1, keepdim=True) > 0.95).float()
        tool_mask = 1.0 - background_mask
        weights = 1.0 + tool_mask * (self.weight - 1.0)
        recon_loss = (torch.pow(outputs - targets, 2) * weights).mean()

        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / outputs.shape[0]
        else:
            kl_loss = torch.tensor(0.0).to(outputs.device)

        total_loss = recon_loss + (self.kl_weight * kl_loss)
        return total_loss, recon_loss, kl_loss

if __name__ == '__main__':
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group("nccl", device_id=local_rank)
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if local_rank == 0:
        all_neat_files = sorted([f for f in os.listdir(NEAT_DIR) if f.endswith('.png')])
        train_files, temp_files = train_test_split(all_neat_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    else:
        train_files, val_files, test_files = [], [], []

    dist.barrier()
    dist.broadcast_object_list(train_files, src=0)
    dist.broadcast_object_list(val_files, src=0)
    dist.broadcast_object_list(test_files, src=0)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = ImagePairDataset(MESSY_DIR, NEAT_DIR, image_files=train_files, transform=train_transform)
    val_dataset = ImagePairDataset(MESSY_DIR, NEAT_DIR, image_files=val_files, transform=val_test_transform)
    test_dataset = ImagePairDataset(MESSY_DIR, NEAT_DIR, image_files=test_files, transform=val_test_transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=train_sampler, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=val_sampler, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=test_sampler, shuffle=False, pin_memory=True)

    model = StochasticViT().to(device)
    model = DDP(model, device_ids=[local_rank])
    model = torch.compile(model)

    criterion = StochasticLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()
    best_val_loss = float('inf')

    if local_rank == 0:
        print(f"\nStarting Stochastic Training on {world_size} GPUs...")

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        running_train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", disable=(local_rank != 0))

        for messy_imgs, neat_imgs in train_progress_bar:
            messy_imgs, neat_imgs = messy_imgs.to(device, non_blocking=True), neat_imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs, mu, logvar = model(messy_imgs, neat_imgs)
                loss, recon, kl = criterion(outputs, neat_imgs, mu, logvar)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            if local_rank == 0:
                train_progress_bar.set_postfix({
                    'L': f"{loss.item():.3f}",
                    'Rec': f"{recon.item():.3f}",
                    'KL': f"{kl.item():.4f}"
                })

        model.eval()
        running_val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", disable=(local_rank != 0))

        with torch.no_grad():
            for messy_imgs, neat_imgs in val_progress_bar:
                messy_imgs, neat_imgs = messy_imgs.to(device, non_blocking=True), neat_imgs.to(device, non_blocking=True)

                with autocast():
                    outputs, mu, logvar = model(messy_imgs, neat_imgs)
                    loss, _, _ = criterion(outputs, neat_imgs, mu, logvar)

                running_val_loss += loss.item()
                if local_rank == 0:
                    val_progress_bar.set_postfix({'val_loss': running_val_loss / (val_progress_bar.n + 1)})

        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader)

        train_loss_tensor = torch.tensor(avg_train_loss).to(device)
        val_loss_tensor = torch.tensor(avg_val_loss).to(device)

        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

        avg_train_loss = train_loss_tensor.item() / world_size
        avg_val_loss = val_loss_tensor.item() / world_size

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        scheduler.step()

        if avg_val_loss < best_val_loss and local_rank == 0:
            best_val_loss = avg_val_loss
            torch.save(model.module.state_dict(), VIT_PATH)
            if local_rank == 0:
                print(f"New best model saved! ({best_val_loss:.4f})")

    if local_rank == 0:
        print("\nFinal Testing skipped (rely on inference script)...")

    dist.destroy_process_group()