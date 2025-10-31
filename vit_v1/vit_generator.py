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
NUM_WORKERS = 16

IMAGE_SIZE = 518
TOOL_WEIGHT = 50
LR = 1e-4

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
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        messy_img_name = self.image_files[idx]
        messy_path = os.path.join(self.messy_dir, messy_img_name)

        neat_img_name = messy_img_name.replace('messy_', 'neat_')
        neat_path = os.path.join(self.neat_dir, neat_img_name)

        messy_image = Image.open(messy_path).convert("RGB")
        neat_image = Image.open(neat_path).convert("RGB")

        if self.transform:
            messy_image = self.transform(messy_image)
            neat_image = self.transform(neat_image)

        return messy_image, neat_image

class ViT(nn.Module):
    def __init__(self, vit_model_name='facebook/dinov2-large'):
        super().__init__()

        vit_model = ViTModel.from_pretrained(vit_model_name)

        self.encoder = vit_model.encoder
        self.embeddings = vit_model.embeddings
        hidden_size = vit_model.config.hidden_size

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

    def forward(self, pixel_values):
        embedding_output = self.embeddings(pixel_values)
        encoder_output = self.encoder(embedding_output)
        sequence_output = encoder_output.last_hidden_state

        batch_size = sequence_output.shape[0]
        num_patches_side = int((sequence_output.shape[1] - 1) ** 0.5)
        hidden_size = sequence_output.shape[2]
        decoder_input = sequence_output[:, 1:].permute(0, 2, 1).reshape(batch_size, hidden_size, num_patches_side, num_patches_side)
        
        return self.decoder(decoder_input)

class WeightedL1Loss(nn.Module):
    def __init__(self, weight=TOOL_WEIGHT):
        super().__init__()
        self.weight = weight

    def forward(self, outputs, targets):
        mask = (targets.mean(dim=1, keepdim=True) < 0.99).float()
        
        weights = 1.0 + mask * (self.weight - 1.0)
        
        loss = torch.abs(outputs - targets) * weights
        
        return loss.mean()

if __name__ == '__main__':
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group("nccl", device_id=local_rank)
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if local_rank == 0:
        all_files = sorted([f for f in os.listdir(MESSY_DIR) if os.path.isfile(os.path.join(MESSY_DIR, f))])
        train_files, temp_files = train_test_split(all_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    else:
        train_files, val_files, test_files = [], [], []

    dist.barrier()

    dist.broadcast_object_list(train_files, src=0)
    dist.broadcast_object_list(val_files, src=0)
    dist.broadcast_object_list(test_files, src=0)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = ImagePairDataset(MESSY_DIR, NEAT_DIR, image_files=train_files, transform=transform)
    val_dataset = ImagePairDataset(MESSY_DIR, NEAT_DIR, image_files=val_files, transform=transform)
    test_dataset = ImagePairDataset(MESSY_DIR, NEAT_DIR, image_files=test_files, transform=transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=train_sampler, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=val_sampler, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=test_sampler, shuffle=False, pin_memory=True)

    model = ViT().to(device)
    model = DDP(model, device_ids=[local_rank])
    model = torch.compile(model)
    criterion = WeightedL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()
    best_val_loss = float('inf')

    if local_rank == 0:
        print(f"\nStarting training on {world_size} GPUs...")

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        running_train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", disable=(local_rank != 0))

        for messy_imgs, neat_imgs in train_progress_bar:
            messy_imgs, neat_imgs = messy_imgs.to(device, non_blocking=True), neat_imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(messy_imgs)
                loss = criterion(outputs, neat_imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            if local_rank == 0:
                train_progress_bar.set_postfix({'train_loss': running_train_loss / (train_progress_bar.n + 1)})

        model.eval()
        running_val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", disable=(local_rank != 0))

        with torch.no_grad():
            for messy_imgs, neat_imgs in val_progress_bar:
                messy_imgs, neat_imgs = messy_imgs.to(device, non_blocking=True), neat_imgs.to(device, non_blocking=True)

                with autocast():
                    outputs = model(messy_imgs)
                    loss = criterion(outputs, neat_imgs)

                running_val_loss += loss.item()
                if local_rank == 0:
                    val_progress_bar.set_postfix({'val_loss': running_val_loss / (val_progress_bar.n + 1)})

        avg_train_loss_proc = running_train_loss / len(train_loader)
        avg_val_loss_proc = running_val_loss / len(val_loader)

        train_loss_tensor = torch.tensor(avg_train_loss_proc).to(device)
        val_loss_tensor = torch.tensor(avg_val_loss_proc).to(device)

        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

        avg_train_loss = train_loss_tensor.item() / world_size
        avg_val_loss = val_loss_tensor.item() / world_size

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        if avg_val_loss < best_val_loss and local_rank == 0:
            best_val_loss = avg_val_loss
            torch.save(model.module.state_dict(), VIT_PATH)
            if local_rank == 0:
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    if local_rank == 0:
        print("\nLoading best model for final testing...")

    dist.barrier()
    model.module.load_state_dict(torch.load(VIT_PATH, map_location=device))
    model.eval()
    test_loss = 0.0
    test_progress_bar = tqdm(test_loader, desc="[Final Test]", disable=(local_rank != 0))

    with torch.no_grad():
        for messy_imgs, neat_imgs in test_progress_bar:
            messy_imgs, neat_imgs = messy_imgs.to(device, non_blocking=True), neat_imgs.to(device, non_blocking=True)

            with autocast():
                outputs = model(messy_imgs)
                loss = criterion(outputs, neat_imgs)

            test_loss += loss.item()

    avg_test_loss_proc = test_loss / len(test_loader)
    test_loss_tensor = torch.tensor(avg_test_loss_proc).to(device)
    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    avg_test_loss = test_loss_tensor.item() / world_size

    if local_rank == 0:
        print(f"Final performance on the test set -> Test Loss: {avg_test_loss:.4f}")

    dist.destroy_process_group()
