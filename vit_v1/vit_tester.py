import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTModel

MODEL_NAME = 'knolling_vit.pth'
INPUT_DIR_NAME = 'test'
OUTPUT_DIR_NAME = INPUT_DIR_NAME
LATENT_DIM = 256
IMAGE_SIZE = 518
NUM_VARIATIONS = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME)
INPUT_DIR = os.path.join(SCRIPT_DIR, INPUT_DIR_NAME)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

if __name__ == '__main__':
    device = torch.device("cuda")
    print(f"Using device: {device}")

    model = StochasticViT().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if list(checkpoint.keys())[0].startswith('module.'):
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_files = [f for f in os.listdir(INPUT_DIR) if f.startswith('messy_') and f.endswith('.png')]

    print(f"\nFound {len(image_files)} test images. Generating {NUM_VARIATIONS} variations per image...")

    for filename in tqdm(image_files, desc="Generating"):
        input_path = os.path.join(INPUT_DIR, filename)

        messy_image = Image.open(input_path).convert("RGB")
        input_tensor = transform(messy_image).unsqueeze(0).to(device)

        for i in range(NUM_VARIATIONS):
            with torch.no_grad():
                output_tensor, _, _ = model(input_tensor)

            output_tensor = output_tensor.squeeze(0).cpu()
            output_tensor = output_tensor * 0.5 + 0.5
            output_tensor = torch.clamp(output_tensor, 0, 1)
            output_image = transforms.ToPILImage()(output_tensor)
            final_image = transforms.Resize((256, 256))(output_image)

            base_name = filename.replace('messy_', 'knolled_').replace('.png', '')
            output_filename = f"{base_name}_v{i}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            final_image.save(output_path)