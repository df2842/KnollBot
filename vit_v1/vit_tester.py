import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTModel

MODEL_NAME = 'knolling_vit.pth'
INPUT_DIR_NAME = 'test'
OUTPUT_DIR_NAME = 'test'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME)
INPUT_DIR = os.path.join(SCRIPT_DIR, INPUT_DIR_NAME)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR_NAME)

IMAGE_SIZE = 518

os.makedirs(OUTPUT_DIR, exist_ok=True)
 
class ViT(nn.Module):
    def __init__(self, vit_model_name='facebook/dinov2-large'):
        super().__init__()

        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.encoder = self.vit.encoder
        self.embeddings = self.vit.embeddings
        hidden_size = self.vit.config.hidden_size

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

if __name__ == '__main__':
    device = torch.device("cuda")

    model = ViT().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_files = [f for f in os.listdir(INPUT_DIR) if f.startswith('messy_') and f.endswith('.png')]

    print(f"\nGenerating Images...")

    for filename in tqdm(image_files, desc="Generating neat images"):
        input_path = os.path.join(INPUT_DIR, filename)
        
        messy_image = Image.open(input_path).convert("RGB")
        input_tensor = transform(messy_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor = output_tensor.squeeze(0).cpu()
        output_tensor = output_tensor * 0.5 + 0.5
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_image = transforms.ToPILImage()(output_tensor)

        final_image = transforms.Resize((256, 256))(output_image)

        output_filename = filename.replace('messy_', 'knolled_')
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        final_image.save(output_path)