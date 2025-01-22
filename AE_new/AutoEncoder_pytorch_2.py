import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import torch.nn as nn
from rasterio.windows import Window
from tqdm import tqdm


class OrthoDataset(Dataset):
    def __init__(self, image_folders, patch_size=224):
        self.image_paths = []
        self.patch_size = patch_size
        self.num_patches = 0
        self.index_mapping = {}

        # Collect image paths
        for folder in image_folders:
            folder_images = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.tif', '.tiff'))
            ]

            self.image_paths.extend(folder_images)

        # Shuffle image paths
        np.random.shuffle(self.image_paths)

        # count patches
        indx = 0
        for i, path in tqdm(enumerate(self.image_paths), total=len(self.image_paths)):
            with rasterio.open(path, 'r') as src:
                w, h = src.width, src.height
            patches_in_current_image = ((w // self.patch_size) * (h // self.patch_size))
            self.num_patches += patches_in_current_image

            self.index_mapping.update({indx + j: (self.image_paths[i], k * self.patch_size, l * self.patch_size)
                                       for j, (k, l) in enumerate([(k, l)
                                                                   for k in range(0, w // self.patch_size)
                                                                   for l in range(0, h // self.patch_size)])})
            indx += patches_in_current_image

    def __getitem__(self, index):
        image_path, w, h = self.index_mapping.get(index)
        with rasterio.open(image_path) as src:
            window = Window(w, h, self.patch_size, self.patch_size)
            img = src.read(window=window, indexes=[1, 2, 3])
            return img, img

    def __len__(self):
        return self.num_patches


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.proj(x)  # (batch_size, embed_dim, h', w')
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self._attention_block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _attention_block(self, x):
        x = x.transpose(0, 1)  # (n_patches, batch_size, embed_dim)
        x, _ = self.attn(x, x, x)
        x = x.transpose(0, 1)  # (batch_size, n_patches, embed_dim)
        return x


class ViTAutoencoder(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            mlp_ratio=4.0,
            dropout=0.1,
            latent_dim=512
    ):
        super().__init__()
        self.patch_size = patch_size

        # Encoder
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Latent space projection
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.latent_proj = nn.Linear(embed_dim * (img_size // patch_size) ** 2, latent_dim)

        # Decoder
        self.latent_upproj = nn.Linear(latent_dim, embed_dim * (img_size // patch_size) ** 2)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Final reconstruction
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim,
                in_channels,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Tanh()
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)

        # Project to latent space
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        latent = self.latent_proj(x)

        return latent

    def decode(self, latent):
        # Project from latent space
        batch_size = latent.shape[0]
        x = self.latent_upproj(latent)
        x = x.reshape(batch_size, -1, self.pos_embed.shape[-1])

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer decoder
        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)

        # Reshape for deconvolution
        h = w = int(x.shape[1] ** 0.5)
        x = x.transpose(1, 2).reshape(batch_size, -1, h, w)

        # Final reconstruction
        x = self.decoder_pred(x)

        return x

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def get_latent_dim(self):
        return self.latent_proj.out_features


