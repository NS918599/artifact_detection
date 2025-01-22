import shutil
import tempfile
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import torch.nn.functional as F
from datetime import datetime
from pytorch_msssim import ssim
from AutoEncoder_pytorch_4 import CNNModel_2
from tqdm import tqdm
import numpy as np
from rasterio.windows import Window
from torch.utils.data import Dataset
import rasterio
import random
import cv2
from PIL import Image, ImageEnhance


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


class OrthoDataset_Improved(Dataset):
    def __init__(self, image_folders, patch_size=224):
        self.augmentations = {
            'rotate': self.rotate_image,
            'noise': self.add_noise,
            'brightness': self.change_brightness,
            'contrast': self.change_contrast,
            'saturation': self.change_saturation,
            'hue': self.change_hue,
            'none': lambda x: x
        }
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
            patches_in_current_image = ((w // self.patch_size) * (h // self.patch_size)) * len(self.augmentations)
            self.num_patches += patches_in_current_image

            self.index_mapping.update(
                {indx + j: (self.image_paths[i], k * self.patch_size, l * self.patch_size, augment)
                 for j, (k, l, augment) in enumerate([(k, l, augment)
                                                      for k in range(0, w // self.patch_size)
                                                      for l in range(0, h // self.patch_size)
                                                      for augment in self.augmentations.keys()])})
            indx += patches_in_current_image

    def __getitem__(self, index):
        image_path, w, h, augment = self.index_mapping.get(index)
        with rasterio.open(image_path) as src:
            window = Window(w, h, self.patch_size, self.patch_size)
            img = src.read(window=window, indexes=[1, 2, 3])
            augmentation_function = self.augmentations[augment]
            img = augmentation_function(img)
            return img, img

    def __len__(self):
        return self.num_patches

    def rotate_image(self, img):
        # Generate a random angle
        angle = random.uniform(0, 360)

        # Calculate the center of the image
        center = (self.patch_size // 2, self.patch_size // 2)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        rotated = cv2.warpAffine(np.moveaxis(img, 0, -1), M, (self.patch_size, self.patch_size))

        return np.moveaxis(rotated, -1, 0)

    def add_noise(self, img):
        ch, row, col = img.shape
        mean = 0
        sigma = 20
        gauss = np.random.normal(mean, sigma, (ch, row, col))
        gauss = gauss.reshape(ch, row, col)
        noisy = img + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def change_brightness(self, image):
        image = Image.fromarray(np.moveaxis(image, 0, -1))

        # Randomly change brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(np.random.uniform(0.5, 1.5))

        return np.moveaxis(np.array(image), -1, 0)

    def change_contrast(self, image):
        image = Image.fromarray(np.moveaxis(image, 0, -1))

        # Randomly change contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(np.random.uniform(0.5, 1.5))

        return np.moveaxis(np.array(image), -1, 0)

    def change_saturation(self, image):
        image = Image.fromarray(np.moveaxis(image, 0, -1))

        # Randomly change saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(np.random.uniform(0.5, 1.5))

        return np.moveaxis(np.array(image), -1, 0)

    def change_hue(self, image):
        image = np.array(Image.fromarray(np.moveaxis(image, 0, -1)).convert('HSV'))
        image[:, :, 0] = (image[:, :, 0].astype(int) + np.random.randint(-10, 10)) % 180
        image = Image.fromarray(image, mode='HSV').convert('RGB')

        return np.moveaxis(np.array(image), -1, 0)


def train_step(model, optimizer, dataloader, device, writer, epoch):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # batch = torch.from_numpy(batch[0]).float().to(device)
        batch = batch[0].float().to(device)
        batch = batch / 255

        if batch.ndim == 3:
            batch = batch.unsqueeze(1)

        image_size = batch.shape[2]
        if image_size % model.patch_size != 0:
            padding = model.patch_size - (image_size % model.patch_size)
            padded_batch = F.pad(batch, (0, padding, 0, padding), mode='reflect')
            batch = padded_batch
            image_size = batch.shape[2]

        # Forward pass
        reconstructed = model(batch)
        # loss = F.mse_loss(reconstructed, batch)
        loss = -ssim(reconstructed, batch, data_range=1.0)
        # loss = -ms_ssim(reconstructed, batch, data_range=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log to TensorBoard every few batches
        if batch_idx % 10 == 0:  # Log every 10 batches
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)
            # Log some example images
            if batch_idx == 0:
                writer.add_images('Input Images', batch[:6], global_step)
                writer.add_images('Reconstructed Images', reconstructed[:6], global_step)

    return total_loss / num_batches


def train(model, train_dataset, val_dataset, optimizer, device, epochs, batch_size, log_dir):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    writer = SummaryWriter(log_dir=log_dir)  # Initialize writer

    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_step(model, optimizer, train_loader, device, writer, epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0] / 255
                # batch = torch.from_numpy(batch).float().to(device)
                batch = batch.float().to(device)
                if batch.ndim == 3:
                    batch = batch.unsqueeze(1)
                image_size = batch.shape[2]
                if image_size % model.patch_size != 0:
                    padding = model.patch_size - (image_size % model.patch_size)
                    padded_batch = F.pad(batch, (0, padding, 0, padding), mode='reflect')
                    batch = padded_batch
                reconstructed = model(batch)
                # loss = F.mse_loss(reconstructed, batch)
                loss = -ssim(reconstructed, batch, data_range=1.0)
                # loss = -ms_ssim(reconstructed, batch, data_range=1.0)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        writer.add_scalar('Loss/validation', val_loss, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))

    writer.close()


# def main():
#     image_folders = [
#         'C:\\Users\\User\\defect_detection\\Data\\05\\140621',
#         'C:\\Users\\User\\defect_detection\\Data\\05\\160521',
#         'C:\\Users\\User\\defect_detection\\Data\\05\\180521',
#         'C:\\Users\\User\\defect_detection\\Data\\05\\190521',
#         'C:\\Users\\User\\defect_detection\\Data\\05\\21-270721',
#         'C:\\Users\\User\\defect_detection\\Data\\05\\211023',
#         'C:\\Users\\User\\defect_detection\\Data\\05\\220821'
#     ]
#
#     # image_folders = [r'C:\Users\User\defect_detection\AE_new\test_data\debug']
#     dataset = OrthoDataset(image_folders=image_folders, patch_size=224)
#     print(len(dataset))
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f'device = {device}')
#     # model = ViTAutoencoder().to(device)
#     model = CNNModel().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
#     batch_size = 100
#     epochs = 50
#
#     # Create log directory with timestamp
#     current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
#     log_dir = os.path.join("runs", current_time)
#     os.makedirs(log_dir, exist_ok=True)
#
#     train(model, train_dataset, val_dataset, optimizer, device, epochs, batch_size, log_dir)


def run_model(data_paths: list, model_class, batch_size: int = 100, epochs: int = 50, is_log: bool = True):
    dataset = OrthoDataset_Improved(image_folders=data_paths, patch_size=224)

    print(f'total data samples in dataset: {len(dataset)} image patches')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create log directory with timestamp
    if is_log:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", current_time)
        os.makedirs(log_dir, exist_ok=True)
    else:
        tmp_dir_name = tempfile.mkdtemp()
        log_dir = tmp_dir_name

    train(model, train_dataset, val_dataset, optimizer, device, epochs, batch_size, log_dir)

    if not is_log:
        shutil.rmtree(tmp_dir_name)


def main():
    run_model(
        data_paths=[
            'C:\\Users\\User\\defect_detection\\Data\\05\\140621',
            'C:\\Users\\User\\defect_detection\\Data\\05\\160521',
            'C:\\Users\\User\\defect_detection\\Data\\05\\180521',
            'C:\\Users\\User\\defect_detection\\Data\\05\\190521',
            'C:\\Users\\User\\defect_detection\\Data\\05\\21-270721',
            'C:\\Users\\User\\defect_detection\\Data\\05\\211023',
            'C:\\Users\\User\\defect_detection\\Data\\05\\220821'
        ],
        # data_paths=[r'C:\Users\User\defect_detection\AE_new\test_data\debug'],
        model_class=CNNModel_2,
        batch_size=2,
        epochs=100,
        is_log=True
    )


if __name__ == "__main__":
    main()
