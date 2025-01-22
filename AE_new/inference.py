import os
from tqdm import tqdm
from AutoEncoder_pytorch_2 import ViTAutoencoder
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf
from rasterio.windows import Window
from pytorch_msssim import ssim, ms_ssim
from shapely import Polygon
import geopandas as gpd
from glob import glob
from AutoEncoder_pytorch_3 import CNNModel

def score_images(folder_path: str, tiles_path: str):
    pass


def split_image(path: str, tile_size=224):
    patches = []
    polygons = []
    with rasterio.open(path) as src:
        w, h = src.width, src.height
        pixel_size_x, _, x_0, _, pixel_size_y, y_0, _, _, _ = src.transform
        for k in range(0, w // tile_size):
            for l in range(0, h // tile_size):
                window = Window(k * tile_size, l * tile_size, tile_size, tile_size)
                img = src.read(window=window, indexes=[1, 2, 3])
                polygon = Polygon([(x_0 + pixel_size_x*k*tile_size, y_0 + pixel_size_y*l*tile_size),
                                   (x_0 + pixel_size_x*k*tile_size, y_0 + pixel_size_y*(l+1)*tile_size),
                                   (x_0 + pixel_size_x*(k+1)*tile_size, y_0 + pixel_size_y*(l+1)*tile_size),
                                   (x_0 + pixel_size_x*(k+1)*tile_size, y_0 + pixel_size_y*l*tile_size)])
                polygons.append(polygon)
                patches.append(img)
    return np.array(patches), np.array(polygons)


def split_image_into_tiles(image_tensor, tile_size=256):
    """
    Splits an image tensor into tiles of specified size.

    Args:
        image_tensor (tf.Tensor): The input image tensor.
        tile_size (int): The size of each tile (default is 256).

    Returns:
        tf.Tensor: A tensor containing the image tiles.
    """
    # Get the dimensions of the image tensor
    height, width, channels = image_tensor.shape

    # Calculate the number of tiles along height and width
    num_tiles_height = height // tile_size
    num_tiles_width = width // tile_size

    # Split the image tensor into tiles
    tiles = tf.image.extract_patches(
        images=tf.expand_dims(image_tensor, 0),
        sizes=[1, tile_size, tile_size, 1],
        strides=[1, tile_size, tile_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    # Reshape the tiles tensor to the desired shape
    tiles = tf.reshape(tiles, (num_tiles_height, num_tiles_width, tile_size, tile_size, channels))

    return tiles


# def split_tensor_into_tiles(tensor, tile_size):
#
#     # Split the tensor into tiles
#     tiles = tensor.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
#
#     # Reshape the tiles to have each tile as a separate tensor
#     tiles = tiles.contiguous().view(-1, 3, tile_size, tile_size)
#
#     return tiles


def rebuild_image_from_tiles(tiles, image_shape, tile_size):
    # Calculate the number of tiles along height and width
    num_tiles_height = image_shape[1] // tile_size
    num_tiles_width = image_shape[2] // tile_size

    # Reshape the tiles to match the original image dimensions
    tiles = tiles.view(num_tiles_height, num_tiles_width, 3, tile_size, tile_size)

    # Permute the dimensions to match the original image dimensions
    tiles = tiles.permute(2, 0, 3, 1, 4).contiguous()

    # Reshape the tiles to form the original image
    image = tiles.view(3, num_tiles_height * tile_size, num_tiles_width * tile_size)

    return image


def class_images(folder_path: str, output_file: str, model_path: str, model_class):
    total_polygon, total_errors = [], []

    model = model_class().to('cuda')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print('loaded model')

    images = glob(os.path.join(folder_path, '*.tif'))
    for img_path in tqdm(images):
        patches, polygons = split_image(img_path, tile_size=224)

        patches_tensor = torch.from_numpy(patches / 255).float().to('cuda')

        reco = model(patches_tensor)
        error = ms_ssim(reco, patches_tensor, data_range=1.0, size_average=False)

        total_polygon.extend(polygons)
        total_errors.extend([error[i].item() for i in range(len(error))])

    print('run model')
    df = gpd.GeoDataFrame({'geometry': total_polygon, 'error': total_errors})
    df.set_crs(epsg=32636, inplace=True)
    df.to_file(output_file)


def main2():
    class_images(
                folder_path=r'C:\Users\User\defect_detection\telem_27102023_part1_sample_05',
                output_file=r'C:\Users\User\defect_detection\telem_27102023_part1_sample_05\result_cnn.geojson',
                model_path=r'C:\Users\User\defect_detection\AE_new\runs\20250119-095603\best_model.pth',
                model_class=CNNModel
                 )


def main():
    ck_path = r'C:\Users\User\defect_detection\AE_new\runs\20250119-095603\best_model.pth'

    model = CNNModel().to('cuda')
    model.load_state_dict(torch.load(ck_path, weights_only=True))
    model.eval()

    print('loaded model')

    img_path = r"C:\Users\User\defect_detection\AE_new\test_data\test-0.5\Ortho_+003_+011.tif"

    patches, polygons = split_image(img_path, tile_size=224)

    patches_tensor = torch.from_numpy(patches / 255).float().to('cuda')

    print('loaded img')

    reco = model(patches_tensor)
    reco_numpy = reco.to('cpu').detach().numpy()

    error = ssim(reco, patches_tensor, data_range=1.0, size_average=False)

    print('run model')

    fig, ax = plt.subplots(4, 8)
    for i in range(16):
        ax[(i // 8) * 2][i % 8].imshow(np.moveaxis(patches[i], 0, -1))
        ax[(i // 8) * 2][i % 8].set_title(round(error[i].item(), 3))
        ax[(i // 8) * 2 + 1][i % 8].imshow(np.moveaxis(reco_numpy[i], 0, -1))
        print(i)
    plt.show()


if __name__ == '__main__':
    # main()
    main2()
