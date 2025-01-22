import torch
from clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


def clipseg():

    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load(r"C:\Users\User\defect_detection\AE_new\playgroud\clipseg_weights\rd64-uni-refined.pth",
                                     map_location=torch.device('cpu')), strict=False)

    # load and normalize image
    input_image = Image.open(r"C:\Users\User\defect_detection\telem_27102023_part1_sample_05\Ortho_+003_+011.tif")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    img = transform(input_image).unsqueeze(0)

    prompts = ['cloud', 'not a cloud']

    # predict
    with torch.no_grad():
        preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]

    # visualize prediction
    _, ax = plt.subplots(1, len(prompts) + 2, figsize=(15, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(input_image)
    [ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))]
    [ax[i + 1].text(0, -15, prompts[i]) for i in range(len(prompts))]
    ax[len(prompts) + 1].imshow((torch.sigmoid(preds[0][0]) > torch.sigmoid(preds[1][0])).char().numpy())

    plt.show()


def main():
    clipseg()


if __name__ == '__main__':
    main()