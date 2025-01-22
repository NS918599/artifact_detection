import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torchvision.models import vit_b_16


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.patch_size = 224

    def encode(self, x):
        x = F.relu(nn.Conv2d(3, 64, kernel_size=3, padding=1)(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(x)

        x = F.relu(nn.Conv2d(64, 32, kernel_size=3, padding=1)(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(x)

        x = F.relu(nn.Conv2d(32, 16, kernel_size=3, padding=1)(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(x)

        x = F.relu(nn.Conv2d(16, 16, kernel_size=3, padding=1)(x))

        return x

    def decode(self, x):
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = F.relu(nn.Conv2d(16, 32, kernel_size=3, padding=1)(x))

        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = F.relu(nn.Conv2d(32, 64, kernel_size=3, padding=1)(x))

        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = F.relu(nn.Conv2d(64, 3, kernel_size=3, padding=1)(x))

        return x

    def forward(self, x):
        latent = self.encode(x)
        decoded = self.decode(latent)
        return decoded


class SatelliteImageAE_ViT_DoublePretrained(nn.Module):
    def __init__(self, latent_dim=128):
        super(SatelliteImageAE_ViT_DoublePretrained, self).__init__()

        self.patch_size = 224

        # Load pre-trained ViT-B/16
        self.encoder = vit_b_16(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Modify the last layer of the encoder to match the desired latent dimension
        encoder_output_features = self.encoder.heads.head.in_features
        self.encoder.heads = nn.Linear(encoder_output_features, latent_dim)

        # Load another pre-trained ViT-B/16 for the decoder
        self.decoder_vit = vit_b_16(pretrained=True)
        for param in self.decoder_vit.parameters():
            param.requires_grad = False
        # Remove the classification head from the decoder ViT
        self.decoder_vit.heads = nn.Identity()

        # Final output layer
        self.output_layer = nn.Conv2d(768, 3, kernel_size=1)

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)

        # Decode with pre-trained ViT
        decoded_features = self.decoder_vit(encoded)

        # Reshape for the convolutional output layer
        decoded_features = decoded_features.permute(0, 2, 3, 1).reshape(x.shape[0], 768, 256, 256)

        # Final output
        decoded = self.output_layer(decoded_features)

        return decoded


class ResNet_VAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        self.patch_size = 224
        super(ResNet_VAE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)

        return x_reconst  # , z, mu, logvar


if __name__ == '__main__':
    model = CNNModel()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
