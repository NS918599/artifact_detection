import math
import torch.nn as nn
import torch.nn.functional as F
import torch


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         print('test')
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


# class PositionalEncoding2D(nn.Module):
#     def __init__(self, d_model, dropout=0.1, height=64, width=64):
#         super(PositionalEncoding2D, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(d_model, height, width)
#         y_position = torch.arange(0, height, dtype=torch.float).unsqueeze(1).unsqueeze(1)
#         x_position = torch.arange(0, width, dtype=torch.float).unsqueeze(0).unsqueeze(2)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#
#         pe[0::2, :, :] = torch.sin(y_position * div_term).transpose(0, 1)
#         pe[1::2, :, :] = torch.cos(x_position * div_term).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe.unsqueeze(0)
#         return self.dropout(x)
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, n_heads, d_model, d_k, d_v):
#         super(MultiHeadAttention, self).__init__()
#         self.n_heads = n_heads
#         self.d_k = d_k
#         self.d_v = d_v
#
#         self.W_Q = nn.Linear(d_model, n_heads * d_k)
#         self.W_K = nn.Linear(d_model, n_heads * d_k)
#         self.W_V = nn.Linear(d_model, n_heads * d_v)
#
#         self.fc = nn.Linear(n_heads * d_v, d_model)
#
#     def forward(self, input_Q, input_K, input_V):
#         batch_size, len_Q, _ = input_Q.size()
#         batch_size, len_K, _ = input_K.size()
#         batch_size, len_V, _ = input_V.size()
#
#         # 1. Dot product attention
#         Q = self.W_Q(input_Q).view(batch_size, len_Q, self.n_heads, self.d_k).permute(0, 2, 1, 3)
#         K = self.W_K(input_K).view(batch_size, len_K, self.n_heads, self.d_k).permute(0, 2, 1, 3)
#         V = self.W_V(input_V).view(batch_size, len_V, self.n_heads, self.d_v).permute(0, 2, 1, 3)
#
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
#         scores = F.softmax(scores, dim=-1)
#         context = torch.matmul(scores, V)
#
#         context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, len_Q, -1)
#         output = self.fc(context)
#         return output

class PatchBasedAttentionBlock(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads):
        super(PatchBasedAttentionBlock, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # Assuming x is a tensor of shape (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape

        # Divide the image into patches
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        x = x.view(batch_size, channels, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        x = x.permute(0, 2, 3, 4, 5,
                      1).contiguous()  # Reshape to (batch_size, num_patches_h, num_patches_w, patch_size, patch_size, channels)
        x = x.view(batch_size, num_patches_h * num_patches_w, -1)  # Flatten patches

        # Apply self-attention
        output, _ = self.attention(x, x, x)

        return output

#
# class Attention(nn.Module):
#     def __init__(self, image_shape, num_heads):
#         super(Attention, self).__init__()
#         self.image_shape = image_shape
#         self.input_size = self.image_shape[1] * self.image_shape[2]
#         self.attention = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=num_heads, batch_first=True)
#
#     def forward(self, x):
#         x = x.flatten(start_dim=2)
#         x = self.attention(x, x, x)
#         x = x.reshape(x.shape[0], *self.image_shape)
#         return x


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Inputs:
            x: input feature maps(B X C X W X H)
        Returns:
            out: self attention value + input feature
            attention: attention map
        """
        batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(x).view(batchsize,-1,width*height) # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(batchsize,C,width,height)
        out = self.gamma*out + x
        return out


class CNNModel_2(nn.Module):
    def __init__(self, n_heads=6, d_model=128, d_k=32, d_v=32):
        super(CNNModel_2, self).__init__()

        self.patch_size = 224

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_drop1 = nn.Dropout(p=0.05)

        self.enc_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_drop2 = nn.Dropout(p=0.05)
        # self.enc_attn1 = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.enc_attn1 = SelfAttention(in_dim=32)
        # self.enc_pos_enc1 = PositionalEncoding2D(d_model, width=self.patch_size // 4, height=self.patch_size // 4)

        self.enc_conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(16)
        self.enc_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_drop3 = nn.Dropout(p=0.05)

        self.enc_conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.enc_drop4 = nn.Dropout(p=0.05)
        # self.enc_attn2 = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.enc_attn2 = SelfAttention(in_dim=16)

        # self.enc_pos_enc2 = PositionalEncoding2D(d_model, width=self.patch_size // 8, height=self.patch_size // 8)

        # Decoder layers
        self.dec_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        # self.dec_attn1 = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.dec_attn1 = SelfAttention(in_dim=32)
        # self.dec_pos_enc1 = PositionalEncoding2D(d_model, width=self.patch_size // 4, height=self.patch_size // 4)
        self.dec_drop1 = nn.Dropout(p=0.05)

        self.dec_conv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        # self.dec_attn2 = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.dec_attn2 = SelfAttention(in_dim=64)
        # self.dec_pos_enc2 = PositionalEncoding(d_model)
        self.dec_drop2 = nn.Dropout(p=0.05)

        self.dec_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn3 = nn.BatchNorm2d(3)
        # self.dec_attn3 = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        # self.dec_attn3 = SelfAttention(in_dim=3)

        # self.dec_pos_enc3 = PositionalEncoding2D(d_model, width=self.patch_size, height=self.patch_size)
        self.dec_drop3 = nn.Dropout(p=0.05)

    def encode(self, x):
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_pool1(x)
        x = self.enc_drop1(x)

        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_pool2(x)
        x = self.enc_drop2(x)
        # x = self.enc_pos_enc1(x)
        x = self.enc_attn1(x)

        x = F.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_pool3(x)
        x = self.enc_drop3(x)

        x = F.relu(self.enc_conv4(x))
        x = self.enc_drop4(x)
        # x = self.enc_pos_enc2(x)
        x = self.enc_attn2(x)

        return x

    def decode(self, x):
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        # x = self.dec_pos_enc1(x)
        x = self.dec_attn1(x)
        x = self.dec_drop1(x)
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))
        # x = self.dec_pos_enc2(x)
        x = self.dec_attn2(x)
        x = self.dec_drop2(x)

        x = F.relu(self.dec_bn3(self.dec_conv3(x)))
        # x = self.dec_pos_enc3(x)
        # x = self.dec_attn3(x)
        x = self.dec_drop3(x)

        return x

    def forward(self, x):
        latent = self.encode(x)
        decoded = self.decode(latent)
        return decoded


if __name__ == '__main__':
    model = CNNModel_2()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    #
    # model = CNNModel_2()
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(pytorch_total_params)

    # model = CNNModel_2()
    # print(list(model.named_parameters()))
    #
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter: {name}, Shape: {param.shape}, params: {param.numel()}")
    #
    # img = torch.randint(0, 255, size=(64, 224, 224)).unsqueeze(0).to('cuda')
    # img = img / 255
    #
    # # model = Attention(image_shape=(64, 224, 224), num_heads=8).to('cuda')
    # model = SelfAttention(in_dim=64).to('cuda')
    #
    # img_a, at = model(img)
    #
    # print(img_a.shape)
