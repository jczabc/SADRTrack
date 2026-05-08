import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import Levit
from timm.models.levit import levit_384
from torchvision import models
from lib.models.layers.patch_embed import PatchEmbed
from lib.models.hip.resnet import resnet18
from lib.models.hip import cbam
from lib.models.hiptrack.levit import LeViT

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=384, kernel_size=1, stride=1, padding=0)
    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)#[4,1024,24,24]
        # x = self.block1(x)
        # r = self.attention(x)
        # x = self.block2(x + r)
        x= self.conv(x)

        return x

class HistoricalPromptEncoder(nn.Module):
    def __init__(self,img_size=384, patch_size=16,embed_layer=PatchEmbed,in_chans=4,embed_dim=320,levit_model=None):
        super().__init__()

        self.levit = levit_model
        # 提取 LeViT 前四层

        self.conv = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv1d = nn.Conv1d(in_channels=576, out_channels=384, kernel_size=1)
        self.conv1d1 = nn.Conv1d(in_channels=384, out_channels=576, kernel_size=1)
        resnet = resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool
        self.le = self.levit
        self.fb_idx = self.levit.fb_idx
    #    self.neck_type == self.levit.neck_type
        self.blocks=self.levit.blocks
        self.num_x = self.levit.num_patches_search#16
        self.num_layers = 2
        BN = True
        st = [2,2]
        backbone_embed_dim = self.levit.embed_dim_list
        if BN: # 如果使用批量归一化 (BN)，则每一层都加上 BatchNorm2d
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)), nn.BatchNorm2d(k)) # 批量归一化层#upsample
                                        for n, k,s in zip(backbone_embed_dim[::-1][0:-1], backbone_embed_dim[::-1][1:], st))# 反卷积层 (上采样)
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)))
                                        for n, k,s in zip(backbone_embed_dim[::-1][0:-1], backbone_embed_dim[::-1][1:], st))


    #    self.le = self.levit.blocks[0:8]
        # self.layer1 = resnet.layer1 # 1/4, 64
        #
        # self.layer2 = resnet.layer2 # 1/8, 128
        # self.layer3 = resnet.layer3 # 1/16, 256
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed1 = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=4, embed_dim=384)
        self.patch_embed2 = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=384)
        num_patches = self.patch_embed.num_patches
        self.fuser = FeatureFusionBlock(768 + 256, 384)

    def forward(self, image,search, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)#torch.Size([4, 3, 384, 384])
       # f = F.interpolate(f, size=(256, 256), mode='bilinear', align_corners=False)
        f_patch = self.patch_embed1(f)# 转换为 LeViT 的 patch 输入格式torch.Size([4, 576, 384])   BNC
        s_patch = self.patch_embed2(search)
        patch_concat = torch.cat([f_patch, s_patch], dim=1)#torch.Size([4, 1152, 384])

        # f_patch = f_patch.flatten(2)  # 转换为 [B, N_patches, embed_dim]
        #
        # f_patch = f_patch.transpose(1, 2)#torch.Size([4, 320, 576])
        # 通过 LeViT 的前四层提取特征
        if self.levit.neck_type == 'FB' or self.levit.neck_type == "MAXF" or self.levit.neck_type == "MAXMINF" or self.levit.neck_type == "MAXMIDF" or self.levit.neck_type == "MINMIDF" or self.levit.neck_type == 'MIDF':
            assert len(self.fb_idx) == 2
            xz1 = self.blocks[0:self.fb_idx[0]](patch_concat)#torch.Size([4, 1152, 384])
            xz2 = self.blocks[self.fb_idx[0]:self.fb_idx[1]](xz1)#torch.Size([4, 288, 512])
            xz = self.blocks[self.fb_idx[1]:](xz2)#torch.Size([4, 72, 768])
            out_list = [xz1, xz2]
        else:
            xz = self.blocks(patch_concat) #[bs, 20, 768]
            out_list = []

        cls = xz.mean(1).unsqueeze(1) #[bs, 1, 768]
        cxz = torch.cat((cls, xz), dim=1)
        out_list.append(cxz)
      #  global_vector = out_list[-1][:, 0:1, :].permute(1, 0,
       #                                                2)  # global vector# 处理输入的 xz_list，最后一个元素是全局向量 # 提取全局向量并调整维度
        fb_features = []  # 用于存储特征图的列表
        for i in range(len(out_list)):  # 遍历 xz_list，处理每一个元素
            if i == len(out_list) - 1:  # 对于最后一个元素，提取除去第一个元素外的特征#torch.Size([4, 73, 768])
                x = out_list[i][:, 1:self.num_x + 1, :]
                B, N, C = x.shape  # 获取 batch size, 特征数, 通道数
                Len = int(N ** 0.5)  # 假设输入 N 的形状是平方数，求出边长
                x = x.permute(0, 2, 1).view(B, C, Len, Len)  # 调整维度为 (B, C, Len, Len)
                fb_features.append(x)  # 将该特征图加入列表
            else:  # 对于其他元素，进行相似的处理
                x = out_list[i][:, 0:self.num_x * (4 ** (len(out_list) - 1 - i)), :]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)  # 将特征图加入列表
        x = fb_features[-1]  # 获取最后一个特征图作为输入
        for i, layer in enumerate(self.layers):  # 通过各层的反卷积进行上采样
            x = layer(x)  # 经过当前层
            x = x + fb_features[-2 - i]  # 与对应的上一层特征图进行残差加法
            if i < self.num_layers - 1:#torch.Size([4, 384, 24, 24])
                x = F.relu(x)  # 除最后一层外，其他层加 ReLU 激活函数------------------------------------------------------>END

        # s_features = []
        # for i in range(len(out_list)):  # 遍历 xz_list，处理每一个元素
        #     if i == len(out_list) - 1:  # 对于最后一个元素，提取除去第一个元素外的特征#torch.Size([4, 73, 768])
        #         s = out_list[i][:, self.num_x + 1:, :]
        #         B, N, C = s.shape  # 获取 batch size, 特征数, 通道数
        #         Len = int(N ** 0.5)  # 假设输入 N 的形状是平方数，求出边长
        #         s = s.permute(0, 2, 1).view(B, C, Len, Len)  # 调整维度为 (B, C, Len, Len)
        #         s_features.append(s)  # 将该特征图加入列表
        #     else:  # 对于其他元素，进行相似的处理
        #         s = out_list[i][:, self.num_x * (4 ** (len(out_list) - 1 - i)):, :]
        #         B, N, C = s.shape
        #         Len = int(N ** 0.5)
        #         s = s.permute(0, 2, 1).view(B, C, Len, Len)
        #         s_features.append(s)  # 将特征图加入列表
        # s = fb_features[-1]  # 获取最后一个特征图作为输入
        # for i, layer in enumerate(self.layers):  # 通过各层的反卷积进行上采样
        #     s = layer(s)  # 经过当前层
        #     s = s + s_features[-2 - i]  # 与对应的上一层特征图进行残差加法
        #     if i < self.num_layers - 1:#torch.Size([4, 384, 24, 24])
        #         s = F.relu(s)  # 除最后一层外，其他层加 ReLU 激活函数------------------------------------------------------>END
        #
        # f_out=torch.cat([x, s], dim=1)



    #    x = self.le(f_patch)  # 经过 LeViT 的前四层
    #    x = x.transpose(1, 2)#torch.Size([4, 384, 256])


        # if self.training:
        #     x = x.reshape(4, 384, 16, 16)
        # else:
        #     x = x.reshape(1, 384, 16, 16)

        # x = self.conv(x)#320 256[4,256,24,24]
   #     x = F.interpolate(x, size=(24, 24), mode='bilinear', align_corners=False)

        # x = self.conv1(f)
        # x = self.bn1(x)
        # x = self.relu(x)   # 1/2, 64
        # x = self.maxpool(x)  # 1/4, 64
        # x = self.layer1(x)   # 1/4, 64
        # x = self.layer2(x) # 1/8, 128
        # x = self.layer3(x) # 1/16, 256
        # x = self.fuser(x, key_f16)#[B,256,24,24]

        return x#torch.Size([4, 768, 24, 24])

class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)
