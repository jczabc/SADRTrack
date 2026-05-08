import torch.nn as nn
import torch

class DualAttention(nn.Module):
    """A Dual Attention block for Dual Attention Unit"""

    def __init__(self, d_model, kernel_size=16, attn_shortcut=True):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.spatial_gating_unit = DualAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class DualAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size=16, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1  # 用于确定实际卷积操作时的有效核大小（考虑了空洞等情况）
        d_p = (d_k - 1) // 2  # 普通深度可分离卷积（conv0）的填充（padding）参数，目的是保持特征图尺寸在卷积前后不变
        # 奇数尺寸的卷积核，合适的填充能保证输入输出尺寸一致
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        # 确定深度可分离空洞卷积（conv_spatial）的实际卷积核尺寸
        dd_p = (dilation * (dd_k - 1) // 2)  # 深度可分离空洞卷积（conv_spatial）的填充参数

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        # 深度可分离卷积（每个输入通道单独用一个卷积核卷积，不进行通道间的混合，降低了计算量和参数量）
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        # 带空洞的深度可分离卷积，通过空洞卷积可以在不增加过多计算量的情况下扩大感受野
        self.conv1 = nn.Conv2d(dim, dim, 1)
        # 对前面经过卷积操作后的特征图进行通道融合或者进一步的特征变换，将多通道的特征进行整合等操作。
        self.reduction = max(dim // reduction, 3)  # 通道压缩的比例
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层
        self.fc = nn.Sequential(  # 全连接层
            nn.Linear(dim, dim // self.reduction, bias=False),  # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False),  # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)  # depth-wise conv
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution
        f_x = self.conv1(attn)  # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)

      #  print(torch.cuda.is_available())

        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u