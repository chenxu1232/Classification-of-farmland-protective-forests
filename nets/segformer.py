# # 8通道整体
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保这里能导入你上一步改好的 backbone
from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

# ==============================================================================
# 1. Coordinate Attention & Detail Extractor
#    作用：针对农田防护林“细长、线性”的特征，强化垂直/水平方向的空间感知。
# ==============================================================================
class CoordAtt(nn.Module):
    """
    Coordinate Attention (Hou et al., CVPR 2021)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        # 限制 mip 最小为 8，防止通道压缩过猛丢失信息
        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        B, C, H, W = x.size()

        # X Avg Pool -> (B, C, H, 1)
        x_h = x.mean(dim=3, keepdim=True)
        # Y Avg Pool -> (B, C, 1, W) -> (B, C, W, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        # Concat -> (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        # Sigmoid -> Attention Map
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))

        return x * a_h * a_w

class CADetailExtractor(nn.Module):
    """
    结合了连续卷积和坐标注意力的细节提取器
    """
    def __init__(self, in_ch, out_ch, reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ca = CoordAtt(out_ch, reduction=reduction)

    def forward(self, x):
        feat = self.conv(x)   # 提取高频纹理
        feat = self.ca(feat)  # 增强线性结构
        return feat

# ==============================================================================
# 2. Frequency Guided Cross-Scale Attention (FG_CSA)
#    作用：利用 FSA 提取的频域权重（反映物候特征）来门控特征融合。
# ==============================================================================
class FG_CSA(nn.Module):
    def __init__(self, low_ch, high_ch, mid_ch, only_small_to_large=True):
        super().__init__()
        self.proj_low = nn.Conv2d(low_ch, mid_ch, 1)
        self.proj_high = nn.Conv2d(high_ch, mid_ch, 1)
        self.proj_v = nn.Conv2d(high_ch, mid_ch, 1)
        
        # 边界门控机制
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(low_ch + high_ch, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(mid_ch, low_ch, 1)
        self.only_small_to_large = only_small_to_large

    def forward(self, low_feat, high_feat, freq_map=None):
        """
        low_feat: 较浅层特征 (e.g., p3)
        high_feat: 较深层特征 (e.g., p4)
        freq_map: (B, C, 1, 1) 来自 FSA 的频域权重
        """
        B, Cl, H, W = low_feat.shape
        
        # 1. 频域引导 (Frequency Gating)
        # 如果有 freq_map，先对高层语义特征进行重加权
        # 这步很关键：利用频域信息抑制那些“看起来像树其实是玉米”的特征
        if freq_map is not None:
            # freq_map 已经在 Head 里投影到了 high_ch 维度
            high_feat = high_feat * freq_map 

        # 2. Cross-Attention
        # Q: Low
        L = self.proj_low(low_feat)
        q = L.flatten(2).transpose(1, 2) # (B, Nl, mid)

        # K, V: High (保持原分辨率计算，节省显存)
        Kf = self.proj_high(high_feat)
        Vf = self.proj_v(high_feat)
        k = Kf.flatten(2).transpose(1, 2)
        v = Vf.flatten(2).transpose(1, 2)

        # Attention Map
        attn = torch.matmul(q, k.transpose(1, 2)) * (mid_ch ** -0.5 if 'mid_ch' in locals() else (k.size(-1)**-0.5))
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, H, W)

        # 3. Boundary Gating Fusion
        high_up = F.interpolate(high_feat, size=(H, W), mode='bilinear', align_corners=False)
        gate_ctx = torch.cat([low_feat, high_up], dim=1)
        gate = self.boundary_gate(gate_ctx)

        fused = low_feat + gate * self.out_conv(out)
        return fused

# ==============================================================================
# 3. Bilateral Guided Aggregation (BGA)
#    作用：最后一步，将“语义信息”和“细节信息”进行双向纠错融合。
# ==============================================================================
class BGA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # 特征预处理
        self.detail_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

        # Gate 生成器
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True) # 输出 gate mask
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True)
        )

        # 最终融合
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, detail, semantic):
        d_feat = self.detail_conv(detail)
        s_feat = self.semantic_conv(semantic)

        d_mask = torch.sigmoid(self.detail_gate(d_feat))
        s_mask = torch.sigmoid(self.semantic_gate(s_feat))

        # 互相引导：语义掩膜过滤细节噪声，细节掩膜补充语义边界
        d_guided = d_feat * s_mask
        s_guided = s_feat * d_mask

        return self.out_conv(d_guided + s_guided)

# ==============================================================================
# 4. Modified SegFormer Head
# ==============================================================================
class SegFormerHead(nn.Module):
    def __init__(self, embed_dims, decoder_channels=128, num_classes=2, phi='b0'):
        super().__init__()
        C1, C2, C3, C4 = embed_dims

        # 1x1 MLP Projects (Unified Channel Dim)
        self.proj1 = nn.Conv2d(C1, decoder_channels, 1, bias=False)
        self.proj2 = nn.Conv2d(C2, decoder_channels, 1, bias=False)
        self.proj3 = nn.Conv2d(C3, decoder_channels, 1, bias=False)
        self.proj4 = nn.Conv2d(C4, decoder_channels, 1, bias=False)

        # ★ 关键：将 Backbone 输出的 freq_map 投影对齐到 decoder_channels
        self.freq_proj = nn.Conv2d(C4, decoder_channels, 1, bias=True)

        # 1. 细节增强分支 (Processing C1)
        self.detail_extractor = CADetailExtractor(C1, decoder_channels, reduction=16)

        # 2. 频域引导跨尺度注意力 (Fusing C4 & C3)
        self.fg_csa = FG_CSA(
            low_ch=decoder_channels,
            high_ch=decoder_channels,
            mid_ch=decoder_channels // 4,
            only_small_to_large=True
        )

        # 3. 中层融合 (Fusing C2)
        self.fuse2_conv = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        # 4. 最终双向融合 (Detail + Semantic)
        self.bga = BGA(decoder_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1), # 防止过拟合
            nn.Conv2d(decoder_channels, num_classes, 1)
        )

    def forward(self, features, freq_map4=None):
        e1, e2, e3, e4 = features

        # --- Step 1: 投影 ---
        p2 = self.proj2(e2) # H/8
        p3 = self.proj3(e3) # H/16
        p4 = self.proj4(e4) # H/32

        # --- Step 2: 频域投影 ---
        freq_dec = None
        if freq_map4 is not None:
            # freq_map4: (B, C4, 1, 1) -> (B, Dec, 1, 1)
            freq_dec = self.freq_proj(freq_map4)

        # --- Step 3: 深层融合 (Stage4 -> Stage3) ---
        # 利用 freq_dec 引导 p4，再注入 p3
        pyr_up_to_e3 = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        fuse_e3 = self.fg_csa(p3, pyr_up_to_e3, freq_map=freq_dec)

        # --- Step 4: 中层融合 (Stage3 -> Stage2) ---
        fuse_e3_up = F.interpolate(fuse_e3, size=p2.shape[2:], mode='bilinear', align_corners=False)
        fuse_e2 = self.fuse2_conv(torch.cat([p2, fuse_e3_up], dim=1))

        # --- Step 5: 细节融合 (Stage2 -> Stage1) ---
        # 细节分支独立处理 e1 (H/4)，保留最丰富的空间信息
        detail = self.detail_extractor(e1) 
        
        # 将语义特征上采样到 H/4
        fuse_e2_up = F.interpolate(fuse_e2, size=detail.shape[2:], mode='bilinear', align_corners=False)
        
        # 使用 BGA 融合 细节(detail) 和 语义(fuse_e2_up)
        fused = self.bga(detail, fuse_e2_up)

        # --- Step 6: 分类 ---
        logits = self.classifier(fused)
        return logits, {} # 返回空 dict 兼容 mmsegmentation 格式

# ==============================================================================
# 5. Main SegFormer Model Wrapper
# ==============================================================================
class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b0', pretrained=False, in_chans=3):
        super(SegFormer, self).__init__()
        
        # 配置各版本的通道数
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512],
            'b2': [64, 128, 320, 512], 'b3': [64, 128, 320, 512],
            'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        
        # 解码器通道数配置 (b5 更大)
        decoder_dim = 192 if phi == 'b5' else 128

        # 实例化 Backbone (传入 in_chans 以支持多波段)
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained=pretrained, in_chans=in_chans)

        # 实例化 Custom Head
        self.decode_head = SegFormerHead(
            embed_dims=self.in_channels,
            decoder_channels=decoder_dim,
            num_classes=num_classes,
            phi=phi
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        # Backbone 前向传播 (返回特征列表 和 频域图)
        features, freq_map4 = self.backbone(inputs)
        
        # Head 前向传播
        x, _ = self.decode_head(features, freq_map4=freq_map4)
        
        # 恢复到原图尺寸
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x

# 测试代码 (Optional)
if __name__ == "__main__":
    # 假设输入是 8 通道 (多时相)
    dummy_input = torch.randn(2, 8, 512, 512)
    model = SegFormer(num_classes=2, phi='b2', in_chans=8, pretrained=False)
    
    output = model(dummy_input)
    print("Output Shape:", output.shape) # Should be [2, 2, 512, 512]
    
    
    
# 原模型        
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

# class MLP(nn.Module):
#     """
#     Linear Embedding
#     """
#     def __init__(self, input_dim=2048, embed_dim=768):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, embed_dim)

#     def forward(self, x):
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x
    
# class ConvModule(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
#         super(ConvModule, self).__init__()
#         self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
#         self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
#         self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def fuseforward(self, x):
#         return self.act(self.conv(x))

# class SegFormerHead(nn.Module):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
#         super(SegFormerHead, self).__init__()
#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

#         self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

#         self.linear_fuse = ConvModule(
#             c1=embedding_dim*4,
#             c2=embedding_dim,
#             k=1,
#         )

#         self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
#         self.dropout        = nn.Dropout2d(dropout_ratio)
    
#     def forward(self, inputs):
#         c1, c2, c3, c4 = inputs

#         ############## MLP decoder on C1-C4 ###########
#         n, _, h, w = c4.shape
        
#         _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

#         _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

#         _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

#         _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

#         _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

#         x = self.dropout(_c)
#         x = self.linear_pred(x)

#         return x

# class SegFormer(nn.Module):
#     # 修改点：将 in_channels 改为 in_chans
#     def __init__(self, num_classes=21, phi='b0', pretrained=False, in_chans=8):
#         super(SegFormer, self).__init__()
#         self.in_channels_config = {
#             'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
#             'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
#         }[phi]
        
#         # 1. 初始化骨干网络
#         self.backbone = {
#             'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
#             'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
#         }[phi](pretrained)
        
#         # 2. 修改第一层 Patch Embedding 以适配指定通道数 (in_chans)
#         # 获取原始层
#         old_layer = self.backbone.patch_embed1.proj
        
#         # 检查是否需要修改 (如果传入的 in_chans 与当前层不符)
#         if old_layer.in_channels != in_chans:
#             # 创建新卷积层
#             new_layer = nn.Conv2d(
#                 in_channels=in_chans,
#                 out_channels=old_layer.out_channels,
#                 kernel_size=old_layer.kernel_size,
#                 stride=old_layer.stride,
#                 padding=old_layer.padding,
#                 bias=old_layer.bias is not None
#             )
            
#             # 3. 处理权重初始化
#             if pretrained:
#                 with torch.no_grad():
#                     # 只有当原始层是RGB(3通道)且我们需要扩展时，才进行部分复制
#                     if old_layer.in_channels == 3:
#                         # 复制前3通道
#                         new_layer.weight[:, :3, :, :] = old_layer.weight
#                         # 后续通道随机初始化
#                         nn.init.kaiming_normal_(new_layer.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
#                     else:
#                         # 如果维度完全对不上且不是3通道，直接全部随机初始化，避免报错
#                         print(f"Warning: Re-initializing input layer from {old_layer.in_channels} to {in_chans} channels.")
#                         nn.init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')
                    
#                     if old_layer.bias is not None:
#                         new_layer.bias = old_layer.bias

#             # 替换层
#             self.backbone.patch_embed1.proj = new_layer

#         self.embedding_dim = {
#             'b0': 256, 'b1': 256, 'b2': 768,
#             'b3': 768, 'b4': 768, 'b5': 768,
#         }[phi]
        
#         # 初始化解码头
#         self.decode_head = SegFormerHead(num_classes, self.in_channels_config, self.embedding_dim)

#     def forward(self, inputs):
#         H, W = inputs.size(2), inputs.size(3)
        
#         x = self.backbone.forward(inputs)
#         x = self.decode_head.forward(x)
        
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x

# 去掉频域
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 确保这里能导入你上一步改好的 backbone
# from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

# # ==============================================================================
# # 1. Coordinate Attention & Detail Extractor
# #    作用：针对农田防护林“细长、线性”的特征，强化垂直/水平方向的空间感知。
# # ==============================================================================
# class CoordAtt(nn.Module):
#     """
#     Coordinate Attention (Hou et al., CVPR 2021)
#     """
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.channels = channels
#         # 限制 mip 最小为 8，防止通道压缩过猛丢失信息
#         mip = max(8, channels // reduction)

#         self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1   = nn.BatchNorm2d(mip)
#         self.act   = nn.ReLU(inplace=True)

#         self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x):
#         B, C, H, W = x.size()

#         # X Avg Pool -> (B, C, H, 1)
#         x_h = x.mean(dim=3, keepdim=True)
#         # Y Avg Pool -> (B, C, 1, W) -> (B, C, W, 1)
#         x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

#         # Concat -> (B, C, H+W, 1)
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)

#         # Split
#         y_h, y_w = torch.split(y, [H, W], dim=2)
#         y_w = y_w.permute(0, 1, 3, 2)

#         # Sigmoid -> Attention Map
#         a_h = torch.sigmoid(self.conv_h(y_h))
#         a_w = torch.sigmoid(self.conv_w(y_w))

#         return x * a_h * a_w

# class CADetailExtractor(nn.Module):
#     """
#     结合了连续卷积和坐标注意力的细节提取器
#     """
#     def __init__(self, in_ch, out_ch, reduction=16):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#         self.ca = CoordAtt(out_ch, reduction=reduction)

#     def forward(self, x):
#         feat = self.conv(x)   # 提取高频纹理
#         feat = self.ca(feat)  # 增强线性结构
#         return feat

# # ==============================================================================
# # 2. Frequency Guided Cross-Scale Attention (FG_CSA)
# #    作用：利用 FSA 提取的频域权重（反映物候特征）来门控特征融合。
# # ==============================================================================
# class FG_CSA(nn.Module):
#     def __init__(self, low_ch, high_ch, mid_ch, only_small_to_large=True):
#         super().__init__()
#         self.proj_low = nn.Conv2d(low_ch, mid_ch, 1)
#         self.proj_high = nn.Conv2d(high_ch, mid_ch, 1)
#         self.proj_v = nn.Conv2d(high_ch, mid_ch, 1)
        
#         # 边界门控机制
#         self.boundary_gate = nn.Sequential(
#             nn.Conv2d(low_ch + high_ch, 1, 3, padding=1),
#             nn.Sigmoid()
#         )
#         self.out_conv = nn.Conv2d(mid_ch, low_ch, 1)
#         self.only_small_to_large = only_small_to_large

#     def forward(self, low_feat, high_feat, freq_map=None):
#         """
#         low_feat: 较浅层特征 (e.g., p3)
#         high_feat: 较深层特征 (e.g., p4)
#         freq_map: (B, C, 1, 1) 来自 FSA 的频域权重
#         """
#         B, Cl, H, W = low_feat.shape
        
#         # 1. 频域引导 (Frequency Gating)
#         # 如果有 freq_map，先对高层语义特征进行重加权
#         # 这步很关键：利用频域信息抑制那些“看起来像树其实是玉米”的特征
#         if freq_map is not None:
#             # freq_map 已经在 Head 里投影到了 high_ch 维度
#             high_feat = high_feat * freq_map 

#         # 2. Cross-Attention
#         # Q: Low
#         L = self.proj_low(low_feat)
#         q = L.flatten(2).transpose(1, 2) # (B, Nl, mid)

#         # K, V: High (保持原分辨率计算，节省显存)
#         Kf = self.proj_high(high_feat)
#         Vf = self.proj_v(high_feat)
#         k = Kf.flatten(2).transpose(1, 2)
#         v = Vf.flatten(2).transpose(1, 2)

#         # Attention Map
#         attn = torch.matmul(q, k.transpose(1, 2)) * (mid_ch ** -0.5 if 'mid_ch' in locals() else (k.size(-1)**-0.5))
#         attn = torch.softmax(attn, dim=-1)
        
#         out = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, H, W)

#         # 3. Boundary Gating Fusion
#         high_up = F.interpolate(high_feat, size=(H, W), mode='bilinear', align_corners=False)
#         gate_ctx = torch.cat([low_feat, high_up], dim=1)
#         gate = self.boundary_gate(gate_ctx)

#         fused = low_feat + gate * self.out_conv(out)
#         return fused

# # ==============================================================================
# # 3. Bilateral Guided Aggregation (BGA)
# #    作用：最后一步，将“语义信息”和“细节信息”进行双向纠错融合。
# # ==============================================================================
# class BGA(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
        
#         # 特征预处理
#         self.detail_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
#         )
#         self.semantic_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
#         )

#         # Gate 生成器
#         self.detail_gate = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 1, bias=True) # 输出 gate mask
#         )
#         self.semantic_gate = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 1, bias=True)
#         )

#         # 最终融合
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
#         )

#     def forward(self, detail, semantic):
#         d_feat = self.detail_conv(detail)
#         s_feat = self.semantic_conv(semantic)

#         d_mask = torch.sigmoid(self.detail_gate(d_feat))
#         s_mask = torch.sigmoid(self.semantic_gate(s_feat))

#         # 互相引导：语义掩膜过滤细节噪声，细节掩膜补充语义边界
#         d_guided = d_feat * s_mask
#         s_guided = s_feat * d_mask

#         return self.out_conv(d_guided + s_guided)

# # ==============================================================================
# # 4. Modified SegFormer Head
# # ==============================================================================
# class SegFormerHead(nn.Module):
#     def __init__(self, embed_dims, decoder_channels=128, num_classes=2, phi='b0'):
#         super().__init__()
#         C1, C2, C3, C4 = embed_dims

#         # 1x1 MLP Projects (Unified Channel Dim)
#         self.proj1 = nn.Conv2d(C1, decoder_channels, 1, bias=False)
#         self.proj2 = nn.Conv2d(C2, decoder_channels, 1, bias=False)
#         self.proj3 = nn.Conv2d(C3, decoder_channels, 1, bias=False)
#         self.proj4 = nn.Conv2d(C4, decoder_channels, 1, bias=False)

#         # ★ 关键：将 Backbone 输出的 freq_map 投影对齐到 decoder_channels
#         self.freq_proj = nn.Conv2d(C4, decoder_channels, 1, bias=True)

#         # 1. 细节增强分支 (Processing C1)
#         self.detail_extractor = CADetailExtractor(C1, decoder_channels, reduction=16)

#         # 2. 频域引导跨尺度注意力 (Fusing C4 & C3)
#         self.fg_csa = FG_CSA(
#             low_ch=decoder_channels,
#             high_ch=decoder_channels,
#             mid_ch=decoder_channels // 4,
#             only_small_to_large=True
#         )

#         # 3. 中层融合 (Fusing C2)
#         self.fuse2_conv = nn.Sequential(
#             nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True)
#         )

#         # 4. 最终双向融合 (Detail + Semantic)
#         self.bga = BGA(decoder_channels)

#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1), # 防止过拟合
#             nn.Conv2d(decoder_channels, num_classes, 1)
#         )

#     def forward(self, features, freq_map4=None):
#         e1, e2, e3, e4 = features

#         # --- Step 1: 投影 ---
#         p2 = self.proj2(e2) # H/8
#         p3 = self.proj3(e3) # H/16
#         p4 = self.proj4(e4) # H/32

#         # --- Step 2: 频域投影 ---
#         freq_dec = None
#         if freq_map4 is not None:
#             # freq_map4: (B, C4, 1, 1) -> (B, Dec, 1, 1)
#             freq_dec = self.freq_proj(freq_map4)

#         # --- Step 3: 深层融合 (Stage4 -> Stage3) ---
#         # 利用 freq_dec 引导 p4，再注入 p3
#         pyr_up_to_e3 = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
#         fuse_e3 = self.fg_csa(p3, pyr_up_to_e3, freq_map=freq_dec)

#         # --- Step 4: 中层融合 (Stage3 -> Stage2) ---
#         fuse_e3_up = F.interpolate(fuse_e3, size=p2.shape[2:], mode='bilinear', align_corners=False)
#         fuse_e2 = self.fuse2_conv(torch.cat([p2, fuse_e3_up], dim=1))

#         # --- Step 5: 细节融合 (Stage2 -> Stage1) ---
#         # 细节分支独立处理 e1 (H/4)，保留最丰富的空间信息
#         detail = self.detail_extractor(e1) 
        
#         # 将语义特征上采样到 H/4
#         fuse_e2_up = F.interpolate(fuse_e2, size=detail.shape[2:], mode='bilinear', align_corners=False)
        
#         # 使用 BGA 融合 细节(detail) 和 语义(fuse_e2_up)
#         fused = self.bga(detail, fuse_e2_up)

#         # --- Step 6: 分类 ---
#         logits = self.classifier(fused)
#         return logits, {} # 返回空 dict 兼容 mmsegmentation 格式



# # ==============================================================================
# # 5. Main SegFormer Model Wrapper (已修复接口匹配问题)
# # ==============================================================================
# class SegFormer(nn.Module):
#     def __init__(self, num_classes=21, phi='b0', pretrained=False, in_chans=3):
#         super(SegFormer, self).__init__()
        
#         # 配置各版本的通道数
#         self.in_channels = {
#             'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512],
#             'b2': [64, 128, 320, 512], 'b3': [64, 128, 320, 512],
#             'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
#         }[phi]
        
#         # 解码器通道数配置 (b5 更大)
#         decoder_dim = 192 if phi == 'b5' else 128

#         # 实例化 Backbone (传入 in_chans 以支持多波段)
#         self.backbone = {
#             'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
#             'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
#         }[phi](pretrained=pretrained, in_chans=in_chans)

#         # 实例化 Custom Head (你的魔改 Head)
#         self.decode_head = SegFormerHead(
#             embed_dims=self.in_channels,
#             decoder_channels=decoder_dim,
#             num_classes=num_classes,
#             phi=phi
#         )

#     def forward(self, inputs):
#         H, W = inputs.size(2), inputs.size(3)
        
#         # -----------------------------------------------------------
#         # 修改点：Backbone 现在只返回 features，不再返回 freq_map4
#         # -----------------------------------------------------------
#         features = self.backbone(inputs)
        
#         # Head 前向传播
#         # 因为 Backbone 去掉了 FSA，这里 freq_map4 传入 None
#         # 你的 FG_CSA 模块会自动处理 None 的情况（只做 Cross Attention，不做频域门控）
#         x, _ = self.decode_head(features, freq_map4=None)
        
#         # 恢复到原图尺寸
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
#         return x

# 只保留CoordAtt 和 CADetailExtractor
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 确保这里能导入你之前改好的 backbone
# from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

# # ==============================================================================
# # 1. Coordinate Attention & Detail Extractor (保留)
# # ==============================================================================
# class CoordAtt(nn.Module):
#     """
#     Coordinate Attention (Hou et al., CVPR 2021)
#     """
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.channels = channels
#         mip = max(8, channels // reduction)

#         self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1   = nn.BatchNorm2d(mip)
#         self.act   = nn.ReLU(inplace=True)

#         self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x):
#         B, C, H, W = x.size()
#         x_h = x.mean(dim=3, keepdim=True)
#         x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)

#         y_h, y_w = torch.split(y, [H, W], dim=2)
#         y_w = y_w.permute(0, 1, 3, 2)

#         a_h = torch.sigmoid(self.conv_h(y_h))
#         a_w = torch.sigmoid(self.conv_w(y_w))

#         return x * a_h * a_w

# class CADetailExtractor(nn.Module):
#     """
#     结合了连续卷积和坐标注意力的细节提取器
#     """
#     def __init__(self, in_ch, out_ch, reduction=16):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#         self.ca = CoordAtt(out_ch, reduction=reduction)

#     def forward(self, x):
#         feat = self.conv(x)   # 提取高频纹理
#         feat = self.ca(feat)  # 增强线性结构
#         return feat

# # ==============================================================================
# # 2. Modified SegFormer Head (消融版本：移除了 FG_CSA 和 BGA)
# # ==============================================================================
# class SegFormerHead(nn.Module):
#     def __init__(self, embed_dims, decoder_channels=128, num_classes=2, phi='b0'):
#         super().__init__()
#         C1, C2, C3, C4 = embed_dims

#         # 1x1 MLP Projects
#         self.proj1 = nn.Conv2d(C1, decoder_channels, 1, bias=False)
#         self.proj2 = nn.Conv2d(C2, decoder_channels, 1, bias=False)
#         self.proj3 = nn.Conv2d(C3, decoder_channels, 1, bias=False)
#         self.proj4 = nn.Conv2d(C4, decoder_channels, 1, bias=False)

#         # 1. 细节增强分支 (保留这个核心模块)
#         self.detail_extractor = CADetailExtractor(C1, decoder_channels, reduction=16)

#         # 2. 深层融合 (替代 FG_CSA) -> 普通的 Concat + Conv
#         self.fuse34_conv = nn.Sequential(
#             nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True)
#         )

#         # 3. 中层融合
#         self.fuse2_conv = nn.Sequential(
#             nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True)
#         )

#         # 4. 最终融合 (替代 BGA) -> 普通的 Concat + Conv
#         self.final_fuse_conv = nn.Sequential(
#             nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True)
#         )

#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(decoder_channels, num_classes, 1)
#         )

#     def forward(self, features, freq_map4=None):
#         e1, e2, e3, e4 = features

#         # --- Step 1: 投影 ---
#         p2 = self.proj2(e2) # H/8
#         p3 = self.proj3(e3) # H/16
#         p4 = self.proj4(e4) # H/32

#         # --- Step 2: 深层融合 (Stage4 -> Stage3) ---
#         # 简单上采样 + 拼接 + 卷积
#         pyr_up_to_e3 = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
#         fuse_e3 = self.fuse34_conv(torch.cat([p3, pyr_up_to_e3], dim=1))

#         # --- Step 3: 中层融合 (Stage3 -> Stage2) ---
#         fuse_e3_up = F.interpolate(fuse_e3, size=p2.shape[2:], mode='bilinear', align_corners=False)
#         fuse_e2 = self.fuse2_conv(torch.cat([p2, fuse_e3_up], dim=1))

#         # --- Step 4: 细节融合 (Stage2 -> Stage1) ---
#         # 细节分支 (保留)
#         detail = self.detail_extractor(e1) 
        
#         # 语义特征上采样
#         fuse_e2_up = F.interpolate(fuse_e2, size=detail.shape[2:], mode='bilinear', align_corners=False)
        
#         # 简单拼接融合 (替代了 BGA)
#         fused = self.final_fuse_conv(torch.cat([detail, fuse_e2_up], dim=1))

#         # --- Step 5: 分类 ---
#         logits = self.classifier(fused)
#         return logits, {}

# # ==============================================================================
# # 3. Main SegFormer Model Wrapper
# # ==============================================================================
# class SegFormer(nn.Module):
#     def __init__(self, num_classes=21, phi='b0', pretrained=False, in_chans=3):
#         super(SegFormer, self).__init__()
        
#         self.in_channels = {
#             'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512],
#             'b2': [64, 128, 320, 512], 'b3': [64, 128, 320, 512],
#             'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
#         }[phi]
        
#         decoder_dim = 192 if phi == 'b5' else 128

#         self.backbone = {
#             'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
#             'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
#         }[phi](pretrained=pretrained, in_chans=in_chans)

#         self.decode_head = SegFormerHead(
#             embed_dims=self.in_channels,
#             decoder_channels=decoder_dim,
#             num_classes=num_classes,
#             phi=phi
#         )

#     def forward(self, inputs):
#         H, W = inputs.size(2), inputs.size(3)
        
#         # 注意：这里适配你之前改好的 Backbone (只返回 features)
#         features = self.backbone(inputs)
        
#         x, _ = self.decode_head(features, freq_map4=None)
        
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
#         return x
    
#只保留BGA

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 确保这里能导入你之前改好的 backbone
# from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

# # ==============================================================================
# # 1. Bilateral Guided Aggregation (BGA) [保留]
# #    作用：最后一步，将“语义信息”和“细节信息”进行双向纠错融合。
# # ==============================================================================
# class BGA(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
        
#         # 特征预处理
#         self.detail_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
#         )
#         self.semantic_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
#         )

#         # Gate 生成器
#         self.detail_gate = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 1, bias=True) # 输出 gate mask
#         )
#         self.semantic_gate = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 1, bias=True)
#         )

#         # 最终融合
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
#         )

#     def forward(self, detail, semantic):
#         d_feat = self.detail_conv(detail)
#         s_feat = self.semantic_conv(semantic)

#         d_mask = torch.sigmoid(self.detail_gate(d_feat))
#         s_mask = torch.sigmoid(self.semantic_gate(s_feat))

#         # 互相引导：语义掩膜过滤细节噪声，细节掩膜补充语义边界
#         d_guided = d_feat * s_mask
#         s_guided = s_feat * d_mask

#         return self.out_conv(d_guided + s_guided)

# # ==============================================================================
# # 2. Modified SegFormer Head (消融版本：仅保留 BGA)
# # ==============================================================================
# class SegFormerHead(nn.Module):
#     def __init__(self, embed_dims, decoder_channels=128, num_classes=2, phi='b0'):
#         super().__init__()
#         C1, C2, C3, C4 = embed_dims

#         # 1x1 MLP Projects (Unified Channel Dim)
#         self.proj1 = nn.Conv2d(C1, decoder_channels, 1, bias=False)
#         self.proj2 = nn.Conv2d(C2, decoder_channels, 1, bias=False)
#         self.proj3 = nn.Conv2d(C3, decoder_channels, 1, bias=False)
#         self.proj4 = nn.Conv2d(C4, decoder_channels, 1, bias=False)

#         # 1. 细节增强分支 (替代 CADetailExtractor)
#         # 移除了 CoordAtt，只保留基础的卷积提取特征，以适配 BGA 的输入
#         self.detail_conv = nn.Sequential(
#             nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True)
#         )

#         # 2. 深层融合 (替代 FG_CSA) -> 普通拼接卷积
#         self.fuse34_conv = nn.Sequential(
#             nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True)
#         )

#         # 3. 中层融合
#         self.fuse2_conv = nn.Sequential(
#             nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True)
#         )

#         # 4. 最终双向融合 (BGA 保留)
#         self.bga = BGA(decoder_channels)

#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(decoder_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(decoder_channels, num_classes, 1)
#         )

#     def forward(self, features, freq_map4=None):
#         e1, e2, e3, e4 = features

#         # --- Step 1: 投影 ---
#         p1 = self.proj1(e1) # H/4
#         p2 = self.proj2(e2) # H/8
#         p3 = self.proj3(e3) # H/16
#         p4 = self.proj4(e4) # H/32

#         # --- Step 2: 语义分支聚合 (Semantic Branch) ---
#         # Stage4 -> Stage3
#         pyr_up_to_e3 = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
#         fuse_e3 = self.fuse34_conv(torch.cat([p3, pyr_up_to_e3], dim=1))

#         # Stage3 -> Stage2
#         fuse_e3_up = F.interpolate(fuse_e3, size=p2.shape[2:], mode='bilinear', align_corners=False)
#         semantic_feat = self.fuse2_conv(torch.cat([p2, fuse_e3_up], dim=1)) # 此时是 H/8

#         # --- Step 3: 细节分支处理 (Detail Branch) ---
#         # 替代了 CADetailExtractor，仅做卷积处理
#         detail_feat = self.detail_conv(p1) # H/4
        
#         # --- Step 4: BGA 融合 ---
#         # 需要将 semantic_feat 上采样到 H/4 以匹配 detail_feat
#         semantic_feat_up = F.interpolate(semantic_feat, size=detail_feat.shape[2:], mode='bilinear', align_corners=False)
        
#         # 
#         # 使用 BGA 融合 细节 和 语义
#         fused = self.bga(detail_feat, semantic_feat_up)

#         # --- Step 5: 分类 ---
#         logits = self.classifier(fused)
#         return logits, {}

# # ==============================================================================
# # 3. Main SegFormer Model Wrapper
# # ==============================================================================
# class SegFormer(nn.Module):
#     def __init__(self, num_classes=21, phi='b0', pretrained=False, in_chans=3):
#         super(SegFormer, self).__init__()
        
#         self.in_channels = {
#             'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512],
#             'b2': [64, 128, 320, 512], 'b3': [64, 128, 320, 512],
#             'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
#         }[phi]
        
#         decoder_dim = 192 if phi == 'b5' else 128

#         self.backbone = {
#             'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
#             'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
#         }[phi](pretrained=pretrained, in_chans=in_chans)

#         self.decode_head = SegFormerHead(
#             embed_dims=self.in_channels,
#             decoder_channels=decoder_dim,
#             num_classes=num_classes,
#             phi=phi
#         )

#     def forward(self, inputs):
#         H, W = inputs.size(2), inputs.size(3)
        
#         # Backbone 只返回特征列表
#         features = self.backbone(inputs)
        
#         # Head 处理
#         x, _ = self.decode_head(features, freq_map4=None)
        
#         # 恢复尺寸
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
#         return x


# 基础跑Transformer




import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


# ==============================================================================
# 1. Coordinate Attention & Detail Extractor
# ==============================================================================
class CoordAtt(nn.Module):
    """
    Coordinate Attention (Hou et al., CVPR 2021)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        B, C, H, W = x.size()

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))

        return x * a_h * a_w


class CADetailExtractor(nn.Module):
    """
    结合连续卷积和坐标注意力的细节提取器
    """
    def __init__(self, in_ch, out_ch, reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ca = CoordAtt(out_ch, reduction=reduction)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.ca(feat)
        return feat


# ==============================================================================
# 2. Frequency Guided Cross-Scale Attention (FG_CSA)
# ==============================================================================
class FG_CSA(nn.Module):
    def __init__(self, low_ch, high_ch, mid_ch):
        super().__init__()
        self.proj_low = nn.Conv2d(low_ch, mid_ch, 1)
        self.proj_high = nn.Conv2d(high_ch, mid_ch, 1)
        self.proj_v = nn.Conv2d(high_ch, mid_ch, 1)
        self.scale = mid_ch ** -0.5

        self.boundary_gate = nn.Sequential(
            nn.Conv2d(low_ch + high_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Conv2d(mid_ch, low_ch, 1)

    def forward(self, low_feat, high_feat, freq_map=None):
        """
        low_feat: 较浅层特征 (e.g., p3)
        high_feat: 较深层特征 (e.g., p4)
        freq_map: (B, C, 1, 1)
        """
        B, _, H, W = low_feat.shape

        if freq_map is not None:
            high_feat = high_feat * freq_map

        q = self.proj_low(low_feat).flatten(2).transpose(1, 2)    # (B, Nl, mid)
        k = self.proj_high(high_feat).flatten(2).transpose(1, 2)  # (B, Nh, mid)
        v = self.proj_v(high_feat).flatten(2).transpose(1, 2)     # (B, Nh, mid)

        attn = torch.matmul(q, k.transpose(1, 2)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, H, W)

        high_up = F.interpolate(high_feat, size=(H, W), mode="bilinear", align_corners=False)
        gate_ctx = torch.cat([low_feat, high_up], dim=1)
        gate = self.boundary_gate(gate_ctx)

        fused = low_feat + gate * self.out_conv(out)
        return fused


# ==============================================================================
# 3. Bilateral Guided Aggregation (BGA)
# ==============================================================================
class BGA(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.detail_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, detail, semantic):
        d_feat = self.detail_conv(detail)
        s_feat = self.semantic_conv(semantic)

        d_mask = torch.sigmoid(self.detail_gate(d_feat))
        s_mask = torch.sigmoid(self.semantic_gate(s_feat))

        d_guided = d_feat * s_mask
        s_guided = s_feat * d_mask

        return self.out_conv(d_guided + s_guided)


# ==============================================================================
# 4. Modified SegFormer Head
# ==============================================================================
class SegFormerHead(nn.Module):
    def __init__(self, embed_dims, decoder_channels=128, num_classes=2, phi='b0'):
        super().__init__()
        C1, C2, C3, C4 = embed_dims

        self.proj2 = nn.Conv2d(C2, decoder_channels, 1, bias=False)
        self.proj3 = nn.Conv2d(C3, decoder_channels, 1, bias=False)
        self.proj4 = nn.Conv2d(C4, decoder_channels, 1, bias=False)

        self.freq_proj = nn.Conv2d(C4, decoder_channels, 1, bias=True)

        self.detail_extractor = CADetailExtractor(C1, decoder_channels, reduction=16)

        self.fg_csa = FG_CSA(
            low_ch=decoder_channels,
            high_ch=decoder_channels,
            mid_ch=decoder_channels // 4,
        )

        self.fuse2_conv = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        self.bga = BGA(decoder_channels)

        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )

    def forward(self, features, freq_map4=None):
        e1, e2, e3, e4 = features

        p2 = self.proj2(e2)
        p3 = self.proj3(e3)
        p4 = self.proj4(e4)

        freq_dec = None
        if freq_map4 is not None:
            freq_dec = self.freq_proj(freq_map4)

        pyr_up_to_e3 = F.interpolate(p4, size=p3.shape[2:], mode="bilinear", align_corners=False)
        fuse_e3 = self.fg_csa(p3, pyr_up_to_e3, freq_map=freq_dec)

        fuse_e3_up = F.interpolate(fuse_e3, size=p2.shape[2:], mode="bilinear", align_corners=False)
        fuse_e2 = self.fuse2_conv(torch.cat([p2, fuse_e3_up], dim=1))

        detail = self.detail_extractor(e1)

        fuse_e2_up = F.interpolate(fuse_e2, size=detail.shape[2:], mode="bilinear", align_corners=False)
        fused = self.bga(detail, fuse_e2_up)

        logits = self.classifier(fused)
        return logits, {}


# ==============================================================================
# 5. Main SegFormer Model Wrapper
# ==============================================================================
class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b0', pretrained=False, in_chans=3):
        super().__init__()

        self.in_channels = {
            'b0': [32, 64, 160, 256],
            'b1': [64, 128, 320, 512],
            'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512],
            'b4': [64, 128, 320, 512],
            'b5': [64, 128, 320, 512],
        }[phi]

        decoder_dim = 192 if phi == 'b5' else 128

        self.backbone = {
            'b0': mit_b0,
            'b1': mit_b1,
            'b2': mit_b2,
            'b3': mit_b3,
            'b4': mit_b4,
            'b5': mit_b5,
        }[phi](pretrained=pretrained, in_chans=in_chans)

        self.decode_head = SegFormerHead(
            embed_dims=self.in_channels,
            decoder_channels=decoder_dim,
            num_classes=num_classes,
            phi=phi,
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        features, freq_map4 = self.backbone(inputs)
        x, _ = self.decode_head(features, freq_map4=freq_map4)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    dummy_input = torch.randn(2, 8, 512, 512)
    model = SegFormer(num_classes=2, phi='b2', in_chans=8, pretrained=False)

    output = model(dummy_input)
    print("Output Shape:", output.shape)
