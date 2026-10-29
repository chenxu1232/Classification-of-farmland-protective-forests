# # 八通道整体                       
import math
import warnings
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Helper Functions
# ==============================================================================
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

# ==============================================================================
# 1. NEW: Channel Attention Adapter (解决多波段输入问题的核心模块)
# ==============================================================================
class ChannelAttentionAdapter(nn.Module):
    """
    用于将多波段（如8通道）数据适配到 SegFormer 的 3通道输入，
    并利用 SE Attention 自动学习波段重要性。
    """
    def __init__(self, in_chans, out_chans=3, reduction=4):
        super().__init__()
        # 通道注意力 (SE Module)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_chans, in_chans // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_chans // reduction, in_chans, bias=False),
            nn.Sigmoid()
        )
        # 降维映射: N -> 3
        self.mapping = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        
        # 初始化
        self._init_weights(in_chans, out_chans)

    def _init_weights(self, in_chans, out_chans):
        # 1. 初始化映射层
        # 我们希望初始状态下，它能尽量保留原始 RGB 的信息（如果前3通道是RGB）
        nn.init.kaiming_normal_(self.mapping.weight, mode='fan_out', nonlinearity='relu')
        
        # 如果你明确知道前3通道是RGB，可以尝试强制初始化（可选，这里用标准Kaiming即可）

    def forward(self, x):
        b, c, _, _ = x.size()
        # 1. SE Attention: 学习波段权重
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x) # 动态加权
        
        # 2. Mapping: 映射到 3 通道，以便后续网络加载预训练权重
        x = self.mapping(x)
        return x

# ==============================================================================
# Existing Modules (OverlapPatchEmbed, Attention, etc.)
# ==============================================================================

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

# ===================== FSA: DCT-based Frequency Self-Attention =====================
class _DCTCache:
    def __init__(self):
        self.cache = {}

    @staticmethod
    def _dct_orthonormal(n, device, dtype):
        k = torch.arange(n, device=device, dtype=dtype).view(1, -1)
        x = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
        mat = torch.cos(math.pi * (x + 0.5) * k / n)
        mat[:, 0] = mat[:, 0] / math.sqrt(n)
        mat[:, 1:] = mat[:, 1:] * math.sqrt(2.0 / n)
        return mat

    def get_operators(self, H, W, kq, kv, device, dtype):
        key = (H, W, kq, kv, device, dtype)
        if key in self.cache:
            return self.cache[key]
        D_H = self._dct_orthonormal(H, device, dtype)
        D_W = self._dct_orthonormal(W, device, dtype)
        kq = min(kq, H, W)
        kv = min(kv, H, W)
        PqH, PqW = D_H[:kq, :], D_W[:kq, :]
        GqH, GqW = D_H[:, :kq], D_W[:, :kq]
        PkH, PkW = D_H[:kv, :], D_W[:kv, :]
        GkH, GkW = D_H[:, :kv], D_W[:, :kv]
        self.cache[key] = (PqH, PqW, GqH, GqW, PkH, PkW, GkH, GkW, kq, kv)
        return self.cache[key]

_DCT = _DCTCache()

class FSA_DCT_Attn_Token(nn.Module):
    def __init__(self, dim, num_heads=4, kq=16, kv=8, qkv_bias=True, use_linear_attn=False, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kq = kq
        self.kv = kv
        self.use_linear = use_linear_attn
        self.eps = eps
        
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        # FSA 频域通道注意分支
        self.freq_pool = nn.AdaptiveAvgPool2d(1)
        self.freq_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.freq_act  = nn.Sigmoid()
        self.last_freq_map = None

    @staticmethod
    def _to_heads_2d(t, heads):
        B, C, h, w = t.shape
        d = C // heads
        return t.view(B, heads, d, h, w).flatten(-2).permute(0, 1, 3, 2)

    @staticmethod
    def _from_heads_2d(t, heads, C, h, w):
        B, _, N, d = t.shape
        return t.permute(0,1,3,2).contiguous().view(B, C, h, w)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        x_sp = x.transpose(1, 2).reshape(B, C, H, W)
        
        PqH, PqW, GqH, GqW, PkH, PkW, GkH, GkW, kq, kv = _DCT.get_operators(
            H, W, self.kq, self.kv, x.device, x.dtype
        )

        q_low = torch.einsum('bchw,kh,lw->bckl', x_sp, PqH, PqW)
        k_low = torch.einsum('bchw,kh,lw->bckl', x_sp, PkH, PkW)
        v_low = torch.einsum('bchw,kh,lw->bckl', x_sp, PkH, PkW)

        # Freq Map Calculation
        freq = self.freq_pool(q_low)
        freq = self.freq_conv(freq)
        freq = self.freq_act(freq)
        self.last_freq_map = freq

        q_map = self.q_proj(q_low)
        k_map = self.k_proj(k_low)
        v_map = self.v_proj(v_low)

        q = self._to_heads_2d(q_map, self.num_heads)
        k = self._to_heads_2d(k_map, self.num_heads)
        v = self._to_heads_2d(v_map, self.num_heads)

        if not self.use_linear:
            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            att = att.softmax(dim=-1)
            out_low = torch.matmul(att, v)
        else:
            phi = lambda t: F.elu(t, alpha=1.0, inplace=False) + 1.0
            qf, kf = phi(q), phi(k)
            kv_ = torch.matmul(kf.transpose(-2, -1), v)
            z_  = torch.sum(kf, dim=-2)
            out_num = torch.matmul(qf, kv_)
            out_den = torch.matmul(qf, z_.unsqueeze(-1)).clamp_min(1e-6)
            out_low = out_num / out_den

        out_low_2d = self._from_heads_2d(out_low, self.num_heads, C, kq, kq)
        out_sp = torch.einsum('bckl,hk,wl->bchw', out_low_2d, GqH, GqW)
        
        out = out_sp.flatten(2).transpose(1, 2).contiguous()
        out = self.out_proj(out)
        return out

# ===================== Standard Attention & Blocks =====================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=GELU, 
                 norm_layer=nn.LayerNorm, sr_ratio=1,
                 use_fsa=False, fsa_kq=16, fsa_kv=8, fsa_linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_fsa:
            self.attn = FSA_DCT_Attn_Token(dim, num_heads, fsa_kq, fsa_kv, qkv_bias, fsa_linear)
        else:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# ==============================================================================
# Main Backbone Class: MixVisionTransformer
# ==============================================================================
class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # ====================================================
        # NEW: Channel Adapter Logic
        # 如果输入不是3通道（比如8通道），则启用 Adapter
        # 如果是3通道，则透传，不增加参数量
        # ====================================================
        if in_chans != 3:
            self.channel_adapter = ChannelAttentionAdapter(in_chans, out_chans=3)
        else:
            self.channel_adapter = nn.Identity()

        # Transformer parts
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Block 1
        # 注意：这里的 in_chans 固定为 3，因为前面的 Adapter 已经把 N 转为 3 了
        # 这样可以完美保留 Pretrained Weights
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        # Block 2
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        # Block 3
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        # Block 4 (With FSA)
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3],
                  # 开启你的 FSA 模块
                  use_fsa=True, fsa_kq=16, fsa_kv=8, fsa_linear=False)
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # NEW: 先通过 Channel Adapter 处理多波段输入
        # x shape: [B, in_chans, H, W] -> [B, 3, H, W]
        x = self.channel_adapter(x)

        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        
        # 提取 FSA 的 freq_map
        freq_map4 = None
        last_attn = self.block4[-1].attn
        if isinstance(last_attn, FSA_DCT_Attn_Token) and last_attn.last_freq_map is not None:
            freq_map4 = last_attn.last_freq_map

        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs, freq_map4

# ==============================================================================
# Backbone Variants (B0 - B5)
# ==============================================================================
# 注意：我在所有类中都添加了 **kwargs 以便你传入 in_chans

class mit_b0(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b0, self).__init__(
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b0_backbone_weights.pth"), strict=False)

class mit_b1(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b1, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b1_backbone_weights.pth"), strict=False)

class mit_b2(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b2, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b2_backbone_weights.pth"), strict=False)

class mit_b3(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b3, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            # 由于加了Adapter，主干第一层依然是3通道，所以可以直接 load，不需要之前的复杂筛选逻辑了
            # 但为了保险（防止pth文件里有无关key），加上 strict=False 即可
            self.load_state_dict(torch.load("model_data/segformer_b3_backbone_weights.pth"), strict=False)

class mit_b4(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b4, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b4_backbone_weights.pth"), strict=False)

class mit_b5(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b5, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            # 同样简化加载逻辑，因为结构已经匹配
            self.load_state_dict(torch.load("model_data/segformer_b5_backbone_weights.pth"), strict=False)

# 原始

# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.

# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# import math
# import warnings
# import numpy as np
# from functools import partial

# import torch
# import torch.nn as nn


# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     # Cut & paste from PyTorch official master until it's in a few official releases - RW
#     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     def norm_cdf(x):
#         # Computes standard normal cumulative distribution function
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.

#     if (mean < a - 2 * std) or (mean > b + 2 * std):
#         warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#                       "The distribution of values may be incorrect.",
#                       stacklevel=2)

#     with torch.no_grad():
#         # Values are generated by using a truncated uniform distribution and
#         # then using the inverse CDF for the normal distribution.
#         # Get upper and lower cdf values
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)

#         # Uniformly fill tensor with values from [l, u], then translate to
#         # [2l-1, 2u-1].
#         tensor.uniform_(2 * l - 1, 2 * u - 1)

#         # Use inverse cdf transform for normal distribution to get truncated
#         # standard normal
#         tensor.erfinv_()

#         # Transform to proper mean, std
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)

#         # Clamp to ensure it's in the proper range
#         tensor.clamp_(min=a, max=b)
#         return tensor


# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     r"""
#     Fills the input Tensor with values drawn from a truncated
#     normal distribution. The values are effectively drawn from the
#     normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
#     with values outside :math:`[a, b]` redrawn until they are within
#     the bounds. The method used for generating the random values works
#     best when :math:`a \leq \text{mean} \leq b`.
#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         mean: the mean of the normal distribution
#         std: the standard deviation of the normal distribution
#         a: the minimum cutoff value
#         b: the maximum cutoff value
#     Examples:
#         >>> w = torch.empty(3, 5)
#         >>> nn.init.trunc_normal_(w)
#     """
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# #--------------------------------------#
# #   Gelu激活函数的实现
# #   利用近似的数学公式
# #--------------------------------------#
# class GELU(nn.Module):
#     def __init__(self):
#         super(GELU, self).__init__()

#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))

# class OverlapPatchEmbed(nn.Module):
#     def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
#         super().__init__()
#         patch_size  = (patch_size, patch_size)
#         self.proj   = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
#                               padding=(patch_size[0] // 2, patch_size[1] // 2))
#         self.norm   = nn.LayerNorm(embed_dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)

#         return x, H, W

# #--------------------------------------------------------------------------------------------------------------------#
# #   Attention机制
# #   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
# #   然后利用 查询向量query 叉乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
# #   然后利用 score 叉乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
# #   
# #   在segformer中，为了减少计算量，首先对特征图进行了浓缩，所有特征层都压缩到原图的1/32。
# #   当输入图片为512, 512时，Block1的特征图为128, 128，此时就先将特征层压缩为16, 16。
# #   在Block1的Attention模块中，相当于将8x8个特征点进行特征浓缩，浓缩为一个特征点。
# #   然后利用128x128个查询向量对16x16个键向量与值向量进行查询。尽管键向量与值向量的数量较少，但因为查询向量的不同，依然可以获得不同的输出。
# #--------------------------------------------------------------------------------------------------------------------#
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim        = dim
#         self.num_heads  = num_heads
#         head_dim        = dim // num_heads
#         self.scale      = qk_scale or head_dim ** -0.5

#         self.q          = nn.Linear(dim, dim, bias=qkv_bias)
        
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr     = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm   = nn.LayerNorm(dim)
#         self.kv         = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
#         self.attn_drop  = nn.Dropout(attn_drop)
        
#         self.proj       = nn.Linear(dim, dim)
#         self.proj_drop  = nn.Dropout(proj_drop)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         if self.sr_ratio > 1:
#             # bs, 16384, 32 => bs, 32, 128, 128
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 8, 4 => 2, bs, 8, 256, 4
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         # bs, 8, 16384, 4 @ bs, 8, 4, 256 => bs, 8, 16384, 256 
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         # bs, 8, 16384, 256  @ bs, 8, 256, 4 => bs, 8, 16384, 4 => bs, 16384, 32
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         # bs, 16384, 32 => bs, 16384, 32
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

# def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
#     """
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob       = 1 - drop_prob
#     shape           = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor   = x.new_empty(shape).bernoulli_(keep_prob)
#     if keep_prob > 0.0 and scale_by_keep:
#         random_tensor.div_(keep_prob)
#     return x * random_tensor

# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None, scale_by_keep=True):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W)
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)

#         return x
    
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
#         super().__init__()
#         out_features    = out_features or in_features
#         hidden_features = hidden_features or in_features
        
#         self.fc1    = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act    = act_layer()
        
#         self.fc2    = nn.Linear(hidden_features, out_features)
        
#         self.drop   = nn.Dropout(drop)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
#         super().__init__()
#         self.norm1      = norm_layer(dim)
        
#         self.attn       = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
#         )
#         self.norm2      = norm_layer(dim)
#         self.mlp        = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

#         self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         x = x + self.drop_path(self.attn(self.norm1(x), H, W))
#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
#         return x

# class MixVisionTransformer(nn.Module):
#     def __init__(self, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
#                  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
#                  depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
#         super().__init__()
#         self.num_classes    = num_classes
#         self.depths         = depths

#         #----------------------------------#
#         #   Transformer模块，共有四个部分
#         #----------------------------------#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
#         #----------------------------------#
#         #   block1
#         #----------------------------------#
#         #-----------------------------------------------#
#         #   对输入图像进行分区，并下采样
#         #   512, 512, 3 => 128, 128, 32 => 16384, 32
#         #-----------------------------------------------#
#         self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
#         #-----------------------------------------------#
#         #   利用transformer模块进行特征提取
#         #   16384, 32 => 16384, 32
#         #-----------------------------------------------#
#         cur = 0
#         self.block1 = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]
#                 )
#                 for i in range(depths[0])
#             ]
#         )
#         self.norm1 = norm_layer(embed_dims[0])
        
#         #----------------------------------#
#         #   block2
#         #----------------------------------#
#         #-----------------------------------------------#
#         #   对输入图像进行分区，并下采样
#         #   128, 128, 32 => 64, 64, 64 => 4096, 64
#         #-----------------------------------------------#
#         self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
#         #-----------------------------------------------#
#         #   利用transformer模块进行特征提取
#         #   4096, 64 => 4096, 64
#         #-----------------------------------------------#
#         cur += depths[0]
#         self.block2 = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]
#                 )
#                 for i in range(depths[1])
#             ]
#         )
#         self.norm2 = norm_layer(embed_dims[1])

#         #----------------------------------#
#         #   block3
#         #----------------------------------#
#         #-----------------------------------------------#
#         #   对输入图像进行分区，并下采样
#         #   64, 64, 64 => 32, 32, 160 => 1024, 160
#         #-----------------------------------------------#
#         self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
#         #-----------------------------------------------#
#         #   利用transformer模块进行特征提取
#         #   1024, 160 => 1024, 160
#         #-----------------------------------------------#
#         cur += depths[1]
#         self.block3 = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]
#                 )
#                 for i in range(depths[2])
#             ]
#         )
#         self.norm3 = norm_layer(embed_dims[2])

#         #----------------------------------#
#         #   block4
#         #----------------------------------#
#         #-----------------------------------------------#
#         #   对输入图像进行分区，并下采样
#         #   32, 32, 160 => 16, 16, 256 => 256, 256
#         #-----------------------------------------------#
#         self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
#         #-----------------------------------------------#
#         #   利用transformer模块进行特征提取
#         #   256, 256 => 256, 256
#         #-----------------------------------------------#
#         cur += depths[2]
#         self.block4 = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]
#                 )
#                 for i in range(depths[3])
#             ]
#         )
#         self.norm4 = norm_layer(embed_dims[3])

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
                
#     def forward(self, x):
#         B = x.shape[0]
#         outs = []

#         #----------------------------------#
#         #   block1
#         #----------------------------------#
#         x, H, W = self.patch_embed1.forward(x)
#         for i, blk in enumerate(self.block1):
#             x = blk.forward(x, H, W)
#         x = self.norm1(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         #----------------------------------#
#         #   block2
#         #----------------------------------#
#         x, H, W = self.patch_embed2.forward(x)
#         for i, blk in enumerate(self.block2):
#             x = blk.forward(x, H, W)
#         x = self.norm2(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         #----------------------------------#
#         #   block3
#         #----------------------------------#
#         x, H, W = self.patch_embed3.forward(x)
#         for i, blk in enumerate(self.block3):
#             x = blk.forward(x, H, W)
#         x = self.norm3(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         #----------------------------------#
#         #   block4
#         #----------------------------------#
#         x, H, W = self.patch_embed4.forward(x)
#         for i, blk in enumerate(self.block4):
#             x = blk.forward(x, H, W)
#         x = self.norm4(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         return outs

# # --------------------------------------------------------------------------------------- #
# #   辅助函数：用于处理权重加载时的通道不匹配问题
# # --------------------------------------------------------------------------------------- #
# def load_pretrained_weights(model, weight_path, in_chans):
#     print(f"Load backbone weights from {weight_path}")
#     state_dict = torch.load(weight_path)
    
#     # 如果输入通道不是3，且预训练权重是3通道，需要调整第一层卷积的权重形状
#     if in_chans != 3:
#         # 获取第一层 Patch Embedding 的权重键名
#         key = 'patch_embed1.proj.weight'
#         if key in state_dict:
#             weight = state_dict[key] # shape: [Out, 3, K, K]
            
#             # 检查预训练权重的通道数是否为3
#             if weight.shape[1] == 3:
#                 print(f"Expanding pre-trained weights from 3 channels to {in_chans} channels.")
                
#                 # 创建新的权重张量
#                 new_weight = torch.empty(weight.shape[0], in_chans, weight.shape[2], weight.shape[3])
                
#                 # 1. 复制原有的 RGB 权重到前 3 个通道
#                 new_weight[:, :3, :, :] = weight
                
#                 # 2. 对新增的通道 (第4到第8通道) 进行 Kaiming 初始化
#                 # 这样可以保持原有特征提取能力，同时让新通道有合理的初始值
#                 nn.init.kaiming_normal_(new_weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                
#                 # 更新 state_dict
#                 state_dict[key] = new_weight
#             else:
#                 print(f"Warning: Pre-trained weight channel size ({weight.shape[1]}) matches input or is not 3. No expansion applied.")

#     model.load_state_dict(state_dict, strict=False)


# # --------------------------------------------------------------------------------------- #
# #   修改后的模型定义 (支持 in_chans 参数)
# # --------------------------------------------------------------------------------------- #

# class mit_b0(MixVisionTransformer):
#     def __init__(self, pretrained=False, in_chans=8):
#         super(mit_b0, self).__init__(
#             in_chans=in_chans,  # 传入通道数
#             embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             load_pretrained_weights(self, "model_data/segformer_b0_backbone_weights.pth", in_chans)

# class mit_b1(MixVisionTransformer):
#     def __init__(self, pretrained=False, in_chans=8):
#         super(mit_b1, self).__init__(
#             in_chans=in_chans,
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             load_pretrained_weights(self, "model_data/segformer_b1_backbone_weights.pth", in_chans)

# class mit_b2(MixVisionTransformer):
#     def __init__(self, pretrained=False, in_chans=8):
#         super(mit_b2, self).__init__(
#             in_chans=in_chans,
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             load_pretrained_weights(self, "model_data/segformer_b2_backbone_weights.pth", in_chans)

# class mit_b3(MixVisionTransformer):
#     def __init__(self, pretrained=False, in_chans=8):
#         super(mit_b3, self).__init__(
#             in_chans=in_chans,
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             load_pretrained_weights(self, "model_data/segformer_b3_backbone_weights.pth", in_chans)

# class mit_b4(MixVisionTransformer):
#     def __init__(self, pretrained=False, in_chans=8):
#         super(mit_b4, self).__init__(
#             in_chans=in_chans,
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             load_pretrained_weights(self, "model_data/segformer_b4_backbone_weights.pth", in_chans)

# class mit_b5(MixVisionTransformer):
#     def __init__(self, pretrained=False, in_chans=8):
#         super(mit_b5, self).__init__(
#             in_chans=in_chans,
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             load_pretrained_weights(self, "model_data/segformer_b5_backbone_weights.pth", in_chans)

# 去掉频域
# import math
# import warnings
# import numpy as np
# from functools import partial
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ==============================================================================
# # Helper Functions
# # ==============================================================================
# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     def norm_cdf(x):
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.

#     if (mean < a - 2 * std) or (mean > b + 2 * std):
#         warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#                       "The distribution of values may be incorrect.",
#                       stacklevel=2)

#     with torch.no_grad():
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)
#         tensor.uniform_(2 * l - 1, 2 * u - 1)
#         tensor.erfinv_()
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)
#         tensor.clamp_(min=a, max=b)
#         return tensor

# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# class GELU(nn.Module):
#     def __init__(self):
#         super(GELU, self).__init__()
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

# # ==============================================================================
# # 1. Channel Attention Adapter (保持用于8通道适配)
# # ==============================================================================
# class ChannelAttentionAdapter(nn.Module):
#     """
#     用于将多波段（如8通道）数据适配到 SegFormer 的 3通道输入，
#     并利用 SE Attention 自动学习波段重要性。
#     """
#     def __init__(self, in_chans, out_chans=3, reduction=4):
#         super().__init__()
#         # 通道注意力 (SE Module)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_chans, in_chans // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_chans // reduction, in_chans, bias=False),
#             nn.Sigmoid()
#         )
#         # 降维映射: N -> 3
#         self.mapping = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        
#         # 初始化
#         self._init_weights(in_chans, out_chans)

#     def _init_weights(self, in_chans, out_chans):
#         # 初始化映射层
#         nn.init.kaiming_normal_(self.mapping.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         # 1. SE Attention: 学习波段权重
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         x = x * y.expand_as(x) # 动态加权
        
#         # 2. Mapping: 映射到 3 通道
#         x = self.mapping(x)
#         return x

# # ==============================================================================
# # Existing Modules (OverlapPatchEmbed, Attention, etc.)
# # ==============================================================================

# class OverlapPatchEmbed(nn.Module):
#     def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
#         super().__init__()
#         patch_size = (patch_size, patch_size)
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
#                               padding=(patch_size[0] // 2, patch_size[1] // 2))
#         self.norm = nn.LayerNorm(embed_dim)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x, H, W

# # ===================== Standard Attention (Original SegFormer) =====================
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None, scale_by_keep=True):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep

#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#         keep_prob = 1 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#         if keep_prob > 0.0 and self.scale_by_keep:
#             random_tensor.div_(keep_prob)
#         return x * random_tensor

# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W)
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)
#         return x

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class Block(nn.Module):
#     # 修改：移除了 use_fsa 等参数
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
#                  drop=0., attn_drop=0., drop_path=0., act_layer=GELU, 
#                  norm_layer=nn.LayerNorm, sr_ratio=1):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
        
#         # 修改：强制使用标准 Attention
#         self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        
#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         x = x + self.drop_path(self.attn(self.norm1(x), H, W))
#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
#         return x

# # ==============================================================================
# # Main Backbone Class: MixVisionTransformer
# # ==============================================================================
# class MixVisionTransformer(nn.Module):
#     def __init__(self, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
#                  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
#                  depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
#         super().__init__()
#         self.num_classes = num_classes
#         self.depths = depths

#         # ====================================================
#         # 8-Channel Adapter Logic
#         # ====================================================
#         if in_chans != 3:
#             self.channel_adapter = ChannelAttentionAdapter(in_chans, out_chans=3)
#         else:
#             self.channel_adapter = nn.Identity()

#         # Transformer parts
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
#         # Block 1
#         self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
#         cur = 0
#         self.block1 = nn.ModuleList([
#             Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, 
#                   qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
#                   norm_layer=norm_layer, sr_ratio=sr_ratios[0])
#             for i in range(depths[0])
#         ])
#         self.norm1 = norm_layer(embed_dims[0])

#         # Block 2
#         self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
#         cur += depths[0]
#         self.block2 = nn.ModuleList([
#             Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, 
#                   qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
#                   norm_layer=norm_layer, sr_ratio=sr_ratios[1])
#             for i in range(depths[1])
#         ])
#         self.norm2 = norm_layer(embed_dims[1])

#         # Block 3
#         self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
#         cur += depths[1]
#         self.block3 = nn.ModuleList([
#             Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, 
#                   qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
#                   norm_layer=norm_layer, sr_ratio=sr_ratios[2])
#             for i in range(depths[2])
#         ])
#         self.norm3 = norm_layer(embed_dims[2])

#         # Block 4
#         # 修改：移除了 use_fsa 等参数，恢复为标准 SegFormer 块
#         self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
#         cur += depths[2]
#         self.block4 = nn.ModuleList([
#             Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, 
#                   qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
#                   norm_layer=norm_layer, sr_ratio=sr_ratios[3])
#             for i in range(depths[3])
#         ])
#         self.norm4 = norm_layer(embed_dims[3])

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # Channel Adapter 处理多波段输入
#         x = self.channel_adapter(x)

#         B = x.shape[0]
#         outs = []

#         # Stage 1
#         x, H, W = self.patch_embed1(x)
#         for i, blk in enumerate(self.block1):
#             x = blk(x, H, W)
#         x = self.norm1(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         # Stage 2
#         x, H, W = self.patch_embed2(x)
#         for i, blk in enumerate(self.block2):
#             x = blk(x, H, W)
#         x = self.norm2(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         # Stage 3
#         x, H, W = self.patch_embed3(x)
#         for i, blk in enumerate(self.block3):
#             x = blk(x, H, W)
#         x = self.norm3(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         # Stage 4
#         x, H, W = self.patch_embed4(x)
#         for blk in self.block4:
#             x = blk(x, H, W)
        
#         # 修改：移除了 freq_map 提取逻辑
#         x = self.norm4(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         return outs

# # ==============================================================================
# # Backbone Variants (B0 - B5)
# # ==============================================================================
# class mit_b0(MixVisionTransformer):
#     def __init__(self, pretrained=False, **kwargs):
#         super(mit_b0, self).__init__(
#             embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, **kwargs)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b0_backbone_weights.pth"), strict=False)

# class mit_b1(MixVisionTransformer):
#     def __init__(self, pretrained=False, **kwargs):
#         super(mit_b1, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, **kwargs)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b1_backbone_weights.pth"), strict=False)

# class mit_b2(MixVisionTransformer):
#     def __init__(self, pretrained=False, **kwargs):
#         super(mit_b2, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, **kwargs)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b2_backbone_weights.pth"), strict=False)

# class mit_b3(MixVisionTransformer):
#     def __init__(self, pretrained=False, **kwargs):
#         super(mit_b3, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, **kwargs)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b3_backbone_weights.pth"), strict=False)

# class mit_b4(MixVisionTransformer):
#     def __init__(self, pretrained=False, **kwargs):
#         super(mit_b4, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, **kwargs)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b4_backbone_weights.pth"), strict=False)

# class mit_b5(MixVisionTransformer):
#     def __init__(self, pretrained=False, **kwargs):
#         super(mit_b5, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, **kwargs)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b5_backbone_weights.pth"), strict=False)
            
# 基础跑Transformer            
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# import math
# import warnings
# import numpy as np
# from functools import partial

# import torch
# import torch.nn as nn


# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     # Cut & paste from PyTorch official master until it's in a few official releases - RW
#     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     def norm_cdf(x):
#         # Computes standard normal cumulative distribution function
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.

#     if (mean < a - 2 * std) or (mean > b + 2 * std):
#         warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#                       "The distribution of values may be incorrect.",
#                       stacklevel=2)

#     with torch.no_grad():
#         # Values are generated by using a truncated uniform distribution and
#         # then using the inverse CDF for the normal distribution.
#         # Get upper and lower cdf values
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)

#         # Uniformly fill tensor with values from [l, u], then translate to
#         # [2l-1, 2u-1].
#         tensor.uniform_(2 * l - 1, 2 * u - 1)

#         # Use inverse cdf transform for normal distribution to get truncated
#         # standard normal
#         tensor.erfinv_()

#         # Transform to proper mean, std
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)

#         # Clamp to ensure it's in the proper range
#         tensor.clamp_(min=a, max=b)
#         return tensor


# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     r"""
#     Fills the input Tensor with values drawn from a truncated
#     normal distribution. The values are effectively drawn from the
#     normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
#     with values outside :math:`[a, b]` redrawn until they are within
#     the bounds. The method used for generating the random values works
#     best when :math:`a \leq \text{mean} \leq b`.
#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         mean: the mean of the normal distribution
#         std: the standard deviation of the normal distribution
#         a: the minimum cutoff value
#         b: the maximum cutoff value
#     Examples:
#         >>> w = torch.empty(3, 5)
#         >>> nn.init.trunc_normal_(w)
#     """
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# #--------------------------------------#
# #   Gelu激活函数的实现
# #   利用近似的数学公式
# #--------------------------------------#
# class GELU(nn.Module):
#     def __init__(self):
#         super(GELU, self).__init__()

#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))

# class OverlapPatchEmbed(nn.Module):
#     def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
#         super().__init__()
#         patch_size  = (patch_size, patch_size)
#         self.proj   = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
#                               padding=(patch_size[0] // 2, patch_size[1] // 2))
#         self.norm   = nn.LayerNorm(embed_dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)

#         return x, H, W

# #--------------------------------------------------------------------------------------------------------------------#
# #   Attention机制
# #   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
# #   然后利用 查询向量query 叉乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
# #   然后利用 score 叉乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
# #   
# #   在segformer中，为了减少计算量，首先对特征图进行了浓缩，所有特征层都压缩到原图的1/32。
# #   当输入图片为512, 512时，Block1的特征图为128, 128，此时就先将特征层压缩为16, 16。
# #   在Block1的Attention模块中，相当于将8x8个特征点进行特征浓缩，浓缩为一个特征点。
# #   然后利用128x128个查询向量对16x16个键向量与值向量进行查询。尽管键向量与值向量的数量较少，但因为查询向量的不同，依然可以获得不同的输出。
# #--------------------------------------------------------------------------------------------------------------------#
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim        = dim
#         self.num_heads  = num_heads
#         head_dim        = dim // num_heads
#         self.scale      = qk_scale or head_dim ** -0.5

#         self.q          = nn.Linear(dim, dim, bias=qkv_bias)
        
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr     = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm   = nn.LayerNorm(dim)
#         self.kv         = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
#         self.attn_drop  = nn.Dropout(attn_drop)
        
#         self.proj       = nn.Linear(dim, dim)
#         self.proj_drop  = nn.Dropout(proj_drop)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         if self.sr_ratio > 1:
#             # bs, 16384, 32 => bs, 32, 128, 128
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 8, 4 => 2, bs, 8, 256, 4
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         # bs, 8, 16384, 4 @ bs, 8, 4, 256 => bs, 8, 16384, 256 
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         # bs, 8, 16384, 256  @ bs, 8, 256, 4 => bs, 8, 16384, 4 => bs, 16384, 32
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         # bs, 16384, 32 => bs, 16384, 32
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

# def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
#     """
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob       = 1 - drop_prob
#     shape           = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor   = x.new_empty(shape).bernoulli_(keep_prob)
#     if keep_prob > 0.0 and scale_by_keep:
#         random_tensor.div_(keep_prob)
#     return x * random_tensor

# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None, scale_by_keep=True):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W)
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)

#         return x
    
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
#         super().__init__()
#         out_features    = out_features or in_features
#         hidden_features = hidden_features or in_features
        
#         self.fc1    = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act    = act_layer()
        
#         self.fc2    = nn.Linear(hidden_features, out_features)
        
#         self.drop   = nn.Dropout(drop)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
#         super().__init__()
#         self.norm1      = norm_layer(dim)
        
#         self.attn       = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
#         )
#         self.norm2      = norm_layer(dim)
#         self.mlp        = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

#         self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         x = x + self.drop_path(self.attn(self.norm1(x), H, W))
#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
#         return x

# class MixVisionTransformer(nn.Module):
#     def __init__(self, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
#                  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
#                  depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
#         super().__init__()
#         self.num_classes    = num_classes
#         self.depths         = depths

#         #----------------------------------#
#         #   Transformer模块，共有四个部分
#         #----------------------------------#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
#         #----------------------------------#
#         #   block1
#         #----------------------------------#
#         #-----------------------------------------------#
#         #   对输入图像进行分区，并下采样
#         #   512, 512, 3 => 128, 128, 32 => 16384, 32
#         #-----------------------------------------------#
#         self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
#         #-----------------------------------------------#
#         #   利用transformer模块进行特征提取
#         #   16384, 32 => 16384, 32
#         #-----------------------------------------------#
#         cur = 0
#         self.block1 = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]
#                 )
#                 for i in range(depths[0])
#             ]
#         )
#         self.norm1 = norm_layer(embed_dims[0])
        
#         #----------------------------------#
#         #   block2
#         #----------------------------------#
#         #-----------------------------------------------#
#         #   对输入图像进行分区，并下采样
#         #   128, 128, 32 => 64, 64, 64 => 4096, 64
#         #-----------------------------------------------#
#         self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
#         #-----------------------------------------------#
#         #   利用transformer模块进行特征提取
#         #   4096, 64 => 4096, 64
#         #-----------------------------------------------#
#         cur += depths[0]
#         self.block2 = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]
#                 )
#                 for i in range(depths[1])
#             ]
#         )
#         self.norm2 = norm_layer(embed_dims[1])

#         #----------------------------------#
#         #   block3
#         #----------------------------------#
#         #-----------------------------------------------#
#         #   对输入图像进行分区，并下采样
#         #   64, 64, 64 => 32, 32, 160 => 1024, 160
#         #-----------------------------------------------#
#         self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
#         #-----------------------------------------------#
#         #   利用transformer模块进行特征提取
#         #   1024, 160 => 1024, 160
#         #-----------------------------------------------#
#         cur += depths[1]
#         self.block3 = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]
#                 )
#                 for i in range(depths[2])
#             ]
#         )
#         self.norm3 = norm_layer(embed_dims[2])

#         #----------------------------------#
#         #   block4
#         #----------------------------------#
#         #-----------------------------------------------#
#         #   对输入图像进行分区，并下采样
#         #   32, 32, 160 => 16, 16, 256 => 256, 256
#         #-----------------------------------------------#
#         self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
#         #-----------------------------------------------#
#         #   利用transformer模块进行特征提取
#         #   256, 256 => 256, 256
#         #-----------------------------------------------#
#         cur += depths[2]
#         self.block4 = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]
#                 )
#                 for i in range(depths[3])
#             ]
#         )
#         self.norm4 = norm_layer(embed_dims[3])

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
                
#     def forward(self, x):
#         B = x.shape[0]
#         outs = []

#         #----------------------------------#
#         #   block1
#         #----------------------------------#
#         x, H, W = self.patch_embed1.forward(x)
#         for i, blk in enumerate(self.block1):
#             x = blk.forward(x, H, W)
#         x = self.norm1(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         #----------------------------------#
#         #   block2
#         #----------------------------------#
#         x, H, W = self.patch_embed2.forward(x)
#         for i, blk in enumerate(self.block2):
#             x = blk.forward(x, H, W)
#         x = self.norm2(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         #----------------------------------#
#         #   block3
#         #----------------------------------#
#         x, H, W = self.patch_embed3.forward(x)
#         for i, blk in enumerate(self.block3):
#             x = blk.forward(x, H, W)
#         x = self.norm3(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         #----------------------------------#
#         #   block4
#         #----------------------------------#
#         x, H, W = self.patch_embed4.forward(x)
#         for i, blk in enumerate(self.block4):
#             x = blk.forward(x, H, W)
#         x = self.norm4(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

#         return outs

# class mit_b0(MixVisionTransformer):
#     def __init__(self, pretrained = False):
#         super(mit_b0, self).__init__(
#             embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b0_backbone_weights.pth"), strict=False)

# class mit_b1(MixVisionTransformer):
#     def __init__(self, pretrained = False):
#         super(mit_b1, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b1_backbone_weights.pth"), strict=False)

# class mit_b2(MixVisionTransformer):
#     def __init__(self, pretrained = False):
#         super(mit_b2, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b2_backbone_weights.pth"), strict=False)

# class mit_b3(MixVisionTransformer):
#     def __init__(self, pretrained = False):
#         super(mit_b3, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b3_backbone_weights.pth"), strict=False)

# class mit_b4(MixVisionTransformer):
#     def __init__(self, pretrained = False):
#         super(mit_b4, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b4_backbone_weights.pth"), strict=False)

# class mit_b5(MixVisionTransformer):
#     def __init__(self, pretrained = False):
#         super(mit_b5, self).__init__(
#             embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#         if pretrained:
#             print("Load backbone weights")
#             self.load_state_dict(torch.load("model_data/segformer_b5_backbone_weights.pth"), strict=False)


