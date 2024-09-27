from typing import Callable, Optional, Union, List, Tuple, Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.profiler import profile, ProfilerActivity

from network.module.CBAM import CBAM


def upsample_size(x: Tensor,
                  size: Any,
                  align_corners: bool = True) -> Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


def upsample_scale(x: Tensor,
                   scale: Any,
                   align_corners: bool = True) -> Tensor:
    return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=align_corners)


class Permute(nn.Module):
    def __init__(self,
                 dims: List[int]) -> None:
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self,
                x: Tensor) -> Tensor:
        return x.permute(self.dims)


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 group: int = 1,
                 dilation: int = 1,
                 bias: Optional[bool] = None,
                 use_refl: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
                 inplace: bool = True) -> None:
        super(ConvNormAct, self).__init__()

        layers: List[nn.Module] = []

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        if use_refl:
            pad = nn.ReflectionPad2d(padding)
            layers.append(pad)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation,
                             groups=group, bias=bias)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation,
                             groups=group, bias=bias)
        layers.append(conv)

        if norm_layer is not None:
            norm = norm_layer(out_channels)
            layers.append(norm)

        if act_layer is not None:
            if hasattr(act_layer, 'inplace'):
                act = act_layer(inplace=inplace)
            else:
                act = act_layer()
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self,
                x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class PPM(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool_dims: Tuple[int, ...] = (1, 2, 3, 6),
                 pool_layer: Optional[Callable[..., nn.Module]] = nn.AdaptiveAvgPool2d,
                 bias: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(PPM, self).__init__()

        inter_channels = in_channels // len(pool_dims)
        ppm_layers: List[nn.Module] = []

        for dim in pool_dims:
            ppm_layers.append(
                nn.Sequential(pool_layer(dim),
                              ConvNormAct(in_channels, inter_channels, 1, 1, 0,
                                          norm_layer=norm_layer, act_layer=act_layer, bias=bias))
            )

        self.ppm_layers = nn.Sequential(*ppm_layers)
        self.ppm_last_layer = ConvNormAct(in_channels * 2, out_channels, 1, 1, 0,
                                          norm_layer=norm_layer, act_layer=act_layer, bias=bias)

    def forward(self,
                x: Tensor) -> Tensor:
        size = x.shape[2:]
        ppm_outs: List[Tensor] = [x]
        for ppm_layer in self.ppm_layers:
            ppm_outs.append(upsample_size(ppm_layer(x), size))

        ppm_out = self.ppm_last_layer(torch.cat(ppm_outs, dim=1))
        return ppm_out


class SE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction: int = 16) -> None:
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            ConvNormAct(in_channels, in_channels // reduction, 1, 1, 0),
            ConvNormAct(in_channels // reduction, in_channels, 1, 1, 0, act_layer=None),
            nn.Sigmoid()
        )

    def forward(self,
                x: Tensor) -> Tensor:
        # B, C, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


class CAPv2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = None,
                 out_channels: int = None,
                 atrous_rates: Tuple[Tuple[int, ...]] = ((1, 1, 1), (1, 2, 1), (1, 4, 1), (1, 3, 5, 1),
                                                         (1, 3, 5, 8, 1)),
                 pool_scale: int = 1,
                 bias: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(CAPv2, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = out_channels

        cap_layers: List[nn.Module] = []

        for dilation_rates in atrous_rates:
            cap_layer: List[nn.Module] = []
            for i, dilation_rate in enumerate(dilation_rates):
                if i == 0:
                    cap_layer.append(ConvNormAct(in_channels, hidden_channels, 3, 1, padding=dilation_rate,
                                                 dilation=dilation_rate, norm_layer=norm_layer, act_layer=act_layer,
                                                 bias=bias))
                elif i == len(dilation_rates) - 1:
                    cap_layer.append(ConvNormAct(hidden_channels, out_channels, 3, 1, padding=dilation_rate,
                                                 dilation=dilation_rate, norm_layer=norm_layer, act_layer=act_layer,
                                                 bias=bias))
                else:
                    cap_layer.append(ConvNormAct(hidden_channels, hidden_channels, 3, 1, padding=dilation_rate,
                                                 dilation=dilation_rate, norm_layer=norm_layer, act_layer=act_layer,
                                                 bias=bias))
            cap_layers.append(nn.Sequential(*cap_layer))

        cap_layers.append(
            nn.Sequential(nn.AdaptiveAvgPool2d(pool_scale),
                          ConvNormAct(in_channels, out_channels, 1, 1, 0,
                                      norm_layer=norm_layer, act_layer=act_layer, bias=bias))
        )

        self.cap_layers = nn.Sequential(*cap_layers)

        fusion_channels = len(self.cap_layers) * out_channels
        self.fuse = nn.Sequential(CBAM(fusion_channels),
                                  ConvNormAct(fusion_channels, out_channels, 1, 1, 0,
                                              norm_layer=norm_layer, act_layer=act_layer, bias=bias),
                                  nn.Dropout(0.5))

    def forward(self,
                x: Tensor) -> Tensor:
        # size = x.shape[2:]
        cap_flow = 0
        cap_outs: List[Tensor] = []

        for cap_layer in self.cap_layers[:-1]:
            cap_out = cap_layer[0](x)
            cap_out = cap_out + cap_flow
            cap_out = cap_layer[1:](cap_out)
            cap_flow = cap_out
            cap_outs.append(cap_out)
        # cap_outs.append(upsample_size(self.cap_layers[-1](x), size))
        cap_outs.append(self.cap_layers[-1](x).expand_as(x))

        cap_out = self.fuse(torch.cat(cap_outs, dim=1))
        return cap_out


class HBD(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bias: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(HBD, self).__init__()

        hbd_layers: List[nn.Module] = [ConvNormAct(in_channels, out_channels, 1, 1, 0,
                                       norm_layer=norm_layer, act_layer=act_layer, bias=bias)]
        self.hbd_layers = nn.Sequential(*hbd_layers)

    def forward(self,
                hbd_input: Tensor) -> Tensor:
        hbd_output = self.hbd_layers(hbd_input)
        return hbd_output


class ConvFusion(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction: int = 16,
                 bias: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(ConvFusion, self).__init__()
        self.res_attn = SE(in_channels * 2, reduction=reduction)
        self.res_conv = ConvNormAct(in_channels * 2, in_channels, 1, 1, 0,
                                    norm_layer=norm_layer, act_layer=act_layer, bias=bias)

    def forward(self,
                low_f: Tensor,
                high_f: Tensor) -> Tensor:
        low_res = low_f - high_f
        high_res = high_f - low_f
        res = torch.cat([low_res, high_res], dim=1)
        res = self.res_attn(res)
        res = self.res_conv(res)
        fused_f = low_f + high_f + res
        return fused_f


class GSDFPN(nn.Module):
    def __init__(self,
                 in_channels: Tuple[int, ...],
                 out_channels: int,
                 skip_top: bool = True,
                 context_layer: Optional[Callable[..., nn.Module]] = CAPv2,
                 boundary_layer: Optional[Callable[..., nn.Module]] = HBD,
                 bias: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(GSDFPN, self).__init__()

        self.skip_top = skip_top
        self.ppm = PPM(in_channels=in_channels[-1], out_channels=out_channels, pool_dims=(1, 2, 3, 6), bias=bias,
                        norm_layer=norm_layer, act_layer=act_layer)
        self.context_layer = context_layer
        self.boundary_layer = boundary_layer

        fpn_in_layers: List[nn.Module] = []
        fpn_out_layers: List[nn.Module] = []
        fuse_layers: List[nn.Module] = []
        if self.context_layer is not None:
            context_layers: List[nn.Module] = []
        if self.boundary_layer is not None:
            boundary_layers: List[nn.Module] = [self.boundary_layer(in_channels=out_channels, out_channels=1, bias=bias,
                                                                    norm_layer=norm_layer, act_layer=act_layer)]

        for i in range(len(in_channels) - self.skip_top):
            fpn_in_layers.append(ConvNormAct(in_channels[i], out_channels, 1, 1, 0,
                                             norm_layer=norm_layer, act_layer=act_layer, bias=bias))
            fpn_out_layers.append(ConvNormAct(out_channels, out_channels, 3, 1, 1,
                                              norm_layer=norm_layer, act_layer=act_layer, bias=bias))
            fuse_layers.append(ConvFusion(out_channels, reduction=16, norm_layer=norm_layer, act_layer=act_layer, bias=bias))
            if self.context_layer is not None:
                context_layers.append(self.context_layer(in_channels=out_channels, out_channels=out_channels,
                                                         pool_scale=1, bias=bias, norm_layer=norm_layer,
                                                         act_layer=act_layer))
            if self.boundary_layer is not None:
                boundary_layers.append(self.boundary_layer(in_channels=out_channels, out_channels=1, bias=bias,
                                                           norm_layer=norm_layer, act_layer=act_layer))

        self.fpn_in_layers = nn.Sequential(*fpn_in_layers)
        self.fpn_out_layers = nn.Sequential(*fpn_out_layers)
        self.fuse_layers = nn.Sequential(*fuse_layers)
        if self.context_layer is not None:
            self.context_layers = nn.Sequential(*context_layers)
        if self.boundary_layer is not None:
            self.boundary_layers = nn.Sequential(*boundary_layers)

        self.layer_num = len(self.fpn_in_layers)

    def forward(self,
                fpn_input: List[Tensor]) -> Union[List[Tensor], Tuple[List[Tensor], List[Tensor]]]:
        fpn_out: List[Tensor] = []

        backbone_f = fpn_input[-1]
        f = self.ppm(backbone_f)
        fpn_out.insert(0, f)

        if self.boundary_layer is not None:
            boundary_outs: List[Tensor] = []
            boundary_out = self.boundary_layers[-1](f)
            boundary_outs.insert(0, boundary_out)

        # Cascaded Network Architecture
        for i in range(self.layer_num):
            backbone_f = fpn_input[-(2 + i)]
            x = self.fpn_in_layers[-(1 + i)](backbone_f)

            size = x.shape[2:]
            f = upsample_size(f, size)
            f = self.fuse_layers[-(1 + i)](x, f)

            if self.boundary_layer is not None:
                boundary_out = upsample_size(boundary_out, size)
                f = f + torch.sigmoid(boundary_out) * f

            if self.context_layer is not None:
                f = self.context_layers[-(1 + i)](f)

            if self.boundary_layer is not None:
                fpn_out.insert(0, self.fpn_out_layers[-(1 + i)](f + torch.sigmoid(boundary_out) * f))

                boundary_out = self.boundary_layers[-(2 + i)](f)
                boundary_outs.insert(0, boundary_out)
            else:
                fpn_out.insert(0, self.fpn_out_layers[-(1 + i)](f))

        if self.boundary_layer is not None:
            return fpn_out, boundary_outs
        else:
            return fpn_out


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 kv_bias: bool = False,
                 kv_scale: float = None,
                 attn_drop_ratio: float = 0.1,
                 proj_drop_ratio: float = 0.1) -> None:
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=kv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)

        self.scale = kv_scale or self.head_dim ** -0.5
        # self.attn_drop = nn.Dropout(attn_drop_ratio)

        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self,
                context_x: Tensor,
                depth_x: Tensor) -> Tensor:
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = context_x.shape

        # kv(): -> [batch_size, num_patches, 2 * total_embed_dim]
        # reshape: -> [batch_size, num_patches, 2, num_heads, embed_dim_per_head]
        # permute: -> [2, batch_size, num_heads, num_patches, embed_dim_per_head]
        q = self.q(context_x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(depth_x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches, embed_dim_per_head]
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # @: multiply -> [batch_size, num_heads, embed_dim_per_head, embed_dim_per_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # permute: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 drop_ratio: float = 0.1,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ELU,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm) -> None:
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv = ConvNormAct(in_features, in_features, kernel_size=3, stride=1, padding=1, group=in_features,
                                bias=True, norm_layer=None, act_layer=None, use_refl=True)
        self.norm = norm_layer(in_features)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # self.drop = nn.Dropout(drop_ratio)

        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self,
                x: Tensor) -> Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)

        x = self.fc2(x)
        # x = self.drop(x)
        return x


class TransBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 kv_bias: bool = False,
                 kv_scale: float = None,
                 layer_scale: float = 1e-6,
                 drop_ratio: float = 0.1,
                 attn_drop_ratio: float = 0.1,
                 drop_path_ratio: float = 0.1,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ELU,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm) -> None:
        super(TransBlock, self).__init__()
        self.norm_context = norm_layer(dim)
        self.norm_depth = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, kv_bias=kv_bias, kv_scale=kv_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm_mlp = norm_layer(dim)
        mlp_hidden_dim = dim * mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop_ratio=drop_ratio, act_layer=act_layer,
                       norm_layer=norm_layer)
        # self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale, requires_grad=True) \
        #     if layer_scale > 0 else None

    def forward(self,
                context_x: Tensor,
                depth_x: Tensor) -> Tensor:
        # [batch_size, channels, height, width]
        # identity: Tensor = depth_x

        B, C, H, W = depth_x.shape
        # [batch_size, height * width, channels]
        context_x = context_x.reshape(B, C, H * W).permute(0, 2, 1)
        context_x = self.norm_context(context_x)
        depth_x = depth_x.reshape(B, C, H * W).permute(0, 2, 1)
        depth_x = self.norm_depth(depth_x)
        # x = depth_x + self.drop_path(self.attn(context_x, depth_x))
        x = self.attn(context_x, depth_x)

        # [batch_size, height, width, channels]
        x = x.reshape(B, H, W, C)
        # x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        x = self.mlp(self.norm_mlp(x))
        # [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)

        # if self.layer_scale is not None:
        #     x = self.layer_scale * x
        # x = identity + self.drop_path(x)
        return x


class GSDEFPN(nn.Module):
    def __init__(self,
                 in_channels: Tuple[int, ...],
                 out_channels: int,
                 skip_top: bool = True,
                 bias: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ELU) -> None:
        super(GSDEFPN, self).__init__()
        self.skip_top = skip_top

        fpn_in_layers: List[nn.Module] = [ConvNormAct(in_channels[0], out_channels, 1, 1, 0,
                                                      norm_layer=norm_layer, act_layer=act_layer, bias=bias)]
        fpn_out_layers: List[nn.Module] = [ConvNormAct(out_channels, out_channels, 3, 1, 1,
                                                       norm_layer=norm_layer, act_layer=act_layer, bias=bias, use_refl=True)]
        attn_layers: List[nn.Module] = [TransBlock(dim=out_channels, num_heads=8, kv_bias=True, layer_scale=0, drop_ratio=0,
                                                   drop_path_ratio=0, attn_drop_ratio=0, act_layer=act_layer)]

        for i in range(len(in_channels) - self.skip_top):
            fpn_in_layers.append(ConvNormAct(in_channels[i + 1], out_channels, 1, 1, 0,
                                             norm_layer=norm_layer, act_layer=act_layer, bias=bias))
            fpn_out_layers.append(ConvNormAct(out_channels, out_channels, 3, 1, 1,
                                              norm_layer=norm_layer, act_layer=act_layer, bias=bias, use_refl=True))
            attn_layers.append(TransBlock(dim=out_channels, num_heads=8, kv_bias=True, layer_scale=0, drop_ratio=0,
                                          drop_path_ratio=0, attn_drop_ratio=0, act_layer=act_layer))

        self.fpn_in_layers = nn.Sequential(*fpn_in_layers)
        self.fpn_out_layers = nn.Sequential(*fpn_out_layers)
        self.attn_layers = nn.Sequential(*attn_layers)

        self.layer_num = len(self.fpn_in_layers)

    def forward(self,
                context_input: List[Tensor],
                fpn_input: List[Tensor]) -> List[Tensor]:
        fpn_out: List[Tensor] = []

        backbone_f = fpn_input[-1]
        f = self.fpn_in_layers[-1](backbone_f)
        f = self.attn_layers[-1](context_input[-1], f)
        fpn_out.insert(0, self.fpn_out_layers[-1](f))

        # Cascaded Network Architecture
        for i in range(1, self.layer_num):
            backbone_f = fpn_input[-(1 + i)]
            x = self.fpn_in_layers[-(1 + i)](backbone_f)

            size = x.shape[2:]
            f = upsample_size(f, size)
            f = x + f

            f = self.attn_layers[-(1 + i)](context_input[-(1 + i)], f)

            fpn_out.insert(0, self.fpn_out_layers[-(1 + i)](f))
        return fpn_out


if __name__ == "__main__":
    test_input = torch.randn(2, 128, 26, 26)
    test_module = SE(in_channels=128)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        x = test_module(test_input)
    print(x.size())

    num_params = sum(p.numel() for p in test_module.parameters())
    print(num_params)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))