from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.misc import Conv2dNormActivation  # 一个包含单层卷积层、Normalizaiton层和激活层的网络结构
from torchvision.ops.stochastic_depth import StochasticDepth  # Drop path

# 继承nn.LayerNorm，主要目的是防止输入输出数据的维度混乱
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


# 创建一个继承nn.Module的Permute网络结构，主要目的是在搭建CNBlock时代码更加统一
class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


# ConvNeXt的单个CNBlock网络结构
class CNBlock(nn.Module):
    def __init__(self,
                 dim,  # 输入的chanels数
                 layer_scale: float,  # 输出数据的缩放比例
                 stochastic_depth_prob: float,  # Drop path的概率
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        # Normalization层默认为eps=1e-6的LN层
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # CNBlock的网络结构
        self.block = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
                                   Permute([0, 2, 3, 1]),
                                   norm_layer(dim),
                                   nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                                   nn.GELU(),
                                   nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
                                   Permute([0, 3, 1, 2]))
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)  # 与输出size相同的缩放tensor
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")  # Drop path

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)  # 缩放
        result = self.stochastic_depth(result)  # Drop path
        result += input  # shortcut残差结构
        return result


# 每个Block的设置
class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,  # 输入chanels数
        out_channels: Optional[int],  # 输出chanels数
        num_layers: int,  # CNBlock的堆叠数量
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


# ConvNeXt的网络结构
class ConvNeXt(nn.Module):
    def __init__(self,
                 block_setting: List[CNBlockConfig],  # 每个Block的输入输出chanels数和排列数量的设置
                 stochastic_depth_prob: float = 0.0,
                 layer_scale: float = 1e-6,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any) -> None:
        super(ConvNeXt, self).__init__()

        self.stochastic_depth_prob = stochastic_depth_prob

        # 判断网络的设置是否正确
        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        # block默认为CNBlock
        if block is None:
            block = CNBlock

        # normalization层默认为LayerNorm2d
        if norm_layer is None:
            # 定制一个eps=1e-6的LayerNorm2d类
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        # layers列表用于储存所有的卷积层网络结构
        layers: List[nn.Module] = []

        # Stem
        # 即第一个4*4的卷积层和LN层
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(Conv2dNormActivation(3,
                                           firstconv_output_channels,
                                           kernel_size=4,
                                           stride=4,
                                           padding=0,
                                           norm_layer=norm_layer,
                                           activation_layer=None,
                                           bias=True))
        # 不使用ConvNormActivation创建第一个4*4的卷积层和LN层
        # layers.append(nn.Sequential(nn.Conv2d(3, firstconv_output_channels, kernel_size=4, padding=0, bias=True),
        #                             norm_layer(firstconv_output_channels)))

        # 统计CNBlock的总数
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        # 记录该CNBlock是第几个
        stage_block_id = 0
        # 循环创建所有的Block即下采样层
        for cnf in block_setting:
            # Bottlenecks
            # stage列表用于储存该Block的所有的CNBlock
            stage: List[nn.Module] = []
            # 循环创建该Block所设定数量的CNBlock
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                # 层数越深的CNBlock的Drop path的概率越大
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                # 将CNBlock压入列表中，并计数加1
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            # 将该Block压入layers中
            layers.append(nn.Sequential(*stage))
            # 若该Block后有下采样层，则将对应的下采样层压入layers
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(nn.Sequential(norm_layer(cnf.input_channels),
                                            nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2)))

        # 所有的卷积网络结构
        self.features = nn.Sequential(*layers)
        # 全局自适应平均池化，将每个chanels的数据都池化成1*1的size
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 创建网络最后的分类器
        lastblock = block_setting[-1]
        # 确定分类器的输入chanels数
        lastconv_output_channels = (lastblock.out_channels if lastblock.out_channels is not None
                                    else lastblock.input_channels)
        # 分类器包括一个LN层、将维度变为2的展平层和一个全连接层，输出的chanels数为分类的类别数
        self.classifier = nn.Sequential(norm_layer(lastconv_output_channels),
                                        nn.Flatten(1),
                                        nn.Linear(lastconv_output_channels, num_classes))

        # 将网络中的卷积层和全连接层进行权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # weight初始为均值为0，标准差为2，范围在[-2, 2]的截断正太分部
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    # bias初始为0
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)  # 所有卷积网络结构
        x = self.avgpool(x)  # 全局平均池化
        x = self.classifier(x)  # 分类器
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _convnext(arch: str,  # 网络名称
              block_setting: List[CNBlockConfig],  # Block设置
              stochastic_depth_prob: float,  # Drop path的概率
              num_classes: int = 1000,
              **kwargs: Any) -> ConvNeXt:
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes, **kwargs)
    return model


def convnext_tiny(*, num_classes=1000, **kwargs: Any) -> ConvNeXt:
    # https://download.pytorch.org/models/convnext_tiny-983f1562.pth
    block_setting = [CNBlockConfig(96, 192, 3),
                     CNBlockConfig(192, 384, 3),
                     CNBlockConfig(384, 768, 9),
                     CNBlockConfig(768, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext("convnext_tiny", block_setting, stochastic_depth_prob, num_classes, **kwargs)


def convnext_small(*, num_classes=1000, **kwargs: Any) -> ConvNeXt:
    # https://download.pytorch.org/models/convnext_small-0c510722.pth
    block_setting = [CNBlockConfig(96, 192, 3),
                     CNBlockConfig(192, 384, 3),
                     CNBlockConfig(384, 768, 27),
                     CNBlockConfig(768, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext("convnext_small", block_setting, stochastic_depth_prob, num_classes, **kwargs)


def convnext_base(*, num_classes=1000, **kwargs: Any) -> ConvNeXt:
    # https://download.pytorch.org/models/convnext_base-6075fbad.pth
    block_setting = [CNBlockConfig(128, 256, 3),
                     CNBlockConfig(256, 512, 3),
                     CNBlockConfig(512, 1024, 27),
                     CNBlockConfig(1024, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext("convnext_base", block_setting, stochastic_depth_prob, num_classes, **kwargs)


def convnext_large(*, num_classes=1000, **kwargs: Any) -> ConvNeXt:
    # https://download.pytorch.org/models/convnext_large-ea097f82.pth
    block_setting = [CNBlockConfig(192, 384, 3),
                     CNBlockConfig(384, 768, 3),
                     CNBlockConfig(768, 1536, 27),
                     CNBlockConfig(1536, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext("convnext_large", block_setting, stochastic_depth_prob, num_classes, **kwargs)


if __name__ == "__main__":
    ConvNeXt_B = convnext_base(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(ConvNeXt_B, x, 'ConvNeXt_B.onnx', verbose=True)
