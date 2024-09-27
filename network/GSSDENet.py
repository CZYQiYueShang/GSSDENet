from network.backbone.backbone import Backbone
from network.module.misc import *


class GSSDENet(nn.Module):
    def __init__(self,
                 backbone_path: str,
                 skip_top: bool = True,
                 max_depth: float = 10.0,
                 context_layer: Optional[Callable[..., nn.Module]] = CAPv2,
                 boundary_layer: Optional[Callable[..., nn.Module]] = HBD,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
                 device: torch.device = 'cpu') -> None:
        super(GSSDENet, self).__init__()
        # params
        self.skip_top = skip_top
        self.max_depth = max_depth
        self.boundary_layer = boundary_layer

        if 'resnet' in backbone_path or 'resnext' in backbone_path:
            basic_dim = 256
        elif 'convnext_base' in backbone_path:
            basic_dim = 128
        elif 'convnext_small' in backbone_path or 'convnext_tiny' in backbone_path:
            basic_dim = 96
        else:
            raise ValueError('No such UperNet model for backbone: %s!' % backbone_path)

        # backbone
        net = Backbone(backbone_path, device=device)
        self.layer0 = net.layer0
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # depth_net = Backbone(backbone_path, device=device)
        # self.depth_layer0 = depth_net.layer0
        # self.depth_layer1 = depth_net.layer1
        # self.depth_layer2 = depth_net.layer2
        # self.depth_layer3 = depth_net.layer3
        # self.depth_layer4 = depth_net.layer4

        self.segment_fpn = GSDFPN((basic_dim, 2 * basic_dim, 4 * basic_dim, 8 * basic_dim), basic_dim,
                                  skip_top=self.skip_top, context_layer=context_layer, boundary_layer=boundary_layer,
                                  bias=False, norm_layer=norm_layer, act_layer=act_layer)
        self.segment_fuse = ConvNormAct((5 - self.skip_top) * basic_dim, basic_dim, 1, 1, 0,
                                        norm_layer=norm_layer, act_layer=act_layer, bias=False)
        self.segment_conv = nn.Sequential(ConvNormAct(basic_dim, basic_dim, 1, 1, 0,
                                                      norm_layer=norm_layer, act_layer=act_layer, bias=False),
                                          nn.Conv2d(basic_dim, 1, 1, 1, 0))
        self.segment_predict_layers = nn.ModuleList([nn.Conv2d(basic_dim, 1, 3, 1, 1),
                                                     nn.Conv2d(basic_dim, 1, 3, 1, 1),
                                                     nn.Conv2d(basic_dim, 1, 3, 1, 1),
                                                     nn.Conv2d(basic_dim, 1, 3, 1, 1)])
        if self.skip_top is not True:
            self.segment_predict_layers.append(nn.Conv2d(basic_dim, 1, 3, 1, 1))
        if self.boundary_layer is not None:
            self.boundary_fuse = ConvNormAct(5 - self.skip_top, 1, 1, 1, 0,
                                             norm_layer=norm_layer, act_layer=act_layer, bias=False)
            self.boundary_final_predict = nn.Conv2d(1, 1, 3, 1, 1)
        self.segment_final_predict = nn.Conv2d(1, 1, 3, 1, 1)

        self.depth_fpn = GSDEFPN((basic_dim, 2 * basic_dim, 4 * basic_dim, 8 * basic_dim), basic_dim,
                                 skip_top=self.skip_top, bias=False, norm_layer=norm_layer, act_layer=nn.ELU)
        self.depth_fuse = nn.Sequential(ConvNormAct((5 - self.skip_top) * basic_dim, basic_dim, 3, 1, 1,
                                                    norm_layer=norm_layer, act_layer=nn.ELU, bias=False, use_refl=True),
                                        ConvNormAct(basic_dim, basic_dim, 3, 1, 1,
                                                    norm_layer=norm_layer, act_layer=nn.ELU, bias=False, use_refl=True))
        self.depth_predict_layers = nn.ModuleList([ConvNormAct(basic_dim, 1, 3, 1, 1,
                                                               norm_layer=None, act_layer=None, bias=True, use_refl=True),
                                                   ConvNormAct(basic_dim, 1, 3, 1, 1,
                                                               norm_layer=None, act_layer=None, bias=True, use_refl=True),
                                                   ConvNormAct(basic_dim, 1, 3, 1, 1,
                                                               norm_layer=None, act_layer=None, bias=True, use_refl=True),
                                                   ConvNormAct(basic_dim, 1, 3, 1, 1,
                                                               norm_layer=None, act_layer=None, bias=True, use_refl=True)])
        if self.skip_top is not True:
            self.depth_predict_layers.append(ConvNormAct(basic_dim, 1, 3, 1, 1,
                                                         norm_layer=None, act_layer=None, bias=True, use_refl=True))
        self.depth_final_predict = ConvNormAct(basic_dim, 1, 3, 1, 1,
                                               norm_layer=None, act_layer=None, bias=True, use_refl=True)

    def forward(self,
                x: Tensor) -> List[Tensor]:
        size = x.shape[2:]

        layer0_feature = self.layer0(x)
        layer1_feature = self.layer1(layer0_feature)
        layer2_feature = self.layer2(layer1_feature)
        layer3_feature = self.layer3(layer2_feature)
        layer4_feature = self.layer4(layer3_feature)

        # depth_layer0_feature = self.depth_layer0(x)
        # depth_layer1_feature = self.depth_layer1(depth_layer0_feature)
        # depth_layer2_feature = self.depth_layer2(depth_layer1_feature)
        # depth_layer3_feature = self.depth_layer3(depth_layer2_feature)
        # depth_layer4_feature = self.depth_layer4(depth_layer3_feature)

        if self.skip_top is True:
            segment_fpn_in_features = [layer1_feature, layer2_feature, layer3_feature, layer4_feature]
            depth_fpn_in_features = [layer1_feature, layer2_feature, layer3_feature, layer4_feature]
        else:
            segment_fpn_in_features = [layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer4_feature]
            depth_fpn_in_features = [layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer4_feature]

        if self.boundary_layer is None:
            segment_fpn_out_features = self.segment_fpn(segment_fpn_in_features)
        else:
            segment_fpn_out_features, boundary_fpn_out_features = self.segment_fpn(segment_fpn_in_features)
            boundary_fuse_features: List[Tensor] = [boundary_fpn_out_features[0]]

        segment_fuse_size = segment_fpn_out_features[0].shape[2:]
        segment_fuse_features: List[Tensor] = [segment_fpn_out_features[0]]
        context_features: List[Tensor] = [segment_fpn_out_features[0]]
        predicts: List[Tensor] = [
            torch.sigmoid(upsample_size(self.segment_predict_layers[0](segment_fpn_out_features[0]), size))]

        for i in range(1, len(segment_fpn_out_features)):
            segment_fpn_out_feature = upsample_size(segment_fpn_out_features[i], segment_fuse_size)
            segment_fuse_features.append(segment_fpn_out_feature)
            context_features.append(segment_fpn_out_features[i])
            predict = self.segment_predict_layers[i](segment_fpn_out_features[i])
            predict = upsample_size(predict, size)
            predicts.append(torch.sigmoid(predict))
            if self.boundary_layer is not None:
                boundary_fuse_feature = upsample_size(boundary_fpn_out_features[i], segment_fuse_size)
                boundary_fuse_features.append(boundary_fuse_feature)

        segment_fuse = self.segment_fuse(torch.cat(segment_fuse_features, dim=1))
        if self.boundary_layer is not None:
            boundary_fuse = self.boundary_fuse(torch.cat(boundary_fuse_features, dim=1))
            boundary_final_predict = self.boundary_final_predict(boundary_fuse)
            boundary_final_predict = upsample_size(boundary_final_predict, size)
            boundary_predict = torch.sigmoid(boundary_final_predict)
            predicts.append(boundary_predict)

            segment_boundary_fuse = torch.sigmoid(boundary_fuse) * segment_fuse
            segment_fuse = segment_fuse + segment_boundary_fuse

        segment_fuse = self.segment_conv(segment_fuse)
        segment_final_predict = self.segment_final_predict(segment_fuse)
        segment_final_predict = upsample_size(segment_final_predict, size)
        predicts.append(torch.sigmoid(segment_final_predict))

        depth_fpn_out_features = self.depth_fpn(context_features, depth_fpn_in_features)
        depth_fuse_size = depth_fpn_out_features[0].shape[2:]
        depth_fuse_features: List[Tensor] = [depth_fpn_out_features[0]]
        predicts.append(torch.sigmoid(upsample_size(self.depth_predict_layers[0](depth_fpn_out_features[0]), size)) * self.max_depth)

        for i in range(1, len(depth_fpn_out_features)):
            depth_fpn_out_feature = upsample_size(depth_fpn_out_features[i], depth_fuse_size)
            depth_fuse_features.append(depth_fpn_out_feature)
            predict = self.depth_predict_layers[i](depth_fpn_out_features[i])
            predict = upsample_size(predict, size)
            predicts.append(torch.sigmoid(predict) * self.max_depth)

        depth_fuse = self.depth_fuse(torch.cat(depth_fuse_features, dim=1))
        depth_final_predict = self.depth_final_predict(depth_fuse)
        depth_final_predict = upsample_size(depth_final_predict, size)
        predicts.append(torch.sigmoid(depth_final_predict) * self.max_depth)

        return predicts


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone_path = '/media/chen/1TB/Projects/GSDDENet-master/ckpt/pretrained/resnet101-deep.pth'
    net = GSSDENet(backbone_path=backbone_path, skip_top=True, device=device).to(device)

    num_params = sum(p.numel() for p in net.parameters())
    print(num_params)

    x = torch.randn(6, 3, 512, 512).to(device)
    outputs = net(x)
    for output in outputs:
        print(output.size())
