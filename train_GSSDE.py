import os
import sys
import argparse
from typing import Optional, Tuple, List, Any

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from dataset import GWDepth
from transform import common, image, mask, depth
from network.GSSDENet import GSSDENet
from loss import GSDDELoss
from optimizer import SGD, AdamW, PolyScheduler, StepScheduler
from eval_GSS import get_val_confmat, confmat_to_metric
from eval_GSSDE import compute_val_metric


parser = argparse.ArgumentParser(description='Training for Glass Surface Detection and Depth Estimation')

parser.add_argument('--network', type=str, default='GSSDENet')
# dataset
parser.add_argument('--dataset_root', type=str, default='./dataset/GW-Depth')
parser.add_argument('--image_scale', type=int, default=512)
parser.add_argument('--crop_scale', type=float, default=0.6)
parser.add_argument('--brightness', type=float, default=0.4)
parser.add_argument('--contrast', type=float, default=0.4)
parser.add_argument('--saturation', type=float, default=0.4)
parser.add_argument('--hue', type=float, default=0.4)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--boundary_map', action='store_true', default=True)
parser.add_argument('--boundary_thickness', type=int, default=8)
parser.add_argument('--remove_edge', action='store_true', default=False)
parser.add_argument('--save_boundary', action='store_true', default=False)
parser.add_argument('--drop_last', action='store_true', default=False)
# network
parser.add_argument('--backbone_path', type=str, default='./ckpt/pretrained/resnet101-deep.pth')
parser.add_argument('--skip_top', action='store_true', default=True)
parser.add_argument('--pretrained_model', type=str, default=None)
# parser.add_argument('--pretrained_model', type=str, default='./ckpt/trained/GSDENet/GSDE/deep101_all-200.pth')
parser.add_argument('--device', type=torch.device, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
# loss function
parser.add_argument('--loss_weight', type=Tuple[float], default=(1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 2))
parser.add_argument('--dice_smooth', type=float, default=1)
parser.add_argument('--variance_focus', type=float, default=0.85)
parser.add_argument('--log_depth_error', action='store_true', default=True)
parser.add_argument('--min_depth', type=float, default=0.2)
parser.add_argument('--max_depth', type=float, default=10.0)
# optimizer
parser.add_argument('--total_epochs', type=int, default=200)
parser.add_argument('--segment_lr', type=float, default=1e-3)
parser.add_argument('--depth_lr', type=float, default=1e-4)
parser.add_argument('--lr_power', type=float, default=0.9)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--segment_weight_decay', type=float, default=5e-4)
parser.add_argument('--depth_weight_decay', type=float, default=1e-4)
parser.add_argument('--use_warmup', action='store_true', default=False)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--warmup_factor', type=float, default=0.001)
parser.add_argument('--scheduler_type', type=str, default='epoch')
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--gamma', type=float, default=0.1)

args = parser.parse_args()
args.num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
# train
args.save_root = './ckpt/trained/%s' % args.network
if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)
print(args)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def get_data_loader(network: str,
                    dataset_root: str,
                    image_scale: int,
                    crop_scale: float = 0.6,
                    brightness: float = 0.4,
                    contrast: float = 0.4,
                    saturation: float = 0.4,
                    hue: float = 0.4,
                    batch_size: int = 1,
                    num_workers: int = 1,
                    boundary_map: bool = False,
                    boundary_thickness: int = 8,
                    remove_edge: bool = False,
                    save_boundary: bool = False,
                    drop_last: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_common_transform_fn = [common.BatchRandomSelect(common.BatchResize(size=image_scale),
                                                          common.BatchRandomResizeCrop(size=image_scale,
                                                                                       scale=crop_scale),
                                                          pro=0.5),
                                 common.BatchRandomHorizontallyFlip(flip_pro=0.5),
                                 common.BatchRandomVerticallyFlip(flip_pro=0.5)]
    val_common_transform_fn = [common.BatchResize(size=image_scale)]

    train_image_transform_fn = [image.RandomChoice([image.ColorJitter(brightness=brightness, contrast=contrast,
                                                                      saturation=saturation, hue=hue),
                                                    image.NullTransform()],
                                                   p=[0.5, 0.5]),
                                image.ToTensor(),
                                image.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    val_image_transform_fn = [image.ToTensor(),
                              image.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    if network == 'GSSDENet':
        mask_transform_fn = [mask.MaskToTensor(),
                             mask.MultiNormalize(),
                             mask.UnSqueeze(dim=0)]

        depth_transform_fn = [depth.DepthToTensor(),
                              depth.Normalize(),
                              depth.UnSqueeze(dim=0)]
    else:
        raise ValueError("No such data loader for %s!" % network)

    train_loader = GWDepth.get_data_loader(network, dataset_root, dataset_type='train', batch_size=batch_size,
                                           num_workers=num_workers, common_transform_fn=train_common_transform_fn,
                                           image_transform_fn=train_image_transform_fn,
                                           mask_transform_fn=mask_transform_fn, depth_transform_fn=depth_transform_fn,
                                           boundary_map=boundary_map, boundary_thickness=boundary_thickness,
                                           remove_edge=remove_edge, save_boundary=save_boundary, is_train=True,
                                           drop_last=drop_last)
    val_loader = GWDepth.get_data_loader(network, dataset_root, dataset_type='val', batch_size=batch_size,
                                         num_workers=num_workers, common_transform_fn=val_common_transform_fn,
                                         image_transform_fn=val_image_transform_fn, mask_transform_fn=mask_transform_fn,
                                         depth_transform_fn=depth_transform_fn)
    print("Successfully got GW-Depth's data loader for %s!" % network)
    return train_loader, val_loader


# # 获取需要训练的网络模型
def get_model(network: str,
              backbone_path: str,
              skip_top: bool = True,
              max_depth: float = 10.0,
              pretrained_model: Optional[str] = None,
              device: torch.device = 'cpu') -> nn.Module:
    if network == 'GSSDENet':
        model = GSSDENet(backbone_path=backbone_path, skip_top=skip_top, max_depth=max_depth, device=device)
    else:
        raise ValueError('No such network model for %s!' % network)

    if pretrained_model is not None:
        pretrained_dict = torch.load(pretrained_model, map_location=device)
        model_dict = model.state_dict()
        print([k for k in model_dict.keys() if (k not in pretrained_dict)])
        print([k for k in pretrained_dict if (k not in model_dict.keys())])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'Prediction' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Successfully loaded pretrained weights of %s!" % network)

    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print("The parameters of the network model of %s are: %d" % (network, num_params))
    print("Successfully got network model of %s!" % network)
    return model


def get_loss_function(network: str,
                      loss_weight: Tuple[float] = (1, 1, 1, 1, 2),
                      boundary_map: bool = True,
                      dice_smooth: float = 1,
                      variance_focus: float = 0.85,
                      log_depth_error: bool = True,
                      min_depth: float = 0.2,
                      max_depth: float = 10.0,
                      device: torch.device = 'cpu') -> nn.Module:
    if network == 'GSSDENet':
        loss_fn = GSDDELoss(weight=loss_weight, boundary_map=boundary_map, dice_smooth=dice_smooth,
                            variance_focus=variance_focus, log_depth_error=log_depth_error, min_depth=min_depth,
                            max_depth=max_depth)
    else:
        raise ValueError("No such loss function for %s!" % network)

    loss_fn.to(device)
    print("Successfully got loss function for %s!" % network)
    return loss_fn


def get_optimizer(network: str,
                  model: nn.Module,
                  train_loader: DataLoader,
                  total_epochs: int,
                  segment_lr: float = 1e-3,
                  depth_lr: float = 1e-4,
                  lr_power: float = 0.9,
                  momentum: float = 0.9,
                  segment_weight_decay: float = 1e-4,
                  depth_weight_decay: float = 1e-4,
                  use_warmup: bool = False,
                  warmup_epochs: int = 10,
                  warmup_factor: float = 0.001,
                  scheduler_type: str = 'epoch',
                  step_size: int = 70,
                  gamma: float = 0.1) -> Tuple[Optimizer, Optimizer, Any, Any]:
    segment_param_dicts = [p for n, p in model.named_parameters() if 'depth' not in n and p.requires_grad]
    depth_param_dicts = [p for n, p in model.named_parameters() if 'depth' in n and p.requires_grad]

    segment_optim = SGD(segment_param_dicts, lr=segment_lr, momentum=momentum, weight_decay=segment_weight_decay)
    segment_scheduler = PolyScheduler(segment_optim, total_epochs, len(train_loader), lr_power=lr_power,
                                      use_warmup=use_warmup, warmup_epochs=warmup_epochs, warmup_factor=warmup_factor,
                                      scheduler_type=scheduler_type).get_poly_scheduler()

    depth_optim = AdamW(depth_param_dicts, lr=depth_lr, weight_decay=depth_weight_decay)
    # depth_scheduler = StepScheduler(depth_optim, step_size, gamma=gamma).get_step_scheduler()
    depth_scheduler = PolyScheduler(depth_optim, total_epochs, len(train_loader), lr_power=lr_power,
                                    use_warmup=use_warmup, warmup_epochs=warmup_epochs, warmup_factor=warmup_factor,
                                    scheduler_type=scheduler_type).get_poly_scheduler()

    print("Successfully got optimizer for %s!" % network)
    return segment_optim, depth_optim, segment_scheduler, depth_scheduler


# 训练函数
def train(network: str,
          train_loader: DataLoader,
          val_loader: DataLoader,
          model: nn.Module,
          loss_fn: nn.Module,
          segment_optim: Optimizer,
          depth_optim: Optimizer,
          segment_scheduler: Any,
          depth_scheduler: Any,
          epochs: int,
          save_root: str,
          boundary_map: bool = True,
          device: torch.device = 'cpu') -> None:
    print("Start training process for %s!" % network)
    best_PA: float = 0.0
    best_sigma_1: float = 0.0
    best_log_out: str = ''
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    map_save_path = os.path.join(save_root, '%d_map.png' % epochs)

    # graph
    train_loss_list: List[float] = []
    IoU_list: List[float] = []
    PA_list: List[float] = []
    FB_list: List[float] = []
    sigma_1_list: List[float] = []
    REL_list: List[float] = []
    RMS_list: List[float] = []

    # train and evaluation in one epoch
    for epoch in range(epochs):
        torch.cuda.empty_cache()

        # train
        model.train()
        running_loss: float = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            if boundary_map is False:
                names, images, masks, depths = data
                targets = (masks.to(device), depths.to(device))
            else:
                names, images, masks, masks_edge, masks_body, depths = data
                targets = (masks.to(device), masks_edge.to(device), depths.to(device))

            predicts = model(images.to(device))
            total_loss, loss_list = loss_fn(predicts, targets)

            segment_optim.zero_grad()
            depth_optim.zero_grad()
            total_loss.backward()
            segment_optim.step()
            depth_optim.step()

            running_loss += total_loss.item()
            loss_log: str = '['
            for loss in loss_list:
                loss_log += '%.4f, ' % loss
            loss_log = loss_log[:-2] + ']'
            train_bar.desc = "train epoch[%d/%d] loss: %s  segment_lr: %.6f  depth_lr: %.6f" % \
                             (epoch + 1, epochs, loss_log, segment_optim.param_groups[-1]['lr'],
                              depth_optim.param_groups[-1]['lr'])

        # eval
        model.eval()
        confmat = 0
        sigma_1_sum: float = 0.0
        REL_sum: float = 0.0
        RMS_sum: float = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for data in val_bar:
                names, images, gt_masks, depths = data

                predicts = model(images.to(device))
                if network == 'GSSDENet':
                    confmat += get_val_confmat(predicts[5].cpu(), gt_masks)
                    sigma_1, REL, RMS = compute_val_metric(predicts[-1].cpu(), depths)
                    sigma_1_sum += sigma_1
                    REL_sum += REL
                    RMS_sum += RMS
                else:
                    raise ValueError('No such net model!')

                val_bar.desc = "val epoch[%d/%d]" % (epoch + 1, epochs)

        train_loss = running_loss / train_steps
        IoU, PA, FB = confmat_to_metric(confmat)
        sigma_1 = sigma_1_sum / val_steps
        REL = REL_sum / val_steps
        RMS = RMS_sum / val_steps
        log_out = ('[epoch %d] train_loss: %.4f  IoU: %.4f  PA: %.4f  FB: %.4f  S1: %.4f  REL: %.4f  RMS: %.4f  '
                   'segment_lr: %.6f  depth_lr: %.6f') % \
                  (epoch + 1, train_loss, IoU, PA, FB, sigma_1, REL, RMS, segment_optim.param_groups[-1]['lr'],
                   depth_optim.param_groups[-1]['lr'])
        print(log_out)

        train_loss_list.append(train_loss)
        IoU_list.append(IoU)
        PA_list.append(PA)
        FB_list.append(FB)
        sigma_1_list.append(sigma_1)
        REL_list.append(REL)
        RMS_list.append(RMS)

        model_save_path = os.path.join(save_root, str(epoch + 1) + '.pth')
        if sigma_1 > best_sigma_1:
            best_sigma_1 = sigma_1
            torch.save(model.state_dict(), model_save_path)
            print('%dth model is saved' % (epoch + 1))
            best_log_out = log_out
        else:
            print('best model: %s' % best_log_out)
            if sigma_1 > 0.898 and IoU > 0.955:
                torch.save(model.state_dict(), model_save_path)
                print('%dth model is saved' % (epoch + 1))
        if PA > best_PA:
            best_PA = PA
            # torch.save(model.state_dict(), model_save_path)
            # print('%dth model is saved' % (epoch + 1))
            # best_log_out = log_out

        segment_scheduler.step()
        depth_scheduler.step()

        plt.clf()
        plt.plot(train_loss_list, label='Training Loss')
        plt.plot(IoU_list, label='IoU')
        plt.plot(PA_list, label='PA')
        plt.plot(FB_list, label='FB')
        plt.plot(sigma_1_list, label='S1')
        plt.plot(REL_list, label='REL')
        plt.plot(RMS_list, label='RMS')
        plt.legend()
        plt.savefig(map_save_path)

    torch.save(model.state_dict(), os.path.join(save_root, '200.pth'))
    print('Best PA is: %f' % best_PA)
    print("Training process for %s has finished!" % network)


def main():
    train_loader, val_loader = get_data_loader(args.network, args.dataset_root, args.image_scale, args.crop_scale,
                                               args.brightness, args.contrast, args.saturation, args.hue,
                                               args.batch_size, args.num_workers, args.boundary_map,
                                               args.boundary_thickness, args.remove_edge, args.save_boundary,
                                               args.drop_last)

    model = get_model(args.network, args.backbone_path, args.skip_top, args.max_depth, args.pretrained_model,
                      args.device)

    loss_fn = get_loss_function(args.network, args.loss_weight, args.boundary_map, args.dice_smooth,
                                args.variance_focus, args.log_depth_error, args.min_depth, args.max_depth, args.device)

    segment_optim, depth_optim, segment_scheduler, depth_scheduler = get_optimizer(args.network, model, train_loader,
                                                                                   args.total_epochs, args.segment_lr,
                                                                                   args.depth_lr, args.lr_power,
                                                                                   args.momentum,
                                                                                   args.segment_weight_decay,
                                                                                   args.depth_weight_decay,
                                                                                   args.use_warmup, args.warmup_epochs,
                                                                                   args.warmup_factor,
                                                                                   args.scheduler_type, args.step_size,
                                                                                   args.gamma)

    train(args.network, train_loader, val_loader, model, loss_fn, segment_optim, depth_optim, segment_scheduler,
          depth_scheduler, args.total_epochs, args.save_root, args.boundary_map, args.device)


if __name__ == '__main__':
    main()
