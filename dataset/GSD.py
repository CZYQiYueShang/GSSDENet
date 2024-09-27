import os
from typing import Optional, Union, List, Tuple, Any

import torch
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import cv2 as cv
import numpy as np
from numpy import ndarray
from PIL import Image


class MaskDataset(Dataset):
    def __init__(self,
                 root_path: str,
                 folder_name: str = 'mask',
                 transform_fn: Optional[List] = None) -> None:
        self.root_path = root_path
        self.transform_fn = transform_fn

        path = os.path.join(self.root_path, folder_name)
        items = os.listdir(path)

        masks: List[str] = []
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                masks.append(item)
        masks.sort()

        self.path = path
        self.masks = masks

    def __getitem__(self,
                    index: int) -> Tuple[str, Any]:
        mask_path = os.path.join(self.path, self.masks[index])
        mask_img = Image.open(mask_path).convert('L')

        if self.transform_fn is not None:
            for idx, transform in enumerate(self.transform_fn):
                mask_img = transform(mask_img)

        mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        return mask_name, mask_img

    def __len__(self) -> int:
        return len(self.masks)


class GSDDataset(Dataset):
    def __init__(self,
                 network: str,
                 root_path: str,
                 common_transform_fn: Optional[List] = None,
                 image_transform_fn: Optional[List] = None,
                 mask_transform_fn: Optional[List] = None,
                 boundary_map: bool = False,
                 boundary_thickness: int = 8,
                 remove_edge: bool = False,
                 save_boundary: bool = False) -> None:
        self.network = network
        self.root_path = root_path
        self.common_transform_fn = common_transform_fn
        self.image_transform_fn = image_transform_fn
        self.mask_transform_fn = mask_transform_fn
        self.boundary_map = boundary_map
        self.boundary_thickness = boundary_thickness
        self.save_boundary = save_boundary
        self.remove_edge = remove_edge

        image_root = os.path.join(self.root_path, 'image')
        image_items = os.listdir(image_root)
        image_items.sort()
        mask_root = os.path.join(self.root_path, 'mask')
        mask_items = os.listdir(mask_root)
        mask_items.sort()

        self.image_root = image_root
        self.mask_root = mask_root
        self.images = image_items
        self.masks = mask_items

    def __getitem__(self,
                    index: int) -> Union[Tuple[str, Tensor, Tensor], Tuple[str, Tensor, Tensor, Tensor, Tensor]]:
        image_path = os.path.join(self.image_root, self.images[index])
        mask_path = os.path.join(self.mask_root, self.masks[index])
        image_img = Image.open(image_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')

        if self.common_transform_fn is not None:
            for idx, transform in enumerate(self.common_transform_fn):
                image_img, mask_img = transform(image_img, mask_img)

        if self.image_transform_fn is not None:
            for idx, transform in enumerate(self.image_transform_fn):
                image_img = transform(image_img)

        if self.mask_transform_fn is not None:
            for idx, transform in enumerate(self.mask_transform_fn):
                mask_img = transform(mask_img)

        name = os.path.splitext(os.path.basename(image_path))[0]

        if self.boundary_map is True:
            if self.network == 'GSSDENet-S':
                mask_tmp = mask_img.squeeze(0)
            else:
                raise ValueError("No such net model's dataset!")
            boundary = self.get_boundary(self.network, mask_tmp, name, self.root_path,
                                         thickness=self.boundary_thickness, remove_edge=self.remove_edge,
                                         save_boundary=self.save_boundary)
            body = self.get_body(self.network, mask_tmp, boundary)
            boundary = torch.from_numpy(boundary).float()
            return name, image_img, mask_img, boundary, body
        return name, image_img, mask_img

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def get_boundary(network: str,
                     mask: Tensor,
                     name: str,
                     root_path: str,
                     thickness: int = 8,
                     remove_edge: bool = False,
                     save_boundary: bool = False) -> ndarray:
        tmp = mask.data.numpy().astype(np.uint8)
        contour, _ = cv.findContours(tmp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv.drawContours(boundary, contour, -1, 1, thickness, lineType=cv.LINE_AA)
        if remove_edge is True:
            contour_image = [np.array([[0, 0],
                                       [boundary.shape[0] - 1, 0],
                                       [boundary.shape[0] - 1, boundary.shape[1] - 1],
                                       [0, boundary.shape[1] - 1]])]
            boundary = cv.drawContours(boundary, contour_image, -1, 0, thickness, lineType=cv.LINE_AA)
        boundary = boundary.astype(np.float32)
        if save_boundary is True:
            save_root = os.path.join(root_path, 'mask', 'boundary')
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            image = torch.tensor(boundary).cpu()
            image = transforms.ToPILImage()(image)
            image.save(os.path.join(save_root, '%s_boundary.png' % name))
        if network == 'GSSDENet-S':
            boundary = np.expand_dims(boundary, axis=0)
        return boundary

    @staticmethod
    def get_body(network: str,
                 mask: Tensor,
                 boundary: ndarray) -> Tensor:
        boundary_valid = boundary == 1
        body = mask.clone()
        body[boundary_valid] = 0
        if network == 'GSSDENet-S':
            body = torch.unsqueeze(body, dim=0)
        return body


def get_data_loader(network: str,
                    root_path: str,
                    batch_size: int = 1,
                    num_workers: int = 1,
                    common_transform_fn: Optional[List] = None,
                    image_transform_fn: Optional[List] = None,
                    mask_transform_fn: Optional[List] = None,
                    boundary_map: bool = False,
                    boundary_thickness: int = 8,
                    remove_edge: bool = False,
                    save_boundary: bool = False,
                    is_train: bool = False,
                    drop_last: bool = False) -> DataLoader:
    dataset = GSDDataset(network, root_path, common_transform_fn=common_transform_fn,
                         image_transform_fn=image_transform_fn, mask_transform_fn=mask_transform_fn,
                         boundary_map=boundary_map, boundary_thickness=boundary_thickness, remove_edge=remove_edge,
                         save_boundary=save_boundary)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=is_train,
                             drop_last=drop_last)
    return data_loader
