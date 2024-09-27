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
                 dataset_type: str = 'val',
                 folder_name: str = 'segmentation',
                 transform_fn: Optional[List] = None) -> None:
        self.transform_fn = transform_fn

        names: List[str] = []
        txt_path = os.path.join(root_path, dataset_type + '.txt')
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                names.append(line[:-1])
        names.sort()

        path = os.path.join(root_path, folder_name)

        self.path = path
        self.names = names

    def __getitem__(self,
                    index: int) -> Tuple[str, Any]:
        mask_path = os.path.join(self.path, self.names[index] + '.png')
        mask_img = Image.open(mask_path).convert('L')

        if self.transform_fn is not None:
            for idx, transform in enumerate(self.transform_fn):
                mask_img = transform(mask_img)

        mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        return mask_name, mask_img

    def __len__(self) -> int:
        return len(self.names)


class DepthDataset(Dataset):
    def __init__(self,
                 root_path: str,
                 dataset_type: str = 'val',
                 folder_name: str = 'depth',
                 transform_fn: Optional[List] = None) -> None:
        self.transform_fn = transform_fn

        names: List[str] = []
        if dataset_type == 'val':
            txt_path = os.path.join(root_path, dataset_type + '.txt')
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    names.append(line[:-1] + '.png')
            path = os.path.join(root_path, folder_name)
        elif dataset_type == 'result':
            path = os.path.join(root_path, dataset_type, folder_name)
            items = os.listdir(path)
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    names.append(item)
        else:
            raise ValueError("No such type for depth dataset!")
        names.sort()

        self.path = path
        self.names = names

    def __getitem__(self,
                    index: int) -> Tuple[str, Any]:
        depth_path = os.path.join(self.path, self.names[index])
        depth_img = Image.open(depth_path)

        if self.transform_fn is not None:
            for idx, transform in enumerate(self.transform_fn):
                depth_img = transform(depth_img)

        depth_name = os.path.splitext(os.path.basename(depth_path))[0]
        return depth_name, depth_img

    def __len__(self) -> int:
        return len(self.names)


class GWDepthDataset(Dataset):
    def __init__(self,
                 network: str,
                 root_path: str,
                 dataset_type: str = 'train',
                 common_transform_fn: Optional[List] = None,
                 image_transform_fn: Optional[List] = None,
                 mask_transform_fn: Optional[List] = None,
                 depth_transform_fn: Optional[List] = None,
                 boundary_map: bool = False,
                 boundary_thickness: int = 8,
                 remove_edge: bool = False,
                 save_boundary: bool = False) -> None:
        self.network = network
        self.root_path = root_path
        self.dataset_type = dataset_type
        self.common_transform_fn = common_transform_fn
        self.image_transform_fn = image_transform_fn
        self.mask_transform_fn = mask_transform_fn
        self.depth_transform_fn = depth_transform_fn
        self.boundary_map = boundary_map
        self.boundary_thickness = boundary_thickness
        self.save_boundary = save_boundary
        self.remove_edge = remove_edge

        names: List[str] = []
        txt_path = os.path.join(self.root_path, dataset_type + '.txt')
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                names.append(line[:-1])
        image_root = os.path.join(self.root_path, 'images')
        mask_root = os.path.join(self.root_path, 'segmentation')
        depth_root = os.path.join(self.root_path, 'depth')

        self.names = names
        self.image_root = image_root
        self.mask_root = mask_root
        self.depth_root = depth_root

    def __getitem__(self,
                    index: int) -> Union[Tuple[str, Tensor, Tensor, Any],
                                         Tuple[str, Tensor, Tensor, Tensor, Tensor, Any]]:
        image_path = os.path.join(self.image_root, self.names[index] + '.png')
        mask_path = os.path.join(self.mask_root, self.names[index] + '.png')
        depth_path = os.path.join(self.depth_root, self.names[index] + '.png')
        image_img = Image.open(image_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')
        depth_img = Image.open(depth_path)

        if self.common_transform_fn is not None:
            for idx, transform in enumerate(self.common_transform_fn):
                image_img, mask_img, depth_img = transform(image_img, mask_img, depth_img)

        if self.image_transform_fn is not None:
            for idx, transform in enumerate(self.image_transform_fn):
                image_img = transform(image_img)

        if self.mask_transform_fn is not None:
            for idx, transform in enumerate(self.mask_transform_fn):
                mask_img = transform(mask_img)

        if self.depth_transform_fn is not None:
            for idx, transform in enumerate(self.depth_transform_fn):
                depth_img = transform(depth_img)

        name = os.path.splitext(os.path.basename(image_path))[0]

        if self.boundary_map is True:
            if 'GSSDENet' in self.network:
                mask_tmp = mask_img.squeeze(0)
            else:
                raise ValueError("No such GW-Depth dataset for %s!" % self.network)
            boundary = self.get_boundary(self.network, mask_tmp, name, self.root_path,
                                         thickness=self.boundary_thickness, remove_edge=self.remove_edge,
                                         save_boundary=self.save_boundary)
            body = self.get_body(self.network, mask_tmp, boundary)
            boundary = torch.from_numpy(boundary).float()
            return name, image_img, mask_img, boundary, body, depth_img
        return name, image_img, mask_img, depth_img

    def __len__(self) -> int:
        return len(self.names)

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
            save_root = os.path.join(root_path, 'segmentation', 'boundary')
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            image = torch.tensor(boundary).cpu()
            image = transforms.ToPILImage()(image)
            image.save(os.path.join(save_root, '%s_boundary.png' % name))
        if 'GSSDENet' in network:
            boundary = np.expand_dims(boundary, axis=0)
        return boundary

    @staticmethod
    def get_body(network: str,
                 mask: Tensor,
                 boundary: ndarray) -> Tensor:
        boundary_valid = boundary == 1
        body = mask.clone()
        body[boundary_valid] = 0
        if 'GSSDENet' in network:
            body = torch.unsqueeze(body, dim=0)
        return body


def get_data_loader(network: str,
                    root_path: str,
                    dataset_type: str = 'train',
                    batch_size: int = 1,
                    num_workers: int = 1,
                    common_transform_fn: Optional[List] = None,
                    image_transform_fn: Optional[List] = None,
                    mask_transform_fn: Optional[List] = None,
                    depth_transform_fn: Optional[List] = None,
                    boundary_map: bool = False,
                    boundary_thickness: int = 8,
                    remove_edge: bool = False,
                    save_boundary: bool = False,
                    is_train: bool = False,
                    drop_last: bool = False) -> DataLoader:
    dataset = GWDepthDataset(network, root_path, dataset_type=dataset_type, common_transform_fn=common_transform_fn,
                             image_transform_fn=image_transform_fn, mask_transform_fn=mask_transform_fn,
                             depth_transform_fn=depth_transform_fn, boundary_map=boundary_map,
                             boundary_thickness=boundary_thickness, remove_edge=remove_edge,
                             save_boundary=save_boundary)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=is_train,
                             drop_last=drop_last)
    return data_loader
