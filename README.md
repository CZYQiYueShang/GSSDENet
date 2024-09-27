# GSSDENet

## GSSDENet: Network for Simultaneous Glass Surface Segmentation and Depth Estimation

### Dataset
Please download from the corresponding pages of [GDD](https://github.com/Mhaiyang/CVPR2020_GDNet) and [GW-Depth](https://github.com/ViktorLiang/GW-Depth) and put them in the `dataset` folder like:
```
dataset/
├── GDD/
├── GW-Depth/
├── GSD.py
├── GWDepth.py
└── __init__.py
```

### Requirements
* Python 3.10
* CUDA 11.7
```
pip install -r requirements.txt
```

### Model
Place models in the `ckpt` folder.
#### Pretrained Models
| Name | Download |
|:----:|:---:|
| ResNet-50 | [Google Drive](https://drive.google.com/file/d/1PtlVlHc5-pU4AlfuWFOsd9H5LGuAbs5T/view?usp=sharing) |
| ResNet-101 | [Google Drive](https://drive.google.com/file/d/1PdOIvflcAiEK7oJatc_OPtS_8Bk328lm/view?usp=sharing) |
| ConvNeXt-T | [Google Drive](https://drive.google.com/file/d/1oLCI2xn7J1oKyNaLams2_5AMKnqURTKg/view?usp=sharing) |
| ConvNeXt-S | [Google Drive](https://drive.google.com/file/d/1Mz66j_8r8h1p5gFqVFYZqrX1jIXLN86A/view?usp=sharing) |
| ConvNeXt-B | [Google Drive](https://drive.google.com/file/d/1o6uxGOB6Smj14Ka8QE7smjDGi75RECC6/view?usp=sharing) |
#### Trained Models
| Network | Backbone | Download |
|:---------------|:----:|:---:|
| GSSDENet-S | ResNet-101 | [Google Drive](https://drive.google.com/file/d/1Ua1IoDuiJlu4A8VOxnbB919gtMGJk3CQ/view?usp=sharing) |
| GSSDENet-S | ConvNeXt-B | [Google Drive](https://drive.google.com/file/d/1vHNlDHHsRWQ4muFEQtdMVyWwVvNEMRE8/view?usp=sharing) |
| GSSDENet | ResNet-101 | [Google Drive](https://drive.google.com/file/d/1-MaMNrlr9IR_5e-wuCUkJYDDB6c_07Br/view?usp=sharing) |

### Test
Use `infer_GSS.py` or `infer_GSSDE.py` to test after downloading the datasets and models.
```
python infer_GSS.py
```

### Contact
E-Mail: chen.zeyuan.tkb_gu@u.tsukuba.ac.jp
