# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import enum
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d
from torchvision.models.regnet import regnet_x_800mf, regnet_y_800mf
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from classy_vision.models import build_model
from .gem_pooling import GlobalGeMPool2d
from pytorch_lightning import LightningModule

import timm
import torch

class TeacherNetwork(LightningModule):
    def __init__(self, args):
        super(TeacherNetwork, self).__init__()
        if "vit" in args.ckpt:
            self.backbone = timm.create_model('vit_base_patch8_224', num_classes=0, pretrained=True)
            self.arch = 'vit'
            self.backbone.avgpool = GlobalGeMPool2d(pooling_param=args.pool_param)
            self.norm = L2Norm()
        elif "classy" in args.ckpt: 
            self.backbone = build_model({"name": "resnext101_32x4d"}).classy_model
            self.arch = 'resnext'
            self.embeddings = nn.Sequential(
                GlobalGeMPool2d(pooling_param=args.pool_param),
                nn.Linear(2048, args.kd_dim),
                L2Norm(),
            )
            self.norm = L2Norm()
        else: 
            self.backbone = resnet50(num_classes=args.kd_dim, zero_init_residual=True)
            self.arch = 'resnet'
            self.backbone.avgpool = GlobalGeMPool2d(pooling_param=args.pool_param)
            self.norm = L2Norm()

    def forward(self, x):
        if self.arch == 'vit':
            x = self.backbone.forward_features(x)
            cls_output_token = x[:,0,:]
            patch_token = x[:,1:,:].reshape(256,28,28,-1)
            patch_token = patch_token.clamp(min=1e-6).permute(0,3,1,2)
            patch_token = self.backbone.avgpool(patch_token)
            feats = torch.cat((cls_output_token, patch_token), dim=1)
            return self.norm(feats)
        elif self.arch == 'resnet':
            x = self.backbone(x)
            return self.norm(x)
        elif self.arch == 'resnext':
            x = self.backbone(x)
            x = self.embeddings(x)
            return self.norm(x)

class get_encoder(LightningModule):
    def __init__(self, args, base_encoder):
        super(get_encoder, self).__init__()
        if "VIT" in args.backbone: #student vit
            self.backbone = timm.create_model('fastvit_t12', num_classes=0, pretrained=True)
            self.arch = 'vit'
            self.backbone.head.global_pool = GlobalGeMPool2d(pooling_param=args.pool_param)
            self.backbone.head.fc = nn.Linear(1024,args.kd_dim)
            self.projector = nn.Sequential(
                nn.Linear(args.kd_dim, args.dim),
                L2Norm()
            )
            self.norm = L2Norm()
        else: #student cnn
            self.arch = 'resnet'
            try:
                self.backbone = base_encoder(num_classes=args.kd_dim, zero_init_residual=True)
            except:
                self.backbone = base_encoder(num_classes=args.kd_dim)
            self.backbone.avgpool = GlobalGeMPool2d(pooling_param=args.pool_param)
            self.projector = nn.Linear(args.kd_dim, args.dim)
            self.norm = L2Norm()
    
    def forward(self, x):
        feat = self.backbone(x)
        emb = self.projector(feat)
        return self.norm(feat), self.norm(emb)

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)

class Implementation(enum.Enum):
    CLASSY_VISION = enum.auto()
    TORCHVISION = enum.auto()
    TIMM = enum.auto()

class Backbone(enum.Enum):
    CV_RESNET18 = ("resnet18", 512, Implementation.CLASSY_VISION)
    CV_RESNET50 = ("resnet50", 2048, Implementation.CLASSY_VISION)
    CV_RESNEXT101 = ("resnext101_32x4d", 2048, Implementation.CLASSY_VISION)

    TV_RESNET50 = (resnet50, 2048, Implementation.TORCHVISION)
    TV_RESNEXT101 = (resnext101_32x8d, 2048, Implementation.TORCHVISION)

    TV_EFFICIENTNET_B0 = (efficientnet_b0, 1280, Implementation.TORCHVISION)
    TV_MOBILENETV3 = (mobilenet_v3_large, 1280, Implementation.TORCHVISION)

    TIMM_FASTVIT_T12 = ("fastvit_t12", 1024, Implementation.TIMM)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser = parser.add_argument_group("Model")
        parser.add_argument(
            "--backbone", default="TV_EFFICIENTNET_B0", choices=[b.name for b in Backbone]
        )
        parser.add_argument("--kd_dim", default=512, type=int)
        parser.add_argument("--dim", type=int)
        parser.add_argument("--pool_param", default=3, type=float)

#     def build(self, kd_dim: int):
#         impl = self.value[2]
#         if impl == Implementation.CLASSY_VISION:
#             model = build_model({"name": self.value[0]})
#             return model.classy_model
#         if impl == Implementation.TORCHVISION:
#             return self.value[0](num_classes=kd_dim, zero_init_residual=True)
#         raise AssertionError("Model implementation not handled: %s" % (self.name,))


# class Model(nn.Module):
#     def __init__(self, backbone: str, kd_dim: int, pool_param: float):
#         super().__init__()
#         self.backbone_type = Backbone[backbone]
#         self.backbone = self.backbone_type.build(dims=kd_dim)
#         impl = self.backbone_type.value[2]
#         if impl == Implementation.CLASSY_VISION:
#             self.embeddings = nn.Sequential(
#                 GlobalGeMPool2d(pool_param),
#                 nn.Linear(self.backbone_type.value[1], kd_dim),
#                 L2Norm(),
#             )
#         elif backbone == "TV_RESNET50":
#             self.backbone.avgpool = GlobalGeMPool2d(pool_param)
#             self.backbone.fc = nn.Identity()
#             self.embeddings = nn.Sequential(
#                     nn.Linear(self.backbone_type.value[1], kd_dim),
#                     L2Norm(),
#                 )
#         elif backbone == 'TV_EFFICIENTNET_B0':
#             self.backbone.avgpool = GlobalGeMPool2d(pool_param)
#             self.backbone.fc = nn.Identity()
#             self.embeddings = L2Norm()
            
#         elif backbone == 'TV_MOBILENETV3':
#             self.backbone.avgpool = GlobalGeMPool2d(pool_param)
#             self.embeddings = L2Norm()

#         elif backbone == 'TV_REGNET_Y_800MF':
#             self.backbone.avgpool = GlobalGeMPool2d(pool_param)
#             self.backbone.fc = nn.Identity()
#             self.embeddings = nn.Sequential(
#                 nn.Linear(784, 512),
#                 L2Norm()
#             )

#     def forward(self, x):
#         x = self.backbone(x)
#         return self.embeddings(x)

    # @classmethod
    # def add_arguments(cls, parser: argparse.ArgumentParser):
    #     parser = parser.add_argument_group("Model")
    #     parser.add_argument(
    #         "--backbone", default="TV_RESNET50", choices=[b.name for b in Backbone]
    #     )
    #     parser.add_argument("--kd_dim", default=512, type=int)
    #     parser.add_argument("--dim", type=int)
    #     parser.add_argument("--pool_param", default=3, type=float)
