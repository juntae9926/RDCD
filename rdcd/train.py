#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
import logging
import os
import numpy as np
import pytorch_lightning as pl
import torch

# Set up our python environment (eg. PYTHONPATH).
from lib import initialize  # noqa

from classy_vision.dataset.transforms import build_transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F

from rdcd.datasets.disc import DISCEvalDataset, DISCTrainDataset, DISCTestDataset
from sscd.datasets.ndec import NDECEvalDataset, NDECTrainDataset, NDECTestDataset
from rdcd.datasets.image_folder import ImageFolder
from rdcd.lib.distributed_util import cross_gpu_batch
from rdcd.transforms.repeated_augmentation import RepeatedAugmentationTransform
from rdcd.models.model import Backbone, TeacherNetwork
from rdcd.transforms.settings import AugmentationSetting
from rdcd.lib.util import call_using_args, parse_bool
from rdcd.lib.pbank import Pbank
from rdcd.models.moco import MoCo
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.models.mobilenetv3 import mobilenet_v3_large

DEBUG = False

if DEBUG:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def add_train_args(parser: ArgumentParser, required=True):
    parser = parser.add_argument_group("Train")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument(
        "--batch_size",
        default=4096,
        type=int,
        help="The global batch size (across all nodes/GPUs, before repeated augmentation)",
    )
    parser.add_argument("--infonce_temperature", default=0.05, type=float)
    parser.add_argument("--entropy_weight", default=30, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--augmentations",
        default="STRONG_BLUR",
        choices=[x.name for x in AugmentationSetting],
    )
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument(
        "--base_learning_rate",
        default=0.3,
        type=float,
        help="Base learning rate, for a batch size of 256. Linear scaling is applied.",
    )
    parser.add_argument(
        "--absolute_learning_rate",
        default=None,
        type=float,
        help="Absolute learning rate (overrides --base_learning_rate).",
    )
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--sync_bn", default=True, type=parse_bool)
    parser.add_argument("--mixup", default=False, type=parse_bool)
    parser.add_argument(
        "--train_image_size", default=224, type=int, help="Image size for training"
    )
    parser.add_argument(
        "--val_image_size", default=288, type=int, help="Image size for validation"
    )
    parser.add_argument(
        "--workers", default=10, type=int, help="Data loader workers per GPU process."
    )
    parser.add_argument("--num_sanity_val_steps", default=0, type=int)

    parser.add_argument('--ckpt', type=str)

    parser.add_argument("--resume", type=str)

    # MoCo params
    parser.add_argument('--queue_size', default=65536, type=int)
    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument("--m", default=0.99, type=float)
    parser.add_argument("--moco_lambda", default=0.1, type=float)
    parser.add_argument("--kld_lambda", default=1, type=float)

    parser.add_argument("--moco", default=False, type=parse_bool)
    parser.add_argument("--kld", default=False, type=parse_bool)
    parser.add_argument("--hn_lambda", default=5, type=int, required=True)



def add_data_args(parser, required=True):
    parser = parser.add_argument_group("Data")
    parser.add_argument("--output_path", required=required, type=str)
    parser.add_argument("--train_dataset_path", required=required, type=str)
    parser.add_argument("--val_dataset_path", required=False, type=str)
    parser.add_argument("--query_dataset_path", required=False, type=str)
    parser.add_argument("--ref_dataset_path", required=False, type=str)


parser = ArgumentParser()
Backbone.add_arguments(parser)
add_train_args(parser)
add_data_args(parser)

class NDECData(pl.LightningDataModule):
    """A data module describing datasets used during training."""

    def __init__(
        self,
        *,
        train_dataset_path,
        val_dataset_path,
        query_dataset_path, 
        ref_dataset_path,
        train_batch_size,
        augmentations: AugmentationSetting,
        train_image_size=224,
        val_image_size=288,
        val_batch_size=64,
        workers=28,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.query_dataset_path = query_dataset_path
        self.ref_dataset_path = ref_dataset_path
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.workers = workers

        self.train_dataset = NDECTrainDataset(
                                train_dataset_path,
                                query_dataset_path, 
                                ref_dataset_path,
                                augmentations=augmentations,
                                supervised=False
                            )
        if val_dataset_path:
            self.val_dataset = self.make_validation_dataset(
                path=self.val_dataset_path,
                size=val_image_size,
            )
        else:
            self.val_dataset = None
        
        self.loader = default_loader
    
    @classmethod
    def make_validation_dataset(
        cls,
        path,
        size=288,
        include_train=False,
        preserve_aspect_ratio=False,
    ):
        transforms = build_transforms(
            [
                {
                    "name": "Resize",
                    "size": size if preserve_aspect_ratio else [size, size],
                },
                {"name": "ToTensor"},
                {
                    "name": "Normalize",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ]
        )
        return NDECEvalDataset(path, transform=transforms, include_train=include_train)

    @classmethod
    def make_test_dataset(
        cls,
        path,
        size=288,
        include_train=True,
        preserve_aspect_ratio=False,
    ):
        transforms = build_transforms(
            [
                {
                    "name": "Resize",
                    "size": size if preserve_aspect_ratio else [size, size],
                },
                {"name": "ToTensor"},
                {
                    "name": "Normalize",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ]
        )
        return NDECTestDataset(path, transform=transforms, include_train=include_train)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.workers,
            persistent_workers=True,
        )


class DISCData(pl.LightningDataModule):
    """A data module describing datasets used during training."""

    def __init__(
        self,
        *,
        train_dataset_path,
        val_dataset_path,
        query_dataset_path, 
        ref_dataset_path,
        train_batch_size,
        augmentations: AugmentationSetting,
        train_image_size=224,
        val_image_size=288,
        val_batch_size=64,
        workers=28,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.query_dataset_path = query_dataset_path
        self.ref_dataset_path = ref_dataset_path
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.workers = workers

        self.train_dataset = DISCTrainDataset(
                                train_dataset_path,
                                query_dataset_path, 
                                ref_dataset_path,
                                augmentations=augmentations,
                                supervised=False
                            )
        if val_dataset_path:
            self.val_dataset = self.make_validation_dataset(
                path=self.val_dataset_path,
                size=val_image_size,
            )
        else:
            self.val_dataset = None
        
        self.loader = default_loader
    
    @classmethod
    def make_validation_dataset(
        cls,
        path,
        size=288,
        include_train=False,
        preserve_aspect_ratio=False,
    ):
        transforms = build_transforms(
            [
                {
                    "name": "Resize",
                    "size": size if preserve_aspect_ratio else [size, size],
                },
                {"name": "ToTensor"},
                {
                    "name": "Normalize",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ]
        )
        return DISCEvalDataset(path, transform=transforms, include_train=include_train)

    @classmethod
    def make_test_dataset(
        cls,
        path,
        size=288,
        include_train=True,
        preserve_aspect_ratio=False,
    ):
        transforms = build_transforms(
            [
                {
                    "name": "Resize",
                    "size": size if preserve_aspect_ratio else [size, size],
                },
                {"name": "ToTensor"},
                {
                    "name": "Normalize",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ]
        )
        return DISCTestDataset(path, transform=transforms, include_train=include_train)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.workers,
            persistent_workers=True,
        )

class RDCD(pl.LightningModule):
    """Training class for RDCD models."""

    def __init__(self, args, train_steps: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = MoCo(args=args, base_encoder=efficientnet_b0, m=args.m, T=args.temp)

        self.teacher = TeacherNetwork(args)
        state_dict = torch.load(args.ckpt)
        #### DINO Teacher ####
        if 'vit' in args.ckpt:
            for k in list(state_dict.keys()):
                state_dict['backbone.' + k] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        ######################
        self.teacher.load_state_dict(state_dict)
        self.teacher.freeze()
        print(self.model)
        print(self.teacher)
        use_mixup = args.mixup
        self.mixup = (
            ContrastiveMixup.from_config(
                {
                    "name": "contrastive_mixup",
                    "mix_prob": 0.05,
                    "mixup_alpha": 2,
                    "cutmix_alpha": 2,
                    "switch_prob": 0.5,
                    "repeated_augmentations": 2,
                    "target_column": "instance_id",
                }
            )
            if use_mixup
            else None
        )
        self.infonce_temperature = args.infonce_temperature
        self.entropy_weight = args.entropy_weight
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.lr = args.absolute_learning_rate or (
            args.base_learning_rate * args.batch_size / 256
        )
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.train_steps = train_steps

        self.moco_lambda = args.moco_lambda
        self.kld_lambda = args.kld_lambda
        self.hn_lambda = args.hn_lambda

        self.moco = args.moco
        self.kd_dim = args.kd_dim
        if args.kld:
            self.kld = Pbank(teacher_feats_dim=self.kd_dim, queue_size=args.queue_size, T=0.04)
        else:
            self.kld = False

    def forward(self, query, key=None):
        if not self.training:
            feat_s_q = self.model(query, eval=True)
            return feat_s_q
        else:
            feat_t_q = self.teacher(query)
            feat_s_q, emb_s_q, loss, stats = self.model(im_q=query, im_k=key)

        return feat_t_q, feat_s_q, loss, stats

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs * self.train_steps,
                max_epochs=self.epochs * self.train_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def training_step(self, batch, batch_idx):
        query, key = batch["input0"], batch["input1"]
        feat_t, feat_s, moco_loss, stats = self(query, key)
        
        total_loss = 0.0

        if self.moco:
            moco_loss = moco_loss * self.moco_lambda
            self.log("moco_loss", moco_loss, on_step=True, on_epoch=True, prog_bar=True)
            total_loss += moco_loss

        if self.kld:
            kld_loss = self.kld(feat_t, feat_s) * self.kld_lambda
            total_loss += kld_loss
            self.log("kld_loss", kld_loss, on_step=True, on_epoch=True, prog_bar=True)

       
        self.log_dict(stats, on_step=True, on_epoch=True)
        self.log("total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        input = batch["input"]
        metadata_keys = ["image_num", "split", "instance_id"]
        batch = {k: v for (k, v) in batch.items() if k in metadata_keys}
        batch["embeddings"] = self(input)
        return batch

    def validation_epoch_end(self, outputs):
        keys = ["embeddings", "image_num", "split", "instance_id"]
        outputs = {k: torch.cat([out[k] for out in outputs]) for k in keys}
        outputs = self._gather(outputs)
        outputs = self.dedup_outputs(outputs)
        if self.current_epoch == 0:
            self.print(
                "Eval dataset size: %d (%d queries, %d index)"
                % (
                    outputs["split"].shape[0],
                    (outputs["split"] == DISCEvalDataset.SPLIT_QUERY).sum(),
                    (outputs["split"] == DISCEvalDataset.SPLIT_REF).sum(),
                )
            )
        dataset: DISCEvalDataset = self.trainer.datamodule.val_dataset
        metrics = dataset.retrieval_eval(
            outputs["embeddings"],
            outputs["image_num"],
            outputs["split"],
        )
        metrics = {k: 0.0 if v is None else v for (k, v) in metrics.items()}
        self.log_dict(metrics, on_epoch=True)

    def on_train_epoch_end(self):
        metrics = []
        for k, v in self.trainer.logged_metrics.items():
            if k.endswith("_step"):
                continue
            if k.endswith("_epoch"):
                k = k[: -len("_epoch")]
            if torch.is_tensor(v):
                v = v.item()
            metrics.append(f"{k}: {v:.3f}")
        metrics = ", ".join(metrics)
        self.print(f"Epoch {self.current_epoch}: {metrics}")

    def on_train_end(self):
        if self.global_rank != 0:
            return
        if not self.logger:
            return
        path = os.path.join(self.logger.log_dir, "model_torchscript.pt")
        self.save_torchscript(path)

    def save_torchscript(self, filename):
        self.eval()
        input = torch.randn((1, 3, 64, 64), device=self.device)
        script = torch.jit.trace(self.model, input)
        torch.jit.save(script, filename)

    def _gather(self, batch):
        batch = self.all_gather(move_data_to_device(batch, self.device))
        return {
            k: v.reshape([-1] + list(v.size()[2:])).cpu() for (k, v) in batch.items()
        }

    @staticmethod
    def dedup_outputs(outputs, key="instance_id"):
        """Deduplicate dataset on instance_id."""
        idx = np.unique(outputs[key].numpy(), return_index=True)[1]
        outputs = {k: v.numpy()[idx] for (k, v) in outputs.items()}
        assert np.unique(outputs[key]).size == outputs["instance_id"].size
        return outputs

    def predict_step(self, batch, batch_idx):
        batch = self.validation_step(batch, batch_idx)

        # Workaround for a CUDA synchronization bug in PyTorch Lightning.
        # Fixed upstream:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/11287
        batch = {k: v.cpu() for (k, v) in batch.items()}

        return batch


def main(args):
    world_size = args.nodes * args.gpus
    if args.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size ({args.batch_size}) must be a multiple of "
            f"the number of GPUs ({world_size})."
        )
    data = DISCData(
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        query_dataset_path=args.query_dataset_path,
        ref_dataset_path=args.ref_dataset_path,
        train_batch_size=args.batch_size // world_size,
        train_image_size=args.train_image_size,
        val_image_size=args.val_image_size,
        augmentations=AugmentationSetting[args.augmentations],
        workers=args.workers,
    )
    model = RDCD(
        args,
        train_steps=len(data.train_dataset) // args.batch_size,
    )
    if args.resume:
        trainer = pl.Trainer(
            devices=args.gpus,
            num_nodes=args.nodes,
            accelerator=args.accelerator,
            max_epochs=args.epochs,
            sync_batchnorm=args.sync_bn,
            default_root_dir=args.output_path,
            strategy="ddp_find_unused_parameters_false",
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            resume_from_checkpoint=args.resume,
            num_sanity_val_steps=args.num_sanity_val_steps,
            callbacks=[LearningRateMonitor()],
        )
    else:
        trainer = pl.Trainer(
            devices=args.gpus,
            num_nodes=args.nodes,
            accelerator=args.accelerator,
            max_epochs=args.epochs,
            sync_batchnorm=args.sync_bn,
            default_root_dir=args.output_path,
            strategy=DDPSpawnPlugin(find_unused_parameters=False),
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            num_sanity_val_steps=args.num_sanity_val_steps,
            callbacks=[LearningRateMonitor()],
        )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
