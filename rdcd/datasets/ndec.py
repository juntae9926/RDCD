# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
from typing import Callable, Dict, Optional
from torchvision.datasets.folder import default_loader
from rdcd.transforms.repeated_augmentation import RepeatedAugmentationTransform

from rdcd.datasets.image_folder import get_image_paths
from rdcd.datasets.isc.descriptor_matching import (
    knn_match_and_make_predictions,
    match_and_make_predictions,
)
from rdcd.datasets.isc.io import read_ground_truth
from rdcd.datasets.isc.metrics import evaluate, Metrics

class NDECTrainDataset:
    """A data module describing datasets used during training."""

    def __init__(
        self,
        train_dataset_path,
        query_dataset_path, 
        ref_dataset_path,
        augmentations,
        train_image_size=224,
        supervised=True,
    ):
        self.train_dataset_path = train_dataset_path
        self.query_dataset_path = query_dataset_path
        self.ref_dataset_path = ref_dataset_path
        self.supervised = supervised

        self.files = get_image_paths(self.train_dataset_path)
        transform = augmentations.get_transformations(train_image_size)
        self.img_transform = RepeatedAugmentationTransform(transform, copies=2)
        self.loader = default_loader

    def __len__(self):
        if self.supervised:
            return len(self.query_paths)
        else:
            return len(self.files)
    
    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.files[idx])
        record = {"input": img, "instance_id": idx}
        record = self.img_transform(record)
        return record
    

class NDECEvalDataset:
    """DISC2021 evaluation dataset."""

    SPLIT_REF = 0
    SPLIT_QUERY = 1
    SPLIT_TRAIN = 2

    def __init__(
        self,
        path: str,
        transform: Callable = None,
        include_train: bool = False,
        # Specific paths for each part of the dataset. If not set, inferred from `path`.
        query_path: Optional[str] = None,
        ref_path: Optional[str] = None,
        train_path: Optional[str] = None,
        gt_path: Optional[str] = None,
    ):

        query_path = os.path.join(path, "final_queries")
        ref_path = os.path.join(path, "references")
        train_path = os.path.join(path, "train") if include_train else None
        gt_path = os.path.join(path, "gt_1k.csv")
        self.files, self.metadata = self.read_files(ref_path, self.SPLIT_REF)
        query_files, query_metadata = self.read_files(query_path, self.SPLIT_QUERY)
        self.files.extend(query_files)
        self.metadata.extend(query_metadata)
        if train_path:
            train_files, train_metadata = self.read_files(train_path, self.SPLIT_TRAIN)
            self.files.extend(train_files)
            self.metadata.extend(train_metadata)
        self.gt = read_ground_truth(gt_path)
        self.transform = transform

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        img = default_loader(filename)
        if self.transform:
            img = self.transform(img)
        sample = {"input": img, "instance_id": idx}
        sample.update(self.metadata[idx])
        return sample

    def __len__(self):
        return len(self.files)

    @classmethod
    def read_files(cls, path, split):
        files = get_image_paths(path)
        names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        metadata = [
            dict(name=name, split=split, image_num=int(name[1:]), target=-1)
            for name in names
        ]
        return files, metadata

    def retrieval_eval(
        self, embedding_array, targets, split, **kwargs
    ) -> Dict[str, float]:
        query_mask = split == self.SPLIT_QUERY
        ref_mask = split == self.SPLIT_REF
        query_ids = targets[query_mask]
        query_embeddings = embedding_array[query_mask, :]
        ref_ids = targets[ref_mask]
        ref_embeddings = embedding_array[ref_mask, :]
        return self.retrieval_eval_splits(
            query_ids, query_embeddings, ref_ids, ref_embeddings, **kwargs
        )

    def retrieval_eval_splits(
        self,
        query_ids,
        query_embeddings,
        ref_ids,
        ref_embeddings,
        use_gpu=False,
        k=10,
        global_candidates=False,
        **kwargs
    ) -> Dict[str, float]:
        query_names = ["Q%05d" % i for i in query_ids]
        ref_names = ["R%06d" % i for i in ref_ids]
        if global_candidates:
            predictions = match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                num_results=k * len(query_names),
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        else:
            predictions = knn_match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                k=k,
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        results: Metrics = evaluate(self.gt, predictions)
        return {
            "uAP": results.average_precision,
            "accuracy-at-1": results.recall_at_rank1,
            "recall-at-p90": results.recall_at_p90 or 0.0,
        }


class NDECTestDataset:
    """DISC2021 evaluation dataset."""

    SPLIT_REF = 0
    SPLIT_QUERY = 1
    SPLIT_TRAIN = 2

    def __init__(
        self,
        path: str,
        transform: Callable = None,
        include_train: bool = False,
        # Specific paths for each part of the dataset. If not set, inferred from `path`.
        query_path: Optional[str] = None,
        ref_path: Optional[str] = None,
        train_path: Optional[str] = None,
        gt_path: Optional[str] = None,
    ):

        query_path = os.path.join(path, "NDEC_test_query")
        ref_path = os.path.join(path, "DISC/images/references")
        train_path = os.path.join(path, "DISC/images/train") if include_train else None
        gt_path = os.path.join(path, "public_ground_truth_h5.csv")
        self.files, self.metadata = self.read_files(ref_path, self.SPLIT_REF)
        query_files, query_metadata = self.read_files(query_path, self.SPLIT_QUERY)
        self.files.extend(query_files)
        self.metadata.extend(query_metadata)
        if train_path:
            print('use train dataset')
            train_files, train_metadata = self.read_files(train_path, self.SPLIT_TRAIN)
            self.files.extend(train_files)
            self.metadata.extend(train_metadata)
        self.gt = read_ground_truth(gt_path)
        self.transform = transform

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        img = default_loader(filename)
        if self.transform:
            img = self.transform(img)
        sample = {"input": img, "instance_id": idx}
        sample.update(self.metadata[idx])
        return sample

    def __len__(self):
        return len(self.files)

    @classmethod
    def read_files(cls, path, split):
        files = get_image_paths(path)
        names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        metadata = [
            dict(name=name, split=split, image_num=int(name[1:]), target=-1)
            for name in names
        ]
        return files, metadata

    def retrieval_eval(
        self, embedding_array, targets, split, **kwargs
    ) -> Dict[str, float]:
        query_mask = split == self.SPLIT_QUERY
        ref_mask = split == self.SPLIT_REF
        query_ids = targets[query_mask]
        query_embeddings = embedding_array[query_mask, :]
        ref_ids = targets[ref_mask]
        ref_embeddings = embedding_array[ref_mask, :]
        return self.retrieval_eval_splits(
            query_ids, query_embeddings, ref_ids, ref_embeddings, **kwargs
        )

    def retrieval_eval_splits(
        self,
        query_ids,
        query_embeddings,
        ref_ids,
        ref_embeddings,
        use_gpu=False,
        k=10,
        global_candidates=False,
        **kwargs
    ) -> Dict[str, float]:
        query_names = ["Q%05d" % i for i in query_ids]
        ref_names = ["R%06d" % i for i in ref_ids]
        if global_candidates:
            predictions = match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                num_results=k * len(query_names),
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        else:
            predictions = knn_match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                k=k,
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        results: Metrics = evaluate(self.gt, predictions)
        return {
            "uAP": results.average_precision,
            "accuracy-at-1": results.recall_at_rank1,
            "recall-at-p90": results.recall_at_p90 or 0.0,
        }
