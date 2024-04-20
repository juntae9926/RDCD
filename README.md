

# Relational Self-supervised Distillation with Compact Descriptors for Image Copy Detection.
The official repository for Relational Self-supervised Distillation with Compact Descriptors for Image Copy Detection.

## Pipeline

![framework](figs/framework.png)

## Implementation
- 
- This code is implemented using Pytorch Lightning


## Datasets used
- DISC (Facebook Image Similarity Challenge 2021)
- NDEC (A Benchmark and Asymmetrical-Similarity Learning for Practical Image Copy
Detection 2023)
- CD10K(copydays + 10k distractors)

## Requirements

### Installation

```bash
pip install -r requirements.txt
(we use /torch 1.6.0 /torchvision 0.7.0 /timm 0.3.2 /cuda 10.1 / 16G or 32G V100 for training and evaluation.
Note that we use torch.cuda.amp to accelerate speed of training which requires pytorch >=1.6)
```

## Teacher Models(Pretrained)
- SSCD: ResNet-50, ResNeXt101
- DINO: ViT-B/8

## Student Models
- EfficientNet-B0
- MobileNet-V3-Large
- FastViT-T12

## How to use

### docker install
```
docker build -t rdcd .
docker run --name rdcd -it --ipc=host --runtime=nvidia rdcd
```

### Library 설치
```
pip install -c pytorch faiss-gpu
pip install Pillow==9.5.0
```

### Training
```
sh train.sh
```

### Evaluation
```
sh disc_eval.sh
```

## Acknowledgement

Codebase from "[SSCD](https://github.com/facebookresearch/sscd-copy-detection)" , [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
