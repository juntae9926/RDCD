

# Relational Self-supervised Distillation with Compact Descriptors for Image Copy Detection.

## Implementation
- This code is implemented with Pytorch Lightning


## Datasets used
- DISC (Facebook Image Similarity Challenge 2021)
- NDEC (A Benchmark and Asymmetrical-Similarity Learning for Practical Image Copy
Detection 2023)
- Copydays

## Unsupervised Knowledge Distillation
- similarity distillation 
- KoLeo regularization
- DirectCLR contrastive learning

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

The codebase is built upon "[SSCD](https://cvpr2022.thecvf.com/)".

## Citation
```
@article{juntae2023,
  title={Self-supervised Knowledge Distillation using Dynamic Memory Bank for Image Copy Detection},
  author={Juntae Kim},
  journal={Master's Thesis},
  year={2023}
}
```
