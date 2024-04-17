CUDA_VISIBLE_DEVICES=6 \
rdcd/disc_eval.py --disc_path /DATA/DISC --gpus=1 \
  --output_path=/ieee_store/diff_models/mobile/256 \
  --size=288 --preserve_aspect_ratio=false --workers=16 \
  --backbone=TV_MOBILENETV3 --dim=256 --checkpoint=/user/mobile/256/epoch=99-step=390599.ckpt

