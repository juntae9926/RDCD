sscd/ndec_eval.py --ndec_path /data/ --gpus=2 \
  --output_path=./ndec_eval/ndec_eval/ours_64 \
  --size=224 --preserve_aspect_ratio=false --workers=16 \
  --backbone=TV_EFFICIENTNET_B0 --dims=64 --checkpoint=/user/ndec_eval/ours_64/epoch=99-step=390599.ckpt
