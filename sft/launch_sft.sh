CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
      --multi-gpu \
      --config_file=../configs/general_acc.yaml \
      sft.py --lr 2e-5 --selekt_alpha 0.05 --weight_decay 0.05 --use-synth-data
