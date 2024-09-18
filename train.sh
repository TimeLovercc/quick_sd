
#!/bin/bash


DEFAULT_COMMAND="python main.py \
  --pretrained_model_name_or_path=\"CompVis/stable-diffusion-v1-4\" \
  --dataset_name=\"lambdalabs/naruto-blip-captions\" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler=\"constant\" --lr_warmup_steps=0 \
  --output_dir=\"output\""

export DEFAULT_COMMAND
export OMP_NUM_THREADS=1


python running.py -p "2,1" "3,1"