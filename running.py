import argparse
import subprocess
import os

DEFAULT_COMMAND = os.environ.get('DEFAULT_COMMAND', """
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 
MKL_NUM_THREADS=1
python main.py
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
  --dataset_name="lambdalabs/naruto-blip-captions"
  --resolution=512 --center_crop --random_flip
  --train_batch_size=4
  --learning_rate=1e-05
  --max_grad_norm=1
  --lr_scheduler="constant" --lr_warmup_steps=0
  --output_dir="output"
""".replace("\n", " ").strip())

def start_process(command):
    """Start a process in the background using nohup"""
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(
            f"nohup {command} > /dev/null 2>&1 &",
            shell=True,
            stdout=devnull,
            stderr=devnull,
            preexec_fn=os.setpgrp
        )
    return process.pid

def main():
    parser = argparse.ArgumentParser(description="Run processes on different GPUs in background")
    parser.add_argument('-p', '--processes', nargs='+', type=int, required=True,
                      help='List of GPU ranks to use')
    parser.add_argument('-b', '--batch_sizes', nargs='+', type=int,
                      help='List of batch sizes for each GPU')

    args = parser.parse_args()

    if args.batch_sizes and len(args.batch_sizes) != len(args.processes):
        print("Error: The number of batch sizes must match the number of processes.")
        return

    try:
        for i, gpu_rank in enumerate(args.processes):
            command = DEFAULT_COMMAND
            
            # Inject respective batch size into the command
            batch_size = args.batch_sizes[i] if args.batch_sizes else 4
            command = command.replace('--train_batch_size=4', f'--train_batch_size={batch_size}')
            
            cuda_visible_devices = str(gpu_rank)
            full_command = f'CUDA_VISIBLE_DEVICES={cuda_visible_devices} OMP_NUM_THREADS=1 {command}'
            
            pid = start_process(full_command)
            print(f"Started process on GPU {gpu_rank} with PID {pid}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()