import argparse
import subprocess
import os

DEFAULT_COMMAND = """python main.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --dataset_name="lambdalabs/naruto-blip-captions" --resolution=512 --center_crop --random_flip --train_batch_size=4 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="output\""""

def run_command(gpu_rank, batch_size=4):
    cmd = DEFAULT_COMMAND.replace('--train_batch_size=4', f'--train_batch_size={batch_size}')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_rank)
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
    )
    return process.pid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--processes', nargs='+', type=int, required=True)
    parser.add_argument('-b', '--batch_sizes', nargs='+', type=int)

    args = parser.parse_args()
    
    if args.batch_sizes and len(args.batch_sizes) != len(args.processes):
        print("Error: Number of batch sizes must match number of processes")
        return

    for i, gpu_rank in enumerate(args.processes):
        batch_size = args.batch_sizes[i] if args.batch_sizes else 4
        pid = run_command(gpu_rank, batch_size)
        print(f"Started process on GPU {gpu_rank} with PID {pid}")

if __name__ == '__main__':
    main()