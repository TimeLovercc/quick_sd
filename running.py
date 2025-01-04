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

def start_process(command, conda_env="torch"):
    """Start a process in the background with conda environment"""
    # Get the path to conda executable
    conda_path = os.path.expanduser("~/miniconda3/bin/conda")
    if not os.path.exists(conda_path):
        conda_path = os.path.expanduser("~/anaconda3/bin/conda")
    
    # Create the full command with conda activation
    full_command = f"""
        source {os.path.dirname(conda_path)}/activate {conda_env} && \
        cd {os.getcwd()} && \
        {command}
    """
    
    # Start the process
    process = subprocess.Popen(
        full_command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp,
        executable='/bin/bash'  # Explicitly use bash
    )
    return process.pid

def main():
    parser = argparse.ArgumentParser(description="Run processes on different GPUs in background")
    parser.add_argument('-p', '--processes', nargs='+', type=int, required=True,
                      help='List of GPU ranks to use')
    parser.add_argument('-b', '--batch_sizes', nargs='+', type=int,
                      help='List of batch sizes for each GPU')
    parser.add_argument('--conda_env', default='torch',
                      help='Conda environment name (default: torch)')

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
            
            pid = start_process(full_command, args.conda_env)
            print(f"Started process on GPU {gpu_rank} with PID {pid}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()