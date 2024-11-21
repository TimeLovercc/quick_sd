import argparse
import subprocess
import os
import time

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

def run_tmux_command(command):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(command)}: {e}")
        return None

def check_and_kill_existing_session(session_name):
    result = subprocess.run(['tmux', 'has-session', '-t', session_name], capture_output=True)
    if result.returncode == 0:
        print(f"Existing session '{session_name}' found. Killing it.")
        subprocess.run(['tmux', 'kill-session', '-t', session_name])
    else:
        print(f"No existing session '{session_name}' found.")

def create_tmux_session(session_name):
    check_and_kill_existing_session(session_name)
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])

def run_commands_in_tmux(session_name, commands):
    # Initial setup
    run_tmux_command(['tmux', 'send-keys', '-t', f'{session_name}:0', 'conda activate torch', 'C-m'])
    run_tmux_command(['tmux', 'send-keys', '-t', f'{session_name}:0', 'cd ~/workspaces/quick_sd/', 'C-m'])
    
    # Run all commands in background except the last one
    for command in commands[:-1]:
        run_tmux_command(['tmux', 'send-keys', '-t', f'{session_name}:0', f'{command} &', 'C-m'])
    
    # Run the last command in foreground
    if commands:
        run_tmux_command(['tmux', 'send-keys', '-t', f'{session_name}:0', commands[-1], 'C-m'])

def main():
    parser = argparse.ArgumentParser(description="Run processes on different GPUs using tmux")
    parser.add_argument('--session_name', default='sd', help='Base name for the tmux session')
    parser.add_argument('-p', '--processes', nargs='+', type=int, help='List of processes in the format "gpu_rank[,command]"')
    parser.add_argument('-b', '--batch_sizes', nargs='+', type=int, help='List of batch sizes for each GPU')

    args = parser.parse_args()

    if args.batch_sizes and len(args.batch_sizes) != len(args.processes):
        print("Error: The number of batch sizes must match the number of processes.")
        return

    try:
        session_name = args.session_name
        create_tmux_session(session_name)

        commands = []
        for i, process in enumerate(args.processes):
            gpu_rank = process
            command = DEFAULT_COMMAND
            
            # Inject respective batch size into the command
            batch_size = args.batch_sizes[i] if args.batch_sizes else 4
            command = command.replace('--train_batch_size=4', f'--train_batch_size={batch_size}')
            
            cuda_visible_devices = str(gpu_rank)
            full_command = f'CUDA_VISIBLE_DEVICES={cuda_visible_devices} OMP_NUM_THREADS=1 {command}'
            commands.append(full_command)

        run_commands_in_tmux(session_name, commands)
        print(f"Tmux session '{session_name}' created with all processes running in single window. Use 'tmux a -t {session_name}' to view.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your tmux sessions and manually clean up if necessary.")

if __name__ == '__main__':
    main()