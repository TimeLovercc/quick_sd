import argparse
import subprocess
import os
import time

DEFAULT_COMMAND = os.environ.get('DEFAULT_COMMAND', """
python main.py
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
  --dataset_name="lambdalabs/naruto-blip-captions"
  --resolution=512 --center_crop --random_flip
  --train_batch_size=4
  --max_train_steps=1000000000
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

def get_available_session_name(base_name):
    for i in range(1, 101):  # Try up to 100 session names
        session_name = f"{base_name}_{i}"
        if not run_tmux_command(['tmux', 'has-session', '-t', session_name]):
            return session_name
    raise RuntimeError("Unable to find an available tmux session name")

def create_tmux_session(session_name):
    result = run_tmux_command(['tmux', 'new-session', '-d', '-s', session_name])
    if result is None:
        raise RuntimeError(f"Failed to create tmux session {session_name}")
    return session_name

def run_command_in_tmux(session_name, window_name, command):
    run_tmux_command(['tmux', 'new-window', '-t', session_name, '-n', window_name])
    run_tmux_command(['tmux', 'send-keys', '-t', f'{session_name}:{window_name}', 'conda activate torch', 'C-m'])
    run_tmux_command(['tmux', 'send-keys', '-t', f'{session_name}:{window_name}', 'cd ~/workspaces/quick_sd/', 'C-m'])
    run_tmux_command(['tmux', 'send-keys', '-t', f'{session_name}:{window_name}', command, 'C-m'])

def main():
    parser = argparse.ArgumentParser(description="Run processes on different GPUs using tmux")
    parser.add_argument('--session_name', default='sd', help='Base name for the tmux session')
    parser.add_argument('-p', '--processes', nargs='+', help='List of processes in the format "gpu_rank,num_gpus[,command]"')
    
    args = parser.parse_args()

    try:
        session_name = get_available_session_name(args.session_name)
        create_tmux_session(session_name)

        for i, process in enumerate(args.processes):
            parts = process.split(',')
            gpu_rank, num_gpus = parts[:2]
            command = ','.join(parts[2:]) if len(parts) > 2 else DEFAULT_COMMAND
            
            cuda_visible_devices = ','.join(str(int(gpu_rank) + j) for j in range(int(num_gpus)))
            full_command = f'CUDA_VISIBLE_DEVICES={cuda_visible_devices} {command}'
            
            window_name = f'gpu_{gpu_rank}'
            run_command_in_tmux(session_name, window_name, full_command)

        print(f"Tmux session '{session_name}' created with all processes. Use 'tmux a -t {session_name}' to view.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your tmux sessions and manually clean up if necessary.")

if __name__ == '__main__':
    main()