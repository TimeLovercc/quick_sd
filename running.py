import argparse
import subprocess
import os
import time
import signal
import atexit
import json
from pathlib import Path
from datetime import datetime

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

class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.pid_file = Path("running_processes.json")
        self.load_existing_processes()
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

    def load_existing_processes(self):
        if self.pid_file.exists():
            try:
                with open(self.pid_file) as f:
                    stored_processes = json.load(f)
                    for pid, info in stored_processes.items():
                        if self.is_process_running(int(pid)):
                            self.processes[int(pid)] = info
            except json.JSONDecodeError:
                print("Warning: Could not load existing process file")

    def save_processes(self):
        with open(self.pid_file, 'w') as f:
            json.dump({str(k): v for k, v in self.processes.items()}, f)

    @staticmethod
    def is_process_running(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def start_process(self, command, gpu_rank):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        stdout_file = log_dir / f"process_{gpu_rank}_{timestamp}.out"
        stderr_file = log_dir / f"process_{gpu_rank}_{timestamp}.err"
        
        with open(stdout_file, 'w') as out, open(stderr_file, 'w') as err:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=out,
                stderr=err,
                preexec_fn=os.setsid
            )
            
        self.processes[process.pid] = {
            'gpu_rank': gpu_rank,
            'command': command,
            'stdout': str(stdout_file),
            'stderr': str(stderr_file),
            'start_time': timestamp
        }
        self.save_processes()
        return process.pid

    def cleanup(self):
        for pid in list(self.processes.keys()):
            try:
                os.killpg(pid, signal.SIGTERM)
                print(f"Terminated process group {pid}")
            except ProcessLookupError:
                pass
        if self.pid_file.exists():
            self.pid_file.unlink()

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}. Cleaning up...")
        self.cleanup()
        exit(0)

def main():
    parser = argparse.ArgumentParser(description="Run processes on different GPUs in background")
    parser.add_argument('-p', '--processes', nargs='+', type=int, required=True,
                      help='List of GPU ranks to use')
    parser.add_argument('-b', '--batch_sizes', nargs='+', type=int,
                      help='List of batch sizes for each GPU')
    parser.add_argument('--kill', action='store_true',
                      help='Kill all running processes and exit')

    args = parser.parse_args()

    process_manager = ProcessManager()
    
    if args.kill:
        process_manager.cleanup()
        return

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
            
            pid = process_manager.start_process(full_command, gpu_rank)
            print(f"Started process on GPU {gpu_rank} with PID {pid}")
            print(f"Logs available at:")
            print(f"  stdout: {process_manager.processes[pid]['stdout']}")
            print(f"  stderr: {process_manager.processes[pid]['stderr']}")

    except Exception as e:
        print(f"An error occurred: {e}")
        process_manager.cleanup()

if __name__ == '__main__':
    main()