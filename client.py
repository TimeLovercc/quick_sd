import subprocess
import time

def create_tmux_session(session_name, servers):
    # Create a new tmux session
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])
    
    # Create windows for each server and run commands
    for i, server in enumerate(servers):
        if i == 0:
            # In the first window, attach to the first server directly
            subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:0', f'assh {server}', 'C-m'])
            subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:0', 'watch -n0.1 nvidia-smi', 'C-m'])
        else:
            # Create a new window for each subsequent server
            subprocess.run(['tmux', 'new-window', '-t', session_name])
            subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:{i}', f'assh {server}', 'C-m'])
            subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:{i}', 'watch -n0.1 nvidia-smi', 'C-m'])

    # Detach from the session
    subprocess.run(['tmux', 'detach-session', '-s', session_name])
    
    print(f'Tmux session "{session_name}" created with windows for servers: {", ".join(servers)}')

if __name__ == "__main__":
    session_name = 'servers'
    servers = ['pcz', 'dgx1', 'gpu01', 'gpu02']
    create_tmux_session(session_name, servers)
