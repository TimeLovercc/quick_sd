import subprocess

def create_tmux_session(session_name, servers):
    # Check if the session already exists
    existing_session_check = subprocess.run(['tmux', 'has-session', '-t', session_name], 
                                             stdout=subprocess.PIPE, 
                                             stderr=subprocess.PIPE)
    
    # If the session exists, kill it
    if existing_session_check.returncode == 0:
        print(f'Session "{session_name}" already exists. Killing it...')
        subprocess.run(['tmux', 'kill-session', '-t', session_name])
    
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

    print(f'Tmux session "{session_name}" created with windows for servers: {", ".join(servers)}')
    print(f'Use `tmux a -t {session_name}` to attach to the session.')

if __name__ == "__main__":
    session_name = 'servers'
    servers = ['pcz', 'dgx1', 'gpu01', 'gpu02']
    create_tmux_session(session_name, servers)
