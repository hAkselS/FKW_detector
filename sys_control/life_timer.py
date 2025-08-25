'''
File:   sys_control/life_timer.py

Spec:   Handle the life timer for the Raspberry Pi system.
        Monitor the system's uptime and trigger a shutdown if necessary.

ID:     lt
'''

from logging import config
import time 
import sys 
import yaml 
import os 
import threading
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import sys_control.shutdown_pi as shutdown_pi


###################################################################
# CONFIGURATION DEFAULTS
config_file = project_root + '/config/config.yaml'
###################################################################

# Global stop signal
stop_timer_event = threading.Event()

def start_timer(allowed_runtime_minutes):
    print(f"\nlt: Starting life timer for {allowed_runtime_minutes} minutes...")
    allowed_runtime_seconds = allowed_runtime_minutes * 60

    for remaining in range(allowed_runtime_seconds, 0, -1):
        if stop_timer_event.is_set():
            print("lt: Timer cancelled early.")
            return  # Exit gracefully

        if remaining % 60 == 0:  # Print a timer pulse every minute
            mins, secs = divmod(remaining, 60)
            print(f'lt: Time remaining: {mins:02}:{secs:02}')

        time.sleep(1)
    try: 
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"lt: Error reading config file: {e}")
        return

    print("lt: Allowed runtime exceeded, shutting down peacefully...")
    config['forced_shutdown'] = False
    with open(config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    shutdown_pi.shutdown('Allowed runtime exceeded')
    sys.exit(0)



def run_life_timer(allowed_runtime_minutes):
    # Run the timer in a separate thread
    timer_thread = threading.Thread(target=start_timer, args=(allowed_runtime_minutes,), daemon=True)
    timer_thread.start()
    return timer_thread