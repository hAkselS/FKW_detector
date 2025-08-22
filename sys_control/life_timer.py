'''
File:   sys_control/life_timer.py

Spec:   Handle the life timer for the Raspberry Pi system.
        Monitor the system's uptime and trigger a shutdown if necessary.

ID:     lt
'''

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

def start_timer():
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"pc: ✗ CRITICAL ERROR: Unexpected error loading config: {e}")
        shutdown_pi.shutdown('Config file not opened')
        sys.exit(1)

    allowed_runtime_minutes = config['allowed_runtime_minutes']
    if 1 <= allowed_runtime_minutes <= 120:
        print(f"lt: Starting life timer for {allowed_runtime_minutes} minutes...")
        allowed_runtime_seconds = allowed_runtime_minutes * 60

        for remaining in range(allowed_runtime_seconds, 0, -1):
            if stop_timer_event.is_set():
                print("lt: Timer cancelled early.")
                return  # Exit gracefully

            if remaining % 60 == 0:  # Print a timer pulse every minute
                mins, secs = divmod(remaining, 60)
                print(f'lt: Time remaining: {mins:02}:{secs:02}')

            time.sleep(1)

        print("lt: Allowed runtime exceeded, shutting down peacefully...")
        config['forced_shutdown'] = False
        with open(config_file, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        shutdown_pi.shutdown('Allowed runtime exceeded')
        sys.exit(0)

    else:
        print(f"lt: ✗ ERROR: Invalid ontime_minuites value: {allowed_runtime_minutes}. Must be between 1 and 120.")
        shutdown_pi.shutdown('Invalid ontime_minuites value')
        sys.exit(1)

def run_life_timer():
    # Run the timer in a separate thread
    timer_thread = threading.Thread(target=start_timer, daemon=True)
    timer_thread.start()
    return timer_thread