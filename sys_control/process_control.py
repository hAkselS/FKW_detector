'''
File:   sys_control/process_control.py

Spec:   Handle all processes and their orderings. 
        Get configs from config.yaml and check for forced shutdown. 
        Create start the emergency shutdown timer.
        Run select audio, then run transform and inference
        on the files selected for sampling. 
        Reset forced shutdown flag, stop forced shutdown timer,
        shutdown the system gracefully.

ID:     pc 

Usage:  python3 sys_control/process_control.py 
'''

import sys
import os
import yaml

# Add project root to sys.path so audio_transform can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import sys_control.transform_and_inference as transform_and_inference
import sys_control.select_audio as select_audio
import sys_control.shutdown_pi as shutdown_pi
import sys_control.life_timer as life_timer


###################################################################
# CONFIGURATION DEFAULTS
config_file = project_root + '/config/config.yaml'
###################################################################

def main():
    print("pc: Starting Process Control")
    # Open the config file 
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"pc: ✗ CRITICAL ERROR: Unexpected error loading config: {e}")
        shutdown_pi.shutdown('Config file not opened')
        sys.exit(1)

    # Check forced shutdown status: True = bad (system failed) False = good (system operated nominally)
    if config['forced_shutdown'] == True:
        print("pc: ✗ CRITICAL ERROR: Forced shutdown flag was upon startup set to True")
        shutdown_pi.shutdown('Forced shutdown flag was found as True')
        sys.exit(1)

    # Grab the allowed runtime minutes for the life timer
    try:    
        allowed_runtime_minutes = config['allowed_runtime_minutes']
        if not (1 <= allowed_runtime_minutes <= 120):
            print(f"pc: ✗ CRITICAL ERROR: Invalid allowed_runtime_minutes value: {allowed_runtime_minutes}. Must be between 1 and 120.")
            shutdown_pi.shutdown('Invalid allowed_runtime_minutes value')
            sys.exit(1)

    except Exception as e:
        print(f"pc: ✗ CRITICAL ERROR: Unexpected error reading allowed_runtime_minutes: {e}")
        shutdown_pi.shutdown('Error reading allowed_runtime_minutes')
        sys.exit(1)

    # Grab the model path and confidence threshold
    try:
        model_path = config['model_path']

        if not os.path.exists(model_path):
            print(f"pc: ✗ CRITICAL ERROR: Model file not found: {model_path}")
            shutdown_pi.shutdown('Model file not found')
            sys.exit(1)

        confidence_threshold = config['confidence_threshold']

        if not (0.0 <= confidence_threshold <= 1.0):
            print(f"pc: ✗ CRITICAL ERROR: Invalid confidence_threshold value: {confidence_threshold}. Must be between 0.0 and 1.0.")
            shutdown_pi.shutdown('Invalid confidence_threshold value')
            sys.exit(1)

    except Exception as e:
        print(f"pc: ✗ CRITICAL ERROR: Unexpected error reading model_path or confidence_threshold: {e}")
        shutdown_pi.shutdown('Error reading model_path or confidence_threshold')
        sys.exit(1)

    # Grab the desired GPIO shutdown pin on time
    try:    
        config_GPIO_on_time = config['shutdown_GPIO_on_time']
    except Exception as e:
        print(f"pc: ✗ ERROR: Unexpected error reading shutdown_GPIO_on_time: {e}... Continuing with default of 15 seconds")

    # Grab the desired GPIO shutdown pin
    try:
        config_GPIO_pin_num = config['shutdown_GPIO_pin_num']
    except Exception as e:
        print(f"pc: ✗ ERROR: Unexpected error reading shutdown_GPIO_pin_num: {e}... Continuing with default of 11")

    # Set the forced shutdown flag true
    config['forced_shutdown'] = True
    with open(config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    # Start the life timer
    life_timer.run_life_timer(allowed_runtime_minutes)

    # Call select audio 
    select_status, select_message, path_to_dive_csv = select_audio.main()
    print(f"\npc: Select Audio Status: {select_status}, Message: {select_message}\n")

    # Call transform and inference with output from select audio
    if select_status:
        trans_status, trans_message = transform_and_inference.process_audio_and_inference(path_to_dive_csv, model_path, confidence_threshold)
        print(f"\npc: Transform and Inference Status: {trans_status}, Message: {trans_message}")


    # Set the forced shutdown flag back to False
    config['forced_shutdown'] = False
    with open(config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
        life_timer.stop_timer_event.set() # Stop the life timer
        if life_timer.stop_timer_event.is_set():
            print("\npc: Timer cancelled by process_control.py")

        shutdown_pi.shutdown('Mission completed successfully', config_GPIO_on_time, config_GPIO_pin_num)
        sys.exit(0)

if __name__ == "__main__":
    main()
