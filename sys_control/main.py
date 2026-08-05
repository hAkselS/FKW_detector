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
        sys.exit(1)

    # Grab the allowed runtime minutes for the life timer
    try:    
        allowed_runtime_minutes = config['allowed_runtime_minutes']
        if not (1 <= allowed_runtime_minutes <= 120):
            print(f"pc: ✗ CRITICAL ERROR: Invalid allowed_runtime_minutes value: {allowed_runtime_minutes}. Must be between 1 and 120.")
            # TODO: make sure that this exits and that pi lager knows about it. 
            sys.exit(1)

    except Exception as e:
        print(f"pc: ✗ CRITICAL ERROR: Unexpected error reading allowed_runtime_minutes: {e}")
        sys.exit(1)

    # Grab the model path and confidence threshold
    try:
        model_path = config['model_path']
        if not os.path.exists(model_path):
            print(f"pc: ✗ CRITICAL ERROR: Model file not found: {model_path}")
            sys.exit(1)

        confidence_threshold = config['confidence_threshold']

        if not (0.0 <= confidence_threshold <= 1.0):
            print(f"pc: ✗ CRITICAL ERROR: Invalid confidence_threshold value: {confidence_threshold}. Must be between 0.0 and 1.0.")
            sys.exit(1)

    except Exception as e:
        print(f"pc: ✗ CRITICAL ERROR: Unexpected error reading model_path or confidence_threshold: {e}")
        sys.exit(1)

    # TODO: consider adding a timer here.
        # This timer could simply report how long the detector ran for
        # or could shutdown the process after a certain amount of time

    # Call select audio 
    select_status, select_message, path_to_dive_csv = select_audio.main()
    print(f"\npc: Select Audio Status: {select_status}, Message: {select_message}\n")

    # Call transform and inference with output from select audio
    if select_status:
        trans_status, trans_message = transform_and_inference.process_audio_and_inference(path_to_dive_csv, model_path, confidence_threshold)
        print(f"\npc: Transform and Inference Status: {trans_status}, Message: {trans_message}")

        # Exit after the program has completed successfully
        # TODO: Ensure that no exception is needed here or add one to indicate a failure
        sys.exit(0)

if __name__ == "__main__":
    main()
