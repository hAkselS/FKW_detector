'''
File:   sys_control/process_control.py

Spec:   Handle all processes and their orderings. 

Usage:  python3 sys_control/process_control.py 

ID:     pc 
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

# print(f'Config file = {config_file}')

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

# Start the life timer
# TODO: GET AND VET THE COUNT DOWN TIME HERE IN PROCESS CONTROL, PASS THE TIME AND START THE TIMING
# life_timer.run_life_timer()


# Set the forced shutdown flag true
config['forced_shutdown'] = True
with open(config_file, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
# TODO: Create a timer the shuts down the pi after X minutes 

# Call select audio 
select_status, select_message, path_to_dive_csv = select_audio.main()
print(f"pc: Select Audio Status: {select_status}, Message: {select_message}\n")

# Call transform and inference with output from select audio
if select_status:
    trans_status, trans_message = transform_and_inference.process_audio_and_inference(path_to_dive_csv)
    print(f"pc: Transform and Inference Status: {trans_status}, Message: {trans_message}")


# Set the forced shutdown flag back to False
config['forced_shutdown'] = False
with open(config_file, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
    life_timer.stop_timer_event.set() # Stop the life timer
    shutdown_pi.shutdown('Mission completed successfully')
    sys.exit(0)