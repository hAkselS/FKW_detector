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



###################################################################
# CONFIGURATION DEFAULTS
config_file = project_root + '/config/config.yaml'
###################################################################

# print(f'Config file = {config_file}')

# TODO: WRITING TO YAML ERASES ALL ITS COMMENTS
# # Open the config file 
# try:
#     with open(config_file, 'r') as file:
#         config = yaml.safe_load(file)
# except Exception as e:
#     print(f"✗ CRITICAL ERROR: Unexpected error loading config: {e}")
#     # TODO: trigger shutdown exit

# # Check forced shutdown status: True = bad (system failed) False = good (system operated nominally)
# if config['forced_shutdown'] == True:
#     print("✗ CRITICAL ERROR: System failed")
#     # TODO: trigger shutdown exit

# # Set the forced shutdown flag true
# config['forced_shutdown'] = True
# with open(config_file, 'w') as file:
#     yaml.dump(config, file, default_flow_style=False)
# TODO: Create a timer the shuts down the pi after X minutes 

# Call select audio 
select_status, select_message, path_to_dive_csv = select_audio.main()
print(f"pc: Select Audio Status: {select_status}, Message: {select_message}\n")

# Call transform and inference with output from select audio
if select_status:
    trans_status, trans_message = transform_and_inference.process_audio_and_inference(path_to_dive_csv)
    print(f"pc: Transform and Inference Status: {trans_status}, Message: {trans_message}")


# Set the forced shutdown flag false
# config['forced_shutdown'] = True
# with open(config_file, 'w') as file:
#     yaml.dump(config, file, default_flow_style=False)