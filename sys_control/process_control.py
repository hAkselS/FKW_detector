'''
File:   sys_control/process_control.py

Spec:   Handle all processes and their orderings. 

Usage:  python3 sys_control/process_control.py 
'''

import sys
import os

# Add project root to sys.path so audio_transform can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import sys_control.transform_and_inference as transform_and_inference
import sys_control.select_audio as select_audio



# TODO: Check to make sure the forced shutdown flag is false 

# TODO: Set the forced shutdown flag true

# TODO: Create a timer the shuts down the pi after X minutes 

# Call select audio 
select_status, select_message, path_to_dive_csv = select_audio.main()
# TODO: don't call transform if there are no new files. 

# Call transform and inference with output from select audio
trans_status, trans_message = transform_and_inference.process_audio_and_inference(path_to_dive_csv)
print(f"Transform and Inference Status: {trans_status}, Message: {trans_message}")

# TODO: Set the forced shutdown flag false