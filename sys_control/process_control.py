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
import audio_transform.audio_to_spectro as audio_to_spectro
import audio_transform.dat_to_wav as dat_to_wav
import image_inference.inference as inference


# TODO: Check to make sure the forced shutdown flag is false 

# TODO: Set the forced shutdown flag true

# TODO: Create a timer the shuts down the pi after X minutes 

# TODO: call select audio 


# TODO: call transform and inference with output from select audio

# TODO: Set the forced shutdown flag false