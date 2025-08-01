'''
File:   system_control/transform_and_inference.py

Spec:   Repeatedly transfrom then infer on selected wave files. 
        This script uses a CSV to know which files to process.
        Upon processing each file, it will update the CSV with a 
        true / false flag for turning the file into a spectrogram
        and for inferencing the spectrogram.

Usage:  python3 system_control/transform_and_inference.py 
'''

import sys
import os

# Add project root to sys.path so audio_transform can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import audio_transform.audio_to_spectro as audio_to_spectro
import image_inference.inference as inference

status, files = audio_to_spectro.process_audio_to_spectrograms('/Users/akselsloan/FKW_detector/scratch_materials/1706_20170709_034442_942.wav', project_root + '/images')

print(f"\nAudio transform status: {status}, Files: {files}")


second_status, message = inference.perform_inference(files, output_directory=project_root + '/images')

print(f"\nImage inference status: {second_status}, Message: {message}")