'''
File:   system_control/transform_and_inference.py

Spec:   Repeatedly transfrom then infer on selected wave files. 
        This script uses a CSV to know which files to process.
        Upon processing each file, it will update the CSV with a 
        true / false flag for turning the file into a spectrogram
        and for inferencing the spectrogram.

Usage:  python3 system_control/transform_and_inference.py <path_to_csv> 
'''

import sys
import os

# Add project root to sys.path so audio_transform can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import audio_transorm.audio_to_spectro as audio_to_spectro

status, number, files = audio_to_spectro.process_audio_to_spectrograms('/Users/akselsloan/FKW_detector/scratch_materials/1706_20170715_220250_436.wav', project_root + '/images')

print(f"Status: {status}, Number of files: {number}, Files: {files}")
print(f"Files are of type: {type(files)}")