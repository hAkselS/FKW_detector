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
import audio_transform.dat_to_wav as dat_to_wav
import image_inference.inference as inference



# second_status, message = inference.perform_inference(files, output_directory=project_root + '/images')

# print(f"\nImage inference status: {second_status}, Message: {message}")

# Test:
# # convert WISPR .dat file to .wav
input_file = "scratch_materials/WISPR_240930_000405.dat"
output_file_path = "scratch_materials/"

status, message, output_file_path = dat_to_wav.convert_dat_to_wav(input_file, output_file_path)
print(f"\nDAT to WAV conversion status: {status}, Message: {message}, Output File: {output_file_path}")

status, message, files = audio_to_spectro.process_audio_to_spectrograms(output_file_path, '/Users/akselsloan/FKW_detector/images')
print(f"\nAudio transform status: {status}, Message: {message}, Files: {files}")
