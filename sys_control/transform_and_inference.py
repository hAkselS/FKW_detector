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
import pandas

# Add project root to sys.path so audio_transform can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import audio_transform.audio_to_spectro as audio_to_spectro
import audio_transform.dat_to_wav as dat_to_wav
import image_inference.inference as inference

# TODO: get input CSV from process control
input_csv = '/Users/akselsloan/FKW_detector/logs/dive_logs/240930_000003-241001_000449.csv'
wave_output_dir = '/Users/akselsloan/FKW_detector/scratch_materials/wave_outputs'
spectro_output_dir = '/Users/akselsloan/FKW_detector/scratch_materials/spectro_outputs'
inference_output_dir = '/Users/akselsloan/FKW_detector/scratch_materials/inference_outputs'


# Read the CSV file
df = pandas.read_csv(input_csv)
for file_name in df.loc[df["selected_for_sampling"] == True, "file_name"]:
    # DEBUG
    # print(f'\nProcessing file: {file_name}')

    # Convert dat to wav 
    dat_status, dat_message, dat_out_path = dat_to_wav.convert_dat_to_wav(file_name, wave_output_dir)
    # DEBUG 
    # print(f'{dat_message}\n')

    # Convert wav to spectrogram
    if dat_status:
        spectro_status, spectro_message, spectro_files_list = audio_to_spectro.process_audio_to_spectrograms(dat_out_path, spectro_output_dir)
        # DEBUG
        # print(f'{spectro_message}\n')

        # Run inference on spectrograms
        if spectro_status:
            inference_status, inference_message = inference.perform_inference(spectro_files_list, inference_output_dir)
            # DEBUG
            print(f'{inference_message}\n')
