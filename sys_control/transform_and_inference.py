'''
File:   system_control/transform_and_inference.py

Spec:   Repeatedly transfrom then infer on selected wave files. 
        This script uses a CSV to know which files to process.
        Upon processing each file, it will update the CSV with a 
        true / false flag for turning the file into a spectrogram
        and for inferencing the spectrogram.

Usage:  python3 system_control/transform_and_inference.py 

ID:     ti
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

###################################################################
# CONFIGURATION DEFAULTS
wave_output_dir = project_root + '/data_products/wave_outputs'
spectro_output_dir = project_root + '/data_products/spectro_outputs'
inference_output_dir = project_root + '/data_products/inference_outputs'
###################################################################


def process_audio_and_inference(input_csv): # Input CSV with file path
    # Get the CSV file name for later use
    csv_filename = os.path.splitext(os.path.basename(input_csv))[0]

    # Read the CSV file
    spectrograms_processed = 0
    df = pandas.read_csv(input_csv)
    for file_name in df.loc[df["selected_for_sampling"] == True, "file_name"]:
        # Convert dat to wav 
        dat_status, dat_message, dat_out_path = dat_to_wav.convert_dat_to_wav(file_name, wave_output_dir)

        # Convert wav to spectrogram
        if dat_status:
            # Log that dat to wave is complete
            df.loc[df["file_name"] == file_name, "dat_to_wave"] = True 

            # Convert wave to spectrogram
            spectro_status, spectro_message, spectro_files_list = audio_to_spectro.process_audio_to_spectrograms(dat_out_path, spectro_output_dir)
        
            # Run inference on spectrograms
            if spectro_status:
                # Log that wave to spectrogram is complete
                df.loc[df["file_name"] == file_name, "wave_to_spectro"] = True

                inference_status, inference_message = inference.perform_inference(spectro_files_list, inference_output_dir+ f'/{csv_filename}_detections.csv')

                if inference_status:
                    # Log that inference is complete
                    df.loc[df["file_name"] == file_name, "image_analyzed"] = True
                    spectrograms_processed += 2 # Two spectrograms processed every time the inference status returns True

        df.to_csv(input_csv, index=False)  # Write updated CSV to memory
        print(f'ti: transform_and_inference: processed {file_name}')
    print(f'ti: transform_and_inference: analyzed {spectrograms_processed} spectrograms')
    
    # TODO: make this message more meaningful 
    return True, f"Processed {spectrograms_processed} spectrograms"

def main():
    
    print('ti: this program should not be run directly')
    
    # TODO: Remove block, this is only used for testing 
    input_csv = '/Users/akselsloan/FKW_detector/logs/dive_logs/240930_000003-241001_000449.csv'
    # Process the files in the CSV given to us by Select Audio
    process_audio_and_inference(input_csv)


if __name__ == "__main__":
    main()