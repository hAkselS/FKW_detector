'''
File:   system_control/transform_and_inference.py

Spec:   Repeatedly transfrom then infer on selected wave files. 
        This script uses a CSV to know which files to process.
        Upon processing each file, it will update the CSV with a 
        true / false flag for turning the file into a spectrogram
        and for inferencing the spectrogram.

Usage:  python3 system_control/transform_and_inference.py <path_to_csv> 
'''